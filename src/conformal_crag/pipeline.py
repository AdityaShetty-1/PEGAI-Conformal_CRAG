"""End-to-end experiment orchestration: builds the corpus/retriever, splits
calibration/test, calibrates each config's threshold, runs every query
through the router, and returns a flat list of RouteResult rows ready for
metrics.py.
"""
from __future__ import annotations

from .conformal import calibrate_threshold
from .data import Dataset, Question, load_dataset, split_calibration_test  # noqa: F401 (Question used in type hints)
from .grader import SufficiencyGrader
from .llm import ExtractiveReader, is_correct
from .noise import NoiseInjector
from .retriever import Retriever
from .router import ConformalCRAGRouter, PipelineConfig, RouteResult
from .web_fallback import WebFallback


def build_components(dataset: Dataset, seed: int, grader_variant: str, closed_domain: bool = False):
    retriever = Retriever(dataset.documents)
    clean_retriever = Retriever(dataset.documents)  # web fallback's "independent" clean index
    noise_injector = NoiseInjector(dataset.documents, seed=seed)
    grader = SufficiencyGrader(variant=grader_variant, seed=seed)
    reader = ExtractiveReader()
    web_fallback = None if closed_domain else WebFallback(clean_retriever)
    router = ConformalCRAGRouter(
        retriever=retriever, grader=grader, reader=reader,
        noise_injector=noise_injector, web_fallback=web_fallback,
    )
    return router


def stage_trace_for_calibration(router: ConformalCRAGRouter, question: Question, config: PipelineConfig, noise_level: float, noise_mode: str) -> list[tuple[float, bool]]:
    """Returns [(score, correct), ...] in the *exact order* router.run()
    would attempt stages (generate -> rewrite -> web), unconditionally
    running every stage the config allows regardless of score. A candidate
    threshold tau is then applied post-hoc by taking the first entry with
    score >= tau -- this must mirror router.run()'s own stopping rule
    exactly, or the calibration set's (score, correctness) pairs describe a
    different decision rule than the one actually deployed at test time,
    which silently breaks the risk guarantee.
    """
    from .llm import STOPWORDS, tokenize
    from .router import _merge_docs

    trace: list[tuple[float, bool]] = []
    retrieved = router._retrieve(question.question, config.k, noise_level, noise_mode)
    s = router.grader.score(question.question, retrieved)
    answer, _ = router.reader.answer(question.question, retrieved)
    trace.append((s, is_correct(answer, question.answer_aliases)))

    if config.allow_rewrite:
        exclude = set(tokenize(question.question)) - STOPWORDS
        entities = router.reader.extract_rewrite_entities(retrieved, exclude)
        if entities:
            new_query = question.question + " " + entities[0]
            retrieved2 = router._retrieve(new_query, config.k, noise_level, noise_mode)
            retrieved = _merge_docs(retrieved, retrieved2)
            s = router.grader.score(new_query, retrieved)
            answer, _ = router.reader.answer(new_query, retrieved)
            trace.append((s, is_correct(answer, question.answer_aliases)))

    if config.allow_web and router.web_fallback is not None:
        web_docs = router.web_fallback.search(question.question, k=2)
        retrieved = _merge_docs(retrieved, web_docs)
        s = router.grader.score(question.question, retrieved)
        answer, _ = router.reader.answer(question.question, retrieved)
        trace.append((s, is_correct(answer, question.answer_aliases)))

    return trace


def run_config_for_seed(
    config: PipelineConfig,
    seed: int,
    grader_variant: str = "autorater_sim",
    noise_level: float = 0.0,
    noise_mode: str = "mixed",
    alpha: float = 0.15,
    delta: float = 0.1,
    closed_domain: bool = False,
) -> list[RouteResult]:
    dataset = load_dataset(seed=seed)
    calib_qs, test_qs = split_calibration_test(dataset, calib_frac=0.5, seed=seed)
    router = build_components(dataset, seed, grader_variant, closed_domain=closed_domain)

    if config.use_calibrated_threshold:
        traces = [stage_trace_for_calibration(router, q, config, noise_level, noise_mode) for q in calib_qs]
        threshold = calibrate_threshold(traces, alpha=alpha, delta=delta)
    else:
        threshold = config.fixed_threshold

    results = [router.run(q, config, threshold, noise_level=noise_level, noise_mode=noise_mode) for q in test_qs]
    return results


def run_experiment_grid(
    configs: list[PipelineConfig],
    seeds: list[int],
    noise_levels: list[float],
    grader_variant: str = "autorater_sim",
    alpha: float = 0.15,
    delta: float = 0.1,
    closed_domain: bool = False,
) -> list[RouteResult]:
    all_results: list[RouteResult] = []
    for config in configs:
        for noise_level in noise_levels:
            for seed in seeds:
                all_results.extend(
                    run_config_for_seed(
                        config, seed, grader_variant=grader_variant, noise_level=noise_level,
                        alpha=alpha, delta=delta, closed_domain=closed_domain,
                    )
                )
    return all_results
