"""The four-action router: generate / rewrite+re-retrieve / web fallback /
abstain, parameterized so the same engine runs Conformal-CRAG and every
baseline in baselines.py by toggling which actions are enabled and whether
the threshold is conformally calibrated or hand-set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .data import Document, Question
from .grader import SufficiencyGrader
from .llm import ExtractiveReader, STOPWORDS, is_correct, tokenize
from .noise import NoiseInjector
from .retriever import Retriever
from .web_fallback import WebFallback


@dataclass
class PipelineConfig:
    name: str
    allow_rewrite: bool = True
    allow_web: bool = True
    allow_abstain: bool = True
    use_calibrated_threshold: bool = True
    fixed_threshold: float = 0.5  # used only when use_calibrated_threshold is False
    k: int = 3


@dataclass
class RouteResult:
    qid: str
    source: str
    config_name: str
    action_trace: list[str]
    final_action: str
    answer: str
    abstained: bool
    correct: bool
    sufficiency_score: float
    web_calls: int
    latency_s: float
    noise_level: float


def _merge_docs(*groups: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for g in groups:
        for d in g:
            base_id = d.doc_id.split("#")[0]
            if base_id in seen:
                continue
            seen.add(base_id)
            out.append(d)
    return out


@dataclass
class ConformalCRAGRouter:
    retriever: Retriever
    grader: SufficiencyGrader
    reader: ExtractiveReader = field(default_factory=ExtractiveReader)
    noise_injector: NoiseInjector | None = None
    web_fallback: WebFallback | None = None  # None => closed-domain mode

    def run(self, question: Question, config: PipelineConfig, threshold: float, noise_level: float = 0.0, noise_mode: str = "mixed") -> RouteResult:
        t0 = time.perf_counter()
        query = question.question
        trace: list[str] = []
        web_calls = 0

        retrieved = self._retrieve(query, config.k, noise_level, noise_mode)
        s = self.grader.score(query, retrieved)

        answer, action, s = self._maybe_generate(query, retrieved, s, threshold, trace, "generate")
        if answer is not None:
            return self._finalize(question, answer, trace, web_calls, s, noise_level, config, t0)

        if config.allow_rewrite:
            trace.append("rewrite")
            exclude = set(tokenize(query)) - STOPWORDS
            entities = self.reader.extract_rewrite_entities(retrieved, exclude)
            if entities:
                new_query = query + " " + entities[0]
                retrieved2 = self._retrieve(new_query, config.k, noise_level, noise_mode)
                retrieved = _merge_docs(retrieved, retrieved2)
                s = self.grader.score(new_query, retrieved)
                answer, _, s = self._maybe_generate(new_query, retrieved, s, threshold, trace, "generate")
                if answer is not None:
                    return self._finalize(question, answer, trace, web_calls, s, noise_level, config, t0)

        if config.allow_web and self.web_fallback is not None:
            trace.append("web_fallback")
            web_calls += 1
            web_docs = self.web_fallback.search(query, k=2)
            retrieved = _merge_docs(retrieved, web_docs)
            s = self.grader.score(query, retrieved)
            answer, _, s = self._maybe_generate(query, retrieved, s, threshold, trace, "generate")
            if answer is not None:
                return self._finalize(question, answer, trace, web_calls, s, noise_level, config, t0)

        if config.allow_abstain:
            trace.append("abstain")
            return self._finalize(question, None, trace, web_calls, s, noise_level, config, t0)

        # Abstain disabled (e.g. standard RAG / CRAG-no-abstain baselines):
        # forced to answer with whatever context survived.
        trace.append("generate_forced")
        final_answer, _ = self.reader.answer(query, retrieved)
        return self._finalize(question, final_answer, trace, web_calls, s, noise_level, config, t0)

    def _retrieve(self, query: str, k: int, noise_level: float, noise_mode: str) -> list[Document]:
        retrieved = self.retriever.retrieve(query, k)
        if self.noise_injector is not None and noise_level > 0:
            retrieved = self.noise_injector.corrupt(retrieved, noise_level, noise_mode)
        return retrieved

    def _maybe_generate(self, query: str, retrieved: list[Document], s: float, threshold: float, trace: list[str], action_name: str):
        if s >= threshold:
            trace.append(action_name)
            answer, _ = self.reader.answer(query, retrieved)
            return answer, action_name, s
        return None, None, s

    def _finalize(self, question: Question, answer: str | None, trace: list[str], web_calls: int, s: float, noise_level: float, config: PipelineConfig, t0: float) -> RouteResult:
        abstained = answer is None
        correct = False if abstained else is_correct(answer, question.answer_aliases)
        return RouteResult(
            qid=question.qid,
            source=question.source,
            config_name=config.name,
            action_trace=trace,
            final_action=trace[-1] if trace else "none",
            answer=answer or "",
            abstained=abstained,
            correct=correct,
            sufficiency_score=s,
            web_calls=web_calls,
            latency_s=time.perf_counter() - t0,
            noise_level=noise_level,
        )
