"""Conformal risk control for threshold selection.

Given a calibration set of (sufficiency score, correctness) pairs, picks the
smallest score threshold tau such that answering every calibration example
with score >= tau has a (1 - delta)-confidence upper bound on its error rate
at or below alpha. Smaller tau -> more coverage (fewer abstentions), so the
smallest admissible tau maximizes coverage subject to the risk guarantee --
this is the finite-sample guarantee CRAG's hand-set cutoffs don't have.

Reference: Clopper-Pearson exact upper confidence bound on a Binomial error
rate (equivalently split conformal risk control, Bates et al. 2021 /
Angelopoulos et al. 2022, specialized to a monotone binary loss).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def risk_upper_confidence_bound(n_wrong: int, n_total: int, delta: float = 0.1) -> float:
    if n_total == 0:
        return 0.0
    if n_wrong >= n_total:
        return 1.0
    return float(stats.beta.ppf(1 - delta, n_wrong + 1, n_total - n_wrong))


def _apply_stopping_rule(traces: list[list[tuple[float, bool]]], tau: float) -> tuple[int, int]:
    """For each query's stage trace, take the first stage whose score clears
    tau (the same early-exit rule router.run() uses); return (n_answered,
    n_wrong) over the calibration set at this tau."""
    n_total = 0
    n_wrong = 0
    for trace in traces:
        for score, correct in trace:
            if score >= tau:
                n_total += 1
                if not correct:
                    n_wrong += 1
                break
    return n_total, n_wrong


def calibrate_threshold(
    traces: list[list[tuple[float, bool]]],
    alpha: float = 0.15,
    delta: float = 0.1,
    min_coverage_n: int = 3,
) -> float:
    """Returns the smallest tau satisfying the risk bound under the router's
    actual early-exit stopping rule, or 1.0 (abstain-all) if none does.

    `traces[i]` is query i's ordered list of (stage_score, stage_correct)
    pairs, exactly as pipeline.stage_trace_for_calibration() produces --
    applying tau to the *max* score reached, or to a single (score, correct)
    pair unrelated to which stage would actually fire at test time, silently
    describes a different decision rule than the deployed one and breaks the
    guarantee; see stage_trace_for_calibration()'s docstring.

    The threshold is *selected* by scanning many candidate values against
    the same calibration set, which is a multiple-comparisons problem: a
    single-shot Clopper-Pearson bound at level delta, applied naively to
    whichever tau happens to look best, understates the true risk. This
    follows the Learn-Then-Test fixed-sequence correction (Angelopoulos et
    al., 2021): delta is Bonferroni-corrected by the number of candidates
    actually tested, so the final guarantee holds simultaneously for every
    candidate considered, not just the one that got picked.
    """
    # Sentinel for "abstain on everything": scores are clipped to [0, 1], so a
    # fallback of exactly 1.0 would still admit any score that saturated at
    # the clip boundary. Use a threshold strictly above the valid range.
    abstain_all = float("inf")
    # A fixed, data-independent grid rather than every unique observed score:
    # Bonferroni-correcting delta by the number of candidates tested makes
    # each additional candidate more expensive, so scanning ~100+ distinct
    # float values (one per query) would make the guarantee needlessly
    # conservative for no real gain in threshold precision.
    grid = [round(0.05 * i, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00
    tested = [tau for tau in grid if _apply_stopping_rule(traces, tau)[0] >= min_coverage_n]
    if not tested:
        return abstain_all
    delta_corrected = delta / len(tested)
    for tau in tested:
        n_total, n_wrong = _apply_stopping_rule(traces, tau)
        ucb = risk_upper_confidence_bound(n_wrong, n_total, delta_corrected)
        if ucb <= alpha:
            return float(tau)
    return abstain_all


def empirical_risk(scores: np.ndarray, correct: np.ndarray, tau: float) -> tuple[float, int]:
    """Observed (uncorrected) error rate and count of answered examples at
    threshold tau -- useful for reporting what a fixed/uncalibrated cutoff
    like CRAG's actually achieves, with no finite-sample guarantee attached."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    mask = scores >= tau
    n_total = int(mask.sum())
    if n_total == 0:
        return 0.0, 0
    n_wrong = int((~correct[mask]).sum())
    return n_wrong / n_total, n_total
