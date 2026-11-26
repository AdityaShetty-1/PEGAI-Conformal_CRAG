"""Evaluation metrics, matching the proposal's metric list exactly: accuracy,
hallucination rate, abstention rate, coverage at fixed risk, latency, and
web fallback calls -- each with a bootstrap confidence interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .router import RouteResult


def results_to_frame(results: list[RouteResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, ci: float = 0.90) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots[i] = sample.mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return float(values.mean()), float(np.quantile(boots, lo_q)), float(np.quantile(boots, hi_q))


def summarize_group(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """One group = one (config, noise_level[, grader_variant]) cell, pooled
    across all seeds and test queries."""
    n_total = len(df)
    answered = df[~df["abstained"]]
    n_answered = len(answered)

    accuracy_vals = df["correct"].astype(float).values
    abstain_vals = df["abstained"].astype(float).values
    halluc_vals = (~df["abstained"] & ~df["correct"]).astype(float).values
    risk_vals = (~answered["correct"]).astype(float).values if n_answered else np.array([])
    latency_vals = df["latency_s"].astype(float).values
    web_call_vals = df["web_calls"].astype(float).values

    acc_mean, acc_lo, acc_hi = bootstrap_ci(accuracy_vals, n_boot, seed)
    abst_mean, abst_lo, abst_hi = bootstrap_ci(abstain_vals, n_boot, seed + 1)
    hall_mean, hall_lo, hall_hi = bootstrap_ci(halluc_vals, n_boot, seed + 2)
    risk_mean, risk_lo, risk_hi = bootstrap_ci(risk_vals, n_boot, seed + 3) if n_answered else (float("nan"),) * 3
    lat_mean, lat_lo, lat_hi = bootstrap_ci(latency_vals, n_boot, seed + 4)
    web_mean, web_lo, web_hi = bootstrap_ci(web_call_vals, n_boot, seed + 5)

    return {
        "n_total": n_total,
        "n_answered": n_answered,
        "coverage": n_answered / n_total if n_total else float("nan"),
        "accuracy_mean": acc_mean, "accuracy_lo": acc_lo, "accuracy_hi": acc_hi,
        "abstention_rate_mean": abst_mean, "abstention_rate_lo": abst_lo, "abstention_rate_hi": abst_hi,
        "hallucination_rate_mean": hall_mean, "hallucination_rate_lo": hall_lo, "hallucination_rate_hi": hall_hi,
        "risk_given_answered_mean": risk_mean, "risk_given_answered_lo": risk_lo, "risk_given_answered_hi": risk_hi,
        "latency_s_mean": lat_mean, "latency_s_lo": lat_lo, "latency_s_hi": lat_hi,
        "web_calls_mean": web_mean, "web_calls_lo": web_lo, "web_calls_hi": web_hi,
    }


def summarize_by(df: pd.DataFrame, group_cols: list[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row.update(summarize_group(g, n_boot=n_boot, seed=seed))
        rows.append(row)
    return pd.DataFrame(rows)
