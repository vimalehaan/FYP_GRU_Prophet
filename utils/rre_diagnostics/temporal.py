"""Temporal structure diagnostics — reuses TMA ACF/PACF utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from utils.tma.acf_pacf import (
    build_acf_matrix,
    build_pacf_matrix,
    cohort_acf_summary,
)
from utils.tma.memory_length import cohort_memory_distribution


def ljung_box_report(
    values: np.ndarray,
    lags: tuple[int, ...] = (10, 20, 48, 96),
) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if len(values) < max(lags) + 5:
        return {"lags": list(lags), "lb_stat": [], "p_value": []}
    result = acorr_ljungbox(values, lags=list(lags), return_df=True)
    return {
        "lags": list(lags),
        "lb_stat": result["lb_stat"].astype(float).tolist(),
        "p_value": result["lb_pvalue"].astype(float).tolist(),
    }


def decorrelation_lag(acf: np.ndarray, threshold: float = 0.1) -> int:
    """First lag where |ACF| drops below threshold (excluding lag 0)."""
    if len(acf) <= 1:
        return 0
    for lag in range(1, len(acf)):
        if abs(float(acf[lag])) < threshold:
            return lag
    return len(acf) - 1


def temporal_profile(
    series_by_container: dict[str, np.ndarray],
    max_lag: int = 96,
) -> dict[str, Any]:
    """Cohort temporal diagnostics."""
    acf_matrix, lags, _ = build_acf_matrix(series_by_container, max_lag)
    acf_summary = cohort_acf_summary(acf_matrix, lags)
    pacf_matrix, pacf_lags, _ = build_pacf_matrix(series_by_container, max_lag)
    pacf_mean = np.nanmean(pacf_matrix, axis=0) if len(pacf_matrix) else np.array([])

    memory = cohort_memory_distribution(series_by_container, max_lag=max_lag)
    mem_dist = memory.get("distribution", {})

    pooled = np.concatenate(
        [s[np.isfinite(s)] for s in series_by_container.values() if len(s) > 0]
    )
    lb = ljung_box_report(pooled)

    acf_mean = np.array(acf_summary["mean"], dtype=float)

    return {
        "acf_mean": acf_mean.tolist(),
        "acf_lags": lags.tolist(),
        "pacf_mean": pacf_mean.tolist(),
        "pacf_lags": pacf_lags.tolist(),
        "acf_ci_lower": acf_summary["ci_lo"].tolist(),
        "acf_ci_upper": acf_summary["ci_hi"].tolist(),
        "memory_length_median": mem_dist.get("decorrelation_lag_0p1", {}).get(
            "median", float("nan")
        ),
        "memory_length_mean": mem_dist.get("decorrelation_lag_0p1", {}).get(
            "mean", float("nan")
        ),
        "integrated_autocorr_time_median": mem_dist.get(
            "integrated_autocorr_time", {}
        ).get("median", float("nan")),
        "decorrelation_lag_0_1": decorrelation_lag(acf_mean, 0.1),
        "decorrelation_lag_0_05": decorrelation_lag(acf_mean, 0.05),
        "mean_abs_acf_lag1_96": float(
            np.nanmean(np.abs(acf_mean[1 : max_lag + 1]))
        ),
        "acf_lag1": float(acf_mean[1]) if len(acf_mean) > 1 else float("nan"),
        "ljung_box": lb,
    }
