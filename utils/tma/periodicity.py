"""Daily periodicity analysis at multiples of 96 lags."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.tma.acf_pacf import acf_at_lag


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(means, 100 * alpha)), float(
        np.percentile(means, 100 * (1.0 - alpha))
    )


def daily_periodicity_report(
    series_by_container: dict[str, np.ndarray],
    daily_lags: tuple[int, ...],
    max_lag: int,
    thresholds: tuple[float, ...],
    bootstrap_n: int,
    bootstrap_seed: int,
    ci_level: float,
) -> dict[str, Any]:
    """Summarize ACF at daily lags across the cohort."""
    report: dict[str, Any] = {"lags": {}}
    for lag in daily_lags:
        vals = np.array([
            acf_at_lag(series, lag, max_lag)
            for series in series_by_container.values()
        ])
        finite = vals[np.isfinite(vals)]
        ci_lo, ci_hi = bootstrap_ci(
            finite, bootstrap_n, bootstrap_seed + lag, ci_level
        )
        lag_entry: dict[str, Any] = {
            "lag": lag,
            "days": lag / 96.0,
            "mean_acf": float(np.mean(finite)) if len(finite) else float("nan"),
            "median_acf": float(np.median(finite)) if len(finite) else float("nan"),
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
            "n_containers": int(len(finite)),
        }
        for thr in thresholds:
            frac = float(np.mean(np.abs(finite) >= thr)) if len(finite) else float(
                "nan"
            )
            lag_entry[f"pct_abs_acf_ge_{thr}"] = frac
        report["lags"][str(lag)] = lag_entry
    return report
