"""Effective memory length estimation."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.tma.acf_pacf import acf_confidence_band, compute_acf


def memory_length_metrics(
    series: np.ndarray,
    max_lag: int,
) -> dict[str, float]:
    """Estimate decorrelation time, half-life, and integrated autocorrelation."""
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 4:
        return {
            "first_insignificant_lag": float("nan"),
            "half_life_lag": float("nan"),
            "decorrelation_lag_0p1": float("nan"),
            "integrated_autocorr_time": float("nan"),
        }

    _, acf_vals = compute_acf(series, max_lag)
    bound = acf_confidence_band(n)

    first_insig = float("nan")
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) < bound:
            first_insig = float(lag)
            break

    half_life = float("nan")
    for lag in range(1, len(acf_vals)):
        if acf_vals[lag] < 0.5:
            half_life = float(lag)
            break

    decorr = float("nan")
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) < 0.1:
            decorr = float(lag)
            break

    # Integrated autocorrelation time (truncated sum)
    positive_lags = acf_vals[1:]
    tau = 1.0 + 2.0 * float(np.nansum(positive_lags))

    return {
        "first_insignificant_lag": first_insig,
        "half_life_lag": half_life,
        "decorrelation_lag_0p1": decorr,
        "integrated_autocorr_time": tau,
    }


def cohort_memory_distribution(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
) -> dict[str, Any]:
    """Distribution of memory-length metrics across containers."""
    rows = []
    for cid, series in series_by_container.items():
        metrics = memory_length_metrics(series, max_lag)
        metrics["container_id"] = cid
        rows.append(metrics)

    def _summ(key: str) -> dict[str, float]:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "iqr": float("nan"),
            }
        q25, q75 = np.percentile(vals, [25, 75])
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
        }

    keys = (
        "first_insignificant_lag",
        "half_life_lag",
        "decorrelation_lag_0p1",
        "integrated_autocorr_time",
    )
    return {
        "per_container": rows,
        "distribution": {k: _summ(k) for k in keys},
    }
