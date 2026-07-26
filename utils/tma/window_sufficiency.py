"""Window sufficiency and cumulative autocorrelation capture."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.tma.acf_pacf import compute_acf


def cumulative_acf_capture(
    series: np.ndarray,
    max_lag: int,
    windows: tuple[int, ...],
) -> dict[str, float]:
    """
    Fraction of cumulative squared ACF (lags 1..max) captured within each window.
    """
    series = series[np.isfinite(series)]
    _, acf_vals = compute_acf(series, max_lag)
    if len(acf_vals) <= 1:
        return {f"pct_capture_within_{w}": float("nan") for w in windows}

    sq = acf_vals[1:] ** 2
    total = float(np.nansum(sq))
    if total < 1e-12:
        return {f"pct_capture_within_{w}": float("nan") for w in windows}

    out: dict[str, float] = {}
    for w in windows:
        end = min(w, len(sq))
        captured = float(np.nansum(sq[:end]))
        out[f"pct_capture_within_{w}"] = captured / total
    return out


def window_sufficiency_report(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
    windows: tuple[int, ...],
) -> dict[str, Any]:
    """Cohort summary of cumulative ACF capture by window."""
    per_container = []
    for cid, series in series_by_container.items():
        row = cumulative_acf_capture(series, max_lag, windows)
        row["container_id"] = cid
        per_container.append(row)

    cohort: dict[str, Any] = {"windows": {}}
    for w in windows:
        key = f"pct_capture_within_{w}"
        vals = np.array([r[key] for r in per_container], dtype=float)
        vals = vals[np.isfinite(vals)]
        cohort["windows"][str(w)] = {
            "mean_pct": float(np.mean(vals)) if len(vals) else float("nan"),
            "median_pct": float(np.median(vals)) if len(vals) else float("nan"),
            "q25_pct": float(np.percentile(vals, 25)) if len(vals) else float("nan"),
            "q75_pct": float(np.percentile(vals, 75)) if len(vals) else float("nan"),
        }
    return {"per_container": per_container, "cohort": cohort}


def hypothetical_window_memory(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
    candidate_windows: tuple[int, ...],
) -> dict[str, Any]:
    """
    Quantify available temporal memory (mean |ACF|) within candidate windows.

    Does NOT claim forecasting performance improvements.
    """
    report: dict[str, Any] = {"windows": {}}
    for w in candidate_windows:
        vals = []
        for series in series_by_container.values():
            _, acf_vals = compute_acf(series, max_lag)
            end = min(w, len(acf_vals) - 1)
            if end >= 1:
                vals.append(float(np.mean(np.abs(acf_vals[1 : end + 1]))))
        arr = np.array(vals, dtype=float)
        report["windows"][str(w)] = {
            "mean_abs_acf_within_window": float(np.mean(arr)) if len(arr) else float(
                "nan"
            ),
            "median_abs_acf_within_window": float(np.median(arr))
            if len(arr)
            else float("nan"),
        }
    return report


def memory_beyond_first_day(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
    day_steps: int = 96,
) -> dict[str, Any]:
    """Compare mean |ACF| within first day vs beyond first day."""
    within = []
    beyond = []
    for series in series_by_container.values():
        _, acf_vals = compute_acf(series, max_lag)
        end_day = min(day_steps, len(acf_vals) - 1)
        if end_day >= 1:
            within.append(float(np.mean(np.abs(acf_vals[1 : end_day + 1]))))
        if len(acf_vals) > day_steps + 1:
            beyond.append(float(np.mean(np.abs(acf_vals[day_steps + 1 :]))))

    within_arr = np.array(within, dtype=float)
    beyond_arr = np.array(beyond, dtype=float)
    return {
        "mean_abs_acf_lags_1_96": float(np.mean(within_arr))
        if len(within_arr)
        else float("nan"),
        "mean_abs_acf_beyond_96": float(np.mean(beyond_arr))
        if len(beyond_arr)
        else float("nan"),
        "within_per_container": within,
        "beyond_per_container": beyond,
    }
