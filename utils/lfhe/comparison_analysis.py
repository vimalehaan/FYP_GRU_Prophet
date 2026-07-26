"""Read-only LFHE comparison analytics — temporal alignment and container selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr


def lag_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Pearson r at integer lags in [-max_lag, +max_lag] (positive lag = pred leads)."""
    a = np.ravel(actual).astype(float)
    p = np.ravel(predicted).astype(float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    rows: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, pp = a[-lag:], p[: n + lag]
        elif lag > 0:
            aa, pp = a[: n - lag], p[lag:]
        else:
            aa, pp = a, p
        if len(aa) < 3 or np.std(aa) < 1e-12 or np.std(pp) < 1e-12:
            r = float("nan")
        else:
            r = float(pearsonr(aa, pp)[0])
        rows.append({"lag": lag, "pearson_r": r})
    return pd.DataFrame(rows)


def sign_agreement_fraction(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.ravel(actual)
    p = np.ravel(predicted)
    mask = (a != 0) | (p != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(p[mask])))


def turning_points(arr: np.ndarray) -> np.ndarray:
    x = np.ravel(arr).astype(float)
    if len(x) < 3:
        return np.array([], dtype=int)
    d = np.diff(x)
    signs = np.sign(d)
    idx: list[int] = []
    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
            idx.append(i)
    return np.array(idx, dtype=int)


def turning_point_agreement(actual: np.ndarray, predicted: np.ndarray, *, tolerance: int = 2) -> float:
    ta = turning_points(actual)
    tp = turning_points(predicted)
    if len(ta) == 0:
        return float("nan")
    matched = 0
    for t in ta:
        if len(tp) == 0:
            break
        if np.min(np.abs(tp - t)) <= tolerance:
            matched += 1
    return float(matched / len(ta))


def peak_timing_shifts(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    prominence: float | None = None,
) -> np.ndarray:
    a = np.ravel(actual).astype(float)
    p = np.ravel(predicted).astype(float)
    prom = prominence if prominence is not None else max(0.05 * np.std(a), 1e-6)
    pa, _ = find_peaks(a, prominence=prom)
    pp, _ = find_peaks(p, prominence=max(0.05 * np.std(p), 1e-6))
    if len(pa) == 0 or len(pp) == 0:
        return np.array([], dtype=int)
    shifts: list[int] = []
    used: set[int] = set()
    for i in pa:
        candidates = [j for j in range(len(pp)) if j not in used]
        if not candidates:
            break
        j = int(min(candidates, key=lambda k: abs(pp[k] - i)))
        used.add(j)
        shifts.append(int(pp[j] - i))
    return np.array(shifts, dtype=int)


def distribution_summary(arr: np.ndarray) -> dict[str, float]:
    x = np.ravel(arr).astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "median": float("nan"), "variance": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "variance": float(np.var(x, ddof=0)),
        "std": float(np.std(x, ddof=0)),
    }


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.ravel(actual).astype(float)
    p = np.ravel(predicted).astype(float)
    mask = np.abs(a) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def paired_metric_table(
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
    metrics: list[tuple[str, str]],
) -> pd.DataFrame:
    """Cohort mean comparison with absolute and percent deltas."""
    rows: list[dict[str, Any]] = []
    for label, col in metrics:
        m = float(mse_ext[col].mean())
        d = float(da_ext[col].mean())
        delta = d - m
        pct = (delta / m * 100.0) if abs(m) > 1e-12 else float("nan")
        rows.append(
            {
                "metric": label,
                "mse_mean": m,
                "da_mse_mean": d,
                "delta_da_minus_mse": delta,
                "pct_change": pct,
            }
        )
    return pd.DataFrame(rows)


def identify_explorer_containers(
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
) -> dict[str, str]:
    """Best std-ratio gain, worst Pearson delta, median |r| container."""
    idx = "container_id"
    merged = mse_ext[[idx, "residual_std_ratio", "residual_pearson_r"]].merge(
        da_ext[[idx, "residual_std_ratio", "residual_pearson_r"]],
        on=idx,
        suffixes=("_mse", "_da"),
    )
    merged["delta_std_ratio"] = merged["residual_std_ratio_da"] - merged["residual_std_ratio_mse"]
    merged["delta_pearson"] = merged["residual_pearson_r_da"] - merged["residual_pearson_r_mse"]

    best_std = str(merged.loc[merged["delta_std_ratio"].idxmax(), idx])
    worst_pearson = str(merged.loc[merged["delta_pearson"].idxmin(), idx])
    med = merged["residual_pearson_r_da"].median()
    median_c = str(merged.loc[(merged["residual_pearson_r_da"] - med).abs().idxmin(), idx])
    return {
        "best_std_ratio_gain": best_std,
        "worst_pearson_delta": worst_pearson,
        "median_pearson_da": median_c,
    }


def temporal_alignment_report(
    actual: np.ndarray,
    pred_mse: np.ndarray,
    pred_da: np.ndarray,
    *,
    max_lag: int = 10,
) -> dict[str, Any]:
    """Measured evidence for amplitude vs alignment decomposition."""
    lag_mse = lag_correlation(actual, pred_mse, max_lag=max_lag)
    lag_da = lag_correlation(actual, pred_da, max_lag=max_lag)
    best_lag_mse = int(lag_mse.loc[lag_mse["pearson_r"].idxmax(), "lag"]) if lag_mse["pearson_r"].notna().any() else 0
    best_lag_da = int(lag_da.loc[lag_da["pearson_r"].idxmax(), "lag"]) if lag_da["pearson_r"].notna().any() else 0
    return {
        "lag_mse": lag_mse,
        "lag_da": lag_da,
        "best_lag_mse": best_lag_mse,
        "best_lag_da": best_lag_da,
        "zero_lag_r_mse": float(lag_mse.loc[lag_mse["lag"] == 0, "pearson_r"].iloc[0]),
        "zero_lag_r_da": float(lag_da.loc[lag_da["lag"] == 0, "pearson_r"].iloc[0]),
        "sign_agreement_mse": sign_agreement_fraction(actual, pred_mse),
        "sign_agreement_da": sign_agreement_fraction(actual, pred_da),
        "turning_point_agreement_mse": turning_point_agreement(actual, pred_mse),
        "turning_point_agreement_da": turning_point_agreement(actual, pred_da),
        "peak_shifts_mse": peak_timing_shifts(actual, pred_mse),
        "peak_shifts_da": peak_timing_shifts(actual, pred_da),
        "std_actual": float(np.std(actual, ddof=0)),
        "std_mse": float(np.std(pred_mse, ddof=0)),
        "std_da": float(np.std(pred_da, ddof=0)),
    }
