"""Long-lag ACF/PACF computation for Temporal Memory Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _finite_series(series: np.ndarray) -> np.ndarray:
    return series[np.isfinite(series)]


def compute_acf(
    series: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lag indices (0..max_lag) and ACF values."""
    from statsmodels.tsa.stattools import acf

    series = _finite_series(series)
    if len(series) < 3:
        empty = np.array([1.0])
        return np.array([0]), empty
    effective = min(max_lag, len(series) - 1)
    values = acf(series, nlags=effective, fft=True)
    lags = np.arange(len(values))
    return lags, values


def compute_pacf(
    series: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lag indices and PACF values (Yule-Walker)."""
    from statsmodels.tsa.stattools import pacf

    series = _finite_series(series)
    if len(series) < 3:
        empty = np.array([1.0])
        return np.array([0]), empty
    effective = min(max_lag, max(1, len(series) // 2 - 1))
    values = pacf(series, nlags=effective, method="ywm")
    lags = np.arange(len(values))
    return lags, values


def acf_confidence_band(n: int, alpha: float = 0.05) -> float:
    """Two-sided normal-approximation ACF significance bound."""
    if n <= 0:
        return float("nan")
    return float(stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))


def cohort_acf_summary(
    acf_matrix: np.ndarray,
    lags: np.ndarray,
) -> dict[str, np.ndarray]:
    """Cohort mean, median, and 95% CI across containers."""
    mean = np.nanmean(acf_matrix, axis=0)
    median = np.nanmedian(acf_matrix, axis=0)
    n = acf_matrix.shape[0]
    se = np.nanstd(acf_matrix, axis=0, ddof=1) / np.sqrt(max(n, 1))
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    return {
        "lags": lags,
        "mean": mean,
        "median": median,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def build_acf_matrix(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack per-container ACF rows aligned to common lag axis."""
    container_ids: list[str] = []
    rows: list[np.ndarray] = []
    ref_lags: np.ndarray | None = None

    for cid, series in series_by_container.items():
        lags, values = compute_acf(series, max_lag)
        if ref_lags is None:
            ref_lags = lags
        elif len(lags) != len(ref_lags):
            padded = np.full(len(ref_lags), np.nan)
            padded[: len(values)] = values
            values = padded
        container_ids.append(cid)
        rows.append(values)

    if ref_lags is None:
        ref_lags = np.array([0])
    matrix = np.vstack(rows) if rows else np.empty((0, len(ref_lags)))
    return matrix, ref_lags, container_ids


def build_pacf_matrix(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack per-container PACF rows aligned to common lag axis."""
    container_ids: list[str] = []
    rows: list[np.ndarray] = []
    ref_lags: np.ndarray | None = None

    for cid, series in series_by_container.items():
        lags, values = compute_pacf(series, max_lag)
        if ref_lags is None:
            ref_lags = lags
        elif len(lags) != len(ref_lags):
            padded = np.full(len(ref_lags), np.nan)
            padded[: len(values)] = values
            values = padded
        container_ids.append(cid)
        rows.append(values)

    if ref_lags is None:
        ref_lags = np.array([0])
    matrix = np.vstack(rows) if rows else np.empty((0, len(ref_lags)))
    return matrix, ref_lags, container_ids


def acf_at_lag(
    series: np.ndarray,
    lag: int,
    max_lag: int,
) -> float:
    """ACF value at a specific lag."""
    _, values = compute_acf(series, max_lag)
    if lag >= len(values):
        return float("nan")
    return float(values[lag])
