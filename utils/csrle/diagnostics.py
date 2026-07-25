"""ACF / Ljung–Box helpers for CSRLE diagnostics."""

from __future__ import annotations

import numpy as np


def avg_abs_pacf(series: np.ndarray, lag_start: int, lag_end: int) -> float:
    from statsmodels.tsa.stattools import pacf

    nlags = min(lag_end, max(1, len(series) // 2 - 1))
    if nlags < lag_start:
        return float("nan")
    vals = pacf(series, nlags=nlags, method="ywm")
    segment = vals[lag_start : min(lag_end + 1, len(vals))]
    return float(np.mean(np.abs(segment))) if len(segment) else float("nan")


def avg_abs_acf(series: np.ndarray, lag_start: int, lag_end: int) -> float:
    from statsmodels.tsa.stattools import acf

    nlags = min(lag_end, len(series) - 1)
    if nlags < lag_start:
        return float("nan")
    vals = acf(series, nlags=nlags, fft=True)
    segment = vals[lag_start : lag_end + 1]
    return float(np.mean(np.abs(segment)))


def ljung_box_reject_fraction(
    series_by_container: dict[str, np.ndarray],
    lag: int = 20,
    alpha: float = 0.05,
) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rejects = 0
    total = 0
    for series in series_by_container.values():
        series = series[np.isfinite(series)]
        if len(series) <= lag + 1:
            continue
        total += 1
        lb = acorr_ljungbox(series, lags=[lag], return_df=True)
        if float(lb["lb_pvalue"].iloc[0]) < alpha:
            rejects += 1
    return rejects / total if total else float("nan")
