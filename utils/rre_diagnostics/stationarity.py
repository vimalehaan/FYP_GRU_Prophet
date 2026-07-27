"""Stationarity tests (ADF, KPSS)."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return {"statistic": float("nan"), "p_value": float("nan")}
    try:
        stat, pval, *_ = adfuller(values, autolag="AIC")
        return {"statistic": float(stat), "p_value": float(pval)}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan")}


def kpss_test(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return {"statistic": float("nan"), "p_value": float("nan")}
    try:
        stat, pval, *_ = kpss(values, regression="c", nlags="auto")
        return {"statistic": float(stat), "p_value": float(pval)}
    except Exception:
        return {"statistic": float("nan"), "p_value": float("nan")}


def stationarity_profile(
    series_by_container: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Per-container and cohort stationarity summaries."""
    adf_rows: list[dict[str, float]] = []
    kpss_rows: list[dict[str, float]] = []
    for cid, series in series_by_container.items():
        finite = series[np.isfinite(series)]
        adf = adf_test(finite)
        kpss = kpss_test(finite)
        adf_rows.append(adf)
        kpss_rows.append(kpss)

    adf_p = [r["p_value"] for r in adf_rows if np.isfinite(r["p_value"])]
    kpss_p = [r["p_value"] for r in kpss_rows if np.isfinite(r["p_value"])]

    return {
        "adf_reject_unit_root_frac": float(np.mean([p < 0.05 for p in adf_p]))
        if adf_p
        else float("nan"),
        "kpss_stationary_frac": float(np.mean([p > 0.05 for p in kpss_p]))
        if kpss_p
        else float("nan"),
        "adf_p_value_median": float(np.median(adf_p)) if adf_p else float("nan"),
        "kpss_p_value_median": float(np.median(kpss_p)) if kpss_p else float("nan"),
        "n_containers_tested": len(adf_rows),
    }
