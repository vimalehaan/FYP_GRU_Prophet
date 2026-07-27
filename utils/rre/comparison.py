"""Paired R0 vs R3 comparison — bootstrap CI and Wilcoxon tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils.csrle.stage2_stats import paired_bootstrap_diff
from utils.rre.config import (
    METRIC_DIRECTION,
    OPTIMIZATION_METRICS,
    PRIMARY_METRICS,
    PRACTICAL_MAE_THRESHOLD,
    SEED_POLICY,
)


def _cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return float("nan")
    std = float(np.std(diffs, ddof=1))
    if std < 1e-12:
        return float("nan")
    return float(np.mean(diffs) / std)


def _wilcoxon_safe(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    diffs = left - right
    if len(diffs) < 5 or np.allclose(diffs, 0):
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n_pairs": int(len(diffs)),
        }
    stat, pval = wilcoxon(left, right, alternative="two-sided")
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "n_pairs": int(len(diffs)),
    }


def analyze_metric_pair(
    left_values: pd.Series,
    right_values: pd.Series,
    metric: str,
    left_id: str,
    right_id: str,
    bootstrap_seed: int = SEED_POLICY["bootstrap_seed"],
) -> dict[str, Any]:
    """Paired analysis: right - left (R3 - R0 when left=R0, right=R3)."""
    aligned = pd.concat([left_values, right_values], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"metric": metric, "n_pairs": 0}
    left = aligned.iloc[:, 0].values.astype(float)
    right = aligned.iloc[:, 1].values.astype(float)
    diff_arr = right - left
    direction = METRIC_DIRECTION.get(metric, "lower")

    boot = paired_bootstrap_diff(
        pd.Series(right, index=aligned.index),
        pd.Series(left, index=aligned.index),
        seed=bootstrap_seed,
    )
    wil = _wilcoxon_safe(right, left)
    improved = diff_arr < 0 if direction == "lower" else diff_arr > 0

    return {
        "metric": metric,
        "left_variant": left_id,
        "right_variant": right_id,
        "direction": direction,
        "mean_delta": float(np.nanmean(diff_arr)),
        "median_delta": float(np.nanmedian(diff_arr)),
        "cohens_d": _cohens_d_paired(diff_arr),
        "fraction_improved": float(np.nanmean(improved)),
        "bootstrap": boot,
        "wilcoxon": wil,
        "significant_wilcoxon_005": bool(wil["pvalue"] < 0.05)
        if np.isfinite(wil["pvalue"])
        else False,
        "significant_bootstrap_95": bool(
            (boot["ci_lower"] > 0 if direction == "higher" else boot["ci_upper"] < 0)
        )
        if np.isfinite(boot.get("ci_lower", float("nan")))
        else False,
    }


def compare_variants(
    eval_r0: pd.DataFrame,
    eval_r3: pd.DataFrame,
    opt_r0: dict[str, Any],
    opt_r3: dict[str, Any],
) -> dict[str, Any]:
    """Full R0 vs R3 paired comparison."""
    merged = eval_r0.merge(
        eval_r3,
        on="container_id",
        suffixes=("_R0", "_R3"),
    )

    forecast_metrics: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        left_col = f"{metric}_R0"
        right_col = f"{metric}_R3"
        if left_col not in merged.columns:
            continue
        forecast_metrics[metric] = analyze_metric_pair(
            merged.set_index("container_id")[left_col],
            merged.set_index("container_id")[right_col],
            metric,
            "R0",
            "R3",
        )

    optimization: dict[str, Any] = {}
    for key in OPTIMIZATION_METRICS:
        v0 = opt_r0.get(key)
        v3 = opt_r3.get(key)
        if v0 is None or v3 is None:
            continue
        delta = float(v3) - float(v0)
        direction = METRIC_DIRECTION.get(key, "lower")
        r3_better = delta < 0 if direction == "lower" else delta > 0
        optimization[key] = {
            "R0": v0,
            "R3": v3,
            "delta_R3_minus_R0": delta,
            "R3_better": r3_better,
        }

    mae_delta = forecast_metrics.get("day1_mae", {}).get("mean_delta", float("nan"))
    return {
        "comparison": "R0_global_zscore vs R3_median_mad",
        "n_containers": int(len(merged)),
        "forecast_metrics": forecast_metrics,
        "optimization_metrics": optimization,
        "practical_significance": {
            "mae_threshold_pct": PRACTICAL_MAE_THRESHOLD,
            "mean_mae_improvement": float(-mae_delta) if np.isfinite(mae_delta) else float("nan"),
            "practically_meaningful": bool(-mae_delta >= PRACTICAL_MAE_THRESHOLD)
            if np.isfinite(mae_delta)
            else False,
        },
    }
