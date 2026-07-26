"""HCERL ablation analysis — paired deltas, bootstrap, Wilcoxon, effect sizes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils.csrle.stage2_stats import paired_bootstrap_diff
from utils.hcerl.config import (
    ABLATION_TRANSITIONS,
    METRIC_DIRECTION,
    PRIMARY_METRICS,
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
    """Paired Wilcoxon signed-rank (left - right)."""
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


def compute_transition_deltas(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_id: str,
    right_id: str,
) -> pd.DataFrame:
    """Per-container deltas for right - left on all primary metrics."""
    merged = left_df.merge(
        right_df,
        on="container_id",
        suffixes=("_left", "_right"),
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        entry: dict[str, Any] = {
            "container_id": row["container_id"],
            "left_variant": left_id,
            "right_variant": right_id,
        }
        for metric in PRIMARY_METRICS:
            lv = row.get(f"{metric}_left")
            rv = row.get(f"{metric}_right")
            if pd.notna(lv) and pd.notna(rv):
                entry[f"delta_{metric}"] = float(rv - lv)
        rows.append(entry)
    return pd.DataFrame(rows)


def analyze_transition(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_id: str,
    right_id: str,
    bootstrap_seed: int = SEED_POLICY["bootstrap_seed"],
) -> dict[str, Any]:
    """Full paired analysis for one ablation step (e.g. V0 → V1)."""
    deltas_df = compute_transition_deltas(left_df, right_df, left_id, right_id)
    merged = left_df.merge(right_df, on="container_id", suffixes=("_left", "_right"))

    metrics_out: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        if f"{metric}_left" not in merged.columns:
            continue
        left_s = merged.set_index("container_id")[f"{metric}_left"]
        right_s = merged.set_index("container_id")[f"{metric}_right"]
        direction = METRIC_DIRECTION.get(metric, "lower")

        # Bootstrap for right - left
        boot = paired_bootstrap_diff(
            right_s, left_s, seed=bootstrap_seed,
        )
        diff_arr = (
            merged[f"{metric}_right"] - merged[f"{metric}_left"]
        ).values.astype(float)

        wil = _wilcoxon_safe(
            merged[f"{metric}_right"].values.astype(float),
            merged[f"{metric}_left"].values.astype(float),
        )

        improved = (
            diff_arr < 0 if direction == "lower" else diff_arr > 0
        )
        metrics_out[metric] = {
            "direction": direction,
            "mean_delta": float(np.nanmean(diff_arr)),
            "median_delta": float(np.nanmedian(diff_arr)),
            "cohens_d": _cohens_d_paired(diff_arr),
            "fraction_improved": float(np.nanmean(improved)),
            "bootstrap": boot,
            "wilcoxon": wil,
            "significant_bootstrap_95": (
                boot["ci_high"] < 0 if direction == "lower" else boot["ci_low"] > 0
            ),
            "significant_wilcoxon_005": (
                wil["pvalue"] < 0.05 if np.isfinite(wil["pvalue"]) else False
            ),
        }

    return {
        "transition": f"{left_id} -> {right_id}",
        "left_variant": left_id,
        "right_variant": right_id,
        "n_containers": int(len(merged)),
        "per_container_deltas": deltas_df,
        "metrics": metrics_out,
    }


def run_full_ablation_analysis(
    variant_eval_dfs: dict[str, pd.DataFrame],
    bootstrap_seed: int = SEED_POLICY["bootstrap_seed"],
) -> dict[str, Any]:
    """Analyze all V0→V1→…→V4 transitions."""
    transitions: list[dict[str, Any]] = []
    for left_id, right_id in ABLATION_TRANSITIONS:
        transitions.append(
            analyze_transition(
                variant_eval_dfs[left_id],
                variant_eval_dfs[right_id],
                left_id,
                right_id,
                bootstrap_seed=bootstrap_seed,
            )
        )

    feature_contribution = _rank_feature_contributions(transitions)
    return {
        "transitions": transitions,
        "feature_contribution_ranking": feature_contribution,
    }


def _rank_feature_contributions(
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank added features by residual Pearson r and MAE impact.

    Uses mean delta on primary metrics; negative MAE delta = improvement.
    """
    feature_map = {
        "v0_baseline -> v1_prophet": "prophet_yhat_scaled",
        "v1_prophet -> v2_volatility": "res_roll_std_12",
        "v2_volatility -> v3_cpu_std": "cpu_std",
        "v3_cpu_std -> v4_full": "is_peak_p90",
    }
    rows: list[dict[str, Any]] = []
    for tr in transitions:
        key = tr["transition"]
        feature = feature_map.get(key, key)
        m = tr["metrics"]
        rows.append({
            "transition": key,
            "added_feature": feature,
            "delta_mae_mean": m.get("day1_mae", {}).get("mean_delta"),
            "delta_rmse_mean": m.get("day1_rmse", {}).get("mean_delta"),
            "delta_pearson_r_mean": m.get("residual_pearson_r", {}).get("mean_delta"),
            "delta_std_ratio_mean": m.get("residual_std_ratio", {}).get("mean_delta"),
            "delta_r2_mean": m.get("residual_r2", {}).get("mean_delta"),
            "mae_improved_fraction": m.get("day1_mae", {}).get("fraction_improved"),
            "pearson_improved_fraction": m.get(
                "residual_pearson_r", {},
            ).get("fraction_improved"),
            "mae_significant": m.get("day1_mae", {}).get("significant_bootstrap_95"),
            "pearson_significant": m.get(
                "residual_pearson_r", {},
            ).get("significant_bootstrap_95"),
        })

    ranked = sorted(
        rows,
        key=lambda r: (
            -(r["delta_pearson_r_mean"] or float("-inf")),
            (r["delta_mae_mean"] or float("inf")),
        ),
    )
    for i, row in enumerate(ranked, start=1):
        row["rank_by_composite"] = i
    return ranked


def build_cohort_comparison_table(
    variant_eval_dfs: dict[str, pd.DataFrame],
    variant_order: tuple[str, ...],
) -> pd.DataFrame:
    """Cohort mean metrics across variants."""
    rows: list[dict[str, Any]] = []
    for vid in variant_order:
        df = variant_eval_dfs[vid]
        row: dict[str, Any] = {"variant_id": vid, "n": len(df)}
        for metric in PRIMARY_METRICS:
            if metric in df.columns:
                row[f"mean_{metric}"] = float(df[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)
