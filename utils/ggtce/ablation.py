"""GGTCE window comparison — paired deltas, bootstrap, Wilcoxon."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils.csrle.stage2_stats import paired_bootstrap_diff
from utils.ggtce.config import (
    METRIC_DIRECTION,
    PRIMARY_METRICS,
    SEED_POLICY,
    VARIANT_ORDER,
    WINDOW_TRANSITIONS,
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


def compute_transition_deltas(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_id: str,
    right_id: str,
) -> pd.DataFrame:
    """Per-container deltas: right − left."""
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
    """Paired bootstrap + Wilcoxon for one window step."""
    deltas_df = compute_transition_deltas(left_df, right_df, left_id, right_id)
    merged = left_df.merge(right_df, on="container_id", suffixes=("_left", "_right"))

    metrics_out: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        col_l, col_r = f"{metric}_left", f"{metric}_right"
        if col_l not in merged.columns:
            continue
        left_s = merged.set_index("container_id")[col_l]
        right_s = merged.set_index("container_id")[col_r]
        direction = METRIC_DIRECTION.get(metric, "lower")

        boot = paired_bootstrap_diff(right_s, left_s, seed=bootstrap_seed)
        diff_arr = (merged[col_r] - merged[col_l]).values.astype(float)
        wil = _wilcoxon_safe(
            merged[col_r].values.astype(float),
            merged[col_l].values.astype(float),
        )
        improved = diff_arr < 0 if direction == "lower" else diff_arr > 0

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


def run_window_comparison(
    variant_eval_dfs: dict[str, pd.DataFrame],
    bootstrap_seed: int = SEED_POLICY["bootstrap_seed"],
) -> dict[str, Any]:
    """Analyze G96→G192, G192→G288, G96→G288."""
    transitions: list[dict[str, Any]] = []
    delta_tables: dict[str, pd.DataFrame] = {}

    for left_id, right_id in WINDOW_TRANSITIONS:
        result = analyze_transition(
            variant_eval_dfs[left_id],
            variant_eval_dfs[right_id],
            left_id,
            right_id,
            bootstrap_seed=bootstrap_seed,
        )
        transitions.append(result)
        delta_tables[f"{left_id}_to_{right_id}"] = result["per_container_deltas"]

    return {"transitions": transitions, "delta_tables": delta_tables}


def build_cohort_comparison_table(
    variant_eval_dfs: dict[str, pd.DataFrame],
    variant_order: tuple[str, ...] = VARIANT_ORDER,
) -> pd.DataFrame:
    """Cohort mean metrics across window variants."""
    rows: list[dict[str, Any]] = []
    for vid in variant_order:
        df = variant_eval_dfs[vid]
        row: dict[str, Any] = {"variant_id": vid, "n": len(df)}
        for metric in PRIMARY_METRICS:
            if metric in df.columns:
                row[f"mean_{metric}"] = float(df[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sequence_reduction_table(
    training_meta: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Quantify sequence-count vs accuracy trade-off."""
    rows: list[dict[str, Any]] = []
    for vid, meta in training_meta.items():
        sc = meta["sequence_counts"]
        rows.append({
            "variant_id": vid,
            "input_window": meta["input_window"],
            "n_sequences": meta["n_sequences"],
            "n_train_sequences": meta["n_train_sequences"],
            "training_seconds": meta["training_seconds"],
            "best_epoch": meta["best_epoch"],
            "epochs_run": meta["epochs_run"],
            "mean_per_container_sequences": sc["mean_per_container"],
        })
    return pd.DataFrame(rows)
