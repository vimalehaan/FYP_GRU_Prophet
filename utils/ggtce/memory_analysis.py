"""GGTCE memory-aware analysis — TMA-linked subgroup comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils.csrle.stage2_stats import cohort_summary, paired_bootstrap_diff
from utils.ggtce.config import SEED_POLICY, TMA_EXPERIMENT
from utils.tma.acf_pacf import acf_at_lag


def load_tma_per_container(repo_root: Path) -> pd.DataFrame:
    """Merge TMA window sufficiency and memory length tables."""
    tma_dir = repo_root / TMA_EXPERIMENT / "statistics"
    with (tma_dir / "cpu_original_window_sufficiency.json").open() as fh:
        import json
        ws = json.load(fh)["per_container"]
    with (tma_dir / "cpu_original_memory_length.json").open() as fh:
        ml = json.load(fh)["per_container"]

    ws_df = pd.DataFrame(ws)
    ml_df = pd.DataFrame(ml)
    merged = ws_df.merge(ml_df, on="container_id", how="outer")
    return merged


def enrich_with_acf_lags(
    memory_df: pd.DataFrame,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    lags: tuple[int, ...] = (96, 192, 288),
    max_lag: int = 672,
) -> pd.DataFrame:
    """Add lag-specific ACF from combined train+val CPU series."""
    rows: list[dict[str, Any]] = []
    for _, row in memory_df.iterrows():
        cid = row["container_id"]
        cpu_col = "cpu" if "cpu" in global_train.columns else "cpu_util_percent"
        train_s = (
            global_train[global_train["container_id"] == cid]
            .sort_values("time_stamp")[cpu_col]
            .values
        )
        val_s = (
            global_val[global_val["container_id"] == cid]
            .sort_values("time_stamp")[cpu_col]
            .values
        )
        series = np.concatenate([train_s, val_s])
        entry = row.to_dict()
        for lag in lags:
            entry[f"acf_lag_{lag}"] = acf_at_lag(series, lag, max_lag)
        rows.append(entry)
    return pd.DataFrame(rows)


def assign_memory_classes(memory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory score: fraction of squared ACF not captured within 96 lags.

    Higher score → stronger long-range temporal dependence beyond G96 window.
    """
    df = memory_df.copy()
    df["memory_score"] = 1.0 - df["pct_capture_within_96"]
    df["memory_score"] = df["memory_score"].clip(0, 1)

    tertiles = df["memory_score"].quantile([1 / 3, 2 / 3]).values
    def _class(score: float) -> str:
        if score <= tertiles[0]:
            return "low"
        if score <= tertiles[1]:
            return "medium"
        return "high"

    df["memory_class"] = df["memory_score"].apply(_class)
    return df


def merge_deltas_with_memory(
    deltas_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    delta_col: str = "delta_day1_mae",
) -> pd.DataFrame:
    """Join per-container metric deltas with memory profile."""
    mem = assign_memory_classes(memory_df)
    cols = [
        "container_id",
        "memory_score",
        "memory_class",
        "pct_capture_within_96",
        "pct_capture_within_192",
        "pct_capture_within_288",
        "acf_lag_96",
        "acf_lag_192",
        "acf_lag_288",
        "integrated_autocorr_time",
        "decorrelation_lag_0p1",
    ]
    cols = [c for c in cols if c in mem.columns]
    return deltas_df.merge(mem[cols], on="container_id", how="left")


def correlation_memory_vs_improvement(
    merged_df: pd.DataFrame,
    delta_col: str = "delta_day1_mae",
) -> dict[str, Any]:
    """Pearson/Spearman correlation: memory score vs metric improvement."""
    df = merged_df.dropna(subset=["memory_score", delta_col])
    x = df["memory_score"].values
    y = (-df[delta_col].values)  # positive = improvement (lower MAE)

    out: dict[str, Any] = {"n": int(len(df))}
    if len(df) >= 5:
        out["pearson_r"] = float(pearsonr(x, y)[0])
        out["pearson_p"] = float(pearsonr(x, y)[1])
        out["spearman_r"] = float(spearmanr(x, y)[0])
        out["spearman_p"] = float(spearmanr(x, y)[1])
    else:
        out["pearson_r"] = out["spearman_r"] = float("nan")

    if "delta_forecast_pearson_r" in df.columns:
        y_r = df["delta_forecast_pearson_r"].values
        if len(df) >= 5:
            out["pearson_r_vs_delta_pearson"] = float(pearsonr(x, y_r)[0])
        else:
            out["pearson_r_vs_delta_pearson"] = float("nan")

    return out


def subgroup_comparison(
    merged_df: pd.DataFrame,
    delta_col: str = "delta_day1_mae",
    bootstrap_seed: int = SEED_POLICY["bootstrap_seed"],
) -> dict[str, Any]:
    """Bootstrap subgroup stats for low/medium/high memory classes."""
    groups: dict[str, Any] = {}
    for cls in ("low", "medium", "high"):
        sub = merged_df[merged_df["memory_class"] == cls]
        if sub.empty:
            continue
        deltas = sub[delta_col].values.astype(float)
        improved = deltas < 0  # lower MAE is better
        boot = paired_bootstrap_diff(
            pd.Series(-deltas, index=sub["container_id"]),
            pd.Series(0.0, index=sub["container_id"]),
            seed=bootstrap_seed,
        )
        groups[cls] = {
            "n": int(len(sub)),
            "mean_delta_mae": float(np.nanmean(deltas)),
            "median_delta_mae": float(np.nanmedian(deltas)),
            "fraction_improved": float(np.nanmean(improved)),
            "bootstrap_improvement": boot,
            "example_containers": sub.nsmallest(3, delta_col)[
                "container_id"
            ].tolist(),
        }
    return groups


def run_memory_analysis(
    repo_root: Path,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    transition_deltas: dict[str, pd.DataFrame],
    primary_transition: tuple[str, str] = ("g96", "g288"),
) -> dict[str, Any]:
    """Full memory-aware co-primary analysis."""
    memory_base = load_tma_per_container(repo_root)
    memory_df = enrich_with_acf_lags(memory_base, global_train, global_val)
    memory_df = assign_memory_classes(memory_df)

    left_id, right_id = primary_transition
    key = f"{left_id}_to_{right_id}"
    deltas = transition_deltas.get(key)
    if deltas is None:
        raise KeyError(f"Missing transition deltas for {key}")

    merged = merge_deltas_with_memory(deltas, memory_df)
    corr_mae = correlation_memory_vs_improvement(merged, "delta_day1_mae")
    corr_pearson = correlation_memory_vs_improvement(
        merged.rename(columns={"delta_forecast_pearson_r": "delta_forecast_pearson_r"}),
        "delta_day1_mae",
    )

    subgroup = subgroup_comparison(merged, "delta_day1_mae")

    examples = {
        "low_memory": memory_df[memory_df["memory_class"] == "low"].nsmallest(
            2, "memory_score",
        )["container_id"].tolist(),
        "medium_memory": (
            memory_df[memory_df["memory_class"] == "medium"]
            .nsmallest(2, "memory_score")["container_id"]
            .tolist()
            if len(memory_df[memory_df["memory_class"] == "medium"]) > 0
            else []
        ),
        "high_memory": memory_df[memory_df["memory_class"] == "high"].nlargest(
            2, "memory_score",
        )["container_id"].tolist(),
    }

    high_benefits_more = (
        subgroup.get("high", {}).get("mean_delta_mae", 0)
        < subgroup.get("low", {}).get("mean_delta_mae", 0)
    )
    low_gain_negligible = abs(
        subgroup.get("low", {}).get("mean_delta_mae", 0),
    ) < 0.02

    return {
        "memory_table": memory_df,
        "merged_deltas": merged,
        "correlation_mae": corr_mae,
        "correlation_pearson_note": corr_pearson,
        "subgroup_comparison": subgroup,
        "examples_by_class": examples,
        "answers": {
            "high_memory_benefits_more": high_benefits_more,
            "low_memory_gain_negligible": low_gain_negligible,
        },
        "primary_transition": key,
    }
