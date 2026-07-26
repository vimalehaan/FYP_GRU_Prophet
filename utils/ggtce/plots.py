"""Publication-quality GGTCE plots (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.ggtce.config import VARIANT_LABELS, VARIANT_ORDER


def _save_pub(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _variant_labels(ids: list[str]) -> list[str]:
    return [VARIANT_LABELS.get(v, v).split("—")[0].strip() for v in ids]


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    out_dir: Path,
) -> None:
    """Training/validation loss curves per variant."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, vid in zip(axes, VARIANT_ORDER):
        hist = histories.get(vid, {})
        epochs = range(1, len(hist.get("loss", [])) + 1)
        ax.plot(epochs, hist.get("loss", []), label="train")
        ax.plot(epochs, hist.get("val_loss", []), label="val")
        ax.set_title(VARIANT_LABELS[vid].split("—")[0].strip())
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("MSE loss")
    fig.suptitle("GGTCE — Training Curves by Input Window")
    fig.tight_layout()
    _save_pub(fig, out_dir / "01_training_curves")


def plot_cohort_comparison(
    cohort_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bars for MAE, RMSE, Pearson r."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = _variant_labels(cohort_table["variant_id"].tolist())

    axes[0].bar(labels, cohort_table["mean_day1_mae"], color="#3498db")
    axes[0].set_title("Day-1 MAE (lower better)")

    axes[1].bar(labels, cohort_table["mean_day1_rmse"], color="#2ecc71")
    axes[1].set_title("Day-1 RMSE (lower better)")

    axes[2].bar(
        labels,
        cohort_table.get("mean_forecast_pearson_r", cohort_table.get("mean_residual_pearson_r")),
        color="#9b59b6",
    )
    axes[2].set_title("Forecast Pearson r (higher better)")

    fig.suptitle("GGTCE — Cohort Mean Metrics")
    fig.tight_layout()
    _save_pub(fig, out_dir / "02_cohort_comparison")


def plot_window_waterfall(
    transitions: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Waterfall of mean Δ MAE per window step."""
    labels = [t["transition"] for t in transitions]
    deltas = [
        t["metrics"].get("day1_mae", {}).get("mean_delta", 0)
        for t in transitions
    ]
    colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean Δ MAE (right − left)")
    ax.set_title("Window Comparison Waterfall")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save_pub(fig, out_dir / "03_window_waterfall_mae")


def plot_metric_evolution(
    cohort_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Line plot G96→G192→G288."""
    x = list(range(len(cohort_table)))
    labels = _variant_labels(cohort_table["variant_id"].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("mean_day1_mae", "Day-1 MAE", axes[0, 0]),
        ("mean_forecast_pearson_r", "Pearson r", axes[0, 1]),
        ("mean_forecast_std_ratio", "Std ratio", axes[1, 0]),
        ("mean_day1_mape", "MAPE", axes[1, 1]),
    ]
    for col, title, ax in specs:
        if col not in cohort_table.columns:
            alt = col.replace("forecast_", "residual_")
            col = alt if alt in cohort_table.columns else col
        ax.plot(x, cohort_table[col], marker="o")
        ax.set_xticks(x, labels)
        ax.set_title(title)
    fig.suptitle("Metric Evolution vs Input Window")
    fig.tight_layout()
    _save_pub(fig, out_dir / "04_metric_evolution")


def plot_bootstrap_distributions(
    transitions: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Bootstrap CI visualization for G96→G288 MAE."""
    g96_g288 = next(
        (t for t in transitions if t["left_variant"] == "g96" and t["right_variant"] == "g288"),
        None,
    )
    if not g96_g288:
        return
    boot = g96_g288["metrics"].get("day1_mae", {}).get("bootstrap", {})
    if not boot:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    mean = boot.get("mean_diff", 0)
    ci_lo, ci_hi = boot.get("ci_low", 0), boot.get("ci_high", 0)
    ax.barh(["G288 − G96 MAE"], [mean], xerr=[[mean - ci_lo], [ci_hi - mean]], color="#3498db")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ MAE (negative = improvement)")
    ax.set_title("Bootstrap 95% CI — G96 vs G288")
    fig.tight_layout()
    _save_pub(fig, out_dir / "05_bootstrap_g96_g288_mae")


def plot_memory_scatter(
    merged_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Memory score vs Δ MAE and Δ Pearson."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df = merged_df.dropna(subset=["memory_score"])

    if "delta_day1_mae" in df.columns:
        y = -df["delta_day1_mae"]
        axes[0].scatter(df["memory_score"], y, alpha=0.6, c="#e74c3c")
        z = np.polyfit(df["memory_score"], y, 1)
        xs = np.linspace(df["memory_score"].min(), df["memory_score"].max(), 50)
        axes[0].plot(xs, np.poly1d(z)(xs), "k--", lw=1)
        axes[0].set_ylabel("MAE improvement (pp)")
        axes[0].set_xlabel("Memory score")
        axes[0].set_title("Memory score vs Δ MAE (G96→G288)")

    if "delta_forecast_pearson_r" in df.columns:
        axes[1].scatter(
            df["memory_score"], df["delta_forecast_pearson_r"], alpha=0.6, c="#9b59b6",
        )
        axes[1].set_xlabel("Memory score")
        axes[1].set_ylabel("Δ Pearson r")
        axes[1].set_title("Memory score vs Δ Pearson r")

    fig.suptitle("Memory-Aware Subgroup Analysis")
    fig.tight_layout()
    _save_pub(fig, out_dir / "06_memory_scatter")


def plot_subgroup_bars(
    subgroup: dict[str, Any],
    out_dir: Path,
) -> None:
    """Bar chart of mean Δ MAE by memory class."""
    classes = [k for k in ("low", "medium", "high") if k in subgroup]
    deltas = [subgroup[k]["mean_delta_mae"] for k in classes]
    colors = ["#95a5a6", "#f39c12", "#c0392b"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(classes, deltas, color=colors[: len(classes)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean Δ MAE (G288 − G96)")
    ax.set_title("Window Benefit by Memory Class")
    fig.tight_layout()
    _save_pub(fig, out_dir / "07_memory_subgroup_bars")


def plot_sequence_count_tradeoff(
    seq_table: pd.DataFrame,
    cohort_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Dual-axis: sequence count vs MAE."""
    if "mean_day1_mae" in seq_table.columns:
        merged = seq_table
    else:
        merged = seq_table.merge(cohort_table, on="variant_id")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    labels = _variant_labels(merged["variant_id"].tolist())
    x = np.arange(len(labels))

    ax1.bar(x - 0.2, merged["n_sequences"], width=0.4, label="Sequences", color="#3498db")
    ax1.set_ylabel("Training sequences")
    ax1.set_xticks(x, labels)

    ax2 = ax1.twinx()
    ax2.plot(x, merged["mean_day1_mae"], "o-", color="#e74c3c", label="Mean MAE")
    ax2.set_ylabel("Mean Day-1 MAE")

    fig.suptitle("Sequence Reduction vs Forecast Accuracy")
    fig.tight_layout()
    _save_pub(fig, out_dir / "08_sequence_tradeoff")


def plot_container_ranking(
    deltas_df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
) -> None:
    """Top/bottom containers by Δ MAE."""
    col = "delta_day1_mae"
    if col not in deltas_df.columns:
        return
    sorted_df = deltas_df.sort_values(col)
    subset = pd.concat([sorted_df.head(top_n // 2), sorted_df.tail(top_n // 2)])
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in subset[col]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["container_id"], subset[col], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ MAE (G288 − G96)")
    ax.set_title("Container Ranking by Window Effect")
    fig.tight_layout()
    _save_pub(fig, out_dir / "09_container_ranking")


def plot_case_study_trajectories(
    narratives: dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot actual vs predicted for case study containers."""
    cases = narratives.get("cases", {})
    n = len(cases)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for ax, (cat, case) in zip(axes.ravel(), cases.items()):
        actual = np.array(case["actual_day1_real"])
        ax.plot(actual, label="Actual", color="k", lw=1.5)
        for vid, pred in case.get("predictions_by_variant", {}).items():
            ax.plot(pred, label=vid, alpha=0.8)
        ax.set_title(f"{cat}: {case['container_id']}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Step (15 min)")
        ax.set_ylabel("CPU %")
    fig.suptitle("Case Study Day-1 Trajectories")
    fig.tight_layout()
    _save_pub(fig, out_dir / "10_case_study_trajectories")


def plot_std_ratio_distribution(
    variant_eval_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Boxplot of std ratio across variants."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for vid in VARIANT_ORDER:
        col = "forecast_std_ratio"
        if col not in variant_eval_dfs[vid].columns:
            col = "residual_std_ratio"
        data.append(variant_eval_dfs[vid][col].dropna().values)
        labels.append(vid.upper())
    ax.boxplot(data, labels=labels)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Prediction std ratio")
    ax.set_title("Std Ratio Distribution by Window")
    fig.tight_layout()
    _save_pub(fig, out_dir / "11_std_ratio_distribution")


def generate_all_plots(
    cohort_table: pd.DataFrame,
    window_comparison: dict[str, Any],
    training_histories: dict[str, dict[str, list[float]]],
    memory_analysis: dict[str, Any],
    seq_table: pd.DataFrame,
    case_narratives: dict[str, Any],
    variant_eval_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Generate full GGTCE plot suite."""
    plot_training_curves(training_histories, out_dir)
    plot_cohort_comparison(cohort_table, out_dir)
    plot_window_waterfall(window_comparison["transitions"], out_dir)
    plot_metric_evolution(cohort_table, out_dir)
    plot_bootstrap_distributions(window_comparison["transitions"], out_dir)

    merged = memory_analysis.get("merged_deltas")
    if merged is not None and not merged.empty:
        plot_memory_scatter(merged, out_dir)
    plot_subgroup_bars(memory_analysis.get("subgroup_comparison", {}), out_dir)
    plot_sequence_count_tradeoff(seq_table, cohort_table, out_dir)

    g96_g288 = window_comparison["delta_tables"].get("g96_to_g288")
    if g96_g288 is not None:
        plot_container_ranking(g96_g288, out_dir)

    plot_case_study_trajectories(case_narratives, out_dir)
    plot_std_ratio_distribution(variant_eval_dfs, out_dir)
