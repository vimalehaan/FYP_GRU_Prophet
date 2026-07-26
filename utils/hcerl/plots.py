"""Publication-quality HCERL plots (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.hcerl.config import VARIANT_LABELS, VARIANT_ORDER


def _save_pub(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overall_comparison(
    cohort_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar chart of cohort mean MAE, RMSE, Pearson r."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [VARIANT_LABELS[v].split("—")[0].strip() for v in cohort_table["variant_id"]]

    axes[0].bar(labels, cohort_table["mean_day1_mae"], color="#3498db")
    axes[0].set_title("Day-1 MAE (lower better)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, cohort_table["mean_day1_rmse"], color="#2ecc71")
    axes[1].set_title("Day-1 RMSE (lower better)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, cohort_table["mean_residual_pearson_r"], color="#9b59b6")
    axes[2].set_title("Residual Pearson r (higher better)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("HCERL — Cohort Mean Metrics by Variant")
    fig.tight_layout()
    _save_pub(fig, out_dir / "01_overall_comparison")


def plot_ablation_waterfall(
    contribution_ranking: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Waterfall of mean MAE delta per ablation step."""
    features = [r["added_feature"] for r in contribution_ranking]
    deltas = [r["delta_mae_mean"] for r in contribution_ranking]
    colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(features, deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean Δ MAE (V_next − V_prev)")
    ax.set_title("Ablation Waterfall — CPU MAE Change per Added Feature")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_pub(fig, out_dir / "02_ablation_waterfall_mae")


def plot_metric_evolution(
    cohort_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Line plot of metric evolution V0→V4."""
    x = list(range(len(cohort_table)))
    labels = [VARIANT_LABELS[v].split("—")[0].strip() for v in cohort_table["variant_id"]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("mean_day1_mae", "Day-1 MAE", axes[0, 0]),
        ("mean_residual_pearson_r", "Residual Pearson r", axes[0, 1]),
        ("mean_residual_std_ratio", "Std ratio", axes[1, 0]),
        ("mean_energy_recovery_ratio", "Energy recovery", axes[1, 1]),
    ]
    for col, title, ax in specs:
        if col in cohort_table.columns:
            ax.plot(x, cohort_table[col], marker="o", lw=2)
            ax.set_xticks(x, labels, rotation=20)
            ax.set_title(title)
    fig.suptitle("Metric Evolution Across Ablation Ladder")
    fig.tight_layout()
    _save_pub(fig, out_dir / "03_metric_evolution")


def plot_std_ratio_distribution(
    variant_eval_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = []
    labels = []
    for vid in VARIANT_ORDER:
        if vid in variant_eval_dfs:
            data.append(variant_eval_dfs[vid]["residual_std_ratio"].dropna().values)
            labels.append(VARIANT_LABELS[vid].split("—")[0].strip())
    ax.boxplot(data, labels=labels)
    ax.set_title("Residual Std Ratio Distribution by Variant")
    ax.set_ylabel("pred_std / actual_std")
    fig.tight_layout()
    _save_pub(fig, out_dir / "04_std_ratio_distribution")


def plot_pearson_scatter(
    variant_eval_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for vid in VARIANT_ORDER:
        if vid not in variant_eval_dfs:
            continue
        df = variant_eval_dfs[vid]
        ax.scatter(
            df["day1_mae"],
            df["residual_pearson_r"],
            alpha=0.6,
            label=VARIANT_LABELS[vid].split("—")[0].strip(),
        )
    ax.set_xlabel("Day-1 MAE")
    ax.set_ylabel("Residual Pearson r")
    ax.legend(fontsize=8)
    ax.set_title("MAE vs Residual Correlation Trade-off")
    fig.tight_layout()
    _save_pub(fig, out_dir / "05_mae_vs_pearson_scatter")


def plot_container_ranking_delta(
    deltas_df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    title: str,
) -> None:
    col = f"delta_{metric}"
    if col not in deltas_df.columns:
        return
    sorted_df = deltas_df.sort_values(col)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sorted_df["container_id"], sorted_df[col])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(col)
    ax.set_title(title)
    fig.tight_layout()
    _save_pub(fig, out_dir / f"06_container_delta_{metric}")


def plot_training_curves(
    histories: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for vid, hist in histories.items():
        label = VARIANT_LABELS.get(vid, vid).split("—")[0].strip()
        axes[0].plot(hist["epoch"], hist["loss"], label=label)
        axes[1].plot(hist["epoch"], hist["val_loss"], label=label)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    fig.suptitle("Training Curves — All Variants")
    fig.tight_layout()
    _save_pub(fig, out_dir / "07_training_curves")


def plot_feature_contribution(
    contribution_ranking: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    features = [r["added_feature"] for r in contribution_ranking]
    pearson = [r["delta_pearson_r_mean"] for r in contribution_ranking]
    ax.bar(features, pearson, color="#8e44ad")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean Δ Pearson r")
    ax.set_title("Per-Feature Contribution — Residual Correlation")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_pub(fig, out_dir / "08_feature_contribution_pearson")


def plot_representative_trajectory(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    container_id: str,
    out_dir: Path,
    suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(len(actual))
    ax.plot(steps, actual, label="Actual", color="black", lw=2)
    for vid, pred in predictions.items():
        ax.plot(steps, pred, label=VARIANT_LABELS.get(vid, vid), alpha=0.8)
    ax.set_xlabel("Day-1 step (15 min)")
    ax.set_ylabel("CPU %")
    ax.set_title(f"Case study — {container_id} ({suffix})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_pub(fig, out_dir / f"09_case_study_{suffix}_{container_id}")


def generate_all_plots(
    cohort_table: pd.DataFrame,
    variant_eval_dfs: dict[str, pd.DataFrame],
    ablation_result: dict[str, Any],
    training_histories: dict[str, pd.DataFrame],
    case_studies: dict[str, Any],
    out_dir: Path,
) -> None:
    plot_overall_comparison(cohort_table, out_dir)
    plot_ablation_waterfall(
        ablation_result["feature_contribution_ranking"], out_dir,
    )
    plot_metric_evolution(cohort_table, out_dir)
    plot_std_ratio_distribution(variant_eval_dfs, out_dir)
    plot_pearson_scatter(variant_eval_dfs, out_dir)
    plot_feature_contribution(
        ablation_result["feature_contribution_ranking"], out_dir,
    )
    plot_training_curves(training_histories, out_dir)

    if ablation_result["transitions"]:
        last = ablation_result["transitions"][-1]
        deltas = last["per_container_deltas"]
        plot_container_ranking_delta(
            deltas, "day1_mae",
            out_dir,
            "V3→V4 Per-Container Δ MAE",
        )

    for name, cs in case_studies.get("trajectories", {}).items():
        plot_representative_trajectory(
            cs["actual"],
            cs["predictions"],
            cs["container_id"],
            out_dir,
            suffix=name,
        )
