"""CSRLE Stage 2 publication-quality plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.csrle.stage2_stats import cohort_summary

CONDITION_LABELS = {
    "control": "A (Real)",
    "b0_lc": "B0",
    "b1_snar": "B1",
    "c0_iid": "C0",
    "c1_iid": "C1",
}

COLORS = {
    "control": "#1f77b4",
    "b0_lc": "#2ca02c",
    "b1_snar": "#ff7f0e",
    "c0_iid": "#9467bd",
    "c1_iid": "#8c564b",
}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_residual_trajectory(
    actual: np.ndarray,
    pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(1, len(actual) + 1)
    ax.plot(steps, actual, label="Actual residual", color="#1f77b4", linewidth=1.5)
    ax.plot(steps, pred, label="Predicted residual", color="#ff7f0e", linewidth=1.5)
    ax.set_xlabel("Horizon step (15 min)")
    ax.set_ylabel("Prophet residual (scaled CPU space)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_distribution_by_condition(
    dfs: dict[str, pd.DataFrame],
    metric: str,
    title: str,
    path: Path,
) -> None:
    rows = []
    for cond, df in dfs.items():
        for v in df[metric].dropna():
            rows.append({"condition": CONDITION_LABELS.get(cond, cond), "value": v})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = plot_df["condition"].unique()
    data = [plot_df.loc[plot_df["condition"] == c, "value"].values for c in conditions]
    ax.boxplot(data, labels=conditions)
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def plot_horizon_mae(
    cross_horizons: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, df in cross_horizons.items():
        ax.plot(
            df["horizon"], df["mae"],
            marker="o",
            label=CONDITION_LABELS.get(cond, cond),
            color=COLORS.get(cond),
        )
    ax.set_xlabel("Horizon h")
    ax.set_ylabel("Cross-cohort MAE at horizon h")
    ax.set_title("Horizon-wise residual MAE (first h steps, pooled)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_paired_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    left_label: str,
    right_label: str,
    path: Path,
) -> None:
    merged = left[["container_id", metric]].merge(
        right[["container_id", metric]],
        on="container_id",
        suffixes=("_left", "_right"),
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        merged[f"{metric}_right"],
        merged[f"{metric}_left"],
        alpha=0.6,
        s=30,
    )
    lims = [
        min(merged[f"{metric}_right"].min(), merged[f"{metric}_left"].min()),
        max(merged[f"{metric}_right"].max(), merged[f"{metric}_left"].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"{right_label} {metric}")
    ax.set_ylabel(f"{left_label} {metric}")
    ax.set_title(f"Paired comparison: {left_label} vs {right_label}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_training_loss(
    histories: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, hist in histories.items():
        ax.plot(hist["loss"], label=f"{CONDITION_LABELS.get(cond, cond)} train")
        ax.plot(hist["val_loss"], linestyle="--", label=f"{CONDITION_LABELS.get(cond, cond)} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Stage 2 GRU training/validation loss by condition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_synthetic_recovery(
    g_hat: np.ndarray,
    r_b: np.ndarray,
    n_eff: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = np.arange(1, len(g_hat) + 1)
    axes[0].plot(steps, r_b, label="R_B (actual residual)", color="#1f77b4")
    axes[0].plot(steps, g_hat, label="Ĝ (GRU pred)", color="#ff7f0e")
    axes[0].set_title("Ĝ vs R_B (primary)")
    axes[0].set_xlabel("Horizon step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, n_eff, label="N_effective", color="#2ca02c")
    axes[1].plot(steps, g_hat, label="Ĝ", color="#ff7f0e")
    axes[1].set_title("Ĝ vs N_effective (supporting)")
    axes[1].set_xlabel("Horizon step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    _save_fig(fig, path)


def select_representative_container(
    ext_df: pd.DataFrame,
    metric: str = "residual_pearson_r",
) -> str:
    """Container closest to cohort median of metric."""
    median = ext_df[metric].median()
    idx = (ext_df[metric] - median).abs().idxmin()
    return str(ext_df.loc[idx, "container_id"])
