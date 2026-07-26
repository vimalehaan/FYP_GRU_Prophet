"""LFHE publication-quality plot exports (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(
    mse_hist: pd.DataFrame,
    da_hist: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mse_hist["val_loss"], label="MSE val_loss", ls="--")
    ax.plot(da_hist["val_loss"], label="DA-MSE val_loss", ls="--")
    ax.plot(mse_hist["loss"], label="MSE train_loss", alpha=0.7)
    ax.plot(da_hist["loss"], label="DA-MSE train_loss", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("LFHE Training Loss — MSE vs DA-MSE (B0)")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_std_ratio_distribution(
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mse_ext["residual_std_ratio"], bins=25, alpha=0.6, label="MSE", color="#1f77b4")
    ax.hist(da_ext["residual_std_ratio"], bins=25, alpha=0.6, label="DA-MSE", color="#ff7f0e")
    ax.axvline(0.220, color="k", ls=":", label="Ridge reference (~0.22)")
    ax.set_xlabel("Std ratio (predicted / actual)")
    ax.set_ylabel("Containers")
    ax.set_title("Residual Std Ratio Distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_metric_scatter(
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
    metric: str,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    m = mse_ext[["container_id", metric]].merge(
        da_ext[["container_id", metric]],
        on="container_id",
        suffixes=("_mse", "_da"),
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(m[f"{metric}_mse"], m[f"{metric}_da"], alpha=0.6, s=35)
    lim = max(m[f"{metric}_mse"].abs().max(), m[f"{metric}_da"].abs().max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.set_xlabel(f"MSE {xlabel}")
    ax.set_ylabel(f"DA-MSE {xlabel}")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def plot_horizon_variance_mean_ci(
    cohort_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    band_order = cohort_df["band"].unique()
    x = np.arange(len(band_order))
    w = 0.35
    for i, arm in enumerate(["mse", "da_mse"]):
        sub = cohort_df[cohort_df["arm"] == arm].set_index("band").reindex(band_order)
        offset = (i - 0.5) * w
        ax.bar(
            x + offset,
            sub["mean_std_ratio"],
            width=w,
            yerr=[
                sub["mean_std_ratio"] - sub["ci_low"],
                sub["ci_high"] - sub["mean_std_ratio"],
            ],
            capsize=3,
            label=arm,
        )
    ax.set_xticks(x, [str(b) for b in band_order], rotation=20)
    ax.set_ylabel("Std ratio")
    ax.set_title("Horizon-wise Variance Recovery (cohort mean ± 95% CI)")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_horizon_variance_boxplot(
    pc_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    bands = [b[0] for b in __import__(
        "utils.lfhe.diagnostics", fromlist=["HORIZON_VARIANCE_BANDS"]
    ).HORIZON_VARIANCE_BANDS]
    data_mse = [
        pc_df[(pc_df["arm"] == "mse") & (pc_df["band"] == b)]["std_ratio"].dropna().values
        for b in bands
    ]
    data_da = [
        pc_df[(pc_df["arm"] == "da_mse") & (pc_df["band"] == b)]["std_ratio"].dropna().values
        for b in bands
    ]
    positions = np.arange(len(bands)) * 3
    bp1 = ax.boxplot(data_mse, positions=positions - 0.4, widths=0.35, patch_artist=True)
    bp2 = ax.boxplot(data_da, positions=positions + 0.4, widths=0.35, patch_artist=True)
    for box in bp1["boxes"]:
        box.set_facecolor("#1f77b4")
        box.set_alpha(0.6)
    for box in bp2["boxes"]:
        box.set_facecolor("#ff7f0e")
        box.set_alpha(0.6)
    ax.set_xticks(positions, bands, rotation=15)
    ax.set_ylabel("Std ratio")
    ax.set_title("Horizon-wise Std Ratio by Band (MSE vs DA-MSE)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["MSE", "DA-MSE"])
    fig.tight_layout()
    _save(fig, path)


def plot_energy_recovery_distribution(
    ec_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, color in [("mse", "#1f77b4"), ("da_mse", "#ff7f0e")]:
        vals = ec_df.loc[ec_df["arm"] == arm, "energy_recovery_ratio"].dropna()
        ax.hist(vals, bins=25, alpha=0.6, label=arm, color=color)
    ax.set_xlabel("Energy Recovery Ratio Σ(ŷ²)/Σ(y²)")
    ax.set_ylabel("Containers")
    ax.set_title("Residual Energy Recovery Distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_representative_trajectory(
    actual: np.ndarray,
    pred_mse: np.ndarray,
    pred_da: np.ndarray,
    title: str,
    path: Path,
) -> None:
    steps = np.arange(1, len(actual) + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, actual, label="Actual residual", lw=1.5)
    ax.plot(steps, pred_mse, label="MSE pred", lw=1.2, alpha=0.85)
    ax.plot(steps, pred_da, label="DA-MSE pred", lw=1.2, alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel("Scaled residual")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_residual_histogram(
    actual: np.ndarray,
    pred: np.ndarray,
    arm: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(actual, bins=30, alpha=0.5, label="Actual", density=True)
    ax.hist(pred, bins=30, alpha=0.5, label=f"{arm} pred", density=True)
    ax.set_xlabel("Scaled residual")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution — {arm}")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
