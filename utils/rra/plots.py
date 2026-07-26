"""Publication plots for Ridge Residual Analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

from utils.csrle.diagnostics import avg_abs_acf, avg_abs_pacf


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distribution_comparison(
    prophet_pooled: np.ndarray,
    remaining_pooled: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        prophet_pooled, bins=50, density=True, alpha=0.55,
        label="Prophet residual", color="#8172B2",
    )
    ax.hist(
        remaining_pooled, bins=50, density=True, alpha=0.55,
        label="Ridge remaining", color="#4C72B0",
    )
    ax.axvline(0.0, color="#7F7F7F", linestyle="--", linewidth=1.0)
    ax.set_title("Pooled Validation Residual Distributions")
    ax.set_xlabel("Scaled residual")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_boxplot_metric(
    diag_df: pd.DataFrame,
    prophet_col: str,
    remaining_col: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [
        diag_df[prophet_col].dropna().values,
        diag_df[remaining_col].dropna().values,
    ]
    ax.boxplot(data, tick_labels=["Prophet", "Ridge remaining"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, path)


def plot_acf_comparison(
    series: np.ndarray,
    title: str,
    path: Path,
    nlags: int = 50,
) -> None:
    from statsmodels.tsa.stattools import acf

    series = series[np.isfinite(series)]
    vals = acf(series, nlags=min(nlags, len(series) - 1), fft=True)
    lags = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(lags, vals, linefmt="#8172B2", markerfmt="o", basefmt=" ")
    ax.axhline(0.0, color="#7F7F7F", linewidth=0.8)
    conf = 1.96 / np.sqrt(len(series))
    ax.axhline(conf, color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    _save_fig(fig, path)


def plot_pacf_comparison(
    series: np.ndarray,
    title: str,
    path: Path,
    nlags: int = 50,
) -> None:
    from statsmodels.tsa.stattools import pacf

    series = series[np.isfinite(series)]
    vals = pacf(series, nlags=min(nlags, max(1, len(series) // 2 - 1)), method="ywm")
    lags = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(lags, vals, linefmt="#4C72B0", markerfmt="o", basefmt=" ")
    ax.axhline(0.0, color="#7F7F7F", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    _save_fig(fig, path)


def plot_fft_comparison(
    prophet_series: np.ndarray,
    remaining_series: np.ndarray,
    title: str,
    path: Path,
    fs: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for series, label, color in (
        (prophet_series, "Prophet residual", "#8172B2"),
        (remaining_series, "Ridge remaining", "#4C72B0"),
    ):
        s = series[np.isfinite(series)]
        if len(s) < 8:
            continue
        freqs, power = periodogram(s, fs=fs)
        ax.semilogy(freqs[1:], power[1:], label=label, color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Frequency (cycles per 15-min step)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)


def plot_case_study_panel(
    container_id: str,
    result: dict[str, Any],
    path: Path,
) -> None:
    """Four-panel case study: time series, ACF, PACF, FFT."""
    prophet = result["prophet_residual_scaled"]
    ridge_pred = result["ridge_prediction_scaled"]
    remaining = result["ridge_remaining_scaled"]
    steps = np.arange(1, len(prophet) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts = axes[0, 0]
    ax_ts.plot(steps, prophet, label="Prophet residual", color="#8172B2", linewidth=1.0)
    ax_ts.plot(steps, ridge_pred, label="Ridge prediction", color="#DD8452", linewidth=1.0)
    ax_ts.plot(steps, remaining, label="Ridge remaining", color="#4C72B0", linewidth=1.0)
    ax_ts.set_title(f"Residuals — {container_id}")
    ax_ts.set_xlabel("Validation step")
    ax_ts.set_ylabel("Scaled value")
    ax_ts.legend(fontsize=8)
    ax_ts.grid(alpha=0.3)

    from statsmodels.tsa.stattools import acf, pacf

    for ax, series, name, color in (
        (axes[0, 1], remaining, "ACF (remaining)", "#4C72B0"),
        (axes[1, 0], remaining, "PACF (remaining)", "#4C72B0"),
    ):
        s = series[np.isfinite(series)]
        nlags = min(30, len(s) - 1)
        if name.startswith("ACF"):
            vals = acf(s, nlags=nlags, fft=True)
        else:
            vals = pacf(s, nlags=nlags, method="ywm")
        ax.stem(np.arange(len(vals)), vals, linefmt=color, markerfmt="o", basefmt=" ")
        ax.axhline(0.0, color="#7F7F7F", linewidth=0.8)
        ax.set_title(name)
        ax.grid(alpha=0.3)

    ax_fft = axes[1, 1]
    for series, label, color in (
        (prophet, "Prophet", "#8172B2"),
        (remaining, "Remaining", "#4C72B0"),
    ):
        s = series[np.isfinite(series)]
        if len(s) >= 8:
            freqs, power = periodogram(s, fs=1.0)
            ax_fft.semilogy(freqs[1:], power[1:], label=label, color=color)
    ax_fft.set_title("Power spectrum")
    ax_fft.set_xlabel("Frequency")
    ax_fft.legend(fontsize=8)
    ax_fft.grid(alpha=0.3)

    fig.suptitle(f"Case study — {container_id}", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, path)


def plot_cohort_acf_paired(
    diag_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ["prophet_avg_abs_acf_1_10", "remaining_avg_abs_acf_1_10"]
    labels = ["Prophet", "Ridge remaining"]
    means = [diag_df[m].mean() for m in metrics]
    ax.bar(labels, means, color=["#8172B2", "#4C72B0"], alpha=0.85)
    ax.set_title("Cohort mean |ACF| (lags 1–10)")
    ax.set_ylabel("Mean |ACF|")
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, path)
