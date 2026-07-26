"""Publication-quality plots for Temporal Memory Analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_CPU = "#2CA02C"
COLOR_PROPHET = "#DD8452"
COLOR_HYBRID = "#4C72B0"
COLOR_MEAN = "#1F77B4"
COLOR_CI = "#AAAAAA"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def plot_cohort_acf(
    summary: dict[str, np.ndarray],
    title: str,
    output_path: Path,
    significance_bound: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    lags = summary["lags"]
    ax.plot(lags, summary["mean"], color=COLOR_MEAN, lw=1.5, label="Cohort mean")
    ax.plot(
        lags,
        summary["median"],
        color=COLOR_PROPHET,
        lw=1.0,
        ls="--",
        alpha=0.8,
        label="Cohort median",
    )
    ax.fill_between(
        lags,
        summary["ci_lo"],
        summary["ci_hi"],
        color=COLOR_CI,
        alpha=0.35,
        label="95% CI",
    )
    if significance_bound is not None:
        ax.axhline(significance_bound, color="red", ls=":", lw=1, label="±95% bound")
        ax.axhline(-significance_bound, color="red", ls=":", lw=1)
    ax.axvline(96, color="gray", ls="--", lw=0.8, alpha=0.7, label="Lag 96 (1 day)")
    _style_axes(ax, title, "Lag (15-min steps)", "ACF")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_cohort_pacf(
    summary: dict[str, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    lags = summary["lags"]
    ax.plot(lags, summary["mean"], color=COLOR_MEAN, lw=1.5, label="Cohort mean")
    ax.plot(
        lags,
        summary["median"],
        color=COLOR_PROPHET,
        lw=1.0,
        ls="--",
        alpha=0.8,
        label="Cohort median",
    )
    ax.fill_between(
        lags,
        summary["ci_lo"],
        summary["ci_hi"],
        color=COLOR_CI,
        alpha=0.35,
        label="95% CI",
    )
    ax.axvline(96, color="gray", ls="--", lw=0.8, alpha=0.7)
    _style_axes(ax, title, "Lag (15-min steps)", "PACF")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_memory_decay(
    summary: dict[str, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    lags = summary["lags"][1:]
    abs_mean = np.abs(summary["mean"][1:])
    ax.semilogy(lags, abs_mean + 1e-6, color=COLOR_MEAN, lw=1.5)
    ax.axvline(96, color="gray", ls="--", lw=0.8, alpha=0.7, label="1-day boundary")
    _style_axes(ax, title, "Lag (15-min steps)", "|ACF| (log scale)")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_daily_periodicity(
    periodicity: dict[str, Any],
    title: str,
    output_path: Path,
) -> None:
    lags = []
    means = []
    ci_lo = []
    ci_hi = []
    for lag_str, entry in sorted(
        periodicity["lags"].items(), key=lambda x: int(x[0])
    ):
        lags.append(int(lag_str))
        means.append(entry["mean_acf"])
        ci_lo.append(entry["bootstrap_ci_lo"])
        ci_hi.append(entry["bootstrap_ci_hi"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(lags))
    ax.bar(x, means, color=COLOR_MEAN, alpha=0.85, label="Mean ACF")
    ax.errorbar(
        x,
        means,
        yerr=[
            np.array(means) - np.array(ci_lo),
            np.array(ci_hi) - np.array(means),
        ],
        fmt="none",
        color="black",
        capsize=4,
        label="Bootstrap 95% CI",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lag}\n({lag // 96}d)" for lag in lags], fontsize=9)
    for thr in (0.1, 0.2, 0.3):
        ax.axhline(thr, ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-thr, ls=":", lw=0.8, alpha=0.5)
    _style_axes(ax, title, "Daily lag (multiples of 96 steps)", "ACF")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_fft_comparison(
    freqs_by_series: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {
        "cpu_original": COLOR_CPU,
        "prophet_residual": COLOR_PROPHET,
        "hybrid_post_forecast_residual": COLOR_HYBRID,
    }
    labels = {
        "cpu_original": "Original CPU",
        "prophet_residual": "Prophet-equivalent residual",
        "hybrid_post_forecast_residual": "Hybrid post-forecast residual",
    }
    for key, (freqs, psd) in freqs_by_series.items():
        if len(freqs) == 0:
            continue
        ax.semilogy(freqs[1:], psd[1:] + 1e-12, color=colors.get(key, "black"), lw=1.2,
                    label=labels.get(key, key))
    daily_f = 1.0 / 96.0
    ax.axvline(daily_f, color="gray", ls="--", lw=0.8, alpha=0.7, label="Daily (1/96)")
    _style_axes(ax, title, "Frequency (cycles per step)", "Mean PSD")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_memory_length_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    finite = values[np.isfinite(values)]
    ax.hist(finite, bins=25, color=COLOR_MEAN, alpha=0.85, edgecolor="white")
    if len(finite):
        ax.axvline(np.median(finite), color=COLOR_PROPHET, ls="--", lw=1.5,
                   label=f"Median={np.median(finite):.1f}")
    _style_axes(ax, title, xlabel, "Count")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_cumulative_capture(
    cohort_report: dict[str, Any],
    title: str,
    output_path: Path,
) -> None:
    windows = sorted(int(k) for k in cohort_report["cohort"]["windows"])
    means = [
        cohort_report["cohort"]["windows"][str(w)]["mean_pct"] * 100 for w in windows
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        [str(w) for w in windows],
        means,
        color=COLOR_MEAN,
        alpha=0.85,
    )
    _style_axes(
        ax,
        title,
        "Window size (lags)",
        "Cumulative squared-ACF capture (%)",
    )
    _save(fig, output_path)


def plot_window_sufficiency_comparison(
    reports_by_series: dict[str, dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    series_names = list(reports_by_series.keys())
    windows = sorted(
        int(k)
        for k in reports_by_series[series_names[0]]["cohort"]["windows"]
    )
    x = np.arange(len(windows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [COLOR_CPU, COLOR_PROPHET, COLOR_HYBRID]
    for i, sname in enumerate(series_names):
        means = [
            reports_by_series[sname]["cohort"]["windows"][str(w)]["mean_pct"] * 100
            for w in windows
        ]
        ax.bar(x + i * width, means, width, label=sname.replace("_", " "), color=colors[i % 3])
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(w) for w in windows])
    _style_axes(ax, title, "Window size (lags)", "Mean capture (%)")
    ax.legend(fontsize=9)
    _save(fig, output_path)


def plot_acf_heatmap(
    acf_matrix: np.ndarray,
    lags: np.ndarray,
    container_ids: list[str],
    title: str,
    output_path: Path,
    max_lag_plot: int = 672,
    lag_stride: int = 4,
) -> None:
    end_idx = min(max_lag_plot + 1, acf_matrix.shape[1])
    data = acf_matrix[:, 1:end_idx:lag_stride]
    fig_h = max(6, min(12, len(container_ids) * 0.06))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Lag (15-min steps)", fontsize=11)
    ax.set_ylabel("Container index", fontsize=11)
    tick_step = max(1, data.shape[1] // 8)
    xticks = np.arange(0, data.shape[1], tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks * lag_stride) + 1, fontsize=8)
    ax.set_yticks(np.arange(0, len(container_ids), max(1, len(container_ids) // 20)))
    fig.colorbar(im, ax=ax, label="ACF")
    _save(fig, output_path)


def plot_case_study_panel(
    bundle: dict[str, Any],
    acf_lags: np.ndarray,
    acf_vals: np.ndarray,
    pacf_vals: np.ndarray,
    fft_freqs: np.ndarray,
    fft_power: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    cid = bundle["container_id"]
    t_cpu = np.arange(len(bundle["y_full"]))
    axes[0, 0].plot(t_cpu, bundle["y_full"], color=COLOR_CPU, lw=0.8)
    axes[0, 0].axvline(bundle["train_len"], color="gray", ls="--", alpha=0.6)
    axes[0, 0].set_title(f"{cid} — Original CPU")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("CPU %")

    t_res = np.arange(len(bundle["prophet_residual_full"]))
    axes[0, 1].plot(t_res, bundle["prophet_residual_full"], color=COLOR_PROPHET, lw=0.8)
    axes[0, 1].axvline(bundle["train_len"], color="gray", ls="--", alpha=0.6)
    axes[0, 1].set_title("Prophet-equivalent residual")
    axes[0, 1].set_xlabel("Step")

    h = bundle["hybrid_post_forecast_residual_day1"]
    axes[1, 0].plot(np.arange(len(h)), h, color=COLOR_HYBRID, lw=1.0)
    axes[1, 0].axhline(0, color="gray", ls=":", lw=0.8)
    axes[1, 0].set_title("Hybrid post-forecast residual (day-1)")
    axes[1, 0].set_xlabel("Validation step")

    axes[1, 1].stem(acf_lags[1:97], acf_vals[1:97], linefmt=COLOR_MEAN, markerfmt=" ", basefmt=" ")
    axes[1, 1].set_title("ACF (first 96 lags)")
    axes[1, 1].set_xlabel("Lag")

    axes[2, 0].stem(
        acf_lags[1:min(97, len(pacf_vals))],
        pacf_vals[1:min(97, len(pacf_vals))],
        linefmt=COLOR_PROPHET,
        markerfmt=" ",
        basefmt=" ",
    )
    axes[2, 0].set_title("PACF (first 96 lags)")
    axes[2, 0].set_xlabel("Lag")

    if len(fft_freqs) > 1:
        axes[2, 1].semilogy(fft_freqs[1:], fft_power[1:] + 1e-12, color=COLOR_CPU, lw=1.0)
    axes[2, 1].set_title("Periodogram")
    axes[2, 1].set_xlabel("Frequency")

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


def plot_hypothetical_windows(
    window_report: dict[str, Any],
    title: str,
    output_path: Path,
) -> None:
    windows = sorted(int(k) for k in window_report["windows"])
    means = [window_report["windows"][str(w)]["mean_abs_acf_within_window"] for w in windows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(windows, means, marker="o", color=COLOR_MEAN, lw=1.5)
    ax.axvline(96, color="gray", ls="--", alpha=0.7, label="Current window (96)")
    _style_axes(
        ax,
        title,
        "Candidate input window (lags)",
        "Mean |ACF| within window",
    )
    ax.legend(fontsize=9)
    _save(fig, output_path)
