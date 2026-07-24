"""Cross-methodology plotting for Hybrid vs Global GRU comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.global_config import DEMO_CONTAINER_ID, INPUT_WINDOW
from utils.global_plotting import get_train_context_real

COLOR_ACTUAL = "#2CA02C"
COLOR_HYBRID = "#C44E52"
COLOR_GLOBAL = "#4C72B0"
COLOR_TRAIN_CONTEXT = "#6C6C6C"
VALIDATION_SHADE = "#4C72B0"
COLOR_REFERENCE_LINE = "#7F7F7F"

FIGURE_DPI = 150
SIDE_BY_SIDE_FIGURE_SIZE = (12.0, 4.5)
SAMPLE_GRID_FIGURE_SIZE = (12.0, 3.0)
HISTOGRAM_FIGURE_SIZE = (8.0, 5.0)
SCATTER_FIGURE_SIZE = (7.5, 7.0)

InferenceCache = dict[str, dict[str, np.ndarray]]


def _save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    """Save a matplotlib figure as PNG and PDF."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def select_comparison_sample_container_ids(
    per_container_delta: pd.DataFrame,
    demo_container_id: str = DEMO_CONTAINER_ID,
    n_samples: int = 4,
) -> list[str]:
    """
    Select demo plus best, worst, and median ΔMAE containers.

    Best = most negative ``delta_day1_mae`` (Global better than Hybrid).
    Worst = most positive ``delta_day1_mae`` (Global worse than Hybrid).
    """
    if per_container_delta.empty:
        return []

    ordered: list[str] = []
    if demo_container_id in set(per_container_delta["container_id"]):
        ordered.append(demo_container_id)

    ranked = per_container_delta.sort_values("delta_day1_mae").reset_index(drop=True)
    candidates = [
        ranked.iloc[0]["container_id"],
        ranked.iloc[-1]["container_id"],
        ranked.iloc[len(ranked) // 2]["container_id"],
    ]
    for container_id in candidates:
        if container_id not in ordered:
            ordered.append(container_id)
        if len(ordered) >= n_samples:
            break

    return ordered[:n_samples]


def _plot_forecast_axis(
    ax: plt.Axes,
    train_context_real: np.ndarray,
    actual_day1_real: np.ndarray,
    day1_pred_real: np.ndarray,
    methodology_label: str,
    methodology_color: str,
    day1_mae: float | None = None,
    show_legend: bool = False,
) -> None:
    """Draw one methodology forecast panel on shared axes."""
    train_context_real = np.ravel(train_context_real)
    actual_day1_real = np.ravel(actual_day1_real)
    day1_pred_real = np.ravel(day1_pred_real)

    n_train = len(train_context_real)
    n_val = len(actual_day1_real)
    x_train = np.arange(n_train)
    x_val = np.arange(n_train, n_train + n_val)
    boundary = n_train

    ax.plot(
        x_train,
        train_context_real,
        color=COLOR_TRAIN_CONTEXT,
        linewidth=1.2,
        label="Train context (actual)",
    )
    ax.plot(
        x_val,
        actual_day1_real,
        color=COLOR_ACTUAL,
        linewidth=1.8,
        label="Validation actual",
    )
    ax.plot(
        x_val,
        day1_pred_real,
        color=methodology_color,
        linewidth=1.6,
        linestyle="--",
        label=f"{methodology_label} predicted",
    )
    ax.axvline(boundary, color="black", linestyle=":", linewidth=1.0)
    ax.axvspan(boundary, boundary + n_val, color=VALIDATION_SHADE, alpha=0.08)

    title = methodology_label
    if day1_mae is not None:
        title += f" (MAE = {day1_mae:.4f}%)"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Timestep (15-min intervals; validation starts at boundary)")
    ax.set_ylabel("CPU utilization (%)")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="best")


def plot_side_by_side_actual_vs_predicted(
    container_id: str,
    train_context_real: np.ndarray,
    hybrid_cache: dict[str, np.ndarray],
    global_cache: dict[str, np.ndarray],
    hybrid_day1_mae: float | None,
    global_day1_mae: float | None,
    output_base: Path,
) -> list[str]:
    """Side-by-side Actual vs Predicted for one container (shared y-axis)."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=SIDE_BY_SIDE_FIGURE_SIZE,
        sharey=True,
    )

    _plot_forecast_axis(
        axes[0],
        train_context_real,
        hybrid_cache["actual_day1_real"],
        hybrid_cache["day1_pred_real"],
        methodology_label="Hybrid Prophet + GRU",
        methodology_color=COLOR_HYBRID,
        day1_mae=hybrid_day1_mae,
        show_legend=True,
    )
    _plot_forecast_axis(
        axes[1],
        train_context_real,
        global_cache["actual_day1_real"],
        global_cache["day1_pred_real"],
        methodology_label="Global GRU",
        methodology_color=COLOR_GLOBAL,
        day1_mae=global_day1_mae,
        show_legend=False,
    )

    fig.suptitle(
        f"Hybrid vs Global GRU — Day-1 Forecast ({container_id})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, output_base)


def plot_side_by_side_samples_panel(
    container_ids: list[str],
    hybrid_inference_cache: InferenceCache,
    global_inference_cache: InferenceCache,
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
    per_container_delta: pd.DataFrame,
    output_base: Path,
) -> list[str]:
    """Multi-row panel: each row is one container, Hybrid left / Global right."""
    if not container_ids:
        return []

    delta_lookup = per_container_delta.set_index("container_id")
    n_containers = len(container_ids)
    fig, axes = plt.subplots(
        n_containers,
        2,
        figsize=(SAMPLE_GRID_FIGURE_SIZE[0], SAMPLE_GRID_FIGURE_SIZE[1] * n_containers),
        sharey="row",
    )
    if n_containers == 1:
        axes = np.array([axes])

    for row_idx, container_id in enumerate(container_ids):
        train_context_real = get_train_context_real(
            container_id,
            global_train,
            scalers,
        )
        hybrid_cache = hybrid_inference_cache[container_id]
        global_cache = global_inference_cache[container_id]
        delta_row = delta_lookup.loc[container_id]
        hybrid_mae = float(delta_row["hybrid_day1_mae"])
        global_mae = float(delta_row["global_day1_mae"])
        delta_mae = float(delta_row["delta_day1_mae"])

        _plot_forecast_axis(
            axes[row_idx, 0],
            train_context_real,
            hybrid_cache["actual_day1_real"],
            hybrid_cache["day1_pred_real"],
            methodology_label="Hybrid",
            methodology_color=COLOR_HYBRID,
            day1_mae=hybrid_mae,
            show_legend=row_idx == 0,
        )
        _plot_forecast_axis(
            axes[row_idx, 1],
            train_context_real,
            global_cache["actual_day1_real"],
            global_cache["day1_pred_real"],
            methodology_label="Global GRU",
            methodology_color=COLOR_GLOBAL,
            day1_mae=global_mae,
            show_legend=False,
        )
        axes[row_idx, 0].set_ylabel(
            f"{container_id}\nΔMAE {delta_mae:+.4f}%\nCPU (%)",
            fontsize=9,
        )

    fig.suptitle(
        "Hybrid vs Global GRU — Sample Containers (Best / Worst / Median ΔMAE)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(fig, output_base)


def plot_mae_delta_histogram(
    per_container_delta: pd.DataFrame,
    output_base: Path,
) -> list[str]:
    """Histogram of per-container Day-1 MAE deltas (Global − Hybrid)."""
    deltas = per_container_delta["delta_day1_mae"].dropna()
    mean_delta = float(deltas.mean())

    fig, ax = plt.subplots(figsize=HISTOGRAM_FIGURE_SIZE)
    ax.hist(
        deltas,
        bins=20,
        color=COLOR_GLOBAL,
        edgecolor="black",
        alpha=0.85,
    )
    ax.axvline(0.0, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=1.5, label="No change (Δ = 0)")
    ax.axvline(
        mean_delta,
        color=COLOR_HYBRID,
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ΔMAE = {mean_delta:+.4f}%",
    )
    ax.set_title(
        "Hybrid vs Global GRU — Per-Container Day-1 MAE Delta Distribution",
        fontsize=12,
    )
    ax.set_xlabel("ΔMAE (Global − Hybrid, real CPU %)")
    ax.set_ylabel("Container count")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def plot_mae_scatter(
    per_container_delta: pd.DataFrame,
    output_base: Path,
) -> list[str]:
    """Scatter Hybrid MAE vs Global MAE with 45° reference line."""
    hybrid_mae = per_container_delta["hybrid_day1_mae"]
    global_mae = per_container_delta["global_day1_mae"]

    fig, ax = plt.subplots(figsize=SCATTER_FIGURE_SIZE)
    ax.scatter(
        hybrid_mae,
        global_mae,
        color=COLOR_GLOBAL,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        s=42,
    )

    min_val = float(min(hybrid_mae.min(), global_mae.min()))
    max_val = float(max(hybrid_mae.max(), global_mae.max()))
    padding = 0.05 * (max_val - min_val) if max_val > min_val else 0.5
    axis_min = max(0.0, min_val - padding)
    axis_max = max_val + padding
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=COLOR_REFERENCE_LINE,
        linestyle="--",
        linewidth=1.5,
        label="Equal MAE (45°)",
    )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        "Hybrid vs Global GRU — Per-Container Day-1 MAE Scatter",
        fontsize=12,
    )
    ax.set_xlabel("Hybrid Prophet + GRU Day-1 MAE (real CPU %)")
    ax.set_ylabel("Global GRU Day-1 MAE (real CPU %)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def plot_mae_distribution_overlay(
    per_container_delta: pd.DataFrame,
    output_base: Path,
) -> list[str]:
    """Overlaid histograms of per-container Day-1 MAE for both methodologies."""
    hybrid_values = per_container_delta["hybrid_day1_mae"].dropna()
    global_values = per_container_delta["global_day1_mae"].dropna()
    combined = pd.concat([hybrid_values, global_values])
    bins = np.linspace(float(combined.min()), float(combined.max()), 21)

    fig, ax = plt.subplots(figsize=HISTOGRAM_FIGURE_SIZE)
    ax.hist(
        hybrid_values,
        bins=bins,
        color=COLOR_HYBRID,
        edgecolor="black",
        alpha=0.55,
        label=f"Hybrid (mean = {hybrid_values.mean():.4f}%)",
    )
    ax.hist(
        global_values,
        bins=bins,
        color=COLOR_GLOBAL,
        edgecolor="black",
        alpha=0.55,
        label=f"Global GRU (mean = {global_values.mean():.4f}%)",
    )
    ax.set_title(
        "Hybrid vs Global GRU — Per-Container Day-1 MAE Distribution",
        fontsize=12,
    )
    ax.set_xlabel("Day-1 MAE (real CPU %)")
    ax.set_ylabel("Container count")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def generate_comparison_plots(
    per_container_delta: pd.DataFrame,
    hybrid_inference_cache: InferenceCache,
    global_inference_cache: InferenceCache,
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
    plots_dir: Path,
    demo_container_id: str = DEMO_CONTAINER_ID,
    sample_container_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Generate all Phase 6 cross-methodology comparison figures.

    Returns mapping of plot category to saved file paths.
    """
    plots_dir = Path(plots_dir)
    side_by_side_dir = plots_dir / "side_by_side"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, list[str]] = {
        "side_by_side_demo": [],
        "side_by_side_samples": [],
        "mae_delta_histogram": [],
        "mae_scatter": [],
        "mae_distribution_overlay": [],
    }

    delta_lookup = per_container_delta.set_index("container_id")

    if demo_container_id not in hybrid_inference_cache:
        raise KeyError(
            f"Demo container {demo_container_id!r} missing from hybrid inference cache."
        )
    if demo_container_id not in global_inference_cache:
        raise KeyError(
            f"Demo container {demo_container_id!r} missing from global inference cache."
        )

    demo_delta = delta_lookup.loc[demo_container_id]
    train_context_real = get_train_context_real(
        demo_container_id,
        global_train,
        scalers,
        input_window=INPUT_WINDOW,
    )
    saved_paths["side_by_side_demo"].extend(
        plot_side_by_side_actual_vs_predicted(
            container_id=demo_container_id,
            train_context_real=train_context_real,
            hybrid_cache=hybrid_inference_cache[demo_container_id],
            global_cache=global_inference_cache[demo_container_id],
            hybrid_day1_mae=float(demo_delta["hybrid_day1_mae"]),
            global_day1_mae=float(demo_delta["global_day1_mae"]),
            output_base=side_by_side_dir / demo_container_id,
        ),
    )

    samples = sample_container_ids or select_comparison_sample_container_ids(
        per_container_delta,
        demo_container_id=demo_container_id,
        n_samples=4,
    )
    sample_ids = [
        container_id
        for container_id in samples
        if container_id in hybrid_inference_cache
        and container_id in global_inference_cache
    ]
    if sample_ids:
        saved_paths["side_by_side_samples"].extend(
            plot_side_by_side_samples_panel(
                container_ids=sample_ids,
                hybrid_inference_cache=hybrid_inference_cache,
                global_inference_cache=global_inference_cache,
                global_train=global_train,
                scalers=scalers,
                per_container_delta=per_container_delta,
                output_base=plots_dir / "side_by_side_samples",
            ),
        )

    saved_paths["mae_delta_histogram"].extend(
        plot_mae_delta_histogram(
            per_container_delta=per_container_delta,
            output_base=plots_dir / "mae_delta_histogram",
        ),
    )
    saved_paths["mae_scatter"].extend(
        plot_mae_scatter(
            per_container_delta=per_container_delta,
            output_base=plots_dir / "mae_scatter",
        ),
    )
    saved_paths["mae_distribution_overlay"].extend(
        plot_mae_distribution_overlay(
            per_container_delta=per_container_delta,
            output_base=plots_dir / "mae_distribution_overlay",
        ),
    )

    return saved_paths
