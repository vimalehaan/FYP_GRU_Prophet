"""Publication-quality plotting for Global GRU baseline evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend for script/notebook-safe figure export.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.global_config import DEMO_CONTAINER_ID, INPUT_WINDOW

COLOR_ACTUAL = "#2CA02C"
COLOR_PREDICTED = "#4C72B0"
COLOR_TRAIN_CONTEXT = "#6C6C6C"
VALIDATION_SHADE = "#4C72B0"

FIGURE_DPI = 150
SINGLE_FIGURE_SIZE = (10.0, 4.5)
SAMPLE_GRID_FIGURE_SIZE = (10.0, 3.2)
ERROR_DIST_FIGURE_SIZE = (8.0, 5.0)

METRIC_COLUMNS = ("day1_mae", "day1_rmse", "day1_mape")


def _save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    """Save a matplotlib figure as PNG and PDF."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def get_train_context_real(
    container_id: str,
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
    input_window: int = INPUT_WINDOW,
    target_column: str = "cpu_scaled",
) -> np.ndarray:
    """Return the last ``input_window`` train-period actual CPU values in real %."""
    train_container = (
        global_train[global_train["container_id"] == container_id]
        .sort_values("time_stamp")
    )
    if train_container.empty:
        raise ValueError(f"Container {container_id!r} not found in global_train.")

    scaled = train_container[target_column].values[-input_window:]
    scaler = scalers[container_id]
    return scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()


def select_sample_container_ids(
    evaluation_df: pd.DataFrame,
    demo_container_id: str = DEMO_CONTAINER_ID,
    n_samples: int = 4,
) -> list[str]:
    """
    Select demo plus best, worst, and median Day-1 MAE containers.

    Returns up to ``n_samples`` unique container IDs (minimum 1 when demo present).
    """
    if evaluation_df.empty:
        return []

    ordered: list[str] = []
    if demo_container_id in set(evaluation_df["container_id"]):
        ordered.append(demo_container_id)

    ranked = evaluation_df.sort_values("day1_mae").reset_index(drop=True)
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


def plot_actual_vs_predicted(
    container_id: str,
    train_context_real: np.ndarray,
    actual_day1_real: np.ndarray,
    day1_pred_real: np.ndarray,
    output_base: Path,
    day1_mae: float | None = None,
) -> list[str]:
    """
    Plot train context, validation actual, and Day-1 predicted CPU utilization.

    Parameters
    ----------
    container_id:
        Container identifier for title annotation.
    train_context_real:
        Last train-window actual CPU % values shown as context.
    actual_day1_real:
        Validation-period actual Day-1 CPU %.
    day1_pred_real:
        Model-predicted Day-1 CPU %.
    output_base:
        Output path without extension (writes .png and .pdf).
    day1_mae:
        Optional Day-1 MAE for subtitle annotation.
    """
    train_context_real = np.ravel(train_context_real)
    actual_day1_real = np.ravel(actual_day1_real)
    day1_pred_real = np.ravel(day1_pred_real)

    n_train = len(train_context_real)
    n_val = len(actual_day1_real)
    x_train = np.arange(n_train)
    x_val = np.arange(n_train, n_train + n_val)
    boundary = n_train

    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)

    ax.plot(
        x_train,
        train_context_real,
        color=COLOR_TRAIN_CONTEXT,
        linewidth=1.5,
        label="Train context (actual)",
    )
    ax.plot(
        x_val,
        actual_day1_real,
        color=COLOR_ACTUAL,
        linewidth=2.0,
        label="Validation actual",
    )
    ax.plot(
        x_val,
        day1_pred_real,
        color=COLOR_PREDICTED,
        linewidth=1.8,
        linestyle="--",
        label="Global GRU predicted",
    )

    ax.axvline(
        boundary,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Train / validation boundary",
    )
    ax.axvspan(
        boundary,
        boundary + n_val,
        color=VALIDATION_SHADE,
        alpha=0.08,
        label="Validation region",
    )

    title = f"Global GRU Day-1 Forecast — {container_id}"
    if day1_mae is not None:
        title += f" (MAE = {day1_mae:.4f}%)"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Timestep (15-min intervals; validation starts at boundary)")
    ax.set_ylabel("CPU utilization (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    return _save_figure(fig, output_base)


def plot_actual_vs_predicted_panel(
    container_ids: list[str],
    inference_cache: dict[str, dict[str, np.ndarray]],
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
    evaluation_df: pd.DataFrame,
    output_base: Path,
) -> list[str]:
    """Multi-container Actual vs Predicted panel (shared styling)."""
    if not container_ids:
        return []

    n_containers = len(container_ids)
    fig, axes = plt.subplots(
        n_containers,
        1,
        figsize=(SAMPLE_GRID_FIGURE_SIZE[0], SAMPLE_GRID_FIGURE_SIZE[1] * n_containers),
    )
    if n_containers == 1:
        axes = [axes]

    mae_lookup = evaluation_df.set_index("container_id")["day1_mae"].to_dict()

    for ax, container_id in zip(axes, container_ids):
        cache = inference_cache[container_id]
        train_context_real = get_train_context_real(
            container_id,
            global_train,
            scalers,
        )
        actual_day1_real = np.ravel(cache["actual_day1_real"])
        day1_pred_real = np.ravel(cache["day1_pred_real"])

        n_train = len(train_context_real)
        n_val = len(actual_day1_real)
        x_train = np.arange(n_train)
        x_val = np.arange(n_train, n_train + n_val)
        boundary = n_train

        ax.plot(x_train, train_context_real, color=COLOR_TRAIN_CONTEXT, linewidth=1.2)
        ax.plot(x_val, actual_day1_real, color=COLOR_ACTUAL, linewidth=1.8, label="Actual")
        ax.plot(
            x_val,
            day1_pred_real,
            color=COLOR_PREDICTED,
            linewidth=1.5,
            linestyle="--",
            label="Predicted",
        )
        ax.axvline(boundary, color="black", linestyle=":", linewidth=1.0)
        ax.axvspan(boundary, boundary + n_val, color=VALIDATION_SHADE, alpha=0.08)

        mae = mae_lookup.get(container_id)
        subtitle = f"{container_id}"
        if mae is not None:
            subtitle += f" — MAE {mae:.4f}%"
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("Timestep (15-min)")
        ax.set_ylabel("CPU (%)")
        ax.grid(alpha=0.3)
        if container_id == container_ids[0]:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Global GRU — Actual vs Predicted (Sample Containers)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(fig, output_base)


def plot_error_distribution(
    evaluation_df: pd.DataFrame,
    output_base: Path,
    metric: str = "day1_mae",
) -> list[str]:
    """Histogram of per-container Day-1 error metric."""
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {METRIC_COLUMNS}")

    values = evaluation_df[metric].dropna()
    mean_value = float(values.mean())

    fig, ax = plt.subplots(figsize=ERROR_DIST_FIGURE_SIZE)
    ax.hist(
        values,
        bins=20,
        color=COLOR_PREDICTED,
        edgecolor="black",
        alpha=0.85,
    )
    ax.axvline(
        mean_value,
        color=COLOR_ACTUAL,
        linestyle="-",
        linewidth=1.5,
        label=f"Mean {metric} = {mean_value:.4f}",
    )
    ax.set_title(
        f"Global GRU — Per-Container Day-1 {metric.upper()} Distribution",
        fontsize=12,
    )
    ax.set_xlabel(f"{metric} (real CPU %)")
    ax.set_ylabel("Container count")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return _save_figure(fig, output_base)


def generate_evaluation_plots(
    evaluation_df: pd.DataFrame,
    inference_cache: dict[str, dict[str, np.ndarray]],
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
    plots_dir: Path,
    demo_container_id: str = DEMO_CONTAINER_ID,
    sample_container_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Generate all Phase 5 Global GRU evaluation figures.

    Returns
    -------
    Mapping of plot category to saved file paths.
    """
    plots_dir = Path(plots_dir)
    actual_vs_predicted_dir = plots_dir / "actual_vs_predicted"
    actual_vs_predicted_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, list[str]] = {
        "actual_vs_predicted": [],
        "actual_vs_predicted_panel": [],
        "error_distribution": [],
    }

    if demo_container_id not in inference_cache:
        raise KeyError(
            f"Demo container {demo_container_id!r} missing from inference cache."
        )

    demo_mae = None
    demo_rows = evaluation_df.loc[
        evaluation_df["container_id"] == demo_container_id,
        "day1_mae",
    ]
    if not demo_rows.empty:
        demo_mae = float(demo_rows.iloc[0])

    demo_cache = inference_cache[demo_container_id]
    saved_paths["actual_vs_predicted"].extend(
        plot_actual_vs_predicted(
            container_id=demo_container_id,
            train_context_real=get_train_context_real(
                demo_container_id,
                global_train,
                scalers,
            ),
            actual_day1_real=demo_cache["actual_day1_real"],
            day1_pred_real=demo_cache["day1_pred_real"],
            output_base=actual_vs_predicted_dir / demo_container_id,
            day1_mae=demo_mae,
        ),
    )

    samples = sample_container_ids or select_sample_container_ids(
        evaluation_df,
        demo_container_id=demo_container_id,
        n_samples=4,
    )
    for container_id in samples:
        if container_id == demo_container_id:
            continue
        if container_id not in inference_cache:
            continue

        row_mae = evaluation_df.loc[
            evaluation_df["container_id"] == container_id,
            "day1_mae",
        ]
        container_mae = float(row_mae.iloc[0]) if not row_mae.empty else None
        cache = inference_cache[container_id]
        saved_paths["actual_vs_predicted"].extend(
            plot_actual_vs_predicted(
                container_id=container_id,
                train_context_real=get_train_context_real(
                    container_id,
                    global_train,
                    scalers,
                ),
                actual_day1_real=cache["actual_day1_real"],
                day1_pred_real=cache["day1_pred_real"],
                output_base=actual_vs_predicted_dir / container_id,
                day1_mae=container_mae,
            ),
        )

    saved_paths["actual_vs_predicted_panel"].extend(
        plot_actual_vs_predicted_panel(
            container_ids=samples,
            inference_cache=inference_cache,
            global_train=global_train,
            scalers=scalers,
            evaluation_df=evaluation_df,
            output_base=plots_dir / "actual_vs_predicted_samples",
        ),
    )

    saved_paths["error_distribution"].extend(
        plot_error_distribution(
            evaluation_df=evaluation_df,
            output_base=plots_dir / "error_distribution",
            metric="day1_mae",
        ),
    )

    return saved_paths
