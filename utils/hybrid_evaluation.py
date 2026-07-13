"""Multi-container Day 1 evaluation for the Hybrid Prophet + GRU model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils.hybrid_inference import HybridInferenceResult, run_hybrid_inference

# Minimum denominator for MAPE (percentage points on the real CPU % scale).
# Prevents division-by-near-zero when actual CPU utilization is 0 or very small.
MAPE_EPSILON = 0.01


def compute_day1_metrics(
    actual_day1_real: np.ndarray,
    day1_final_real: np.ndarray,
) -> tuple[float, float]:
    """Return Day 1 MAE and RMSE in real CPU percent."""
    day1_mae = float(
        mean_absolute_error(actual_day1_real, day1_final_real)
    )
    day1_rmse = float(
        np.sqrt(
            mean_squared_error(actual_day1_real, day1_final_real)
        )
    )
    return day1_mae, day1_rmse


def compute_day1_mape(
    actual_day1_real: np.ndarray,
    day1_final_real: np.ndarray,
    epsilon: float = MAPE_EPSILON,
) -> float:
    """
    Return Day 1 MAPE in real CPU percent.

    Uses an epsilon floor on the denominator:

        MAPE = mean(|actual - predicted| / max(|actual|, epsilon)) * 100

    Near-zero actual CPU values are common after inverse MinMax transform.
    A raw MAPE denominator of 0 produces undefined or extreme percentages.
    Flooring the denominator at ``epsilon`` percentage points keeps MAPE
    finite and comparable across containers without excluding timesteps.
    """
    actual = np.ravel(actual_day1_real)
    predicted = np.ravel(day1_final_real)
    denominator = np.maximum(np.abs(actual), epsilon)
    return float(
        np.mean(np.abs(actual - predicted) / denominator) * 100.0
    )


def _evaluable_container_ids(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    min_train_steps: int = 2,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return container IDs ready for inference and skipped IDs with reasons."""
    train_ids = set(global_train["container_id"])
    val_ids = set(global_val["container_id"])
    evaluable: list[str] = []
    skipped: list[tuple[str, str]] = []

    for container_id in selected_containers:
        if container_id not in train_ids or container_id not in val_ids:
            skipped.append(
                (container_id, "missing from frozen train/val data")
            )
            continue
        if container_id not in scalers:
            skipped.append(
                (container_id, "missing per-container scaler")
            )
            continue
        train_steps = int(
            (global_train["container_id"] == container_id).sum()
        )
        if train_steps < min_train_steps:
            skipped.append(
                (
                    container_id,
                    f"insufficient train steps ({train_steps})",
                )
            )
            continue
        evaluable.append(container_id)

    return evaluable, skipped


def evaluate_selected_containers(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    res_mean: float,
    res_std: float,
) -> tuple[
    pd.DataFrame,
    dict[str, HybridInferenceResult],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Evaluate Day 1 Hybrid forecasts for every container in selected_containers.

    Returns
    -------
    evaluation_df:
        Per-container Day 1 metrics and step counts.
    inference_results:
        Mapping of container_id to full inference outputs.
    failures:
        List of (container_id, error_message) for containers that failed.
    skipped:
        List of (container_id, reason) for containers excluded before inference.
    """
    rows: list[dict[str, Any]] = []
    inference_results: dict[str, HybridInferenceResult] = {}
    failures: list[tuple[str, str]] = []

    evaluable_ids, skipped = _evaluable_container_ids(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        scalers=scalers,
    )

    for container_id in evaluable_ids:
        try:
            result = run_hybrid_inference(
                container_id=container_id,
                global_train=global_train,
                global_val=global_val,
                hybrid_gru_model=hybrid_gru_model,
                scalers=scalers,
                res_mean=res_mean,
                res_std=res_std,
            )
            day1_mae, day1_rmse = compute_day1_metrics(
                result.actual_day1_real,
                result.day1_final_real,
            )
            day1_mape = compute_day1_mape(
                result.actual_day1_real,
                result.day1_final_real,
            )
            inference_results[container_id] = result
            rows.append({
                "container_id": container_id,
                "day1_mae": day1_mae,
                "day1_rmse": day1_rmse,
                "day1_mape": day1_mape,
                "train_steps": result.n_train_steps,
                "validation_steps": result.n_val_steps,
            })
        except Exception as exc:
            failures.append((container_id, str(exc)))

    evaluation_df = pd.DataFrame(rows)
    if not evaluation_df.empty:
        evaluation_df = evaluation_df.sort_values(
            "container_id"
        ).reset_index(drop=True)

    return evaluation_df, inference_results, failures, skipped


def summarize_evaluation_metrics(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute aggregate mean, std, min, and max for Day 1 metrics."""
    metric_columns = ["day1_mae", "day1_rmse", "day1_mape"]
    summary = evaluation_df[metric_columns].agg(["mean", "std", "min", "max"])
    return summary
