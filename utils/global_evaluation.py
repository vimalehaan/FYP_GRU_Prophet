"""Multi-container Day 1 evaluation for the Global GRU baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.global_config import FORECAST_HORIZON, INPUT_WINDOW, MAPE_EPSILON
from utils.global_inference import GlobalInferenceResult, run_global_inference
from utils.hybrid_evaluation import (
    _evaluable_container_ids,
    compute_day1_mape,
    compute_day1_metrics,
    summarize_evaluation_metrics,
)

__all__ = [
    "_evaluable_container_ids",
    "compute_day1_mape",
    "compute_day1_metrics",
    "evaluate_selected_containers",
    "summarize_evaluation_metrics",
]


def evaluate_selected_containers(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    global_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> tuple[
    pd.DataFrame,
    dict[str, GlobalInferenceResult],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Evaluate Day 1 Global GRU forecasts for every container in selected_containers.

    Returns
    -------
    evaluation_df:
        Per-container Day 1 metrics (MAE, RMSE, MAPE) and step counts.
    inference_results:
        Mapping of container_id to GlobalInferenceResult.
    failures:
        List of (container_id, error_message) for containers that failed.
    skipped:
        List of (container_id, reason) for containers excluded before inference.
    """
    rows: list[dict[str, Any]] = []
    inference_results: dict[str, GlobalInferenceResult] = {}
    failures: list[tuple[str, str]] = []

    evaluable_ids, skipped = _evaluable_container_ids(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        scalers=scalers,
    )

    for container_id in evaluable_ids:
        try:
            result = run_global_inference(
                container_id=container_id,
                global_train=global_train,
                global_val=global_val,
                model=global_gru_model,
                scalers=scalers,
                input_window=input_window,
                forecast_horizon=forecast_horizon,
            )
            day1_mae, day1_rmse = compute_day1_metrics(
                result.actual_day1_real,
                result.day1_pred_real,
            )
            day1_mape = compute_day1_mape(
                result.actual_day1_real,
                result.day1_pred_real,
                epsilon=MAPE_EPSILON,
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
            "container_id",
        ).reset_index(drop=True)

    return evaluation_df, inference_results, failures, skipped
