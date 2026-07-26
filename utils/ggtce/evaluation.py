"""GGTCE evaluation — extended Global GRU metrics."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.csrle.stage2_stats import cohort_summary, residual_cohort_summary
from utils.global_config import FORECAST_HORIZON, GLOBAL_TARGET, MAPE_EPSILON
from utils.global_evaluation import evaluate_selected_containers
from utils.global_inference import GlobalInferenceResult
from utils.hybrid_evaluation import compute_day1_mape, compute_day1_metrics


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a) < 1e-12:
        return float("nan")
    return float(r2_score(a, b))


def _actual_scaled(
    container_id: str,
    global_val: pd.DataFrame,
    forecast_horizon: int = FORECAST_HORIZON,
    target: str = GLOBAL_TARGET,
) -> np.ndarray:
    val_container = (
        global_val[global_val["container_id"] == container_id]
        .sort_values("time_stamp")
    )
    return val_container[target].values[:forecast_horizon]


def compute_ggtce_container_metrics(
    result: GlobalInferenceResult,
    container_id: str,
    global_val: pd.DataFrame,
) -> dict[str, float]:
    """Day-1 real + scaled forecast diagnostics for one container."""
    actual_real = np.ravel(result.actual_day1_real)
    pred_real = np.ravel(result.day1_pred_real)
    actual_scaled = _actual_scaled(container_id, global_val)
    pred_scaled = np.ravel(result.day1_pred_scaled)

    day1_mae, day1_rmse = compute_day1_metrics(actual_real, pred_real)
    day1_mape = compute_day1_mape(actual_real, pred_real, epsilon=MAPE_EPSILON)

    res_mae = float(mean_absolute_error(actual_scaled, pred_scaled))
    res_rmse = float(np.sqrt(mean_squared_error(actual_scaled, pred_scaled)))
    pearson_r = _safe_corr(actual_scaled, pred_scaled)
    r2 = _safe_r2(actual_scaled, pred_scaled)
    std_a = float(np.std(actual_scaled))
    std_p = float(np.std(pred_scaled))
    std_ratio = std_p / std_a if std_a > 1e-12 else float("nan")

    return {
        "container_id": container_id,
        "day1_mae": day1_mae,
        "day1_rmse": day1_rmse,
        "day1_mape": day1_mape,
        "forecast_mae_scaled": res_mae,
        "forecast_rmse_scaled": res_rmse,
        "forecast_pearson_r": pearson_r,
        "forecast_r2": r2,
        "forecast_std_ratio": std_ratio,
        "residual_mae_scaled": res_mae,
        "residual_rmse_scaled": res_rmse,
        "residual_pearson_r": pearson_r,
        "residual_r2": r2,
        "residual_std_ratio": std_ratio,
        "train_steps": result.n_train_steps,
        "validation_steps": result.n_val_steps,
        "input_window": result.input_window,
    }


def evaluate_ggtce_variant(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    global_gru_model: Any,
    scalers: dict[str, Any],
    input_window: int,
    variant_id: str,
    forecast_horizon: int = FORECAST_HORIZON,
) -> tuple[
    pd.DataFrame,
    dict[str, GlobalInferenceResult],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Evaluate one GGTCE variant with extended metrics."""
    base_df, inference_results, failures, skipped = evaluate_selected_containers(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        global_gru_model=global_gru_model,
        scalers=scalers,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    rows: list[dict[str, Any]] = []
    for container_id, result in inference_results.items():
        metrics = compute_ggtce_container_metrics(result, container_id, global_val)
        metrics["variant_id"] = variant_id
        rows.append(metrics)

    evaluation_df = pd.DataFrame(rows)
    if not evaluation_df.empty:
        evaluation_df = evaluation_df.sort_values("container_id").reset_index(drop=True)

    return evaluation_df, inference_results, failures, skipped


def summarize_variant_cohort(evaluation_df: pd.DataFrame) -> dict[str, Any]:
    """Cohort summaries for primary metrics."""
    summary: dict[str, Any] = {"n_containers": int(len(evaluation_df))}
    numeric_cols = [
        c for c in evaluation_df.columns
        if c not in {"container_id", "variant_id"} and pd.api.types.is_numeric_dtype(
            evaluation_df[c],
        )
    ]
    for col in numeric_cols:
        summary[col] = cohort_summary(evaluation_df[col])

    residual_cols = [
        "residual_mae_scaled",
        "residual_rmse_scaled",
        "residual_pearson_r",
        "residual_r2",
        "residual_std_ratio",
    ]
    if all(c in evaluation_df.columns for c in residual_cols):
        summary["residual_block"] = residual_cohort_summary(
            evaluation_df[residual_cols],
        )
    return summary


def save_inference_cache(
    inference_results: dict[str, GlobalInferenceResult],
    path: Path,
) -> None:
    """Save Day-1 arrays for case studies (no raw ndarrays in JSON)."""
    cache = {
        cid: {
            "actual_day1_real": np.ravel(res.actual_day1_real).tolist(),
            "day1_pred_real": np.ravel(res.day1_pred_real).tolist(),
            "day1_pred_scaled": np.ravel(res.day1_pred_scaled).tolist(),
            "input_window": res.input_window,
        }
        for cid, res in inference_results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)
