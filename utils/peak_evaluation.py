"""Peak-subset evaluation metrics for Peak-Aware Hybrid and Global research."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.global_inference import GlobalInferenceResult
from utils.hybrid_evaluation import compute_day1_metrics, summarize_evaluation_metrics
from utils.hybrid_inference import HybridInferenceResult
from utils.peak_detection import label_peak_timesteps

GLOBAL_INFERENCE_CACHE_PRED_KEY = "day1_pred_real"
HYBRID_INFERENCE_CACHE_PRED_KEY = "day1_final_real"


def label_day1_peak_timesteps(
    actual_day1_real: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Return a boolean mask for Day-1 validation timesteps meeting the peak rule.

    Peaks are defined on **actual** real CPU percent in the Day-1 forecast
    horizon using train-fitted per-container P90 thresholds.
    """
    actual = np.ravel(actual_day1_real)
    return label_peak_timesteps(actual, threshold)


def _subset_mae_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    """Compute MAE, RMSE, and step count for a boolean timestep mask."""
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    mask = np.ravel(mask).astype(bool)
    n_steps = int(mask.sum())
    if n_steps == 0:
        return float("nan"), float("nan"), 0

    actual_sub = actual[mask]
    predicted_sub = predicted[mask]
    mae = float(mean_absolute_error(actual_sub, predicted_sub))
    rmse = float(np.sqrt(mean_squared_error(actual_sub, predicted_sub)))
    return mae, rmse, n_steps


def compute_peak_subset_metrics_from_arrays(
    container_id: str,
    actual_day1_real: np.ndarray,
    day1_final_real: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute per-container peak metrics from Day-1 actual/predicted arrays."""
    actual = np.ravel(actual_day1_real)
    predicted = np.ravel(day1_final_real)
    peak_mask = label_day1_peak_timesteps(actual, threshold)
    non_peak_mask = ~peak_mask

    peak_mae, peak_rmse, n_peak = _subset_mae_rmse(
        actual, predicted, peak_mask
    )
    non_peak_mae, non_peak_rmse, n_non_peak = _subset_mae_rmse(
        actual, predicted, non_peak_mask
    )
    overall_mae, overall_rmse = compute_day1_metrics(actual, predicted)

    day1_steps = int(len(actual))
    return {
        "container_id": container_id,
        "threshold": float(threshold),
        "day1_steps": day1_steps,
        "peak_steps": n_peak,
        "non_peak_steps": n_non_peak,
        "peak_fraction": float(n_peak / day1_steps) if day1_steps else float("nan"),
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
        "non_peak_mae": non_peak_mae,
        "non_peak_rmse": non_peak_rmse,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
    }


def compute_peak_subset_metrics_from_global_arrays(
    container_id: str,
    actual_day1_real: np.ndarray,
    day1_pred_real: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute per-container peak metrics from Global Day-1 actual/predicted arrays."""
    return compute_peak_subset_metrics_from_arrays(
        container_id=container_id,
        actual_day1_real=actual_day1_real,
        day1_final_real=day1_pred_real,
        threshold=threshold,
    )


def compute_peak_subset_metrics_from_global_result(
    container_id: str,
    inference_result: GlobalInferenceResult,
    threshold: float,
) -> dict[str, Any]:
    """
    Compute per-container peak and non-peak Day-1 metrics for Global GRU inference.

    Parameters
    ----------
    container_id:
        Container identifier (``GlobalInferenceResult`` does not store this).
    inference_result:
        Output from frozen ``run_global_inference``.
    threshold:
        Train-fitted P90 threshold in real CPU percent for the container.
    """
    return compute_peak_subset_metrics_from_global_arrays(
        container_id=container_id,
        actual_day1_real=inference_result.actual_day1_real,
        day1_pred_real=inference_result.day1_pred_real,
        threshold=threshold,
    )


def compute_peak_subset_metrics(
    inference_result: HybridInferenceResult,
    threshold: float,
) -> dict[str, Any]:
    """
    Compute per-container peak and non-peak Day-1 metrics.

    Parameters
    ----------
    inference_result:
        Output from frozen ``run_hybrid_inference``.
    threshold:
        Train-fitted P90 threshold in real CPU percent for the container.

    Returns
    -------
    Dictionary with overall, peak-only, and non-peak MAE/RMSE plus step counts.
    """
    return compute_peak_subset_metrics_from_arrays(
        container_id=inference_result.container_id,
        actual_day1_real=inference_result.actual_day1_real,
        day1_final_real=inference_result.day1_final_real,
        threshold=threshold,
    )


def compute_peak_subset_metrics_from_cache(
    inference_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build per-container peak metrics from a lightweight inference cache."""
    rows: list[dict[str, Any]] = []
    for container_id, arrays in inference_cache.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        rows.append(
            compute_peak_subset_metrics_from_arrays(
                container_id=container_id,
                actual_day1_real=arrays["actual_day1_real"],
                day1_final_real=arrays["day1_final_real"],
                threshold=thresholds[container_id],
            )
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values("container_id").reset_index(drop=True)


def compute_peak_subset_metrics_from_global_cache(
    inference_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build per-container peak metrics from a Global inference cache."""
    rows: list[dict[str, Any]] = []
    for container_id, arrays in inference_cache.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        if GLOBAL_INFERENCE_CACHE_PRED_KEY not in arrays:
            raise KeyError(
                f"Global cache for {container_id} missing "
                f"{GLOBAL_INFERENCE_CACHE_PRED_KEY!r}"
            )
        rows.append(
            compute_peak_subset_metrics_from_global_arrays(
                container_id=container_id,
                actual_day1_real=arrays["actual_day1_real"],
                day1_pred_real=arrays[GLOBAL_INFERENCE_CACHE_PRED_KEY],
                threshold=thresholds[container_id],
            )
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values("container_id").reset_index(drop=True)


def build_peak_subset_comparison_per_container(
    treatment_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge treatment and baseline per-container peak metrics with deltas."""
    merged = treatment_df.merge(
        baseline_df,
        on="container_id",
        suffixes=("_treatment", "_baseline"),
    )
    merged["delta_peak_mae"] = (
        merged["peak_mae_treatment"] - merged["peak_mae_baseline"]
    )
    merged["delta_peak_rmse"] = (
        merged["peak_rmse_treatment"] - merged["peak_rmse_baseline"]
    )
    merged["delta_non_peak_mae"] = (
        merged["non_peak_mae_treatment"] - merged["non_peak_mae_baseline"]
    )
    merged["delta_non_peak_rmse"] = (
        merged["non_peak_rmse_treatment"] - merged["non_peak_rmse_baseline"]
    )
    return merged.sort_values("container_id").reset_index(drop=True)


def compute_peak_subset_metrics_batch(
    inference_results: dict[str, HybridInferenceResult],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build per-container peak-subset metrics for all inference results."""
    rows: list[dict[str, Any]] = []
    for container_id, result in inference_results.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        rows.append(
            compute_peak_subset_metrics(result, thresholds[container_id])
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values("container_id").reset_index(drop=True)


def compute_peak_subset_metrics_batch_global(
    inference_results: dict[str, GlobalInferenceResult],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build per-container peak-subset metrics for Global inference results."""
    rows: list[dict[str, Any]] = []
    for container_id, result in inference_results.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        rows.append(
            compute_peak_subset_metrics_from_global_result(
                container_id=container_id,
                inference_result=result,
                threshold=thresholds[container_id],
            )
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values("container_id").reset_index(drop=True)


def summarize_peak_subset_metrics(
    inference_results: dict[str, HybridInferenceResult],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Compute cohort-level pooled peak and non-peak Day-1 metrics.

    Peak and non-peak MAE/RMSE are computed on **pooled timesteps** across all
    containers, matching the Phase 3 evaluation protocol.
    """
    peak_actual_chunks: list[np.ndarray] = []
    peak_predicted_chunks: list[np.ndarray] = []
    non_peak_actual_chunks: list[np.ndarray] = []
    non_peak_predicted_chunks: list[np.ndarray] = []
    per_container_peak_steps = 0
    per_container_non_peak_steps = 0
    containers_evaluated = 0

    for container_id, result in inference_results.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        actual = np.ravel(result.actual_day1_real)
        predicted = np.ravel(result.day1_final_real)
        peak_mask = label_day1_peak_timesteps(actual, thresholds[container_id])

        peak_actual_chunks.append(actual[peak_mask])
        peak_predicted_chunks.append(predicted[peak_mask])
        non_peak_actual_chunks.append(actual[~peak_mask])
        non_peak_predicted_chunks.append(predicted[~peak_mask])

        per_container_peak_steps += int(peak_mask.sum())
        per_container_non_peak_steps += int((~peak_mask).sum())
        containers_evaluated += 1

    peak_actual = np.concatenate(peak_actual_chunks)
    peak_predicted = np.concatenate(peak_predicted_chunks)
    non_peak_actual = np.concatenate(non_peak_actual_chunks)
    non_peak_predicted = np.concatenate(non_peak_predicted_chunks)

    peak_mae = float(mean_absolute_error(peak_actual, peak_predicted))
    peak_rmse = float(
        np.sqrt(mean_squared_error(peak_actual, peak_predicted))
    )
    non_peak_mae = float(
        mean_absolute_error(non_peak_actual, non_peak_predicted)
    )
    non_peak_rmse = float(
        np.sqrt(mean_squared_error(non_peak_actual, non_peak_predicted))
    )

    total_steps = per_container_peak_steps + per_container_non_peak_steps
    return {
        "containers_evaluated": containers_evaluated,
        "total_day1_steps": total_steps,
        "peak_steps": per_container_peak_steps,
        "non_peak_steps": per_container_non_peak_steps,
        "peak_step_fraction": (
            float(per_container_peak_steps / total_steps)
            if total_steps
            else float("nan")
        ),
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
        "non_peak_mae": non_peak_mae,
        "non_peak_rmse": non_peak_rmse,
    }


def summarize_peak_subset_metrics_from_global_cache(
    inference_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Compute cohort-level pooled peak and non-peak Day-1 metrics from a cache.

    Expects each cache entry to contain ``actual_day1_real`` and
    ``day1_pred_real`` arrays.
    """
    peak_actual_chunks: list[np.ndarray] = []
    peak_predicted_chunks: list[np.ndarray] = []
    non_peak_actual_chunks: list[np.ndarray] = []
    non_peak_predicted_chunks: list[np.ndarray] = []
    per_container_peak_steps = 0
    per_container_non_peak_steps = 0
    containers_evaluated = 0

    for container_id, arrays in inference_cache.items():
        if container_id not in thresholds:
            raise KeyError(
                f"Missing peak threshold for container {container_id}"
            )
        actual = np.ravel(arrays["actual_day1_real"])
        predicted = np.ravel(arrays[GLOBAL_INFERENCE_CACHE_PRED_KEY])
        peak_mask = label_day1_peak_timesteps(actual, thresholds[container_id])

        peak_actual_chunks.append(actual[peak_mask])
        peak_predicted_chunks.append(predicted[peak_mask])
        non_peak_actual_chunks.append(actual[~peak_mask])
        non_peak_predicted_chunks.append(predicted[~peak_mask])

        per_container_peak_steps += int(peak_mask.sum())
        per_container_non_peak_steps += int((~peak_mask).sum())
        containers_evaluated += 1

    peak_actual = np.concatenate(peak_actual_chunks)
    peak_predicted = np.concatenate(peak_predicted_chunks)
    non_peak_actual = np.concatenate(non_peak_actual_chunks)
    non_peak_predicted = np.concatenate(non_peak_predicted_chunks)

    peak_mae = float(mean_absolute_error(peak_actual, peak_predicted))
    peak_rmse = float(
        np.sqrt(mean_squared_error(peak_actual, peak_predicted))
    )
    non_peak_mae = float(
        mean_absolute_error(non_peak_actual, non_peak_predicted)
    )
    non_peak_rmse = float(
        np.sqrt(mean_squared_error(non_peak_actual, non_peak_predicted))
    )

    total_steps = per_container_peak_steps + per_container_non_peak_steps
    return {
        "containers_evaluated": containers_evaluated,
        "total_day1_steps": total_steps,
        "peak_steps": per_container_peak_steps,
        "non_peak_steps": per_container_non_peak_steps,
        "peak_step_fraction": (
            float(per_container_peak_steps / total_steps)
            if total_steps
            else float("nan")
        ),
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
        "non_peak_mae": non_peak_mae,
        "non_peak_rmse": non_peak_rmse,
    }


def compare_overall_metrics(
    treatment_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    metric_columns: tuple[str, ...] = ("day1_mae", "day1_rmse", "day1_mape"),
) -> pd.DataFrame:
    """
    Compare aggregate overall Day-1 metrics between treatment and control.

    Expects summaries produced by ``summarize_evaluation_metrics``.
    """
    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        if metric not in treatment_summary.columns:
            raise KeyError(f"Metric {metric} missing from treatment summary")
        if metric not in control_summary.columns:
            raise KeyError(f"Metric {metric} missing from control summary")

        control_mean = float(control_summary.loc["mean", metric])
        treatment_mean = float(treatment_summary.loc["mean", metric])
        rows.append({
            "metric": metric,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "delta_treatment_minus_control": treatment_mean - control_mean,
        })

    return pd.DataFrame(rows)


def compare_peak_subset_summaries(
    treatment_summary: dict[str, Any],
    control_summary: dict[str, Any],
) -> pd.DataFrame:
    """Compare pooled peak and non-peak summaries between two models."""
    rows: list[dict[str, Any]] = []
    for subset, mae_key, rmse_key in (
        ("peak", "peak_mae", "peak_rmse"),
        ("non_peak", "non_peak_mae", "non_peak_rmse"),
    ):
        control_mae = float(control_summary[mae_key])
        treatment_mae = float(treatment_summary[mae_key])
        control_rmse = float(control_summary[rmse_key])
        treatment_rmse = float(treatment_summary[rmse_key])
        rows.append({
            "subset": subset,
            "metric": "mae",
            "control": control_mae,
            "treatment": treatment_mae,
            "delta_treatment_minus_control": treatment_mae - control_mae,
        })
        rows.append({
            "subset": subset,
            "metric": "rmse",
            "control": control_rmse,
            "treatment": treatment_rmse,
            "delta_treatment_minus_control": treatment_rmse - control_rmse,
        })
    return pd.DataFrame(rows)


def build_overall_evaluation_summary(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Thin wrapper around frozen ``summarize_evaluation_metrics``."""
    return summarize_evaluation_metrics(evaluation_df)
