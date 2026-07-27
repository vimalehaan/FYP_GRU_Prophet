"""Production Hybrid evaluation with extended metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from production.utils.config import INPUT_WINDOW, MAPE_EPSILON
from production.utils.inference import (
    run_known_container_inference,
    run_unseen_container_inference,
)
from utils.hybrid_inference import HybridInferenceResult


def compute_extended_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = MAPE_EPSILON,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, Pearson r, and R² on flattened arrays."""
    actual_flat = np.ravel(actual).astype(float)
    predicted_flat = np.ravel(predicted).astype(float)

    mae = float(mean_absolute_error(actual_flat, predicted_flat))
    rmse = float(np.sqrt(mean_squared_error(actual_flat, predicted_flat)))
    denominator = np.maximum(np.abs(actual_flat), epsilon)
    mape = float(
        np.mean(np.abs(actual_flat - predicted_flat) / denominator) * 100.0
    )

    if len(actual_flat) < 2 or np.std(actual_flat) < 1e-12:
        pearson_r = float("nan")
    else:
        pearson_r = float(pearsonr(actual_flat, predicted_flat).statistic)

    r2 = float(r2_score(actual_flat, predicted_flat))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson_r": pearson_r,
        "r2": r2,
    }


def _metrics_from_result(
    result: HybridInferenceResult,
    split_label: Literal["known", "unseen"],
) -> dict[str, Any]:
    """Extract Day 1 metrics and metadata from an inference result."""
    cpu_metrics = compute_extended_metrics(
        result.actual_day1_real,
        result.day1_final_real,
    )
    return {
        "container_id": result.container_id,
        "split": split_label,
        "train_steps": result.n_train_steps,
        "validation_steps": result.n_val_steps,
        "day1_mae": cpu_metrics["mae"],
        "day1_rmse": cpu_metrics["rmse"],
        "day1_mape": cpu_metrics["mape"],
        "day1_pearson_r": cpu_metrics["pearson_r"],
        "day1_r2": cpu_metrics["r2"],
    }


def evaluate_known_containers(
    container_ids: list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    res_mean: float,
    res_std: float,
    input_window: int = INPUT_WINDOW,
) -> tuple[pd.DataFrame, dict[str, HybridInferenceResult], list[tuple[str, str]]]:
    """Evaluate all known production training containers."""
    rows: list[dict[str, Any]] = []
    results: dict[str, HybridInferenceResult] = {}
    failures: list[tuple[str, str]] = []

    train_ids = set(global_train["container_id"])
    val_ids = set(global_val["container_id"])

    for cid in container_ids:
        if cid not in train_ids or cid not in val_ids:
            failures.append((cid, "missing from production train/val data"))
            continue
        if cid not in scalers:
            failures.append((cid, "missing scaler"))
            continue
        train_steps = int((global_train["container_id"] == cid).sum())
        if train_steps < input_window:
            failures.append(
                (cid, f"insufficient train steps ({train_steps} < {input_window})")
            )
            continue
        try:
            result = run_known_container_inference(
                container_id=cid,
                global_train=global_train,
                global_val=global_val,
                hybrid_gru_model=hybrid_gru_model,
                scalers=scalers,
                res_mean=res_mean,
                res_std=res_std,
                input_window=input_window,
            )
            rows.append(_metrics_from_result(result, "known"))
            results[cid] = result
        except Exception as exc:
            failures.append((cid, str(exc)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("container_id").reset_index(drop=True)
    return df, results, failures


def evaluate_unseen_containers(
    container_ids: list[str],
    unseen_raw: pd.DataFrame,
    hybrid_gru_model: Any,
    res_mean: float,
    res_std: float,
    input_window: int = INPUT_WINDOW,
) -> tuple[pd.DataFrame, dict[str, HybridInferenceResult], list[tuple[str, str]]]:
    """Evaluate completely unseen holdout containers."""
    rows: list[dict[str, Any]] = []
    results: dict[str, HybridInferenceResult] = {}
    failures: list[tuple[str, str]] = []

    for cid in container_ids:
        raw = unseen_raw[unseen_raw["container_id"] == cid]
        if raw.empty:
            failures.append((cid, "missing from unseen resampled data"))
            continue
        train_len = int(len(raw) * 0.8)
        if train_len < input_window:
            failures.append(
                (cid, f"insufficient historical steps ({train_len} < {input_window})")
            )
            continue
        try:
            result = run_unseen_container_inference(
                container_id=cid,
                unseen_raw=unseen_raw,
                hybrid_gru_model=hybrid_gru_model,
                res_mean=res_mean,
                res_std=res_std,
                input_window=input_window,
            )
            rows.append(_metrics_from_result(result, "unseen"))
            results[cid] = result
        except Exception as exc:
            failures.append((cid, str(exc)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("container_id").reset_index(drop=True)
    return df, results, failures


def summarize_metrics_by_split(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics grouped by known vs unseen split."""
    metric_cols = [
        "day1_mae",
        "day1_rmse",
        "day1_mape",
        "day1_pearson_r",
        "day1_r2",
    ]
    if evaluation_df.empty:
        return pd.DataFrame()
    return (
        evaluation_df.groupby("split")[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
    )


def save_prediction_records(
    results: dict[str, HybridInferenceResult],
    split_label: str,
    predictions_dir: Any,
) -> None:
    """Save per-container prediction arrays for notebook loading."""
    import pickle
    from pathlib import Path

    out_dir = Path(predictions_dir) / split_label
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid, result in results.items():
        record = {
            "container_id": cid,
            "split": split_label,
            "actual_day1_real": result.actual_day1_real,
            "day1_final_real": result.day1_final_real,
            "prophet_day1_real": result.prophet_day1_real,
            "day1_residual": result.day1_residual,
            "train_timestamps": result.train_container["time_stamp"].values,
            "val_timestamps": result.val_container["time_stamp"].values[:96],
            "train_cpu_real": result.scaler.inverse_transform(
                result.train_container["cpu_scaled"].values.reshape(-1, 1)
            ).ravel(),
        }
        with (out_dir / f"{cid}.pkl").open("wb") as fh:
            pickle.dump(record, fh)
