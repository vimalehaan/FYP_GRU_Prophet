#!/usr/bin/env python3
"""Verify run_global_inference matches manual temporal-holdout logic."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEMO_CONTAINER_ID,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    INPUT_WINDOW,
    MAPE_EPSILON,
)
from utils.global_artifacts import load_global_artifacts  # noqa: E402
from utils.global_inference import run_global_inference  # noqa: E402
from utils.hybrid_evaluation import compute_day1_mape, compute_day1_metrics  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Global GRU single-container inference protocol.",
    )
    parser.add_argument(
        "--container-id",
        default=DEMO_CONTAINER_ID,
        help=f"Container to verify (default: {DEMO_CONTAINER_ID})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained Global GRU model",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to Global GRU metadata JSON",
    )
    return parser.parse_args()


def _load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected)
    ].copy()
    global_val = val_df[
        val_df["container_id"].isin(selected)
    ].copy()
    return global_train, global_val, scalers


def _manual_inline_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    model,
    scalers: dict,
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> dict[str, np.ndarray]:
    """Replicate temporal holdout inference without run_global_inference()."""
    train_container = (
        global_train[global_train["container_id"] == container_id]
        .sort_values("time_stamp")
    )
    val_container = (
        global_val[global_val["container_id"] == container_id]
        .sort_values("time_stamp")
    )

    feature_list = list(GLOBAL_FEATURES)
    input_features = train_container[feature_list].values[-input_window:]
    x_input = input_features.reshape(1, input_window, len(feature_list))

    day1_pred_scaled = model.predict(x_input, verbose=0)[0]
    actual_day1_scaled = val_container[GLOBAL_TARGET].values[:forecast_horizon]

    scaler = scalers[container_id]
    day1_pred_real = scaler.inverse_transform(
        day1_pred_scaled.reshape(-1, 1),
    ).ravel()
    actual_day1_real = scaler.inverse_transform(
        actual_day1_scaled.reshape(-1, 1),
    ).ravel()

    return {
        "day1_pred_scaled": day1_pred_scaled,
        "day1_pred_real": day1_pred_real,
        "actual_day1_real": actual_day1_real,
    }


def _compare(name: str, legacy: np.ndarray, refactored: np.ndarray) -> None:
    if legacy.shape != refactored.shape:
        raise AssertionError(
            f"{name}: shape mismatch {legacy.shape} vs {refactored.shape}"
        )
    if not np.allclose(legacy, refactored, rtol=0.0, atol=0.0):
        max_diff = float(np.max(np.abs(legacy - refactored)))
        raise AssertionError(f"{name}: values differ (max abs diff = {max_diff})")
    print(f"  OK  {name} {legacy.shape}")


def main() -> None:
    args = _parse_args()
    container_id = args.container_id

    print("=== Global GRU Inference Verification ===")
    print(f"Container: {container_id}")
    print()

    global_train, global_val, scalers = _load_cohort()
    model, _ = load_global_artifacts(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )

    print("Running manual inline inference...")
    manual = _manual_inline_inference(
        container_id,
        global_train,
        global_val,
        model,
        scalers,
    )

    print("Running run_global_inference()...")
    result = run_global_inference(
        container_id=container_id,
        global_train=global_train,
        global_val=global_val,
        model=model,
        scalers=scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    print("Comparing outputs...")
    _compare(
        "day1_pred_scaled",
        manual["day1_pred_scaled"],
        result.day1_pred_scaled,
    )
    _compare(
        "day1_pred_real",
        manual["day1_pred_real"],
        result.day1_pred_real,
    )
    _compare(
        "actual_day1_real",
        manual["actual_day1_real"],
        result.actual_day1_real,
    )

    manual_mae, manual_rmse = compute_day1_metrics(
        manual["actual_day1_real"],
        manual["day1_pred_real"],
    )
    manual_mape = compute_day1_mape(
        manual["actual_day1_real"],
        manual["day1_pred_real"],
        epsilon=MAPE_EPSILON,
    )
    module_mae, module_rmse = compute_day1_metrics(
        result.actual_day1_real,
        result.day1_pred_real,
    )
    module_mape = compute_day1_mape(
        result.actual_day1_real,
        result.day1_pred_real,
        epsilon=MAPE_EPSILON,
    )

    assert np.isclose(manual_mae, module_mae, atol=1e-12)
    assert np.isclose(manual_rmse, module_rmse, atol=1e-12)
    assert np.isclose(manual_mape, module_mape, atol=1e-12)

    print()
    print(f"Day 1 MAE  : {module_mae:.4f}")
    print(f"Day 1 RMSE : {module_rmse:.4f}")
    print(f"Day 1 MAPE : {module_mape:.4f}")
    print()
    print("All Global GRU inference checks passed.")


if __name__ == "__main__":
    main()
