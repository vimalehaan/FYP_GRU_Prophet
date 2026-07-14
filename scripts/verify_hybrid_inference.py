#!/usr/bin/env python3
"""Verify run_hybrid_inference matches legacy inline notebook logic."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from prophet import Prophet

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.hybrid_config import DEFAULT_INPUT_WINDOW, DAY1_HORIZON, FORECAST_HORIZON  # noqa: E402
from utils.hybrid_inference import run_hybrid_inference  # noqa: E402


TARGET_CID = "c_11461"
INPUT_WINDOW = DEFAULT_INPUT_WINDOW


def _load_inputs():
    train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data/train_df.parquet"))
    val_df = pd.read_parquet(os.path.join(REPO_ROOT, "data/val_df.parquet"))
    selected = np.load(
        os.path.join(REPO_ROOT, "data/selected_containers.npy"),
        allow_pickle=True,
    )
    with open(os.path.join(REPO_ROOT, "data/scalers.pkl"), "rb") as handle:
        scalers = pickle.load(handle)
    with open(os.path.join(REPO_ROOT, "models/residual_stats.pkl"), "rb") as handle:
        residual_stats = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    import tensorflow as tf

    model = tf.keras.models.load_model(
        os.path.join(REPO_ROOT, "models/hybrid_gru.keras")
    )
    return (
        global_train,
        global_val,
        model,
        scalers,
        residual_stats["res_mean"],
        residual_stats["res_std"],
    )


def _legacy_inline(
    target_cid: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    global_model,
    scalers: dict,
    res_mean: float,
    res_std: float,
) -> dict[str, np.ndarray]:
    """Replicate pre-refactor notebook cells 25-44."""
    train_container = global_train[
        global_train["container_id"] == target_cid
    ].sort_values("time_stamp")

    val_container = global_val[
        global_val["container_id"] == target_cid
    ].sort_values("time_stamp")

    prophet_train = pd.DataFrame({
        "ds": train_container["time_stamp"],
        "y": train_container["cpu_scaled"],
    })

    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
    )
    prophet_model.fit(prophet_train)

    future_df = pd.DataFrame({
        "ds": val_container["time_stamp"].values[:FORECAST_HORIZON],
    })
    forecast = prophet_model.predict(future_df)

    val_container["prophet_pred"] = forecast["yhat"].values

    train_forecast = prophet_model.predict(prophet_train[["ds"]])
    train_container["prophet_pred"] = train_forecast["yhat"].values
    train_container["residual"] = (
        train_container["cpu_scaled"] - train_container["prophet_pred"]
    )
    train_container["residual_scaled"] = (
        train_container["residual"] - res_mean
    ) / res_std

    residual_input = train_container["residual_scaled"].values[-INPUT_WINDOW:]
    X_input = np.column_stack([residual_input]).reshape(1, INPUT_WINDOW, 1)

    day1_residual_scaled = global_model.predict(X_input)[0]
    day1_residual = (day1_residual_scaled * res_std) + res_mean
    day1_prophet = forecast["yhat"].values[:DAY1_HORIZON]
    day1_final_scaled = day1_prophet + day1_residual

    day2_input_residuals = np.concatenate([
        residual_input[DAY1_HORIZON:],
        day1_residual_scaled,
    ])
    X_day2 = day2_input_residuals.reshape(1, INPUT_WINDOW, 1)

    day2_residual_scaled = global_model.predict(X_day2)[0]
    day2_residual = (day2_residual_scaled * res_std) + res_mean
    day2_prophet = forecast["yhat"].values[DAY1_HORIZON:FORECAST_HORIZON]
    available_len = len(day2_prophet)
    day2_final_scaled = day2_prophet + day2_residual[:available_len]

    actual_day1_scaled = val_container["cpu_scaled"].values[:DAY1_HORIZON]
    actual_day2_scaled = val_container["cpu_scaled"].values[
        DAY1_HORIZON:DAY1_HORIZON + available_len
    ]

    scaler = scalers[target_cid]
    day1_final_real = scaler.inverse_transform(day1_final_scaled.reshape(-1, 1))
    actual_day1_real = scaler.inverse_transform(actual_day1_scaled.reshape(-1, 1))
    prophet_day1_real = scaler.inverse_transform(day1_prophet.reshape(-1, 1))
    day2_final_real = scaler.inverse_transform(day2_final_scaled.reshape(-1, 1))
    actual_day2_real = scaler.inverse_transform(actual_day2_scaled.reshape(-1, 1))
    prophet_day2_real = scaler.inverse_transform(day2_prophet.reshape(-1, 1))

    return {
        "day1_final_scaled": day1_final_scaled,
        "day1_final_real": day1_final_real,
        "actual_day1_real": actual_day1_real,
        "prophet_day1_real": prophet_day1_real,
        "day2_final_scaled": day2_final_scaled,
        "day2_final_real": day2_final_real,
        "actual_day2_real": actual_day2_real,
        "prophet_day2_real": prophet_day2_real,
        "day1_residual_scaled": day1_residual_scaled,
        "day2_residual_scaled": day2_residual_scaled,
        "available_len": np.array([available_len]),
    }


def _compare(name: str, legacy: np.ndarray, refactored: np.ndarray) -> None:
    if legacy.shape != refactored.shape:
        raise AssertionError(
            f"{name}: shape mismatch {legacy.shape} vs {refactored.shape}"
        )
    if not np.allclose(legacy, refactored, rtol=0.0, atol=0.0):
        max_diff = np.max(np.abs(legacy - refactored))
        raise AssertionError(f"{name}: values differ (max abs diff = {max_diff})")
    print(f"  OK  {name} {legacy.shape}")


def main() -> None:
    print("Loading inputs...")
    global_train, global_val, model, scalers, res_mean, res_std = _load_inputs()

    print("Running legacy inline inference...")
    legacy = _legacy_inline(
        TARGET_CID,
        global_train,
        global_val,
        model,
        scalers,
        res_mean,
        res_std,
    )

    print("Running run_hybrid_inference...")
    result = run_hybrid_inference(
        container_id=TARGET_CID,
        global_train=global_train,
        global_val=global_val,
        hybrid_gru_model=model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
        input_window=INPUT_WINDOW,
    )

    print("Comparing outputs...")
    _compare("day1_final_scaled", legacy["day1_final_scaled"], result.day1_final_scaled)
    _compare("day1_final_real", legacy["day1_final_real"], result.day1_final_real)
    _compare("actual_day1_real", legacy["actual_day1_real"], result.actual_day1_real)
    _compare("prophet_day1_real", legacy["prophet_day1_real"], result.prophet_day1_real)
    _compare("day2_final_scaled", legacy["day2_final_scaled"], result.day2_final_scaled)
    _compare("day2_final_real", legacy["day2_final_real"], result.day2_final_real)
    _compare("actual_day2_real", legacy["actual_day2_real"], result.actual_day2_real)
    _compare("prophet_day2_real", legacy["prophet_day2_real"], result.prophet_day2_real)
    _compare(
        "day1_residual_scaled",
        legacy["day1_residual_scaled"],
        result.day1_residual_scaled,
    )
    _compare(
        "day2_residual_scaled",
        legacy["day2_residual_scaled"],
        result.day2_residual_scaled,
    )
    assert int(legacy["available_len"][0]) == result.available_len

    print("\nAll predictions identical.")
    print(f"Container: {TARGET_CID}")
    print(f"Day 1 steps: {len(result.day1_final_real)}")
    print(f"Day 2 steps: {len(result.day2_final_real)}")


if __name__ == "__main__":
    main()
