"""Single-container Global GRU direct CPU inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.global_config import (
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    INPUT_WINDOW,
)


@dataclass
class GlobalInferenceResult:
    """Outputs from a single-container Global GRU temporal-holdout inference run."""

    day1_pred_real: np.ndarray
    actual_day1_real: np.ndarray
    day1_pred_scaled: np.ndarray
    input_window: int
    n_train_steps: int
    n_val_steps: int


def run_global_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    model: Any,
    scalers: dict[str, MinMaxScaler],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    features: tuple[str, ...] = GLOBAL_FEATURES,
    target: str = GLOBAL_TARGET,
) -> GlobalInferenceResult:
    """
    Run Global GRU Day 1 inference for one container (temporal holdout protocol).

    Uses the last ``input_window`` train-period feature steps as model input and
    compares the forecast against the first ``forecast_horizon`` validation steps.

    Parameters
    ----------
    container_id:
        Container identifier.
    global_train:
        Train-period dataframe (pre-filtered cohort).
    global_val:
        Validation-period dataframe (pre-filtered cohort).
    model:
        Trained Global GRU Keras model.
    scalers:
        Per-container MinMax scalers keyed by ``container_id``.
    input_window:
        Number of input timesteps (default 96).
    forecast_horizon:
        Day 1 forecast length (default 96).
    features:
        Input feature columns.
    target:
        Scaled target column name.

    Returns
    -------
    GlobalInferenceResult with scaled and real CPU % Day 1 arrays.
    """
    train_container = (
        global_train[global_train["container_id"] == container_id]
        .sort_values("time_stamp")
    )
    val_container = (
        global_val[global_val["container_id"] == container_id]
        .sort_values("time_stamp")
    )

    if train_container.empty:
        raise ValueError(
            f"Container {container_id!r} not found in global_train."
        )
    if val_container.empty:
        raise ValueError(
            f"Container {container_id!r} not found in global_val."
        )
    if len(train_container) < input_window:
        raise ValueError(
            f"Container {container_id!r} has {len(train_container)} train steps; "
            f"need at least {input_window} for inference."
        )
    if len(val_container) < forecast_horizon:
        raise ValueError(
            f"Container {container_id!r} has {len(val_container)} val steps; "
            f"need at least {forecast_horizon} for Day 1 evaluation."
        )
    if container_id not in scalers:
        raise KeyError(
            f"No scaler found for container {container_id!r}."
        )

    feature_list = list(features)
    missing_features = [
        col for col in feature_list if col not in train_container.columns
    ]
    if missing_features:
        raise ValueError(
            f"Container {container_id!r} train data missing features: "
            f"{missing_features}"
        )

    input_features = (
        train_container[feature_list]
        .values[-input_window:]
    )
    X_input = input_features.reshape(1, input_window, len(feature_list))

    day1_pred_scaled = model.predict(X_input, verbose=0)[0]
    actual_day1_scaled = (
        val_container[target]
        .values[:forecast_horizon]
    )

    scaler = scalers[container_id]
    day1_pred_real = scaler.inverse_transform(
        day1_pred_scaled.reshape(-1, 1),
    ).ravel()
    actual_day1_real = scaler.inverse_transform(
        actual_day1_scaled.reshape(-1, 1),
    ).ravel()

    return GlobalInferenceResult(
        day1_pred_real=day1_pred_real,
        actual_day1_real=actual_day1_real,
        day1_pred_scaled=day1_pred_scaled,
        input_window=input_window,
        n_train_steps=len(train_container),
        n_val_steps=len(val_container),
    )
