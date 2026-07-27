"""Production inference for known and unseen containers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from production.utils.config import INPUT_WINDOW, MIN_ROWS_RESAMPLED
from production.utils.preprocess import temporal_split_single_container
from utils.hybrid_config import DAY1_HORIZON, FORECAST_HORIZON
from utils.hybrid_inference import HybridInferenceResult, run_hybrid_inference


def prepare_unseen_container_frames(
    container_raw: pd.DataFrame,
    container_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Build train/val frames for an unseen container from raw resampled data.

    Uses only the historical 80% for scaler fitting and Prophet training.
    """
    group = container_raw.sort_values("time_stamp").copy()
    if len(group) < MIN_ROWS_RESAMPLED:
        raise ValueError(
            f"Container {container_id} has insufficient rows ({len(group)})"
        )

    train_part, val_part = temporal_split_single_container(group)
    if train_part["cpu"].nunique() <= 1 or train_part["cpu"].std() == 0:
        raise ValueError(f"Container {container_id} has degenerate train CPU")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_part["cpu_scaled"] = scaler.fit_transform(train_part[["cpu"]])
    val_part["cpu_scaled"] = scaler.transform(val_part[["cpu"]])

    cpu_mean = float(train_part["cpu_scaled"].mean())
    cpu_std = float(train_part["cpu_scaled"].std())
    cpu_max = float(train_part["cpu_scaled"].max())
    cpu_min = float(train_part["cpu_scaled"].min())

    for part in (train_part, val_part):
        part["container_id"] = container_id
        part["cpu_mean"] = cpu_mean
        part["cpu_std"] = cpu_std
        part["cpu_max"] = cpu_max
        part["cpu_min"] = cpu_min
        part["hour"] = part["time_stamp"].dt.hour

    return train_part, val_part, scaler


def run_known_container_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    res_mean: float,
    res_std: float,
    input_window: int = INPUT_WINDOW,
) -> HybridInferenceResult:
    """Run inference on a known (training) container using saved scalers."""
    return run_hybrid_inference(
        container_id=container_id,
        global_train=global_train,
        global_val=global_val,
        hybrid_gru_model=hybrid_gru_model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
        input_window=input_window,
        day1_horizon=DAY1_HORIZON,
        forecast_horizon=FORECAST_HORIZON,
    )


def run_unseen_container_inference(
    container_id: str,
    unseen_raw: pd.DataFrame,
    hybrid_gru_model: Any,
    res_mean: float,
    res_std: float,
    input_window: int = INPUT_WINDOW,
) -> HybridInferenceResult:
    """
    Run production inference on a completely unseen container.

    Prophet and scaler are fit only on the container's historical 80%.
    """
    container_raw = unseen_raw[
        unseen_raw["container_id"] == container_id
    ]
    train_part, val_part, scaler = prepare_unseen_container_frames(
        container_raw,
        container_id,
    )
    scalers = {container_id: scaler}
    global_train = train_part
    global_val = val_part

    return run_hybrid_inference(
        container_id=container_id,
        global_train=global_train,
        global_val=global_val,
        hybrid_gru_model=hybrid_gru_model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
        input_window=input_window,
        day1_horizon=DAY1_HORIZON,
        forecast_horizon=FORECAST_HORIZON,
    )
