"""RRE inference — Hybrid Prophet + GRU with variant-specific scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW, FORECAST_HORIZON
from utils.hybrid_inference import HybridInferenceResult
from utils.rre.scaling import ScalingStats


def run_rre_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    scaling_stats: ScalingStats,
    input_window: int = DEFAULT_INPUT_WINDOW,
    day1_horizon: int = DAY1_HORIZON,
    forecast_horizon: int = FORECAST_HORIZON,
) -> HybridInferenceResult:
    """Run Hybrid inference with R0 or R3 residual scaling."""
    train_container = global_train[
        global_train["container_id"] == container_id
    ].sort_values("time_stamp")

    val_container = global_val[
        global_val["container_id"] == container_id
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
        "ds": val_container["time_stamp"].values[:forecast_horizon],
    })
    forecast = prophet_model.predict(future_df)
    val_container = val_container.copy()
    val_container["prophet_pred"] = forecast["yhat"].values

    train_forecast = prophet_model.predict(prophet_train[["ds"]])
    train_container = train_container.copy()
    train_container["prophet_pred"] = train_forecast["yhat"].values
    train_container["residual"] = (
        train_container["cpu_scaled"] - train_container["prophet_pred"]
    )
    train_container["residual_scaled"] = scaling_stats.transform(
        train_container["residual"]
    )

    residual_input = train_container["residual_scaled"].values[-input_window:]
    X_input = residual_input.reshape(1, input_window, 1)

    day1_residual_scaled = hybrid_gru_model.predict(X_input, verbose=0)[0]
    day1_residual = scaling_stats.inverse(day1_residual_scaled)
    day1_prophet = forecast["yhat"].values[:day1_horizon]
    day1_final_scaled = day1_prophet + day1_residual

    day2_input_residuals = np.concatenate([
        residual_input[day1_horizon:],
        day1_residual_scaled,
    ])
    X_day2 = day2_input_residuals.reshape(1, input_window, 1)
    day2_residual_scaled = hybrid_gru_model.predict(X_day2, verbose=0)[0]
    day2_residual = scaling_stats.inverse(day2_residual_scaled)
    day2_prophet = forecast["yhat"].values[day1_horizon:forecast_horizon]
    available_len = len(day2_prophet)
    day2_final_scaled = day2_prophet + day2_residual[:available_len]

    actual_day1_scaled = val_container["cpu_scaled"].values[:day1_horizon]
    actual_day2_scaled = val_container["cpu_scaled"].values[
        day1_horizon : day1_horizon + available_len
    ]

    scaler = scalers[container_id]
    day1_final_real = scaler.inverse_transform(day1_final_scaled.reshape(-1, 1))
    actual_day1_real = scaler.inverse_transform(actual_day1_scaled.reshape(-1, 1))
    prophet_day1_real = scaler.inverse_transform(day1_prophet.reshape(-1, 1))
    day2_final_real = scaler.inverse_transform(day2_final_scaled.reshape(-1, 1))
    actual_day2_real = scaler.inverse_transform(actual_day2_scaled.reshape(-1, 1))
    prophet_day2_real = scaler.inverse_transform(day2_prophet.reshape(-1, 1))

    return HybridInferenceResult(
        container_id=container_id,
        train_container=train_container,
        val_container=val_container,
        prophet_train=prophet_train,
        prophet_model=prophet_model,
        forecast=forecast,
        train_forecast=train_forecast,
        residual_input=residual_input,
        X_input=X_input,
        day1_residual_scaled=day1_residual_scaled,
        day1_residual=day1_residual,
        day1_prophet=day1_prophet,
        day1_final_scaled=day1_final_scaled,
        day2_input_residuals=day2_input_residuals,
        X_day2=X_day2,
        day2_residual_scaled=day2_residual_scaled,
        day2_residual=day2_residual,
        day2_prophet=day2_prophet,
        available_len=available_len,
        day2_final_scaled=day2_final_scaled,
        actual_day1_scaled=actual_day1_scaled,
        actual_day2_scaled=actual_day2_scaled,
        scaler=scaler,
        day1_final_real=day1_final_real,
        actual_day1_real=actual_day1_real,
        prophet_day1_real=prophet_day1_real,
        day2_final_real=day2_final_real,
        actual_day2_real=actual_day2_real,
        prophet_day2_real=prophet_day2_real,
        input_window=input_window,
        day1_horizon=day1_horizon,
        n_train_steps=len(train_container),
        n_val_steps=len(val_container),
    )
