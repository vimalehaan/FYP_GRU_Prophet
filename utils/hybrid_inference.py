"""Single-container Hybrid Prophet + GRU inference."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler


INPUT_WINDOW = 288
DAY1_HORIZON = 96
FORECAST_HORIZON = 192


@dataclass
class HybridInferenceResult:
    """Complete outputs from a single-container Hybrid inference run."""

    container_id: str
    train_container: pd.DataFrame
    val_container: pd.DataFrame
    prophet_train: pd.DataFrame
    prophet_model: Any
    forecast: pd.DataFrame
    train_forecast: pd.DataFrame
    residual_input: np.ndarray
    X_input: np.ndarray
    day1_residual_scaled: np.ndarray
    day1_residual: np.ndarray
    day1_prophet: np.ndarray
    day1_final_scaled: np.ndarray
    day2_input_residuals: np.ndarray
    X_day2: np.ndarray
    day2_residual_scaled: np.ndarray
    day2_residual: np.ndarray
    day2_prophet: np.ndarray
    available_len: int
    day2_final_scaled: np.ndarray
    actual_day1_scaled: np.ndarray
    actual_day2_scaled: np.ndarray
    scaler: MinMaxScaler
    day1_final_real: np.ndarray
    actual_day1_real: np.ndarray
    prophet_day1_real: np.ndarray
    day2_final_real: np.ndarray
    actual_day2_real: np.ndarray
    prophet_day2_real: np.ndarray
    input_window: int
    day1_horizon: int
    n_train_steps: int
    n_val_steps: int


def run_hybrid_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    res_mean: float,
    res_std: float,
    input_window: int = INPUT_WINDOW,
    day1_horizon: int = DAY1_HORIZON,
    forecast_horizon: int = FORECAST_HORIZON,
) -> HybridInferenceResult:
    """
    Run the full Hybrid Prophet + GRU inference pipeline for one container.

    Fits Prophet on the container train period, forecasts the validation period,
    predicts residuals with the trained GRU, combines components, and inverse-
    transforms results to real CPU utilization percent.
    """
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

    val_container["prophet_pred"] = forecast["yhat"].values

    train_forecast = prophet_model.predict(
        prophet_train[["ds"]]
    )

    train_container["prophet_pred"] = train_forecast["yhat"].values
    train_container["residual"] = (
        train_container["cpu_scaled"]
        - train_container["prophet_pred"]
    )
    train_container["residual_scaled"] = (
        train_container["residual"] - res_mean
    ) / res_std

    residual_input = (
        train_container["residual_scaled"]
        .values[-input_window:]
    )

    X_input = np.column_stack([residual_input])
    X_input = X_input.reshape(1, input_window, 1)

    day1_residual_scaled = hybrid_gru_model.predict(X_input)[0]

    day1_residual = (day1_residual_scaled * res_std) + res_mean

    day1_prophet = forecast["yhat"].values[:day1_horizon]

    day1_final_scaled = day1_prophet + day1_residual

    day2_input_residuals = np.concatenate([
        residual_input[day1_horizon:],
        day1_residual_scaled,
    ])

    X_day2 = day2_input_residuals.reshape(1, input_window, 1)

    day2_residual_scaled = hybrid_gru_model.predict(X_day2)[0]

    day2_residual = (day2_residual_scaled * res_std) + res_mean

    day2_prophet = forecast["yhat"].values[day1_horizon:forecast_horizon]

    available_len = len(day2_prophet)

    day2_final_scaled = (
        day2_prophet
        + day2_residual[:available_len]
    )

    actual_day1_scaled = (
        val_container["cpu_scaled"]
        .values[:day1_horizon]
    )

    actual_day2_scaled = (
        val_container["cpu_scaled"]
        .values[day1_horizon:day1_horizon + available_len]
    )

    scaler = scalers[container_id]

    day1_final_real = scaler.inverse_transform(
        day1_final_scaled.reshape(-1, 1)
    )
    actual_day1_real = scaler.inverse_transform(
        actual_day1_scaled.reshape(-1, 1)
    )
    prophet_day1_real = scaler.inverse_transform(
        day1_prophet.reshape(-1, 1)
    )

    day2_final_real = scaler.inverse_transform(
        day2_final_scaled.reshape(-1, 1)
    )
    actual_day2_real = scaler.inverse_transform(
        actual_day2_scaled.reshape(-1, 1)
    )
    prophet_day2_real = scaler.inverse_transform(
        day2_prophet.reshape(-1, 1)
    )

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
