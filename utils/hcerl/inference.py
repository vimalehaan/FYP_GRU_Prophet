"""HCERL multi-feature Hybrid inference (Day-1 evaluation scope)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from utils.hcerl.features import (
    FeatureStats,
    build_inference_input_matrix,
    enrich_inference_train_container,
)
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW, FORECAST_HORIZON
from utils.hybrid_inference import HybridInferenceResult


def run_hcerl_inference(
    container_id: str,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, MinMaxScaler],
    stats: FeatureStats,
    features: list[str],
    input_window: int = DEFAULT_INPUT_WINDOW,
    day1_horizon: int = DAY1_HORIZON,
    forecast_horizon: int = FORECAST_HORIZON,
) -> HybridInferenceResult:
    """
    Run Hybrid Prophet + context-enriched GRU inference for one container.

    Context features in the input window use **past train data only** (causal).
    Day-1 evaluation matches the frozen Hybrid protocol.
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
    val_container = val_container.copy()
    val_container["prophet_pred"] = forecast["yhat"].values

    train_forecast = prophet_model.predict(prophet_train[["ds"]])
    train_container = train_container.copy()
    train_container["prophet_pred"] = train_forecast["yhat"].values

    enriched_train = enrich_inference_train_container(
        train_container,
        stats,
        container_id,
        global_train,
    )

    X_input = build_inference_input_matrix(enriched_train, features, input_window)
    residual_input = enriched_train["residual_scaled"].values[-input_window:]

    day1_residual_scaled = hybrid_gru_model.predict(X_input, verbose=0)[0]
    day1_residual = (day1_residual_scaled * stats.res_std) + stats.res_mean

    day1_prophet = forecast["yhat"].values[:day1_horizon]
    day1_final_scaled = day1_prophet + day1_residual

    # Day-2 recursive path — residual channel only update for API compatibility.
    day2_input_residuals = np.concatenate([
        residual_input[day1_horizon:],
        day1_residual_scaled,
    ])
    if len(features) == 1:
        X_day2 = day2_input_residuals.reshape(1, input_window, 1)
    else:
        # Rebuild multi-feature window: extend enriched train with predicted residuals
        # for Day-2 sliding (prophet/yhat/vol/peak from known train tail only).
        X_day2 = X_input.copy()
        X_day2[0, :, 0] = day2_input_residuals

    day2_residual_scaled = hybrid_gru_model.predict(X_day2, verbose=0)[0]
    day2_residual = (day2_residual_scaled * stats.res_std) + stats.res_mean
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
