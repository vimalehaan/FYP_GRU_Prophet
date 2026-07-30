"""Walk-forward monitoring with leakage-safe origin forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from utils.adaptive_lifecycle.config import AFMLFConfig
from utils.adaptive_lifecycle.rolling_metrics import OriginMetrics, compute_day1_metrics
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW


@dataclass
class LeakageAssertion:
    name: str
    passed: bool
    detail: str


def assert_leakage_free_origin(
    origin_index: int,
    n_rows: int,
    horizon: int,
    train_cutoff_index: int | None = None,
) -> list[LeakageAssertion]:
    checks = [
        LeakageAssertion(
            "origin_within_series",
            0 < origin_index < n_rows - horizon,
            f"origin={origin_index}, n_rows={n_rows}, horizon={horizon}",
        ),
        LeakageAssertion(
            "future_window_available",
            origin_index + horizon <= n_rows,
            f"origin+horizon={origin_index + horizon}",
        ),
    ]
    if train_cutoff_index is not None:
        checks.append(
            LeakageAssertion(
                "train_cutoff_before_origin",
                train_cutoff_index <= origin_index,
                f"train_cutoff={train_cutoff_index}, origin={origin_index}",
            )
        )
    return checks


def build_container_series(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    container_id: str,
) -> pd.DataFrame:
    parts = [
        train_df[train_df["container_id"] == container_id],
        val_df[val_df["container_id"] == container_id],
    ]
    series = pd.concat(parts, ignore_index=True).sort_values("time_stamp")
    return series.reset_index(drop=True)


def generate_origin_indices(n_rows: int, config: AFMLFConfig) -> list[int]:
    start = config.min_history_steps
    indices = list(range(start, n_rows - config.forecast_horizon + 1, config.walk_forward_stride))
    if config.max_origins_per_container is not None:
        indices = indices[: config.max_origins_per_container]
    return indices


def forecast_at_origin(
    series: pd.DataFrame,
    origin_index: int,
    gru_model: Any,
    res_mean: float,
    res_std: float,
    config: AFMLFConfig,
) -> OriginMetrics:
    checks = assert_leakage_free_origin(
        origin_index,
        len(series),
        config.forecast_horizon,
    )
    if not all(c.passed for c in checks):
        failed = [c for c in checks if not c.passed]
        raise ValueError(f"Leakage assertion failed: {failed}")

    history = series.iloc[:origin_index].copy()
    future = series.iloc[origin_index : origin_index + config.forecast_horizon].copy()
    container_id = str(series["container_id"].iloc[0])

    scaler = MinMaxScaler(feature_range=(0, 1))
    history["cpu_scaled"] = scaler.fit_transform(history[["cpu"]])
    future["cpu_scaled"] = scaler.transform(future[["cpu"]])

    prophet_train = pd.DataFrame({
        "ds": history["time_stamp"],
        "y": history["cpu_scaled"],
    })
    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
    )
    prophet_model.fit(prophet_train)

    future_ds = pd.DataFrame({"ds": future["time_stamp"].values})
    forecast = prophet_model.predict(future_ds)
    train_forecast = prophet_model.predict(prophet_train[["ds"]])

    history["prophet_pred"] = train_forecast["yhat"].values
    history["residual"] = history["cpu_scaled"] - history["prophet_pred"]
    history["residual_scaled"] = (history["residual"] - res_mean) / res_std

    residual_input = history["residual_scaled"].values[-config.input_window :]
    X_input = residual_input.reshape(1, config.input_window, 1)
    day1_residual_scaled = gru_model.predict(X_input, verbose=0)[0]
    day1_residual = (day1_residual_scaled * res_std) + res_mean

    day1_prophet = forecast["yhat"].values[: config.forecast_horizon]
    day1_final_scaled = day1_prophet + day1_residual

    actual_scaled = future["cpu_scaled"].values
    prophet_only_real = scaler.inverse_transform(day1_prophet.reshape(-1, 1)).ravel()
    hybrid_real = scaler.inverse_transform(day1_final_scaled.reshape(-1, 1)).ravel()
    actual_real = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).ravel()

    hybrid_mae, hybrid_rmse = compute_day1_metrics(actual_real, hybrid_real)
    prophet_mae, prophet_rmse = compute_day1_metrics(actual_real, prophet_only_real)

    return OriginMetrics(
        container_id=container_id,
        origin_index=origin_index,
        origin_timestamp=str(history["time_stamp"].iloc[-1]),
        hybrid_mae=hybrid_mae,
        hybrid_rmse=hybrid_rmse,
        prophet_mae=prophet_mae,
        prophet_rmse=prophet_rmse,
        n_history_steps=len(history),
    )


def run_walk_forward_monitoring(
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    gru_model: Any,
    res_mean: float,
    res_std: float,
    config: AFMLFConfig,
    failures: list[dict[str, Any]] | None = None,
    on_progress: Callable[..., None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    failure_log = failures if failures is not None else []

    origin_plan: list[tuple[str, int]] = []
    for cid in container_ids:
        series = build_container_series(train_df, val_df, cid)
        for origin in generate_origin_indices(len(series), config):
            origin_plan.append((cid, origin))

    total = len(origin_plan)
    done = 0
    container_idx = 0
    last_cid: str | None = None

    for cid in container_ids:
        series = build_container_series(train_df, val_df, cid)
        origins = generate_origin_indices(len(series), config)
        if cid != last_cid:
            container_idx += 1
            last_cid = cid

        for origin in origins:
            try:
                metrics = forecast_at_origin(
                    series,
                    origin,
                    gru_model,
                    res_mean,
                    res_std,
                    config,
                )
                rows.append(metrics.__dict__)
            except Exception as exc:
                failure_log.append({
                    "container_id": cid,
                    "origin_index": origin,
                    "error": str(exc),
                })
                if on_progress is not None:
                    on_progress("failure", cid, origin, str(exc))
            done += 1
            if on_progress is not None:
                on_progress("step", container_idx, len(container_ids), cid, origin, done, total)

    return pd.DataFrame(rows)
