"""Prophet residual generation and Hybrid GRU training."""

from __future__ import annotations

from typing import Any

import pandas as pd
from prophet import Prophet

from utils.hybrid_config import DAY1_HORIZON
from utils.sequence_utils import create_residual_sequences


def generate_prophet_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-container Prophet models on the train period and extract residuals.

    Parameters
    ----------
    df:
        Train-period dataframe with ``container_id``, ``time_stamp``,
        and ``cpu_scaled`` columns.

    Returns
    -------
    DataFrame with ``prophet_pred`` and ``residual`` columns appended.
    """
    prophet_outputs = []

    for _, group in df.groupby("container_id"):
        group = group.sort_values("time_stamp")

        prophet_df = pd.DataFrame({
            "ds": group["time_stamp"],
            "y": group["cpu_scaled"],
        })

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
        )
        model.fit(prophet_df)

        forecast = model.predict(prophet_df[["ds"]])

        group["prophet_pred"] = forecast["yhat"].values
        group["residual"] = (
            group["cpu_scaled"]
            - group["prophet_pred"]
        )

        prophet_outputs.append(group)

    return pd.concat(
        prophet_outputs,
        ignore_index=True,
    )


def build_hybrid_gru_model(
    input_window: int,
    n_features: int,
    forecast_horizon: int = DAY1_HORIZON,
) -> Any:
    """Build the Hybrid GRU architecture for a given input window."""
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.models import Sequential

    return Sequential([
        GRU(
            256,
            return_sequences=True,
            input_shape=(input_window, n_features),
        ),
        Dropout(0.2),
        GRU(
            128,
            return_sequences=True,
        ),
        Dropout(0.2),
        GRU(64),
        Dense(128, activation="relu"),
        Dense(forecast_horizon),
    ])


def train_hybrid_gru(
    global_train: pd.DataFrame,
    input_window: int,
    forecast_horizon: int = DAY1_HORIZON,
    epochs: int = 100,
    batch_size: int = 64,
    verbose: int = 0,
) -> tuple[Any, float, float, pd.DataFrame]:
    """
    Train the Hybrid GRU on Prophet residuals for the given input window.

    Returns the trained model, global residual statistics, and the residual
    dataframe used for sequence generation.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    train_residual_df = generate_prophet_residuals(global_train)

    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())

    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    X_all, y_all, _ = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    model = build_hybrid_gru_model(
        input_window=input_window,
        n_features=len(features),
        forecast_horizon=forecast_horizon,
    )
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=verbose,
    )

    return model, res_mean, res_std, train_residual_df
