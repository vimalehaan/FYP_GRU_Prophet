"""CSRLE Stage 2 residual baselines: ZERO, PERSISTENCE, RIDGE."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_training import generate_prophet_residuals
from utils.sequence_utils import create_residual_sequences


def train_ridge_residual_model(
    global_train: Any,
    res_mean: float,
    res_std: float,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
    alpha: float = 1.0,
) -> Ridge:
    """Train Ridge 96→96 residual predictor on chronologically first 80% of sequences."""
    train_residual_df = generate_prophet_residuals(global_train)
    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    X_all, y_all, _ = create_residual_sequences(
        train_residual_df,
        features=["residual_scaled"],
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx].reshape(split_idx, -1)
    y_train = y_all[:split_idx]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def predict_baseline_residuals(
    residual_input_scaled: np.ndarray,
    baseline: str,
    ridge_model: Ridge | None = None,
    forecast_horizon: int = DAY1_HORIZON,
) -> np.ndarray:
    """Return z-scored residual forecasts for one container Day-1 window."""
    if baseline == "zero":
        return np.zeros(forecast_horizon, dtype=float)
    if baseline == "persistence":
        return np.full(forecast_horizon, float(residual_input_scaled[-1]))
    if baseline == "ridge":
        if ridge_model is None:
            raise ValueError("ridge_model required for ridge baseline")
        x = residual_input_scaled.reshape(1, -1)
        return ridge_model.predict(x)[0]
    raise ValueError(f"Unknown baseline: {baseline}")


def scaled_to_unscaled_residual(
    pred_scaled: np.ndarray,
    res_mean: float,
    res_std: float,
) -> np.ndarray:
    return (pred_scaled * res_std) + res_mean
