"""Ridge training and remaining-residual computation for RRA."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.csrle.stage2_baselines import train_ridge_residual_model
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_training import generate_prophet_residuals
from utils.sequence_utils import create_residual_sequences


def train_ridge_diagnostic_model(
    train_df: pd.DataFrame,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
    alpha: float = 1.0,
) -> tuple[Ridge, float, float, dict[str, Any]]:
    """
    Train Ridge on CSRLE B0-LC train data using the frozen 96→96 protocol.

    Returns the fitted model, global residual mean/std, and training metadata.
    """
    train_residual_df = generate_prophet_residuals(train_df)
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    if res_std < 1e-12:
        res_std = 1.0

    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    x_all, y_all, _ = create_residual_sequences(
        train_residual_df,
        features=["residual_scaled"],
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    split_idx = int(len(x_all) * 0.8)
    x_train = x_all[:split_idx].reshape(split_idx, -1)
    y_train = y_all[:split_idx]

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    meta: dict[str, Any] = {
        "alpha": alpha,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "internal_val_split": "chronological_80_20",
        "n_sequences_total": int(len(x_all)),
        "n_sequences_train": int(split_idx),
        "n_features": int(x_train.shape[1]),
        "res_mean": res_mean,
        "res_std": res_std,
        "ridge_coef_norm": float(np.linalg.norm(model.coef_)),
        "ridge_intercept_mean": float(np.mean(model.intercept_)),
    }
    return model, res_mean, res_std, meta


def save_ridge_artifacts(
    model: Ridge,
    res_mean: float,
    res_std: float,
    meta: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist Ridge model and training metadata under ``output_dir/ridge/``."""
    ridge_dir = output_dir / "ridge"
    models_dir = ridge_dir / "models"
    training_dir = ridge_dir / "training"
    models_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    with (models_dir / "ridge_model.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    stats = {"res_mean": res_mean, "res_std": res_std}
    with (training_dir / "residual_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)

    with (training_dir / "ridge_training_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2)


def _fit_prophet_val_residuals(
    train_c: pd.DataFrame,
    val_c: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Prophet on train; return val scaled residuals and Prophet forecast."""
    prophet_train = pd.DataFrame({
        "ds": train_c["time_stamp"],
        "y": train_c["cpu_scaled"],
    })
    model = Prophet(daily_seasonality=True, weekly_seasonality=False)
    model.fit(prophet_train)

    val_len = len(val_c)
    future = pd.DataFrame({"ds": val_c["time_stamp"].values})
    forecast = model.predict(future)

    actual_scaled = val_c["cpu_scaled"].values
    prophet_scaled = forecast["yhat"].values[:val_len]
    residual_scaled = actual_scaled - prophet_scaled
    return residual_scaled, prophet_scaled, actual_scaled


def compute_ridge_remaining_series(
    train_residual_scaled: np.ndarray,
    val_residual_scaled: np.ndarray,
    ridge_model: Ridge,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Ridge in non-overlapping blocks across the validation period.

    Returns ``ridge_pred_scaled`` and ``remaining_scaled`` aligned to val length.
    """
    history = np.asarray(train_residual_scaled, dtype=float)
    val = np.asarray(val_residual_scaled, dtype=float)
    ridge_preds: list[float] = []
    remaining: list[float] = []
    val_idx = 0
    n_val = len(val)

    while val_idx < n_val:
        if len(history) < input_window:
            break
        x = history[-input_window:].reshape(1, -1)
        pred_block = ridge_model.predict(x)[0]
        block_len = min(forecast_horizon, n_val - val_idx)
        actual_block = val[val_idx : val_idx + block_len]
        pred_slice = pred_block[:block_len]
        ridge_preds.extend(pred_slice.tolist())
        remaining.extend((actual_block - pred_slice).tolist())
        history = np.concatenate([history, actual_block])
        val_idx += block_len

    return np.asarray(ridge_preds), np.asarray(remaining)


def compute_container_ridge_analysis(
    container_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ridge_model: Ridge,
    res_mean: float,
    res_std: float,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
) -> dict[str, Any]:
    """Prophet + Ridge pipeline diagnostics for one container."""
    train_c = train_df[train_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    val_c = val_df[val_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )

    train_residual_df = generate_prophet_residuals(train_c)
    train_res_scaled = (
        (train_residual_df["residual"].values - res_mean) / res_std
    )

    prophet_val, prophet_pred_val, actual_val = _fit_prophet_val_residuals(
        train_c, val_c
    )
    ridge_pred, remaining = compute_ridge_remaining_series(
        train_res_scaled,
        prophet_val,
        ridge_model,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    n = min(len(prophet_val), len(ridge_pred))
    prophet_slice = prophet_val[:n]
    ridge_slice = ridge_pred[:n]
    remaining_slice = remaining[:n]

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(pearsonr(a, b)[0])

    std_prophet = float(np.std(prophet_slice))
    std_remaining = float(np.std(remaining_slice))

    return {
        "container_id": container_id,
        "n_val": int(n),
        "prophet_residual_scaled": prophet_slice,
        "ridge_prediction_scaled": ridge_slice,
        "ridge_remaining_scaled": remaining_slice,
        "prophet_pred_scaled": prophet_pred_val[:n],
        "actual_scaled": actual_val[:n],
        "ridge_pearson_r": _safe_corr(prophet_slice, ridge_slice),
        "ridge_mae_scaled": float(mean_absolute_error(prophet_slice, ridge_slice)),
        "ridge_rmse_scaled": float(
            np.sqrt(mean_squared_error(prophet_slice, ridge_slice))
        ),
        "prophet_mean": float(np.mean(prophet_slice)),
        "prophet_std": std_prophet,
        "prophet_var": float(np.var(prophet_slice)),
        "remaining_mean": float(np.mean(remaining_slice)),
        "remaining_std": std_remaining,
        "remaining_var": float(np.var(remaining_slice)),
        "remaining_std_ratio": (
            std_remaining / std_prophet if std_prophet > 1e-12 else float("nan")
        ),
        "var_reduction_fraction": (
            1.0 - (std_remaining**2) / (std_prophet**2)
            if std_prophet > 1e-12
            else float("nan")
        ),
    }
