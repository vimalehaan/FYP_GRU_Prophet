"""LFHE training pipeline — injectable MSE or DA-MSE loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from utils.csrle.stage2_training import build_condition_scalers
from utils.hybrid_artifacts import save_hybrid_artifacts
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_training import build_hybrid_gru_model, generate_prophet_residuals
from utils.lfhe.loss import da_mse_loss_registered, make_mse_loss
from utils.sequence_utils import create_residual_sequences


def set_global_seeds(seed: int | None) -> None:
    """Fix seeds when provided; Stage 2 parity uses ``None`` (no TF seed)."""
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def train_lfhe_gru(
    train_df: pd.DataFrame,
    output_dir: Path,
    *,
    arm_id: str,
    loss_kind: str,
    lambda_disp: float = 1.0,
    epsilon: float = 1e-6,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 10,
    seed: int | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """
    Train Hybrid GRU on synthetic B0 data with frozen MSE or DA-MSE loss.

    Mirrors ``utils.csrle.control_reproduction.train_control_hybrid`` except
    for the injectable loss function.
    """
    set_global_seeds(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("training", "models", "config", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    train_residual_df = generate_prophet_residuals(train_df)
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    x_all, y_all, _ = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    split_idx = int(len(x_all) * 0.8)
    x_train, y_train = x_all[:split_idx], y_all[:split_idx]
    x_val, y_val = x_all[split_idx:], y_all[split_idx:]

    model = build_hybrid_gru_model(
        input_window=input_window,
        n_features=len(features),
        forecast_horizon=forecast_horizon,
    )

    if loss_kind == "mse":
        loss: Any = make_mse_loss()
    elif loss_kind == "da_mse":
        loss = da_mse_loss_registered
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    model.compile(optimizer="adam", loss=loss, metrics=["mae"])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=verbose,
    )

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)
    model_path = output_dir / "models" / "hybrid_gru.keras"
    stats_path = output_dir / "models" / "residual_stats.pkl"
    save_hybrid_artifacts(
        model,
        res_mean,
        res_std,
        model_path=model_path,
        residual_stats_path=stats_path,
        input_window=input_window,
    )

    pd.DataFrame(hist).to_csv(output_dir / "training" / "training_history.csv", index=False)

    meta = {
        "arm_id": arm_id,
        "loss_kind": loss_kind,
        "lambda_disp": lambda_disp if loss_kind == "da_mse" else 0.0,
        "epsilon": epsilon,
        "total_sequences": int(len(x_all)),
        "train_sequences": int(len(x_train)),
        "val_sequences": int(len(x_val)),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_train_loss": float(hist["loss"][best_epoch - 1]),
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "res_mean": res_mean,
        "res_std": res_std,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "optimizer": "adam",
        "seed": seed,
        "model_path": str(model_path),
        "stats_path": str(stats_path),
    }
    with (output_dir / "training" / "training_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    with (output_dir / "config" / "arm_config.json").open("w") as fh:
        json.dump(
            {
                "arm_id": arm_id,
                "loss_kind": loss_kind,
                "lambda_disp": lambda_disp,
                "epsilon": epsilon,
            },
            fh,
            indent=2,
        )

    scalers = build_condition_scalers(train_df)
    return {
        "model": model,
        "res_mean": res_mean,
        "res_std": res_std,
        "meta": meta,
        "scalers": scalers,
        "history": hist,
    }
