"""Production Hybrid GRU training with history capture."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from production.utils.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_MAX,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    LOSS,
    OPTIMIZER,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    TRAINING_METRICS,
)
from utils.hybrid_training import (
    build_hybrid_gru_model,
    generate_prophet_residuals,
)
from utils.sequence_utils import create_residual_sequences


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def train_production_hybrid(
    production_train: pd.DataFrame,
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    epochs: int = EPOCHS_MAX,
    batch_size: int = BATCH_SIZE,
    verbose: int = 1,
) -> tuple[Any, float, float, pd.DataFrame, dict[str, Any]]:
    """
    Train production Hybrid GRU using frozen baseline methodology.

    Sequences are built from production train-period data only.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    set_random_seeds(RANDOM_SEED)

    train_residual_df = generate_prophet_residuals(production_train)
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    if res_std < 1e-12:
        res_std = 1.0

    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    X_all, y_all, sample_cids = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    split_idx = int(len(X_all) * SEQUENCE_VAL_SPLIT)
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
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=list(TRAINING_METRICS),
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=verbose,
    )

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)

    metadata: dict[str, Any] = {
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "res_mean": res_mean,
        "res_std": res_std,
        "n_sequences_total": int(len(X_all)),
        "n_train_sequences": int(len(X_train)),
        "n_val_sequences": int(len(X_val)),
        "n_unique_containers": int(len(set(sample_cids.tolist()))),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "batch_size": batch_size,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "sequence_val_split": SEQUENCE_VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "history": hist,
    }
    return model, res_mean, res_std, train_residual_df, metadata
