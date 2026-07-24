"""Global GRU model construction and training (Phase 0.5 locked specification)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from utils.global_config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    GRU_UNITS,
    INPUT_WINDOW,
    LOSS,
    MODEL_VARIANT,
    OPTIMIZER,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    TRAINING_METRICS,
)
from utils.sequence_utils import create_longterm_sequences


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)


def build_global_gru_model(
    input_window: int = INPUT_WINDOW,
    n_features: int = len(GLOBAL_FEATURES),
    forecast_horizon: int = FORECAST_HORIZON,
) -> Any:
    """
    Build the locked Global GRU v1 architecture for direct CPU forecasting.

    Architecture (Phase 0.5 — matches ``global_gru_model.ipynb``):
    GRU(128) -> Dropout(0.2) -> GRU(64) -> Dense(64, relu) -> Dense(forecast_horizon)
    """
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.models import Sequential

    return Sequential([
        GRU(
            GRU_UNITS[0],
            return_sequences=True,
            input_shape=(input_window, n_features),
        ),
        Dropout(DROPOUT),
        GRU(GRU_UNITS[1]),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(forecast_horizon),
    ])


def train_global_gru(
    global_train: pd.DataFrame,
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    features: tuple[str, ...] = GLOBAL_FEATURES,
    target: str = GLOBAL_TARGET,
    epochs: int = EPOCHS_MAX,
    batch_size: int = BATCH_SIZE,
    random_seed: int = RANDOM_SEED,
    verbose: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Train Global GRU v1 on train-period sequences only.

    Sequences are built from ``global_train`` — never pass validation-period
    data. The 80/20 index split is used for early stopping only.

    Returns
    -------
    model:
        Trained Keras model.
    training_metadata:
        Dictionary of hyperparameters, sequence counts, and training history
        summary.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    set_random_seeds(random_seed)

    feature_list = list(features)
    X_all, y_all, sample_cids = create_longterm_sequences(
        global_train,
        features=feature_list,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    if len(X_all) != EXPECTED_N_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_N_SEQUENCES} sequences from global_train, "
            f"got {len(X_all)}. Ensure only train-period data is passed."
        )

    split_idx = int(len(X_all) * SEQUENCE_VAL_SPLIT)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    if len(X_train) != EXPECTED_N_TRAIN or len(X_val) != EXPECTED_N_VAL:
        raise ValueError(
            f"Unexpected train/val sequence split: "
            f"train={len(X_train)}, val={len(X_val)}; "
            f"expected {EXPECTED_N_TRAIN}/{EXPECTED_N_VAL}."
        )

    model = build_global_gru_model(
        input_window=input_window,
        n_features=len(feature_list),
        forecast_horizon=forecast_horizon,
    )
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=list(TRAINING_METRICS),
    )

    early_stop = EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=SHUFFLE,
        callbacks=[early_stop],
        verbose=verbose,
    )

    final_epoch = len(history.history.get("loss", []))
    val_losses = history.history.get("val_loss", [])
    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = float(min(val_losses))
    else:
        best_epoch = None
        best_val_loss = None

    training_metadata: dict[str, Any] = {
        "model_variant": MODEL_VARIANT,
        "random_seed": random_seed,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "features": feature_list,
        "target": target,
        "n_sequences": int(len(X_all)),
        "n_train_sequences": int(len(X_train)),
        "n_val_sequences": int(len(X_val)),
        "n_unique_containers": int(len(set(sample_cids.tolist()))),
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "metrics": list(TRAINING_METRICS),
        "epochs_max": epochs,
        "epochs_run": final_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "batch_size": batch_size,
        "shuffle": SHUFFLE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "sequence_val_split": SEQUENCE_VAL_SPLIT,
        "gru_units": list(GRU_UNITS),
        "dense_units": DENSE_UNITS,
        "dropout": DROPOUT,
        "final_train_loss": (
            float(history.history["loss"][-1]) if final_epoch else None
        ),
        "final_val_loss": (
            float(history.history["val_loss"][-1]) if final_epoch else None
        ),
        "final_train_mae": (
            float(history.history["mae"][-1])
            if final_epoch and "mae" in history.history
            else None
        ),
        "final_val_mae": (
            float(history.history["val_mae"][-1])
            if final_epoch and "val_mae" in history.history
            else None
        ),
    }

    return model, training_metadata
