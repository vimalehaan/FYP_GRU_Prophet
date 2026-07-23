"""Peak-aware Global GRU training pipeline."""

from __future__ import annotations

import types
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.global_peak_aware_config import (
    BASE_LOSS,
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
    MODEL_VARIANT,
    NON_PEAK_WEIGHT,
    OPTIMIZER,
    PEAK_PERCENTILE,
    PEAK_WEIGHT,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    TRAINING_LOSS,
    TRAINING_METRICS,
)
from utils.global_training import build_global_gru_model, set_random_seeds
from utils.peak_detection import (
    align_with_longterm_sequences,
    assert_binary_weight_values,
    assert_expected_sequence_counts,
    build_sequence_weight_matrix,
    compute_peak_thresholds,
    summarize_weight_matrix,
)
from utils.sequence_utils import create_longterm_sequences


def _weighted_train_step(self: Any, data: Any) -> dict[str, Any]:
    """
    Timestep-weighted MSE train step.

    Keras default MSE only accepts 1D ``sample_weight`` for vector targets.
    Peak-aware training requires per-timestep weights of shape
    ``(batch, forecast_horizon)``.
    """
    import keras
    import tensorflow as tf

    x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        sq_error = tf.square(y - y_pred)
        weighted_error = sq_error * sample_weight
        loss = tf.reduce_sum(weighted_error) / tf.reduce_sum(sample_weight)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    for metric in self.metrics:
        if metric.name == "loss":
            metric.update_state(loss)
        else:
            metric.update_state(y, y_pred)

    return {metric.name: metric.result() for metric in self.metrics}


def _install_weighted_train_step(model: Any) -> None:
    """Bind the weighted train step to the compiled Keras model."""
    model.train_step = types.MethodType(_weighted_train_step, model)


def train_global_gru_peak_aware(
    global_train: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    features: tuple[str, ...] = GLOBAL_FEATURES,
    target: str = GLOBAL_TARGET,
    epochs: int = EPOCHS_MAX,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
    random_seed: int = RANDOM_SEED,
    early_stopping_patience: int | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float], np.ndarray]:
    """
    Train Peak-Aware Global GRU with timestep-weighted MSE.

    Follows the locked Stage 1 design: identical ``create_longterm_sequences``
    construction and frozen ``build_global_gru_model`` architecture as baseline,
    with per-timestep ``sample_weight`` on ``y_train`` only.

    Parameters
    ----------
    global_train:
        Train-period dataframe for the selected container cohort.
    scalers:
        Per-container MinMax scalers for peak labelling on real CPU %.
    input_window:
        GRU input context length.
    forecast_horizon:
        GRU target horizon length.
    features:
        Input feature column names.
    target:
        Target column name.
    epochs:
        Maximum training epochs.
    batch_size:
        Training batch size.
    verbose:
        Keras ``fit`` verbosity.
    random_seed:
        Random seed applied before model build and training.

    Returns
    -------
    model:
        Trained Keras GRU model.
    training_metadata:
        Hyperparameters, sequence counts, peak-aware settings, and history.
    peak_thresholds:
        Train-fitted per-container P90 thresholds.
    weight_matrix:
        Full sequence weight matrix ``W_all`` of shape ``(N, forecast_horizon)``.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    set_random_seeds(random_seed)

    feature_list = list(features)
    peak_thresholds = compute_peak_thresholds(global_train, scalers)

    X_all, y_all, sample_cids = create_longterm_sequences(
        global_train,
        features=feature_list,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    W_all, weight_cids = build_sequence_weight_matrix(
        global_train=global_train,
        scalers=scalers,
        thresholds=peak_thresholds,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    if len(X_all) != EXPECTED_N_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_N_SEQUENCES} sequences from global_train, "
            f"got {len(X_all)}. Ensure only train-period data is passed."
        )

    assert_expected_sequence_counts(len(X_all), W_all)
    assert_binary_weight_values(W_all)

    if not np.array_equal(sample_cids, weight_cids):
        raise AssertionError(
            "sample_cids from create_longterm_sequences do not match "
            "weight matrix container order"
        )
    if W_all.shape != y_all.shape:
        raise AssertionError(
            f"Weight matrix shape {W_all.shape} != target shape {y_all.shape}"
        )

    alignment = align_with_longterm_sequences(
        global_train=global_train,
        scalers=scalers,
        thresholds=peak_thresholds,
        features=feature_list,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    if not alignment["cid_arrays_equal"]:
        raise AssertionError(
            "Peak-aware sample_cids do not align with baseline sequences"
        )

    split_idx = int(len(X_all) * SEQUENCE_VAL_SPLIT)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    W_train = W_all[:split_idx]
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
    _install_weighted_train_step(model)
    model.compile(
        optimizer=OPTIMIZER,
        loss=BASE_LOSS,
        metrics=list(TRAINING_METRICS),
    )

    early_stop = EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        patience=(
            early_stopping_patience
            if early_stopping_patience is not None
            else EARLY_STOPPING_PATIENCE
        ),
        restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    )
    patience_used = (
        early_stopping_patience
        if early_stopping_patience is not None
        else EARLY_STOPPING_PATIENCE
    )

    history = model.fit(
        X_train,
        y_train,
        sample_weight=W_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=SHUFFLE,
        callbacks=[early_stop],
        verbose=verbose,
    )

    final_epoch = len(history.history.get("loss", []))
    weight_summary = summarize_weight_matrix(W_all)

    val_loss_history = history.history.get("val_loss", [])
    if val_loss_history:
        best_epoch_idx = int(np.argmin(val_loss_history))
        best_epoch = best_epoch_idx + 1
        stopped_epoch = final_epoch
        best_val_loss = float(val_loss_history[best_epoch_idx])
    else:
        best_epoch = None
        stopped_epoch = final_epoch if final_epoch else None
        best_val_loss = None

    training_metadata: dict[str, Any] = {
        "model_variant": MODEL_VARIANT,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "features": feature_list,
        "target": target,
        "optimizer": OPTIMIZER,
        "loss": TRAINING_LOSS,
        "loss_implementation": "custom_train_step_timestep_weighted_mse",
        "base_loss": BASE_LOSS,
        "peak_percentile": PEAK_PERCENTILE,
        "peak_weight": PEAK_WEIGHT,
        "non_peak_weight": NON_PEAK_WEIGHT,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": patience_used,
        "early_stopping_restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        "epochs_max": epochs,
        "epochs_run": final_epoch,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss,
        "batch_size": batch_size,
        "shuffle": SHUFFLE,
        "random_seed": random_seed,
        "sequence_val_split": SEQUENCE_VAL_SPLIT,
        "n_sequences_total": int(len(X_all)),
        "n_sequences_train": int(len(X_train)),
        "n_sequences_val": int(len(X_val)),
        "n_unique_containers": int(len(set(sample_cids.tolist()))),
        "n_peak_thresholds": len(peak_thresholds),
        "weight_summary": weight_summary,
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
        "validation_unweighted": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "frozen_modules_used": [
            "utils.global_training.build_global_gru_model",
            "utils.global_training.set_random_seeds",
            "utils.sequence_utils.create_longterm_sequences",
        ],
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    return model, training_metadata, peak_thresholds, W_all
