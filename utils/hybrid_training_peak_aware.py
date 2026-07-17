"""Peak-aware Hybrid GRU training pipeline."""

from __future__ import annotations

import random
import types
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.hybrid_training import (
    build_hybrid_gru_model,
    generate_prophet_residuals,
)
from utils.peak_config import (
    BATCH_SIZE,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    TRAINING_METRICS,
)
from utils.peak_detection import (
    align_with_residual_sequences,
    assert_binary_weight_values,
    assert_expected_sequence_counts,
    build_sequence_weight_matrix,
    compute_peak_thresholds,
)
from utils.sequence_utils import create_residual_sequences


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


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def train_hybrid_gru_peak_aware(
    global_train: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    epochs: int = EPOCHS_MAX,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
    random_seed: int = RANDOM_SEED,
) -> tuple[Any, float, float, pd.DataFrame, dict[str, float], np.ndarray]:
    """
    Train the Peak-Aware Hybrid GRU with timestep-weighted MSE.

    Follows the locked Phase 3 training protocol: identical Prophet
    residual generation and sequence construction as baseline, with
    per-timestep ``sample_weight`` on ``y_train`` only.

    Parameters
    ----------
    global_train:
        Train-period dataframe for the selected container cohort.
    scalers:
        Per-container MinMax scalers for peak labelling on real CPU %.
    input_window:
        GRU input context length.
    forecast_horizon:
        GRU residual target horizon length.
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
    res_mean:
        Global train residual mean (same computation as baseline).
    res_std:
        Global train residual std (same computation as baseline).
    train_residual_df:
        Residual dataframe used for sequence generation.
    peak_thresholds:
        Train-fitted per-container P90 thresholds.
    weight_matrix:
        Full sequence weight matrix ``W_all`` of shape ``(N, forecast_horizon)``.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    set_random_seeds(random_seed)

    train_residual_df = generate_prophet_residuals(global_train)

    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())

    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    peak_thresholds = compute_peak_thresholds(global_train, scalers)

    features = ["residual_scaled"]
    X_all, y_all, sample_cids = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
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

    assert_expected_sequence_counts(len(X_all), W_all)
    assert_binary_weight_values(W_all)
    if not np.array_equal(sample_cids, weight_cids):
        raise AssertionError(
            "sample_cids from create_residual_sequences do not match "
            "weight matrix container order"
        )
    if W_all.shape != y_all.shape:
        raise AssertionError(
            f"Weight matrix shape {W_all.shape} != target shape {y_all.shape}"
        )

    alignment = align_with_residual_sequences(
        global_train=global_train,
        train_residual_df=train_residual_df,
        scalers=scalers,
        thresholds=peak_thresholds,
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

    model = build_hybrid_gru_model(
        input_window=input_window,
        n_features=len(features),
        forecast_horizon=forecast_horizon,
    )
    _install_weighted_train_step(model)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=list(TRAINING_METRICS),
    )

    early_stop = EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    )
    model.fit(
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

    return (
        model,
        res_mean,
        res_std,
        train_residual_df,
        peak_thresholds,
        W_all,
    )
