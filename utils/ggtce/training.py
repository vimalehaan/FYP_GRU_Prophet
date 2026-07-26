"""GGTCE variant training — Global GRU with variable input window."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from utils.global_config import FORECAST_HORIZON, GLOBAL_FEATURES, GLOBAL_TARGET
from utils.global_training import build_global_gru_model, set_random_seeds
from utils.ggtce.config import TRAINING_CONFIG
from utils.sequence_utils import create_longterm_sequences


def count_sequences(global_train: pd.DataFrame, input_window: int) -> dict[str, int]:
    """Count sliding-window sequences without materializing arrays."""
    total = 0
    per_c: list[int] = []
    horizon = FORECAST_HORIZON
    for _, group in global_train.groupby("container_id"):
        n = len(group)
        seq = max(0, n - input_window - horizon + 1)
        per_c.append(seq)
        total += seq
    return {
        "total": total,
        "train_80pct": int(total * TRAINING_CONFIG["internal_val_split"]),
        "val_20pct": total - int(total * TRAINING_CONFIG["internal_val_split"]),
        "min_per_container": min(per_c) if per_c else 0,
        "mean_per_container": float(np.mean(per_c)) if per_c else 0.0,
    }


def train_ggtce_variant(
    global_train: pd.DataFrame,
    variant_id: str,
    input_window: int,
    forecast_horizon: int = FORECAST_HORIZON,
    features: tuple[str, ...] = GLOBAL_FEATURES,
    target: str = GLOBAL_TARGET,
    epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
    verbose: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Train one GGTCE variant with frozen Global GRU hyperparameters.

    Unlike ``train_global_gru``, does not assert fixed sequence counts.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    set_random_seeds(random_seed)
    feature_list = list(features)

    seq_counts = count_sequences(global_train, input_window)
    if seq_counts["min_per_container"] <= 0:
        raise ValueError(
            f"Variant {variant_id}: at least one container has zero sequences "
            f"for input_window={input_window}."
        )

    X_all, y_all, sample_cids = create_longterm_sequences(
        global_train,
        features=feature_list,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    split_idx = int(len(X_all) * TRAINING_CONFIG["internal_val_split"])
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    model = build_global_gru_model(
        input_window=input_window,
        n_features=len(feature_list),
        forecast_horizon=forecast_horizon,
    )
    model.compile(
        optimizer=TRAINING_CONFIG["optimizer"],
        loss=TRAINING_CONFIG["loss"],
        metrics=TRAINING_CONFIG["metrics"],
    )

    early_stop = EarlyStopping(
        monitor=TRAINING_CONFIG["early_stopping"]["monitor"],
        patience=TRAINING_CONFIG["early_stopping"]["patience"],
        restore_best_weights=TRAINING_CONFIG["early_stopping"]["restore_best_weights"],
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=TRAINING_CONFIG["shuffle"],
        callbacks=[early_stop],
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - t0

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)
    meta: dict[str, Any] = {
        "variant_id": variant_id,
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "features": feature_list,
        "target": target,
        "sequence_counts": seq_counts,
        "n_sequences": int(len(X_all)),
        "n_train_sequences": int(len(X_train)),
        "n_val_sequences": int(len(X_val)),
        "n_unique_containers": int(len(set(sample_cids.tolist()))),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "training_seconds": float(train_seconds),
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "training_config": TRAINING_CONFIG,
        "random_seed": random_seed,
        "history": hist,
    }
    return model, meta


def save_variant_artifacts(
    model: Any,
    metadata: dict[str, Any],
    variant_dir: Any,
    input_window: int,
) -> None:
    """Save model and metadata under variant directory."""
    from utils.global_artifacts import save_global_artifacts

    models_dir = variant_dir / "models"
    save_global_artifacts(
        model=model,
        metadata=metadata,
        model_path=models_dir / "global_gru.keras",
        metadata_path=models_dir / "global_gru_metadata.json",
    )
    metadata["input_window"] = input_window


def load_variant_artifacts(variant_dir: Any) -> tuple[Any, dict[str, Any], int]:
    """Load trained model and metadata."""
    from utils.global_artifacts import load_global_artifacts

    model, metadata = load_global_artifacts(
        model_path=variant_dir / "models" / "global_gru.keras",
        metadata_path=variant_dir / "models" / "global_gru_metadata.json",
    )
    input_window = int(metadata.get("input_window", 96))
    return model, metadata, input_window
