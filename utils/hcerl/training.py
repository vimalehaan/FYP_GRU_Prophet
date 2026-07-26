"""HCERL variant training — frozen Hybrid GRU protocol with multi-feature input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.hcerl.config import TRAINING_CONFIG
from utils.hcerl.features import FeatureStats, prepare_training_frame
from utils.hybrid_config import DAY1_HORIZON
from utils.hybrid_training import build_hybrid_gru_model
from utils.sequence_utils import create_residual_sequences


def train_hcerl_variant(
    global_train: pd.DataFrame,
    variant_id: str,
    features: list[str],
    input_window: int = TRAINING_CONFIG["input_window"],
    forecast_horizon: int = TRAINING_CONFIG["forecast_horizon"],
    epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    verbose: int = 0,
) -> tuple[Any, FeatureStats, pd.DataFrame, dict[str, Any]]:
    """
    Train one HCERL variant with cumulative context features.

    Prophet generation, GRU architecture, optimizer, loss, epochs, and early
    stopping match the frozen baseline Hybrid pipeline exactly.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    enriched_df, stats = prepare_training_frame(
        global_train, variant_id, features,
    )

    X_all, y_all, sample_cids = create_residual_sequences(
        enriched_df,
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
        optimizer=TRAINING_CONFIG["optimizer"],
        loss=TRAINING_CONFIG["loss"],
        metrics=TRAINING_CONFIG["metrics"],
    )

    early_stop = EarlyStopping(
        monitor=TRAINING_CONFIG["early_stopping"]["monitor"],
        patience=TRAINING_CONFIG["early_stopping"]["patience"],
        restore_best_weights=TRAINING_CONFIG["early_stopping"]["restore_best_weights"],
    )
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

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)
    meta: dict[str, Any] = {
        "variant_id": variant_id,
        "features": features,
        "n_features": len(features),
        "input_shape": [input_window, len(features)],
        "total_sequences": int(len(X_all)),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_train_loss": float(hist["loss"][best_epoch - 1]),
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "training_config": TRAINING_CONFIG,
        "history": hist,
    }
    return model, stats, enriched_df, meta


def save_variant_artifacts(
    model: Any,
    stats: FeatureStats,
    variant_dir: Path,
    input_window: int = TRAINING_CONFIG["input_window"],
) -> None:
    """Persist model and feature statistics under a variant directory."""
    import pickle

    models_dir = variant_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(models_dir / "hybrid_gru.keras"))

    payload = {
        "res_mean": stats.res_mean,
        "res_std": stats.res_std,
        "yhat_mean": stats.yhat_mean,
        "yhat_std": stats.yhat_std,
        "input_window": input_window,
        "feature_stats": stats.to_dict(),
        "features": stats.feature_columns,
        "variant_id": stats.variant_id,
    }
    with open(models_dir / "feature_stats.pkl", "wb") as handle:
        pickle.dump(payload, handle)


def load_variant_artifacts(variant_dir: Path) -> tuple[Any, FeatureStats, list[str], int]:
    """Load trained model, feature stats, feature list, and input window."""
    import pickle

    import tensorflow as tf

    models_dir = variant_dir / "models"
    model = tf.keras.models.load_model(str(models_dir / "hybrid_gru.keras"))
    with open(models_dir / "feature_stats.pkl", "rb") as handle:
        payload = pickle.load(handle)
    stats = FeatureStats.from_dict(payload["feature_stats"])
    features = list(payload["features"])
    input_window = int(payload["input_window"])
    return model, stats, features, input_window
