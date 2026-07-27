"""RRE variant training — frozen Hybrid GRU with scaling-only change."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.hybrid_training import build_hybrid_gru_model, generate_prophet_residuals
from utils.rre.config import TRAINING_CONFIG
from utils.rre.scaling import ScalingStats, apply_scaling, fit_scaling_stats
from utils.sequence_utils import create_residual_sequences


def train_rre_variant(
    global_train: pd.DataFrame,
    variant_id: str,
    input_window: int = TRAINING_CONFIG["input_window"],
    forecast_horizon: int = TRAINING_CONFIG["forecast_horizon"],
    epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    verbose: int = 0,
) -> tuple[Any, ScalingStats, pd.DataFrame, dict[str, Any]]:
    """Train one RRE variant; only residual scaling differs from Hybrid baseline."""
    from tensorflow.keras.callbacks import EarlyStopping

    train_residual_df = generate_prophet_residuals(global_train)
    stats = fit_scaling_stats(train_residual_df, variant_id)
    enriched_df = apply_scaling(train_residual_df, stats)

    features = TRAINING_CONFIG["features"]
    X_all, y_all, _ = create_residual_sequences(
        enriched_df,
        features=features,
        target=TRAINING_CONFIG["target"],
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
        n_features=TRAINING_CONFIG["n_features"],
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
    best_train = float(hist["loss"][best_epoch - 1])
    best_val = float(hist["val_loss"][best_epoch - 1])
    meta: dict[str, Any] = {
        "variant_id": variant_id,
        "scaling_stats": stats.to_dict(),
        "total_sequences": int(len(X_all)),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "generalisation_gap": float(best_val - best_train),
        "convergence_speed": best_epoch,
        "training_config": TRAINING_CONFIG,
        "history": hist,
    }
    return model, stats, enriched_df, meta


def save_variant_artifacts(
    model: Any,
    stats: ScalingStats,
    variant_dir: Path,
    input_window: int = TRAINING_CONFIG["input_window"],
) -> None:
    """Persist model and scaling statistics."""
    import pickle

    models_dir = variant_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(models_dir / "hybrid_gru.keras"))
    payload = {
        "scaling_stats": stats.to_dict(),
        "input_window": input_window,
        "variant_id": stats.variant_id,
    }
    with open(models_dir / "scaling_stats.pkl", "wb") as handle:
        pickle.dump(payload, handle)
