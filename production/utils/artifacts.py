"""Production Hybrid artifact persistence."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import tensorflow as tf

from production.utils.config import (
    CACHE_DIR,
    CONFIG_DIR,
    DATASET_PREPROCESS_METADATA_PATH,
    FORECAST_HORIZON,
    HYBRID_DIR,
    INPUT_WINDOW,
    LOGS_DIR,
    METADATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    MODEL_PATH,
    PREDICTIONS_DIR,
    PRODUCTION_CONFIG_PATH,
    PROPHET_DAILY_SEASONALITY,
    PROPHET_METADATA_PATH,
    PROPHET_WEEKLY_SEASONALITY,
    REPORTS_DIR,
    RESIDUAL_STATS_PATH,
    SCALERS_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_METADATA_PATH,
)


def ensure_production_dirs() -> None:
    """Create production artifact directories."""
    for path in (
        HYBRID_DIR,
        CONFIG_DIR,
        MODELS_DIR,
        METADATA_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_production_artifacts(
    model: Any,
    scalers: dict[str, Any],
    res_mean: float,
    res_std: float,
    training_metadata: dict[str, Any],
    container_split_meta: dict[str, Any],
    preprocess_meta: dict[str, Any],
) -> None:
    """Persist all production Hybrid artifacts."""
    ensure_production_dirs()

    model.save(str(MODEL_PATH))

    with SCALERS_PATH.open("wb") as fh:
        pickle.dump(scalers, fh)

    with RESIDUAL_STATS_PATH.open("wb") as fh:
        pickle.dump(
            {
                "res_mean": res_mean,
                "res_std": res_std,
                "input_window": INPUT_WINDOW,
                "forecast_horizon": FORECAST_HORIZON,
            },
            fh,
        )

    prophet_meta = {
        "daily_seasonality": PROPHET_DAILY_SEASONALITY,
        "weekly_seasonality": PROPHET_WEEKLY_SEASONALITY,
        "fitted_during_training": True,
        "n_train_containers": preprocess_meta.get("n_train_containers_retained"),
        "note": "Per-container Prophet models are refit at inference on historical data.",
    }
    with PROPHET_METADATA_PATH.open("wb") as fh:
        pickle.dump(prophet_meta, fh)

    production_config = {
        "model_type": "hybrid_prophet_gru",
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "optimizer": training_metadata.get("optimizer"),
        "loss": training_metadata.get("loss"),
        "batch_size": training_metadata.get("batch_size"),
        "early_stopping_patience": training_metadata.get(
            "early_stopping_patience"
        ),
        "prophet": prophet_meta,
        "residual_normalization": "global",
        "artifact_paths": {
            "model": "production/hybrid/models/model.keras",
            "scalers": "production/hybrid/metadata/scalers.pkl",
            "residual_stats": "production/hybrid/metadata/residual_stats.pkl",
            "production_config": "production/hybrid/config/production_config.json",
        },
    }
    with PRODUCTION_CONFIG_PATH.open("w") as fh:
        json.dump(production_config, fh, indent=2)

    history = training_metadata.pop("history", None)
    training_metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
    with TRAINING_METADATA_PATH.open("w") as fh:
        json.dump(training_metadata, fh, indent=2, default=str)

    if history is not None:
        hist_df = pd.DataFrame(history)
        hist_df.index.name = "epoch"
        hist_df.reset_index(inplace=True)
        hist_df["epoch"] = hist_df["epoch"] + 1
        hist_df.to_csv(TRAINING_HISTORY_PATH, index=False)

    split_record = {
        "container_split": container_split_meta,
        "preprocess": preprocess_meta,
    }
    with DATASET_PREPROCESS_METADATA_PATH.open("w") as fh:
        json.dump(split_record, fh, indent=2)


def load_production_model() -> Any:
    """Load the production Keras model."""
    return tf.keras.models.load_model(str(MODEL_PATH))


def load_production_scalers() -> dict[str, Any]:
    """Load per-container MinMax scalers for training containers."""
    with SCALERS_PATH.open("rb") as fh:
        return pickle.load(fh)


def load_residual_stats() -> tuple[float, float, int, int]:
    """Load global residual statistics and window sizes."""
    with RESIDUAL_STATS_PATH.open("rb") as fh:
        stats = pickle.load(fh)
    return (
        float(stats["res_mean"]),
        float(stats["res_std"]),
        int(stats.get("input_window", INPUT_WINDOW)),
        int(stats.get("forecast_horizon", FORECAST_HORIZON)),
    )
