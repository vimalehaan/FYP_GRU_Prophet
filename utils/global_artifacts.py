"""Save and load Global GRU training artifacts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf

from utils.global_config import (
    BASELINE_SPEC_REFERENCE,
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
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
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
)


def _baseline_spec_snapshot() -> dict[str, Any]:
    """Return locked Phase 0.5 baseline fields for metadata provenance."""
    return {
        "model_variant": MODEL_VARIANT,
        "spec_reference": BASELINE_SPEC_REFERENCE,
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "features": list(GLOBAL_FEATURES),
        "target": GLOBAL_TARGET,
        "architecture": (
            "GRU(128, return_sequences=True) -> Dropout(0.2) -> "
            "GRU(64) -> Dense(64, relu) -> Dense(96)"
        ),
        "gru_units": list(GRU_UNITS),
        "dense_units": DENSE_UNITS,
        "dropout": DROPOUT,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "metrics": list(TRAINING_METRICS),
        "epochs_max": EPOCHS_MAX,
        "batch_size": BATCH_SIZE,
        "shuffle": SHUFFLE,
        "random_seed": RANDOM_SEED,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_restore_best_weights": (
            EARLY_STOPPING_RESTORE_BEST_WEIGHTS
        ),
        "sequence_val_split": SEQUENCE_VAL_SPLIT,
    }


def _environment_snapshot() -> dict[str, str]:
    """Capture runtime environment for reproducibility metadata."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
    }


def _to_json_serializable(value: Any) -> Any:
    """Convert metadata values to JSON-safe types."""
    if isinstance(value, dict):
        return {
            str(key): _to_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_global_artifacts(
    model: Any,
    metadata: dict[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """
    Persist the trained Global GRU Keras model and JSON metadata record.

    Enriches ``metadata`` with ``saved_at``, ``environment``, and
    ``baseline_spec`` when those keys are not already present.

    Returns
    -------
    The metadata dict written to disk (after enrichment).
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_to_write = dict(metadata)
    metadata_to_write.setdefault(
        "saved_at",
        datetime.now(timezone.utc).isoformat(),
    )
    metadata_to_write.setdefault("environment", _environment_snapshot())
    metadata_to_write.setdefault("baseline_spec", _baseline_spec_snapshot())
    metadata_to_write["model_path"] = str(model_path)
    metadata_to_write["metadata_path"] = str(metadata_path)

    serializable_metadata = _to_json_serializable(metadata_to_write)

    model.save(str(model_path))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_metadata, handle, indent=2)
        handle.write("\n")

    return serializable_metadata


def load_global_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Load the trained Global GRU model and metadata JSON record."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Global GRU model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Global GRU metadata not found: {metadata_path}"
        )

    model = tf.keras.models.load_model(str(model_path))

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    return model, metadata
