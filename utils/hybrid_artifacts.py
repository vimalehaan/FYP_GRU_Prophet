"""Save and load Hybrid Prophet + GRU training artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import tensorflow as tf

from utils.hybrid_config import DEFAULT_INPUT_WINDOW

DEFAULT_MODEL_PATH = Path("models/hybrid_gru.keras")
DEFAULT_RESIDUAL_STATS_PATH = Path("models/residual_stats.pkl")


def save_hybrid_artifacts(
    hybrid_gru_model: Any,
    res_mean: float,
    res_std: float,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    residual_stats_path: str | Path = DEFAULT_RESIDUAL_STATS_PATH,
    input_window: int = DEFAULT_INPUT_WINDOW,
) -> None:
    """Persist the trained GRU model and global residual statistics."""
    model_path = Path(model_path)
    residual_stats_path = Path(residual_stats_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    hybrid_gru_model.save(str(model_path))

    with open(residual_stats_path, "wb") as handle:
        pickle.dump(
            {
                "res_mean": res_mean,
                "res_std": res_std,
                "input_window": input_window,
            },
            handle,
        )


def load_hybrid_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    residual_stats_path: str | Path = DEFAULT_RESIDUAL_STATS_PATH,
) -> tuple[Any, float, float, int]:
    """Load the trained GRU model, residual statistics, and input window."""
    model_path = Path(model_path)
    residual_stats_path = Path(residual_stats_path)

    hybrid_gru_model = tf.keras.models.load_model(str(model_path))

    with open(residual_stats_path, "rb") as handle:
        residual_stats = pickle.load(handle)

    input_window = int(
        residual_stats.get("input_window", DEFAULT_INPUT_WINDOW)
    )

    return (
        hybrid_gru_model,
        residual_stats["res_mean"],
        residual_stats["res_std"],
        input_window,
    )
