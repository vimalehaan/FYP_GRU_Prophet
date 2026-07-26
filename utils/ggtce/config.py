"""Frozen GGTCE protocol constants."""

from __future__ import annotations

from typing import Any

from utils.global_config import (
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
    OPTIMIZER,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    TRAINING_METRICS,
)

PROTOCOL_VERSION = "ggtce_v1.0"
EXPERIMENT_NAME = "ggtce"

VARIANT_WINDOWS: dict[str, int] = {
    "g96": 96,
    "g192": 192,
    "g288": 288,
}

VARIANT_LABELS: dict[str, str] = {
    "g96": "G96 — Input 96 (baseline reproduction)",
    "g192": "G192 — Input 192",
    "g288": "G288 — Input 288",
}

VARIANT_ORDER: tuple[str, ...] = ("g96", "g192", "g288")

GGTCE_VARIANTS: dict[str, dict[str, Any]] = {
    vid: {
        "variant_id": vid,
        "label": VARIANT_LABELS[vid],
        "input_window": VARIANT_WINDOWS[vid],
        "forecast_horizon": FORECAST_HORIZON,
        "input_shape": [VARIANT_WINDOWS[vid], len(GLOBAL_FEATURES)],
    }
    for vid in VARIANT_ORDER
}

WINDOW_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("g96", "g192"),
    ("g192", "g288"),
    ("g96", "g288"),
)

TRAINING_CONFIG: dict[str, Any] = {
    "forecast_horizon": FORECAST_HORIZON,
    "features": list(GLOBAL_FEATURES),
    "target": GLOBAL_TARGET,
    "epochs": EPOCHS_MAX,
    "batch_size": BATCH_SIZE,
    "optimizer": OPTIMIZER,
    "loss": "mse",
    "metrics": list(TRAINING_METRICS),
    "early_stopping": {
        "monitor": EARLY_STOPPING_MONITOR,
        "patience": EARLY_STOPPING_PATIENCE,
        "restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    },
    "internal_val_split": SEQUENCE_VAL_SPLIT,
    "shuffle": SHUFFLE,
    "random_seed": RANDOM_SEED,
    "gru_units": list(GRU_UNITS),
    "dense_units": DENSE_UNITS,
    "dropout": DROPOUT,
}

SEED_POLICY: dict[str, Any] = {
    "random_seed": RANDOM_SEED,
    "bootstrap_seed": 12345,
    "note": "Independent fresh initialization per variant; seed 42 for reproducibility.",
}

FROZEN_GLOBAL_BASELINE = (
    "experiments/global_gru_baseline_2026-07-17_121748"
)
TMA_EXPERIMENT = "experiments/temporal_memory_analysis_2026-07-26_170052"

PRIMARY_METRICS: tuple[str, ...] = (
    "day1_mae",
    "day1_rmse",
    "day1_mape",
    "forecast_mae_scaled",
    "forecast_rmse_scaled",
    "forecast_pearson_r",
    "forecast_r2",
    "forecast_std_ratio",
)

METRIC_DIRECTION: dict[str, str] = {
    "day1_mae": "lower",
    "day1_rmse": "lower",
    "day1_mape": "lower",
    "forecast_mae_scaled": "lower",
    "forecast_rmse_scaled": "lower",
    "forecast_pearson_r": "higher",
    "forecast_r2": "higher",
    "forecast_std_ratio": "higher",
}

G96_REPLICATION_TOLERANCE: dict[str, float] = {
    "mean_mae_abs_delta": 0.05,
    "rank_corr_min": 0.95,
}
