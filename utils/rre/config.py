"""Frozen RRE protocol constants."""

from __future__ import annotations

from typing import Any

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW

PROTOCOL_VERSION = "rre_v1.0"
EXPERIMENT_NAME = "residual_scaling_experiment"
STAGE0_REFERENCE = "experiments/rre_diagnostics_2026-07-27_120759"

VARIANT_ORDER: tuple[str, ...] = ("R0", "R3")

VARIANTS: dict[str, dict[str, Any]] = {
    "R0": {
        "variant_id": "R0",
        "label": "R0 — Global z-score (frozen Hybrid baseline)",
        "scaling_method": "global_zscore",
        "formula": "(r - global_mean) / global_std",
        "inverse": "pred * global_std + global_mean",
    },
    "R3": {
        "variant_id": "R3",
        "label": "R3 — Robust median/MAD scaling",
        "scaling_method": "median_mad",
        "formula": "(r - global_median) / global_mad",
        "inverse": "pred * global_mad + global_median",
    },
}

TRAINING_CONFIG: dict[str, Any] = {
    "input_window": DEFAULT_INPUT_WINDOW,
    "forecast_horizon": DAY1_HORIZON,
    "epochs": 100,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "metrics": ["mae"],
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
    },
    "internal_val_split": "chronological_first_80_pct",
    "shuffle": False,
    "features": ["residual_scaled"],
    "target": "residual_scaled",
    "n_features": 1,
}

SEED_POLICY: dict[str, Any] = {
    "gru_training_seed": None,
    "bootstrap_seed": 12345,
    "note": "Matches frozen Hybrid train_hybrid_gru (no TF seed).",
}

FROZEN_BASELINE_REF = "experiments/baseline_reference_2026-07-14"
PEAK_THRESHOLDS_REF = (
    "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl"
)

PRIMARY_METRICS: tuple[str, ...] = (
    "day1_mae",
    "day1_rmse",
    "peak_mae",
    "peak_rmse",
    "residual_mae_scaled",
    "residual_rmse_scaled",
    "residual_pearson_r",
    "residual_r2",
    "residual_std_ratio",
)

OPTIMIZATION_METRICS: tuple[str, ...] = (
    "best_epoch",
    "epochs_run",
    "best_val_loss",
    "final_val_loss",
    "best_train_loss",
    "final_train_loss",
    "generalisation_gap",
    "convergence_speed",
)

METRIC_DIRECTION: dict[str, str] = {
    "day1_mae": "lower",
    "day1_rmse": "lower",
    "peak_mae": "lower",
    "peak_rmse": "lower",
    "residual_mae_scaled": "lower",
    "residual_rmse_scaled": "lower",
    "residual_pearson_r": "higher",
    "residual_r2": "higher",
    "residual_std_ratio": "higher",
    "best_epoch": "lower",
    "epochs_run": "lower",
    "best_val_loss": "lower",
    "final_val_loss": "lower",
    "best_train_loss": "lower",
    "final_train_loss": "lower",
    "generalisation_gap": "lower",
    "convergence_speed": "lower",
}

PRACTICAL_MAE_THRESHOLD = 0.05
