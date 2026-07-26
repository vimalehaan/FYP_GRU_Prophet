"""Frozen HCERL protocol constants and variant feature definitions."""

from __future__ import annotations

from typing import Any

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW

PROTOCOL_VERSION = "hcerl_v1.0"
EXPERIMENT_NAME = "hcerl"

# Cumulative ablation ladder — each variant adds one channel.
VARIANT_FEATURES: dict[str, list[str]] = {
    "v0_baseline": ["residual_scaled"],
    "v1_prophet": ["residual_scaled", "prophet_yhat_scaled"],
    "v2_volatility": [
        "residual_scaled",
        "prophet_yhat_scaled",
        "res_roll_std_12",
    ],
    "v3_cpu_std": [
        "residual_scaled",
        "prophet_yhat_scaled",
        "res_roll_std_12",
        "cpu_std",
    ],
    "v4_full": [
        "residual_scaled",
        "prophet_yhat_scaled",
        "res_roll_std_12",
        "cpu_std",
        "is_peak_p90",
    ],
}

VARIANT_LABELS: dict[str, str] = {
    "v0_baseline": "V0 — Residual only (96×1)",
    "v1_prophet": "V1 — + Prophet forecast (96×2)",
    "v2_volatility": "V2 — + Rolling residual std (96×3)",
    "v3_cpu_std": "V3 — + Container cpu_std (96×4)",
    "v4_full": "V4 — + Peak P90 (96×5)",
}

VARIANT_ORDER: tuple[str, ...] = (
    "v0_baseline",
    "v1_prophet",
    "v2_volatility",
    "v3_cpu_std",
    "v4_full",
)

HCERL_VARIANTS: dict[str, dict[str, Any]] = {
    vid: {
        "variant_id": vid,
        "label": VARIANT_LABELS[vid],
        "features": VARIANT_FEATURES[vid],
        "n_features": len(VARIANT_FEATURES[vid]),
        "input_shape": [DEFAULT_INPUT_WINDOW, len(VARIANT_FEATURES[vid])],
    }
    for vid in VARIANT_ORDER
}

ABLATION_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("v0_baseline", "v1_prophet"),
    ("v1_prophet", "v2_volatility"),
    ("v2_volatility", "v3_cpu_std"),
    ("v3_cpu_std", "v4_full"),
)

# Frozen training hyperparameters — identical to baseline Hybrid GRU.
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
    "volatility_window": 12,
    "peak_percentile": 90,
}

SEED_POLICY: dict[str, Any] = {
    "gru_training_seed": None,
    "bootstrap_seed": 12345,
    "wilcoxon_seed_note": "Wilcoxon is deterministic given paired inputs",
    "note": (
        "Matches frozen train_hybrid_gru (no TF seed). Cross-variant fairness "
        "relies on identical protocol, not identical initial weights."
    ),
}

FROZEN_BASELINE_REF = (
    "experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv"
)
PEAK_THRESHOLDS_REF = (
    "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl"
)

PRIMARY_METRICS: tuple[str, ...] = (
    "day1_mae",
    "day1_rmse",
    "day1_mape",
    "peak_mae",
    "peak_rmse",
    "residual_mae_scaled",
    "residual_rmse_scaled",
    "residual_pearson_r",
    "residual_r2",
    "residual_std_ratio",
    "energy_recovery_ratio",
)

METRIC_DIRECTION: dict[str, str] = {
    "day1_mae": "lower",
    "day1_rmse": "lower",
    "day1_mape": "lower",
    "peak_mae": "lower",
    "peak_rmse": "lower",
    "residual_mae_scaled": "lower",
    "residual_rmse_scaled": "lower",
    "residual_pearson_r": "higher",
    "residual_r2": "higher",
    "residual_std_ratio": "higher",
    "energy_recovery_ratio": "higher",
}
