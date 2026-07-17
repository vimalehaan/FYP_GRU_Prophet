"""Peak-Aware Hybrid Prophet + GRU configuration constants."""

from __future__ import annotations

from pathlib import Path

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW

# --- Locked peak definition (Phase 2 Task 7) ---

PEAK_PERCENTILE: int = 90
PEAK_RULE: str = "cpu_real >= P90(container_train)"
PEAK_VALUE_SPACE: str = "real_cpu_percent"
PEAK_THRESHOLD_SCOPE: str = "per_container"
PEAK_THRESHOLD_FIT_PERIOD: str = "train_only"

# --- Locked peak-aware training (Phase 3) ---

PEAK_WEIGHT: float = 5.0
NON_PEAK_WEIGHT: float = 1.0
WEIGHTING_GRANULARITY: str = "timestep"
WEIGHT_SCOPE: str = "forecast_horizon_targets_only"

MODEL_VARIANT: str = "hybrid_prophet_gru_peak_aware_v1"

# --- Shared Hybrid geometry (must match frozen baseline) ---

INPUT_WINDOW: int = DEFAULT_INPUT_WINDOW
FORECAST_HORIZON: int = DAY1_HORIZON

# --- Training hyperparameters (must match baseline_metadata.json) ---

OPTIMIZER: str = "adam"
BASE_LOSS: str = "mse"
TRAINING_METRICS: tuple[str, ...] = ("mae",)
EPOCHS_MAX: int = 100
BATCH_SIZE: int = 64
SHUFFLE: bool = False
RANDOM_SEED: int = 42
SEQUENCE_VAL_SPLIT: float = 0.8

EARLY_STOPPING_MONITOR: str = "val_loss"
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = True

# --- Verified sequence counts (frozen train_df, input_window=96) ---

EXPECTED_N_SEQUENCES: int = 41_650
EXPECTED_N_TRAIN: int = 33_320
EXPECTED_N_VAL: int = 8_330

# --- Research references ---

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REFERENCE_DIR = REPO_ROOT / "experiments/baseline_reference_2026-07-14"
PEAK_DEFINITION_REFERENCE = (
    REPO_ROOT / "experiments/peak_exploration_2026-07-14/peak_definition_decision.json"
)
PHASE3_DESIGN_REFERENCE = (
    REPO_ROOT / "experiments/peak_aware_design_2026-07-14/phase3_design.json"
)
PHASE4_WORKSPACE_DIR = REPO_ROOT / "experiments/peak_aware_implementation_2026-07-14"

DATA_TRAIN = REPO_ROOT / "data/train_df.parquet"
DATA_VAL = REPO_ROOT / "data/val_df.parquet"
DATA_SCALERS = REPO_ROOT / "data/scalers.pkl"
DATA_SELECTED_CONTAINERS = REPO_ROOT / "data/selected_containers.npy"

# --- Experiment artifact filenames (under experiments/peak_aware_<timestamp>/) ---

PEAK_AWARE_MODEL_FILENAME: str = "hybrid_gru_peak_aware.keras"
RESIDUAL_STATS_FILENAME: str = "residual_stats.pkl"
PEAK_THRESHOLDS_FILENAME: str = "peak_thresholds.pkl"
PEAK_CONFIG_FILENAME: str = "peak_config.json"
TRAINING_METADATA_FILENAME: str = "training_metadata.json"
EXPERIMENT_METADATA_FILENAME: str = "experiment_metadata.json"
FAIRNESS_CHECK_FILENAME: str = "fairness_check.json"
