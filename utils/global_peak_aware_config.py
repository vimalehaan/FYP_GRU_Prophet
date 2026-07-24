"""Peak-Aware Global GRU configuration constants (Stage 2 treatment track)."""

from __future__ import annotations

from pathlib import Path

from utils.global_config import (
    BATCH_SIZE,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEMO_CONTAINER_ID,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_GRU_REFERENCE_DIR,
    GLOBAL_TARGET,
    GRU_UNITS,
    INPUT_WINDOW,
    MAPE_EPSILON,
    OPTIMIZER,
    RANDOM_SEED,
    REPO_ROOT,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    STALE_SELECTED_CONTAINER_ID,
    TRAINING_METRICS,
)
from utils.peak_config import (
    NON_PEAK_WEIGHT,
    PEAK_PERCENTILE,
    PEAK_RULE,
    PEAK_THRESHOLD_FIT_PERIOD,
    PEAK_THRESHOLD_SCOPE,
    PEAK_VALUE_SPACE,
    PEAK_WEIGHT,
    WEIGHTING_GRANULARITY,
    WEIGHT_SCOPE,
)

# --- Treatment model identity ---

MODEL_VARIANT: str = "global_gru_peak_aware_v1"
BASE_LOSS: str = "mse"
TRAINING_LOSS: str = "timestep_weighted_mse"

# --- Architecture builder (read-only reuse policy) ---

ARCHITECTURE_BUILDER_MODULE: str = "utils.global_training.build_global_gru_model"

# --- Research references ---

DESIGN_LOCK_REFERENCE: Path = (
    REPO_ROOT / "experiments/peak_aware_global_design_2026-07-17/design_lock.json"
)
PEAK_DEFINITION_REFERENCE: Path = (
    REPO_ROOT / "experiments/peak_exploration_2026-07-14/peak_definition_decision.json"
)
CONTROL_REFERENCE_DIR: Path = GLOBAL_GRU_REFERENCE_DIR
CONTROL_BASELINE_METADATA: Path = (
    CONTROL_REFERENCE_DIR / "config/baseline_metadata.json"
)
CONTROL_MODEL_PATH: Path = CONTROL_REFERENCE_DIR / "models/global_gru.keras"
STAGE2_WORKSPACE_DIR: Path = (
    REPO_ROOT / "experiments/peak_aware_global_implementation_2026-07-17"
)
STAGE2_PLAN_REFERENCE: Path = (
    REPO_ROOT / "docs/peak-aware-global/stage-02-implementation-plan.md"
)
STAGE3_PLAN_REFERENCE: Path = (
    REPO_ROOT / "docs/peak-aware-global/stage-03-evaluation-plan.md"
)
PRIMARY_EXPERIMENT_DIR: Path = (
    REPO_ROOT / "experiments/peak_aware_global_validation_vs1_2026-07-17_170939"
)
ARCHIVED_EXPERIMENT_DIR: Path = (
    REPO_ROOT / "experiments/peak_aware_global_2026-07-17_164737"
)

# --- Forbidden write targets (fairness static checks) ---

PRODUCTION_MODEL_PATH: Path = REPO_ROOT / "models/global_gru.keras"
PRODUCTION_METADATA_PATH: Path = REPO_ROOT / "models/global_gru_metadata.json"

FROZEN_GLOBAL_MODULES: tuple[str, ...] = (
    "utils.global_config",
    "utils.global_training",
    "utils.global_inference",
    "utils.global_evaluation",
    "utils.global_artifacts",
    "utils.global_plotting",
)

# --- Experiment artifact filenames ---

PEAK_AWARE_MODEL_FILENAME: str = "global_gru_peak_aware.keras"
PEAK_AWARE_METADATA_FILENAME: str = "global_gru_peak_aware_metadata.json"
PEAK_THRESHOLDS_FILENAME: str = "peak_thresholds.pkl"
PEAK_CONFIG_FILENAME: str = "peak_config.json"
TRAINING_METADATA_FILENAME: str = "training_metadata.json"
EXPERIMENT_METADATA_FILENAME: str = "experiment_metadata.json"
FAIRNESS_CHECK_FILENAME: str = "fairness_check.json"
REFERENCE_METADATA_FILENAME: str = "reference_metadata.json"

# --- Experiment directory layout (relative to experiment root) ---

MODELS_SUBDIR: str = "models"
PEAK_SUBDIR: str = "peak"
TRAINING_SUBDIR: str = "training"
VERIFICATION_SUBDIR: str = "verification"
REFERENCE_SUBDIR: str = "reference"
EVALUATION_SUBDIR: str = "evaluation"

EXPERIMENT_OUTPUT_PREFIX: str = "peak_aware_global"


def experiment_output_dir(timestamp: str) -> Path:
    """Return ``experiments/peak_aware_global_<timestamp>/`` under repo root."""
    return REPO_ROOT / "experiments" / f"{EXPERIMENT_OUTPUT_PREFIX}_{timestamp}"
