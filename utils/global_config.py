"""Global GRU baseline configuration constants (Phase 0.5 locked specification)."""

from __future__ import annotations

from pathlib import Path

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW

# --- Model identity ---

MODEL_VARIANT: str = "global_gru_v1"
BASELINE_SPEC_REFERENCE: str = "docs/global-gru-baseline/phase-03-5-baseline-specification.md"

# --- Shared forecast geometry (read-only from frozen hybrid_config) ---

INPUT_WINDOW: int = DEFAULT_INPUT_WINDOW
FORECAST_HORIZON: int = DAY1_HORIZON

# --- Global GRU features and target (Phase 0.5 locked) ---

GLOBAL_FEATURES: tuple[str, ...] = ("cpu_scaled", "cpu_mean", "cpu_std")
GLOBAL_TARGET: str = "cpu_scaled"

# --- Training hyperparameters (existing notebook values — no tuning) ---

OPTIMIZER: str = "adam"
LOSS: str = "mse"
TRAINING_METRICS: tuple[str, ...] = ("mae",)
EPOCHS_MAX: int = 50
BATCH_SIZE: int = 256
SHUFFLE: bool = False
RANDOM_SEED: int = 42
SEQUENCE_VAL_SPLIT: float = 0.8

EARLY_STOPPING_MONITOR: str = "val_loss"
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = True

# --- Model form (Phase 0.5 locked) ---

GRU_UNITS: tuple[int, ...] = (128, 64)
DENSE_UNITS: int = 64
DROPOUT: float = 0.2

# --- Evaluation (shared protocol with Hybrid) ---

DEMO_CONTAINER_ID: str = "c_11461"
STALE_SELECTED_CONTAINER_ID: str = "c_14674"
# Must match utils.hybrid_evaluation.MAPE_EPSILON (frozen Hybrid baseline).
MAPE_EPSILON: float = 0.01

# --- Verified sequence counts (global_train, 100 selected, Task 1) ---

EXPECTED_N_SEQUENCES: int = 41_650
EXPECTED_N_TRAIN: int = 33_320
EXPECTED_N_VAL: int = 8_330

# --- Repository paths ---

REPO_ROOT = Path(__file__).resolve().parents[1]

HYBRID_BASELINE_REFERENCE_DIR = (
    REPO_ROOT / "experiments/baseline_reference_2026-07-14"
)

# Official Global GRU baseline (patience=10 correction promoted 2026-07-17).
PRIMARY_GLOBAL_BASELINE_DIR = (
    REPO_ROOT / "experiments/global_gru_baseline_2026-07-17_121748"
)
# Previous official reference (patience=3 — superseded implementation oversight).
SUPERSEDED_GLOBAL_BASELINE_DIR = (
    REPO_ROOT / "experiments/global_gru_reference_2026-07-17"
)
GLOBAL_GRU_REFERENCE_DIR = PRIMARY_GLOBAL_BASELINE_DIR
PHASE_0_5_SPEC_PATH = REPO_ROOT / "docs/global-gru-baseline/phase-03-5-baseline-specification.md"

DATA_TRAIN = REPO_ROOT / "data/train_df.parquet"
DATA_VAL = REPO_ROOT / "data/val_df.parquet"
DATA_SCALERS = REPO_ROOT / "data/scalers.pkl"
DATA_SELECTED_CONTAINERS = REPO_ROOT / "data/selected_containers.npy"

DEFAULT_MODEL_PATH = REPO_ROOT / "models/global_gru.keras"
DEFAULT_METADATA_PATH = REPO_ROOT / "models/global_gru_metadata.json"

# --- Experiment artifact filenames (under experiments/global_gru_baseline_<timestamp>/) ---

GLOBAL_GRU_MODEL_FILENAME: str = "global_gru.keras"
GLOBAL_GRU_METADATA_FILENAME: str = "global_gru_metadata.json"
TRAINING_METADATA_FILENAME: str = "training_metadata.json"
EXPERIMENT_METADATA_FILENAME: str = "experiment_metadata.json"
BASELINE_METADATA_FILENAME: str = "baseline_metadata.json"
