"""Production Hybrid configuration and artifact paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = REPO_ROOT / "production"
HYBRID_DIR = PRODUCTION_DIR / "hybrid"

# Artifact layout under production/hybrid/
CONFIG_DIR = HYBRID_DIR / "config"
MODELS_DIR = HYBRID_DIR / "models"
METADATA_DIR = HYBRID_DIR / "metadata"
METRICS_DIR = HYBRID_DIR / "metrics"
PREDICTIONS_DIR = HYBRID_DIR / "predictions"
LOGS_DIR = HYBRID_DIR / "logs"
REPORTS_DIR = HYBRID_DIR / "reports"
CACHE_DIR = HYBRID_DIR / "cache"

# Container-level holdout (fixed once created).
CONTAINER_SPLIT_SEED = 42
TRAIN_CONTAINER_FRACTION = 0.90
UNSEEN_CONTAINER_FRACTION = 0.10

# Temporal split (per container) — identical to frozen preprocessing.
TRAIN_SPLIT_RATIO = 0.8
MIN_ROWS_RESAMPLED = 200

# Frozen Hybrid GRU hyperparameters (utils/hybrid_training.py).
INPUT_WINDOW = 96
FORECAST_HORIZON = 96
EPOCHS_MAX = 100
BATCH_SIZE = 64
OPTIMIZER = "adam"
LOSS = "mse"
TRAINING_METRICS = ("mae",)
SEQUENCE_VAL_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Prophet (utils/hybrid_training.generate_prophet_residuals).
PROPHET_DAILY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False

# Research data source (read-only).
DATA_RESAMPLED = REPO_ROOT / "data" / "df_resampled.parquet"

# Production cache and artifact files.
TRAIN_DF_PATH = CACHE_DIR / "train_df.parquet"
VAL_DF_PATH = CACHE_DIR / "val_df.parquet"
UNSEEN_RAW_PATH = CACHE_DIR / "unseen_resampled.parquet"

MODEL_PATH = MODELS_DIR / "model.keras"
SCALERS_PATH = METADATA_DIR / "scalers.pkl"
RESIDUAL_STATS_PATH = METADATA_DIR / "residual_stats.pkl"
PROPHET_METADATA_PATH = METADATA_DIR / "prophet_metadata.pkl"
PRODUCTION_CONFIG_PATH = CONFIG_DIR / "production_config.json"
TRAINING_METADATA_PATH = CONFIG_DIR / "training_metadata.json"
TRAINING_HISTORY_PATH = LOGS_DIR / "training_history.csv"
TRAIN_CONTAINER_IDS_PATH = METADATA_DIR / "train_container_ids.json"
UNSEEN_CONTAINER_IDS_PATH = METADATA_DIR / "unseen_container_ids.json"
CONTAINER_SPLIT_METADATA_PATH = METADATA_DIR / "container_split_metadata.json"
DATASET_PREPROCESS_METADATA_PATH = METADATA_DIR / "dataset_preprocess_metadata.json"
EVALUATION_RESULTS_PATH = METRICS_DIR / "evaluation_results.csv"

MAPE_EPSILON = 0.01

# Backward-compatible alias used by older code paths.
PRODUCTION_ROOT = HYBRID_DIR
