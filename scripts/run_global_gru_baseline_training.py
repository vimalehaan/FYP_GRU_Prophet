#!/usr/bin/env python3
"""Train Global GRU v1 baseline and save production + experiment artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_artifacts import save_global_artifacts  # noqa: E402
from utils.global_config import (  # noqa: E402
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    FORECAST_HORIZON,
    GLOBAL_GRU_METADATA_FILENAME,
    GLOBAL_GRU_MODEL_FILENAME,
    INPUT_WINDOW,
    RANDOM_SEED,
    TRAINING_METADATA_FILENAME,
)
from utils.global_training import train_global_gru  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Global GRU v1 and save model + metadata artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip pre-training data-split verification gate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment output directory",
    )
    return parser.parse_args()


def _run_data_split_verification() -> None:
    script = REPO_ROOT / "scripts/verify_global_gru_data_split.py"
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=REPO_ROOT,
    )


def _experiment_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return REPO_ROOT / f"experiments/global_gru_baseline_{stamp}"


def main() -> None:
    args = _parse_args()

    if not args.skip_verification:
        print("Running pre-training data-split verification...")
        _run_data_split_verification()

    train_df = pd.read_parquet(DATA_TRAIN)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    global_train = train_df[train_df["container_id"].isin(selected)].copy()

    print("Training Global GRU v1 on global_train only...")
    model, training_metadata = train_global_gru(
        global_train=global_train,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        random_seed=args.seed,
        verbose=1,
    )

    print("Saving production artifacts...")
    saved_metadata = save_global_artifacts(
        model=model,
        metadata=training_metadata,
        model_path=DEFAULT_MODEL_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
    )

    experiment_dir = _experiment_dir(args.output_dir)
    models_dir = experiment_dir / "models"
    training_dir = experiment_dir / "training"
    models_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(DEFAULT_MODEL_PATH, models_dir / GLOBAL_GRU_MODEL_FILENAME)
    shutil.copy2(
        DEFAULT_METADATA_PATH,
        models_dir / GLOBAL_GRU_METADATA_FILENAME,
    )

    training_record_path = training_dir / TRAINING_METADATA_FILENAME
    with training_record_path.open("w", encoding="utf-8") as handle:
        json.dump(saved_metadata, handle, indent=2)
        handle.write("\n")

    print("Training complete.")
    print(f"  Production model    : {DEFAULT_MODEL_PATH}")
    print(f"  Production metadata : {DEFAULT_METADATA_PATH}")
    print(f"  Experiment dir      : {experiment_dir}")
    print(f"  Epochs run          : {saved_metadata.get('epochs_run')}")
    print(f"  Final val loss      : {saved_metadata.get('final_val_loss')}")


if __name__ == "__main__":
    main()
