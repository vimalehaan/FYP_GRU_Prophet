#!/usr/bin/env python3
"""Run Peak-Aware Global GRU training and save experiment artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_peak_aware_config import (  # noqa: E402
    CONTROL_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DESIGN_LOCK_REFERENCE,
    EPOCHS_MAX,
    EXPECTED_N_SEQUENCES,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    MODEL_VARIANT,
    NON_PEAK_WEIGHT,
    PEAK_AWARE_METADATA_FILENAME,
    PEAK_AWARE_MODEL_FILENAME,
    PEAK_CONFIG_FILENAME,
    PEAK_DEFINITION_REFERENCE,
    PEAK_PERCENTILE,
    PEAK_RULE,
    PEAK_THRESHOLDS_FILENAME,
    PEAK_THRESHOLD_FIT_PERIOD,
    PEAK_THRESHOLD_SCOPE,
    PEAK_VALUE_SPACE,
    PEAK_WEIGHT,
    RANDOM_SEED,
    REFERENCE_METADATA_FILENAME,
    REFERENCE_SUBDIR,
    STAGE2_WORKSPACE_DIR,
    TRAINING_METADATA_FILENAME,
    WEIGHTING_GRANULARITY,
    WEIGHT_SCOPE,
)
from utils.global_training_peak_aware import train_global_gru_peak_aware  # noqa: E402
from utils.peak_detection import save_peak_thresholds  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Peak-Aware Global GRU and save timestamped artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_MAX,
        help=f"Maximum training epochs (default: {EPOCHS_MAX})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run with 2 epochs (overrides --epochs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment output directory",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Keras fit verbosity (default: 1)",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_global_train() -> pd.DataFrame:
    train_df = pd.read_parquet(DATA_TRAIN)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    return train_df[train_df["container_id"].isin(selected)].copy()


def _build_peak_config() -> dict[str, object]:
    return {
        "peak_percentile": PEAK_PERCENTILE,
        "peak_weight": PEAK_WEIGHT,
        "non_peak_weight": NON_PEAK_WEIGHT,
        "peak_rule": PEAK_RULE,
        "value_space": PEAK_VALUE_SPACE,
        "threshold_scope": PEAK_THRESHOLD_SCOPE,
        "threshold_fit_period": PEAK_THRESHOLD_FIT_PERIOD,
        "weighting_granularity": WEIGHTING_GRANULARITY,
        "weight_scope": WEIGHT_SCOPE,
        "peak_definition_reference": str(
            PEAK_DEFINITION_REFERENCE.relative_to(REPO_ROOT)
        ),
        "design_lock_reference": str(
            DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)
        ),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "model_variant": MODEL_VARIANT,
    }


def _environment_snapshot() -> dict[str, str]:
    import tensorflow as tf

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
    }


def _build_reference_metadata(trained_at: str) -> dict[str, object]:
    return {
        "global_baseline_reference": str(
            CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)
        ),
        "baseline_version": "global_gru_v1",
        "design_lock_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
        "implementation_version": MODEL_VARIANT,
        "frozen_control_path": str(
            (CONTROL_REFERENCE_DIR / "models/global_gru.keras").relative_to(REPO_ROOT)
        ),
        "timestamp": trained_at,
        "stage": 2,
        "fairness_checks_required": 10,
    }


def _update_stage2_workspace(training_run_dir: Path) -> None:
    workspace_meta_path = STAGE2_WORKSPACE_DIR / "experiment_metadata.json"
    if not workspace_meta_path.exists():
        return

    workspace_meta = json.loads(workspace_meta_path.read_text(encoding="utf-8"))
    workspace_meta["training_run_dir"] = str(
        training_run_dir.relative_to(REPO_ROOT)
    )
    workspace_meta_path.write_text(
        json.dumps(workspace_meta, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    epochs = 2 if args.smoke else args.epochs

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "experiments" / f"peak_aware_global_{run_timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    models_dir = experiment_dir / "models"
    peak_dir = experiment_dir / "peak"
    training_dir = experiment_dir / "training"
    verification_dir = experiment_dir / "verification"
    reference_dir = experiment_dir / REFERENCE_SUBDIR

    for directory in (models_dir, peak_dir, training_dir, verification_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / PEAK_AWARE_MODEL_FILENAME
    model_metadata_path = models_dir / PEAK_AWARE_METADATA_FILENAME
    peak_thresholds_path = peak_dir / PEAK_THRESHOLDS_FILENAME
    peak_config_path = peak_dir / PEAK_CONFIG_FILENAME
    training_metadata_path = training_dir / TRAINING_METADATA_FILENAME
    experiment_metadata_path = experiment_dir / "experiment_metadata.json"
    reference_metadata_path = reference_dir / REFERENCE_METADATA_FILENAME

    print("=" * 72)
    print("Peak-Aware Global GRU — Training Experiment")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Random seed          : {args.seed}")
    print(f"Max epochs           : {epochs}")
    print(f"Smoke run            : {args.smoke}")
    print(f"Input window         : {INPUT_WINDOW}")
    print(f"Forecast horizon     : {FORECAST_HORIZON}")
    print()

    global_train = _load_global_train()
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    print(f"Train containers     : {global_train['container_id'].nunique()}")
    print(f"Expected sequences   : {EXPECTED_N_SEQUENCES}")
    print()
    print("Starting training (weighted Global GRU)...")
    print("-" * 72)

    trained_at = _iso_timestamp()
    model, training_metadata, peak_thresholds, _weight_matrix = (
        train_global_gru_peak_aware(
            global_train=global_train,
            scalers=scalers,
            epochs=epochs,
            verbose=args.verbose,
            random_seed=args.seed,
        )
    )

    model.save(str(model_path))
    save_peak_thresholds(peak_thresholds, peak_thresholds_path)

    peak_config = _build_peak_config()
    peak_config_path.write_text(
        json.dumps(peak_config, indent=2) + "\n",
        encoding="utf-8",
    )

    training_metadata["experiment_dir"] = str(experiment_dir.relative_to(REPO_ROOT))
    training_metadata["smoke_run"] = bool(args.smoke)
    training_metadata_path.write_text(
        json.dumps(training_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    model_metadata = {
        **training_metadata,
        "saved_at": trained_at,
        "environment": _environment_snapshot(),
        "model_path": str(model_path.relative_to(experiment_dir)),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "design_lock_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
    }
    model_metadata_path.write_text(
        json.dumps(model_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.smoke:
        reference_metadata = _build_reference_metadata(trained_at)
        reference_metadata_path.write_text(
            json.dumps(reference_metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    experiment_metadata: dict[str, object] = {
        "experiment_type": "peak_aware_global",
        "stage": 2,
        "timestamp": run_timestamp,
        "smoke_run": bool(args.smoke),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "design_lock_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
        "peak_definition_reference": str(
            PEAK_DEFINITION_REFERENCE.relative_to(REPO_ROOT)
        ),
        "implementation_workspace": str(
            STAGE2_WORKSPACE_DIR.relative_to(REPO_ROOT)
        ),
        "data_artifacts": [
            str(DATA_TRAIN.relative_to(REPO_ROOT)),
            str(DATA_SCALERS.relative_to(REPO_ROOT)),
            str(DATA_SELECTED_CONTAINERS.relative_to(REPO_ROOT)),
        ],
        "outputs": {
            "model": str(model_path.relative_to(experiment_dir)),
            "model_metadata": str(model_metadata_path.relative_to(experiment_dir)),
            "peak_thresholds": str(peak_thresholds_path.relative_to(experiment_dir)),
            "peak_config": str(peak_config_path.relative_to(experiment_dir)),
            "training_metadata": str(
                training_metadata_path.relative_to(experiment_dir)
            ),
            **(
                {
                    "reference_metadata": str(
                        reference_metadata_path.relative_to(experiment_dir)
                    )
                }
                if not args.smoke
                else {}
            ),
        },
        "status": {
            "training": "complete",
            "fairness_verification": "pending",
            "evaluation": "pending",
        },
        "training_summary": {
            "epochs_run": training_metadata.get("epochs_run"),
            "final_val_loss": training_metadata.get("final_val_loss"),
            "peak_weight_fraction": training_metadata.get("weight_summary", {}).get(
                "peak_weight_fraction"
            ),
        },
    }
    experiment_metadata_path.write_text(
        json.dumps(experiment_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _update_stage2_workspace(experiment_dir)

    import tensorflow as tf

    _ = tf.keras.models.load_model(str(model_path))

    print("-" * 72)
    print("Training complete")
    print(f"Epochs run           : {training_metadata.get('epochs_run')}")
    print(
        f"Final train loss     : {training_metadata.get('final_train_loss'):.6f}"
    )
    print(
        f"Final val loss       : {training_metadata.get('final_val_loss'):.6f}"
    )
    print()
    print("Saved artifacts:")
    print(f"  Model              : {model_path}")
    print(f"  Model metadata     : {model_metadata_path}")
    print(f"  Peak thresholds    : {peak_thresholds_path}")
    print(f"  Peak config        : {peak_config_path}")
    print(f"  Training metadata  : {training_metadata_path}")
    print(f"  Experiment metadata: {experiment_metadata_path}")
    if not args.smoke:
        print(f"  Reference metadata : {reference_metadata_path}")
    print()
    print("Stage 3 evaluation not run (by design).")


if __name__ == "__main__":
    main()
