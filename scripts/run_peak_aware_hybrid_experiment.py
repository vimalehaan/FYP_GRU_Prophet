#!/usr/bin/env python3
"""Run Peak-Aware Hybrid Prophet + GRU training and save experiment artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_training_peak_aware import train_hybrid_gru_peak_aware  # noqa: E402
from utils.peak_config import (  # noqa: E402
    BASELINE_REFERENCE_DIR,
    BATCH_SIZE,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    MODEL_VARIANT,
    NON_PEAK_WEIGHT,
    OPTIMIZER,
    PEAK_AWARE_MODEL_FILENAME,
    PEAK_CONFIG_FILENAME,
    PEAK_DEFINITION_REFERENCE,
    PEAK_PERCENTILE,
    PEAK_RULE,
    PEAK_THRESHOLDS_FILENAME,
    PEAK_VALUE_SPACE,
    PEAK_WEIGHT,
    PHASE3_DESIGN_REFERENCE,
    PHASE4_WORKSPACE_DIR,
    RANDOM_SEED,
    RESIDUAL_STATS_FILENAME,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    TRAINING_METADATA_FILENAME,
    WEIGHTING_GRANULARITY,
    WEIGHT_SCOPE,
)
from utils.peak_detection import (  # noqa: E402
    save_peak_thresholds,
    summarize_weight_matrix,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Peak-Aware Hybrid GRU and save timestamped artifacts.",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment output directory",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_residual_stats(
    path: Path,
    res_mean: float,
    res_std: float,
    input_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "res_mean": res_mean,
                "res_std": res_std,
                "input_window": input_window,
                "model_variant": MODEL_VARIANT,
            },
            handle,
        )


def _build_peak_config() -> dict[str, object]:
    return {
        "peak_percentile": PEAK_PERCENTILE,
        "peak_weight": PEAK_WEIGHT,
        "non_peak_weight": NON_PEAK_WEIGHT,
        "peak_rule": PEAK_RULE,
        "value_space": PEAK_VALUE_SPACE,
        "threshold_fit_period": "train_only",
        "weighting_granularity": WEIGHTING_GRANULARITY,
        "weight_scope": WEIGHT_SCOPE,
        "phase_2_reference": str(
            PEAK_DEFINITION_REFERENCE.relative_to(REPO_ROOT)
        ),
        "baseline_reference": str(
            BASELINE_REFERENCE_DIR.relative_to(REPO_ROOT)
        ),
    }


def _update_phase4_workspace(training_run_dir: Path) -> None:
    workspace_meta_path = PHASE4_WORKSPACE_DIR / "experiment_metadata.json"
    if not workspace_meta_path.exists():
        return

    workspace_meta = json.loads(workspace_meta_path.read_text())
    workspace_meta["training_run_dir"] = str(
        training_run_dir.relative_to(REPO_ROOT)
    )
    workspace_meta["tasks"]["task_4"] = "complete"
    workspace_meta_path.write_text(json.dumps(workspace_meta, indent=2))


def main() -> None:
    args = _parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "experiments" / f"peak_aware_{run_timestamp}"
    )
    experiment_dir = experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    models_dir = experiment_dir / "models"
    peak_dir = experiment_dir / "peak"
    training_dir = experiment_dir / "training"
    verification_dir = experiment_dir / "verification"
    evaluation_dir = experiment_dir / "evaluation"

    for directory in (models_dir, peak_dir, training_dir, verification_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / PEAK_AWARE_MODEL_FILENAME
    residual_stats_path = models_dir / RESIDUAL_STATS_FILENAME
    peak_thresholds_path = peak_dir / PEAK_THRESHOLDS_FILENAME
    peak_config_path = peak_dir / PEAK_CONFIG_FILENAME
    training_metadata_path = training_dir / TRAINING_METADATA_FILENAME
    experiment_metadata_path = experiment_dir / "experiment_metadata.json"

    print("=" * 72)
    print("Peak-Aware Hybrid Prophet + GRU — Training Experiment")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Random seed          : {args.seed}")
    print(f"Max epochs           : {args.epochs}")
    print(f"Input window         : {INPUT_WINDOW}")
    print(f"Forecast horizon     : {FORECAST_HORIZON}")
    print()

    train_df = pd.read_parquet(DATA_TRAIN)
    selected_containers = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected_containers)
    ].copy()

    print(f"Train containers     : {global_train['container_id'].nunique()}")
    print(f"Expected sequences   : {EXPECTED_N_SEQUENCES}")
    print()
    print("Starting training (Prophet residuals + weighted GRU)...")
    print("-" * 72)

    trained_at = _iso_timestamp()
    model, res_mean, res_std, _, peak_thresholds, weight_matrix = (
        train_hybrid_gru_peak_aware(
            global_train=global_train,
            scalers=scalers,
            epochs=args.epochs,
            verbose=1,
            random_seed=args.seed,
        )
    )

    history = model.history.history
    epochs_run = len(history.get("loss", []))
    weight_summary = summarize_weight_matrix(weight_matrix)

    model.save(str(model_path))
    _save_residual_stats(
        path=residual_stats_path,
        res_mean=res_mean,
        res_std=res_std,
        input_window=INPUT_WINDOW,
    )
    save_peak_thresholds(peak_thresholds, peak_thresholds_path)

    peak_config = _build_peak_config()
    peak_config_path.write_text(json.dumps(peak_config, indent=2))

    training_metadata: dict[str, object] = {
        "model_variant": MODEL_VARIANT,
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "optimizer": OPTIMIZER,
        "loss": "weighted_mse",
        "loss_implementation": "custom_train_step_timestep_weighted_mse",
        "base_loss": "mse",
        "peak_weight": PEAK_WEIGHT,
        "non_peak_weight": NON_PEAK_WEIGHT,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        "epochs_max": args.epochs,
        "epochs_run": epochs_run,
        "batch_size": BATCH_SIZE,
        "shuffle": SHUFFLE,
        "random_seed": args.seed,
        "sequence_val_split": SEQUENCE_VAL_SPLIT,
        "res_mean": res_mean,
        "res_std": res_std,
        "n_sequences_total": EXPECTED_N_SEQUENCES,
        "n_sequences_train": EXPECTED_N_TRAIN,
        "n_sequences_val": EXPECTED_N_VAL,
        "n_peak_thresholds": len(peak_thresholds),
        "weight_summary": weight_summary,
        "final_train_loss": float(history["loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_train_mae": float(history["mae"][-1]),
        "final_val_mae": float(history["val_mae"][-1]),
        "validation_unweighted": True,
        "trained_at": trained_at,
        "frozen_modules_used": [
            "utils.hybrid_training.generate_prophet_residuals",
            "utils.hybrid_training.build_hybrid_gru_model",
            "utils.sequence_utils.create_residual_sequences",
        ],
    }
    training_metadata_path.write_text(json.dumps(training_metadata, indent=2))

    experiment_metadata: dict[str, object] = {
        "experiment_type": "peak_aware_hybrid",
        "phase": 4,
        "timestamp": run_timestamp,
        "baseline_reference": str(BASELINE_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "peak_definition_reference": str(
            PEAK_DEFINITION_REFERENCE.relative_to(REPO_ROOT)
        ),
        "design_document": str(
            PHASE3_DESIGN_REFERENCE.relative_to(REPO_ROOT)
        ),
        "implementation_workspace": str(
            PHASE4_WORKSPACE_DIR.relative_to(REPO_ROOT)
        ),
        "data_artifacts": [
            str(DATA_TRAIN.relative_to(REPO_ROOT)),
            str(DATA_SCALERS.relative_to(REPO_ROOT)),
            str(DATA_SELECTED_CONTAINERS.relative_to(REPO_ROOT)),
        ],
        "outputs": {
            "model": str(model_path.relative_to(experiment_dir)),
            "residual_stats": str(residual_stats_path.relative_to(experiment_dir)),
            "peak_thresholds": str(peak_thresholds_path.relative_to(experiment_dir)),
            "peak_config": str(peak_config_path.relative_to(experiment_dir)),
            "training_metadata": str(
                training_metadata_path.relative_to(experiment_dir)
            ),
        },
        "status": {
            "training": "complete",
            "fairness_verification": "pending",
            "evaluation": "pending",
        },
        "training_summary": {
            "epochs_run": epochs_run,
            "res_mean": res_mean,
            "res_std": res_std,
            "final_val_loss": training_metadata["final_val_loss"],
        },
    }
    experiment_metadata_path.write_text(json.dumps(experiment_metadata, indent=2))
    _update_phase4_workspace(experiment_dir)

    print("-" * 72)
    print("Training complete")
    print(f"Epochs run           : {epochs_run}")
    print(f"Final train loss     : {training_metadata['final_train_loss']:.6f}")
    print(f"Final val loss       : {training_metadata['final_val_loss']:.6f}")
    print(f"res_mean             : {res_mean}")
    print(f"res_std              : {res_std}")
    print()
    print("Saved artifacts:")
    print(f"  Model              : {model_path}")
    print(f"  Residual stats     : {residual_stats_path}")
    print(f"  Peak thresholds    : {peak_thresholds_path}")
    print(f"  Peak config        : {peak_config_path}")
    print(f"  Training metadata  : {training_metadata_path}")
    print(f"  Experiment metadata: {experiment_metadata_path}")
    print()
    print("Phase 5 evaluation not run (by design).")


if __name__ == "__main__":
    main()
