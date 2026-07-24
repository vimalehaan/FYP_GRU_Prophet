#!/usr/bin/env python3
"""Run VS-1 validation study — Peak-Aware Global training dynamics (patience 10)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    FORECAST_HORIZON,
    INPUT_WINDOW,
)
from utils.global_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.global_peak_aware_config import (  # noqa: E402
    CONTROL_REFERENCE_DIR,
    EPOCHS_MAX,
    EVALUATION_SUBDIR,
    MODELS_SUBDIR,
    PEAK_AWARE_METADATA_FILENAME,
    PEAK_AWARE_MODEL_FILENAME,
    PEAK_CONFIG_FILENAME,
    PEAK_THRESHOLDS_FILENAME,
    PRIMARY_EXPERIMENT_DIR,
    RANDOM_SEED,
    TRAINING_METADATA_FILENAME,
)
from utils.global_training_peak_aware import train_global_gru_peak_aware  # noqa: E402
from utils.peak_evaluation import GLOBAL_INFERENCE_CACHE_PRED_KEY  # noqa: E402

VS1_PATIENCE = 10
PRIMARY_PATIENCE = 3
STUDY_LABEL = "Validation Study – Training Dynamics"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VS-1: Peak-Aware Global with EarlyStopping patience=10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override VS-1 experiment directory",
    )
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=PRIMARY_EXPERIMENT_DIR,
        help="Locked primary experiment (read-only reference)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Keras fit verbosity",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_global_train() -> pd.DataFrame:
    train_df = pd.read_parquet(DATA_TRAIN)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    return train_df[train_df["container_id"].isin(selected)].copy()


def _copy_primary_peak_artifacts(primary_dir: Path, peak_dir: Path) -> None:
    """Copy peak artifacts from primary run (no refit)."""
    peak_dir.mkdir(parents=True, exist_ok=True)
    for filename in (PEAK_THRESHOLDS_FILENAME, PEAK_CONFIG_FILENAME):
        src = primary_dir / "peak" / filename
        dst = peak_dir / filename
        if src.exists():
            shutil.copy2(src, dst)


def _save_inference_cache(inference_results: dict, path: Path) -> None:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for container_id, result in inference_results.items():
        cache[container_id] = {
            "actual_day1_real": np.ravel(result.actual_day1_real),
            GLOBAL_INFERENCE_CACHE_PRED_KEY: np.ravel(result.day1_pred_real),
        }
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)


def _plot_learning_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], label="Train loss (weighted)", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val loss (unweighted)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{STUDY_LABEL} — Learning Curves (patience={VS1_PATIENCE})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_summary_mean(summary_path: Path) -> dict[str, float]:
    summary = pd.read_csv(summary_path, index_col=0)
    return {
        "day1_mae": float(summary.loc["mean", "day1_mae"]),
        "day1_rmse": float(summary.loc["mean", "day1_rmse"]),
        "day1_mape": float(summary.loc["mean", "day1_mape"]),
    }


def _training_row(metadata: dict, eval_summary: dict[str, float] | None) -> dict:
    row = {
        "epochs_run": metadata.get("epochs_run"),
        "best_epoch": metadata.get("best_epoch"),
        "stopped_epoch": metadata.get("stopped_epoch"),
        "best_val_loss": metadata.get("best_val_loss"),
        "final_train_loss": metadata.get("final_train_loss"),
        "final_val_loss": metadata.get("final_val_loss"),
        "early_stopping_patience": metadata.get("early_stopping_patience"),
    }
    if eval_summary:
        row.update(eval_summary)
    return row


def main() -> None:
    args = _parse_args()
    primary_dir = args.primary_dir.resolve()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT
        / "experiments"
        / f"peak_aware_global_validation_vs1_{run_timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    models_dir = experiment_dir / MODELS_SUBDIR
    peak_dir = experiment_dir / "peak"
    training_dir = experiment_dir / "training"
    evaluation_dir = experiment_dir / EVALUATION_SUBDIR
    plots_dir = experiment_dir / "plots"
    for directory in (models_dir, peak_dir, training_dir, evaluation_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    validation_metadata_path = experiment_dir / "validation_study_metadata.json"
    validation_summary_path = experiment_dir / "validation_study_summary.json"
    training_history_path = training_dir / "training_history.json"
    training_metadata_path = training_dir / TRAINING_METADATA_FILENAME
    model_path = models_dir / PEAK_AWARE_MODEL_FILENAME
    model_metadata_path = models_dir / PEAK_AWARE_METADATA_FILENAME

    print("=" * 72)
    print(STUDY_LABEL)
    print("=" * 72)
    print(f"VS-1 directory      : {experiment_dir}")
    print(f"Primary reference   : {primary_dir}")
    print(f"Only change         : EarlyStopping patience {PRIMARY_PATIENCE} → {VS1_PATIENCE}")
    print()

    validation_metadata_path.write_text(
        json.dumps(
            {
                "study_type": "VS-1-training-dynamics",
                "study_label": STUDY_LABEL,
                "question": (
                    "Was poor performance primarily caused by premature early stopping?"
                ),
                "single_change": {
                    "early_stopping_patience": {
                        "primary": PRIMARY_PATIENCE,
                        "validation": VS1_PATIENCE,
                    }
                },
                "primary_experiment": str(primary_dir.relative_to(REPO_ROOT)),
                "control_reference": str(
                    CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)
                ),
                "started_at": _iso_timestamp(),
                "random_seed": RANDOM_SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _copy_primary_peak_artifacts(primary_dir, peak_dir)

    global_train = _load_global_train()
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    print("Training Peak-Aware Global GRU (patience=10)...")
    t0 = time.perf_counter()
    model, training_metadata, _peak_thresholds, _w_all = train_global_gru_peak_aware(
        global_train=global_train,
        scalers=scalers,
        epochs=EPOCHS_MAX,
        verbose=args.verbose,
        random_seed=RANDOM_SEED,
        early_stopping_patience=VS1_PATIENCE,
    )
    train_seconds = time.perf_counter() - t0

    history = training_metadata.pop("training_history", {})
    training_history_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    training_metadata.update(
        {
            "study_type": "VS-1-training-dynamics",
            "validation_study": True,
            "primary_experiment_reference": str(
                primary_dir.relative_to(REPO_ROOT)
            ),
            "early_stopping_patience_primary": PRIMARY_PATIENCE,
            "training_runtime_seconds": round(train_seconds, 2),
            "trained_at": _iso_timestamp(),
        }
    )
    training_metadata_path.write_text(
        json.dumps(training_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    model.save(str(model_path))
    model_metadata = {
        **training_metadata,
        "saved_at": _iso_timestamp(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "model_path": str(model_path.relative_to(experiment_dir)),
    }
    model_metadata_path.write_text(
        json.dumps(model_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_learning_curves(history, plots_dir / "learning_curves.png")

    print("Running Day-1 evaluation (best EarlyStopping weights)...")
    val_df = pd.read_parquet(DATA_VAL)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    eval_t0 = time.perf_counter()
    evaluation_df, inference_results, failures, skipped = evaluate_selected_containers(
        selected_containers=selected,
        global_train=global_train,
        global_val=global_val,
        global_gru_model=model,
        scalers=scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )
    eval_seconds = time.perf_counter() - eval_t0

    if failures:
        raise RuntimeError(f"VS-1 evaluation failures: {failures}")

    summary = summarize_evaluation_metrics(evaluation_df)
    eval_df_path = evaluation_dir / "evaluation_df.csv"
    eval_summary_path = evaluation_dir / "evaluation_summary.csv"
    evaluation_df.to_csv(eval_df_path, index=False)
    summary.round(6).to_csv(eval_summary_path)

    eval_metadata = {
        "evaluation_type": "vs1_validation_study",
        "study_label": STUDY_LABEL,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "evaluated_containers": int(len(evaluation_df)),
        "skipped_containers": [
            {"container_id": cid, "reason": reason} for cid, reason in skipped
        ],
        "runtime_seconds": round(eval_seconds, 2),
        "completed_at": _iso_timestamp(),
    }
    (evaluation_dir / "evaluation_metadata.json").write_text(
        json.dumps(eval_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_inference_cache(
        inference_results,
        evaluation_dir / "treatment_inference_cache.pkl",
    )

    vs1_eval = _read_summary_mean(eval_summary_path)
    primary_training = json.loads(
        (primary_dir / "training" / TRAINING_METADATA_FILENAME).read_text()
    )
    primary_eval = _read_summary_mean(
        primary_dir / "evaluation" / "evaluation_summary.csv"
    )
    baseline_training = json.loads(
        (
            CONTROL_REFERENCE_DIR / "training" / TRAINING_METADATA_FILENAME
        ).read_text()
    )
    baseline_eval = _read_summary_mean(
        CONTROL_REFERENCE_DIR / "evaluation" / "evaluation_summary.csv"
    )

    validation_summary = {
        "study_label": STUDY_LABEL,
        "completed_at": _iso_timestamp(),
        "primary_experiment": str(primary_dir.relative_to(REPO_ROOT)),
        "vs1_experiment": str(experiment_dir.relative_to(REPO_ROOT)),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "comparison": {
            "frozen_global_baseline": {
                **_training_row(baseline_training, baseline_eval),
                "role": "control_reference",
            },
            "primary_peak_aware_global": {
                **_training_row(primary_training, primary_eval),
                "role": "locked_primary_patience_3",
            },
            "vs1_peak_aware_global": {
                **_training_row(training_metadata, vs1_eval),
                "role": "validation_patience_10",
            },
        },
        "deltas_vs_primary": {
            "epochs_run": (
                (training_metadata.get("epochs_run") or 0)
                - (primary_training.get("epochs_run") or 0)
            ),
            "best_val_loss": (
                (training_metadata.get("best_val_loss") or 0)
                - (primary_training.get("final_val_loss") or 0)
            ),
            "day1_mae": vs1_eval["day1_mae"] - primary_eval["day1_mae"],
            "day1_rmse": vs1_eval["day1_rmse"] - primary_eval["day1_rmse"],
            "day1_mape": vs1_eval["day1_mape"] - primary_eval["day1_mape"],
        },
        "deltas_vs_baseline": {
            "day1_mae": vs1_eval["day1_mae"] - baseline_eval["day1_mae"],
            "day1_rmse": vs1_eval["day1_rmse"] - baseline_eval["day1_rmse"],
            "day1_mape": vs1_eval["day1_mape"] - baseline_eval["day1_mape"],
        },
    }
    validation_summary_path.write_text(
        json.dumps(validation_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    validation_meta = json.loads(validation_metadata_path.read_text())
    validation_meta["completed_at"] = _iso_timestamp()
    validation_meta["outputs"] = {
        "training_history": str(training_history_path.relative_to(REPO_ROOT)),
        "training_metadata": str(training_metadata_path.relative_to(REPO_ROOT)),
        "evaluation_summary": str(eval_summary_path.relative_to(REPO_ROOT)),
        "evaluation_df": str(eval_df_path.relative_to(REPO_ROOT)),
        "learning_curves": str(
            (plots_dir / "learning_curves.png").relative_to(REPO_ROOT)
        ),
        "validation_study_summary": str(
            validation_summary_path.relative_to(REPO_ROOT)
        ),
    }
    validation_metadata_path.write_text(
        json.dumps(validation_meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 72)
    print("VS-1 complete")
    print(f"Epochs run           : {training_metadata.get('epochs_run')}")
    print(f"Best epoch           : {training_metadata.get('best_epoch')}")
    print(f"Stopped epoch        : {training_metadata.get('stopped_epoch')}")
    print(f"Best val loss        : {training_metadata.get('best_val_loss'):.6f}")
    print()
    print("Day-1 evaluation (mean):")
    print(f"  MAE                : {vs1_eval['day1_mae']:.4f}")
    print(f"  RMSE               : {vs1_eval['day1_rmse']:.4f}")
    print(f"  MAPE               : {vs1_eval['day1_mape']:.2f}")
    print()
    print("Primary (patience=3) MAE:", f"{primary_eval['day1_mae']:.4f}")
    print("Baseline Global MAE   :", f"{baseline_eval['day1_mae']:.4f}")
    print()
    print("Saved:")
    print(f"  {training_history_path}")
    print(f"  {training_metadata_path}")
    print(f"  {eval_summary_path}")
    print(f"  {eval_df_path}")
    print(f"  {plots_dir / 'learning_curves.png'}")
    print(f"  {validation_summary_path}")


if __name__ == "__main__":
    main()
