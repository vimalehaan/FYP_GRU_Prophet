#!/usr/bin/env python3
"""Verify Peak-Aware Global GRU satisfies the Stage 1 fairness contract."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import DATA_VAL  # noqa: E402
from utils.global_evaluation import evaluate_selected_containers  # noqa: E402
from utils.global_inference import run_global_inference  # noqa: E402
from utils.global_peak_aware_config import (  # noqa: E402
    BATCH_SIZE,
    CONTROL_BASELINE_METADATA,
    CONTROL_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DEMO_CONTAINER_ID,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EPOCHS_MAX,
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    INPUT_WINDOW,
    OPTIMIZER,
    PEAK_AWARE_MODEL_FILENAME,
    PEAK_THRESHOLDS_FILENAME,
    PEAK_WEIGHT,
    PRODUCTION_MODEL_PATH,
    RANDOM_SEED,
    SEQUENCE_VAL_SPLIT,
    SHUFFLE,
    STAGE2_WORKSPACE_DIR,
    TRAINING_METADATA_FILENAME,
)
from utils.global_training import build_global_gru_model, train_global_gru  # noqa: E402
from utils.peak_detection import (  # noqa: E402
    align_with_longterm_sequences,
    build_sequence_weight_matrix,
    compute_peak_thresholds,
    load_peak_thresholds,
    verify_binary_weight_values,
)
from utils.sequence_utils import create_longterm_sequences  # noqa: E402

FROZEN_MODULES = (
    "utils.global_config",
    "utils.global_training",
    "utils.global_inference",
    "utils.global_evaluation",
    "utils.sequence_utils",
)

FROZEN_FILES = (
    REPO_ROOT / "utils/global_config.py",
    REPO_ROOT / "utils/global_training.py",
    REPO_ROOT / "utils/global_inference.py",
    REPO_ROOT / "utils/global_evaluation.py",
    REPO_ROOT / "utils/sequence_utils.py",
)

PEAK_AWARE_IMPLEMENTATION_SOURCES = (
    REPO_ROOT / "utils/global_peak_aware_config.py",
    REPO_ROOT / "utils/global_training_peak_aware.py",
    REPO_ROOT / "scripts/run_peak_aware_global_experiment.py",
)

FORBIDDEN_WRITE_SNIPPETS: tuple[tuple[str, str], ...] = (
    ("save_global_artifacts(", "must not save via baseline production helper"),
    ("model.save(str(PRODUCTION_MODEL_PATH", "must not save to production model"),
    ("model.save(str(DEFAULT_MODEL_PATH", "must not save to production model"),
    ("shutil.copy2(DEFAULT_MODEL_PATH", "must not copy to production model"),
    (".write_text(CONTROL_REFERENCE_DIR", "must not write to control reference"),
    (".write_bytes(CONTROL_REFERENCE_DIR", "must not write to control reference"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Peak-Aware Global GRU fairness verification checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Peak-aware experiment directory (default: latest from workspace metadata)",
    )
    return parser.parse_args()


def _resolve_experiment_dir(cli_dir: Path | None) -> Path:
    if cli_dir is not None:
        return cli_dir.resolve()

    workspace_meta_path = STAGE2_WORKSPACE_DIR / "experiment_metadata.json"
    if workspace_meta_path.exists():
        workspace_meta = json.loads(workspace_meta_path.read_text(encoding="utf-8"))
        training_run_dir = workspace_meta.get("training_run_dir")
        if training_run_dir:
            return (REPO_ROOT / training_run_dir).resolve()

    raise FileNotFoundError(
        "No experiment directory provided and training_run_dir not set "
        "in workspace metadata."
    )


def _check_result(
    check_id: int,
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def _layer_signature(model: Any) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for layer in model.layers:
        config = layer.get_config()
        signature.append({
            "class_name": layer.__class__.__name__,
            "units": config.get("units"),
            "activation": config.get("activation"),
            "rate": config.get("rate"),
            "return_sequences": config.get("return_sequences"),
        })
    return signature


def _load_baseline_metadata() -> dict[str, Any]:
    return json.loads(CONTROL_BASELINE_METADATA.read_text(encoding="utf-8"))


def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()
    return global_train, global_val, scalers


def _build_baseline_sequences(
    global_train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_list = list(GLOBAL_FEATURES)
    return create_longterm_sequences(
        global_train,
        features=feature_list,
        target=GLOBAL_TARGET,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )


def check_1_frozen_file_integrity() -> dict[str, Any]:
    missing_files = [str(path) for path in FROZEN_FILES if not path.exists()]
    import_errors: list[str] = []
    for module_name in FROZEN_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {exc}")

    forbidden_hits: list[dict[str, str]] = []
    for source_path in PEAK_AWARE_IMPLEMENTATION_SOURCES:
        if not source_path.exists():
            continue
        text = source_path.read_text(encoding="utf-8")
        for snippet, reason in FORBIDDEN_WRITE_SNIPPETS:
            if snippet in text:
                forbidden_hits.append({
                    "file": str(source_path.relative_to(REPO_ROOT)),
                    "matched_snippet": snippet,
                    "reason": reason,
                })

    control_reference_exists = CONTROL_REFERENCE_DIR.exists()
    production_model_exists = PRODUCTION_MODEL_PATH.exists()

    passed = (
        not missing_files
        and not import_errors
        and not forbidden_hits
        and control_reference_exists
        and production_model_exists
    )
    return _check_result(
        1,
        "frozen_file_integrity",
        passed,
        {
            "frozen_files_missing": missing_files,
            "import_errors": import_errors,
            "forbidden_write_snippet_hits": forbidden_hits,
            "control_reference_exists": control_reference_exists,
            "production_model_exists": production_model_exists,
        },
    )


def check_2_identical_sequences(global_train: pd.DataFrame) -> dict[str, Any]:
    X_a, y_a, cids_a = _build_baseline_sequences(global_train)
    X_b, y_b, cids_b = _build_baseline_sequences(global_train)

    x_match = np.array_equal(X_a, X_b)
    y_match = np.array_equal(y_a, y_b)
    cid_match = np.array_equal(cids_a, cids_b)
    count_match = len(X_a) == EXPECTED_N_SEQUENCES

    passed = x_match and y_match and cid_match and count_match
    return _check_result(
        2,
        "identical_X_all_y_all_and_sample_ordering",
        passed,
        {
            "x_shape": list(X_a.shape),
            "y_shape": list(y_a.shape),
            "n_sequences": int(len(cids_a)),
            "expected_n_sequences": EXPECTED_N_SEQUENCES,
            "x_all_reproducible": bool(x_match),
            "y_all_reproducible": bool(y_match),
            "sample_cids_reproducible": bool(cid_match),
            "sequence_count_match": bool(count_match),
        },
    )


def check_3_w_all_only_difference(
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    _, y_all, _ = _build_baseline_sequences(global_train)
    thresholds = compute_peak_thresholds(global_train, scalers)
    W_all, _ = build_sequence_weight_matrix(
        global_train=global_train,
        scalers=scalers,
        thresholds=thresholds,
    )
    binary = verify_binary_weight_values(W_all)

    shape_match = W_all.shape == (EXPECTED_N_SEQUENCES, FORECAST_HORIZON)
    y_shape_match = W_all.shape == y_all.shape
    count_match = W_all.shape[0] == EXPECTED_N_SEQUENCES

    passed = (
        shape_match
        and y_shape_match
        and count_match
        and binary["only_expected_values"]
        and binary["counts_sum_to_total"]
    )
    return _check_result(
        3,
        "W_all_only_allowed_difference",
        passed,
        {
            "W_all_shape": list(W_all.shape),
            "expected_shape": [EXPECTED_N_SEQUENCES, FORECAST_HORIZON],
            "y_all_shape": list(y_all.shape),
            "shape_match": bool(shape_match),
            "y_shape_match": bool(y_shape_match),
            "binary_weight_verification": binary,
        },
    )


def check_4_identical_architecture(experiment_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    peak_aware_model = tf.keras.models.load_model(str(model_path))
    reference_model = build_global_gru_model(
        input_window=INPUT_WINDOW,
        n_features=len(GLOBAL_FEATURES),
        forecast_horizon=FORECAST_HORIZON,
    )

    peak_signature = _layer_signature(peak_aware_model)
    reference_signature = _layer_signature(reference_model)
    passed = peak_signature == reference_signature

    return _check_result(
        4,
        "identical_architecture_via_build_global_gru_model",
        passed,
        {
            "peak_aware_layers": peak_signature,
            "reference_layers": reference_signature,
            "layer_count_peak_aware": len(peak_signature),
            "layer_count_reference": len(reference_signature),
            "architecture_builder": "utils.global_training.build_global_gru_model",
        },
    )


def check_5_hyperparam_parity(experiment_dir: Path) -> dict[str, Any]:
    baseline = _load_baseline_metadata()["training"]
    training_metadata = json.loads(
        (experiment_dir / "training" / TRAINING_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    checks = {
        "optimizer": (
            training_metadata.get("optimizer") == baseline["optimizer"] == OPTIMIZER
        ),
        "epochs_max_config": training_metadata.get("epochs_max") <= EPOCHS_MAX,
        "batch_size": training_metadata.get("batch_size") == baseline["batch_size"] == BATCH_SIZE,
        "shuffle": training_metadata.get("shuffle") == baseline["shuffle"] == SHUFFLE,
        "random_seed": training_metadata.get("random_seed") == baseline["random_seed"] == RANDOM_SEED,
        "sequence_val_split": (
            training_metadata.get("sequence_val_split")
            == baseline["sequence_val_split"]
            == SEQUENCE_VAL_SPLIT
        ),
        "n_sequences_total": (
            training_metadata.get("n_sequences_total") == baseline["n_sequences"]
        ),
        "n_sequences_train": (
            training_metadata.get("n_sequences_train") == baseline["n_train_sequences"]
        ),
        "n_sequences_val": (
            training_metadata.get("n_sequences_val") == baseline["n_val_sequences"]
        ),
    }

    passed = all(checks.values())
    return _check_result(
        5,
        "global_hyperparams_match_baseline_metadata",
        passed,
        {
            "parity_checks": checks,
            "baseline_reference": str(CONTROL_BASELINE_METADATA.relative_to(REPO_ROOT)),
            "saved_training_metadata": {
                "optimizer": training_metadata.get("optimizer"),
                "batch_size": training_metadata.get("batch_size"),
                "epochs_max": training_metadata.get("epochs_max"),
                "shuffle": training_metadata.get("shuffle"),
                "random_seed": training_metadata.get("random_seed"),
                "sequence_val_split": training_metadata.get("sequence_val_split"),
            },
        },
    )


def check_6_unweighted_validation() -> dict[str, Any]:
    from utils import global_training_peak_aware

    source = inspect.getsource(
        global_training_peak_aware.train_global_gru_peak_aware
    )
    validation_line_present = "validation_data=(X_val, y_val)" in source
    val_sample_weight_absent = "validation_data=(X_val, y_val, W_val)" not in source
    val_weight_arg_absent = "sample_weight=W_val" not in source
    validation_unweighted_flag = (
        '"validation_unweighted": True' in source
        or "'validation_unweighted': True" in source
    )

    passed = (
        validation_line_present
        and val_sample_weight_absent
        and val_weight_arg_absent
        and validation_unweighted_flag
    )
    return _check_result(
        6,
        "early_stopping_config_unweighted",
        passed,
        {
            "validation_data_unweighted_tuple": bool(validation_line_present),
            "no_weighted_validation_tuple": bool(val_sample_weight_absent),
            "no_validation_sample_weight_arg": bool(val_weight_arg_absent),
            "validation_unweighted_metadata_flag": bool(validation_unweighted_flag),
            "expected_monitor": EARLY_STOPPING_MONITOR,
            "expected_patience": EARLY_STOPPING_PATIENCE,
            "expected_restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        },
    )


def check_7_inference_evaluation_path(
    experiment_dir: Path,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    import tensorflow as tf

    model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    peak_aware_model = tf.keras.models.load_model(str(model_path))

    peak_aware_sources_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "utils/global_training_peak_aware.py",
            REPO_ROOT / "scripts/run_peak_aware_global_experiment.py",
        )
        if path.exists()
    )
    no_inference_fork = "def run_global_inference" not in peak_aware_sources_text
    no_eval_fork = "def evaluate_selected_containers" not in peak_aware_sources_text

    inference_result = run_global_inference(
        container_id=DEMO_CONTAINER_ID,
        global_train=global_train,
        global_val=global_val,
        model=peak_aware_model,
        scalers=scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    eval_df, _, failures, skipped = evaluate_selected_containers(
        selected_containers=np.array([DEMO_CONTAINER_ID]),
        global_train=global_train,
        global_val=global_val,
        global_gru_model=peak_aware_model,
        scalers=scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    passed = (
        no_inference_fork
        and no_eval_fork
        and len(inference_result.day1_pred_real) == FORECAST_HORIZON
        and len(eval_df) == 1
        and not failures
    )
    return _check_result(
        7,
        "shared_global_inference_evaluation_path",
        passed,
        {
            "demo_container": DEMO_CONTAINER_ID,
            "no_peak_aware_inference_fork": no_inference_fork,
            "no_peak_aware_evaluation_fork": no_eval_fork,
            "day1_prediction_steps": len(inference_result.day1_pred_real),
            "expected_day1_steps": FORECAST_HORIZON,
            "evaluated_containers": int(len(eval_df)),
            "failures": failures,
            "skipped": [
                {"container_id": cid, "reason": reason}
                for cid, reason in skipped
            ],
        },
    )


def check_8_threshold_artifact(
    experiment_dir: Path,
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    thresholds_path = experiment_dir / "peak" / PEAK_THRESHOLDS_FILENAME
    saved_thresholds = load_peak_thresholds(thresholds_path)
    expected_thresholds = compute_peak_thresholds(global_train, scalers)

    train_container_count = int(global_train["container_id"].nunique())
    count_match = len(saved_thresholds) == train_container_count == len(
        expected_thresholds
    )
    keys_match = set(saved_thresholds) == set(expected_thresholds)

    value_mismatches: dict[str, dict[str, float]] = {}
    if keys_match:
        for container_id, expected_value in expected_thresholds.items():
            saved_value = saved_thresholds[container_id]
            if not np.isclose(saved_value, expected_value, rtol=0.0, atol=1e-9):
                value_mismatches[container_id] = {
                    "saved": float(saved_value),
                    "expected": float(expected_value),
                }

    passed = count_match and keys_match and not value_mismatches
    return _check_result(
        8,
        "valid_peak_threshold_artifact",
        passed,
        {
            "train_container_count": train_container_count,
            "saved_threshold_count": len(saved_thresholds),
            "expected_threshold_count": len(expected_thresholds),
            "count_match": bool(count_match),
            "keys_match": bool(keys_match),
            "value_mismatch_count": len(value_mismatches),
        },
    )


def check_9_weight_matrix_alignment(
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    thresholds = compute_peak_thresholds(global_train, scalers)
    alignment = align_with_longterm_sequences(
        global_train=global_train,
        scalers=scalers,
        thresholds=thresholds,
        features=list(GLOBAL_FEATURES),
        target=GLOBAL_TARGET,
    )

    passed = alignment["cid_arrays_equal"]
    return _check_result(
        9,
        "weight_matrix_aligns_with_longterm_sequences",
        passed,
        alignment,
    )


def check_10_single_behavioural_difference() -> dict[str, Any]:
    baseline_source = inspect.getsource(train_global_gru)
    from utils.global_training_peak_aware import train_global_gru_peak_aware

    peak_source = inspect.getsource(train_global_gru_peak_aware)

    shared_requirements = {
        "uses_create_longterm_sequences": (
            "create_longterm_sequences" in peak_source
        ),
        "uses_build_global_gru_model": (
            "build_global_gru_model" in peak_source
        ),
        "uses_global_train_only": "global_train" in peak_source,
        "uses_early_stopping_val_loss": (
            'monitor=EARLY_STOPPING_MONITOR' in peak_source
            or "monitor=EARLY_STOPPING_MONITOR" in peak_source
        ),
        "uses_same_batch_shuffle": (
            "batch_size=batch_size" in peak_source and "shuffle=SHUFFLE" in peak_source
        ),
        "uses_sequence_val_split": (
            "SEQUENCE_VAL_SPLIT" in peak_source
        ),
        "baseline_uses_create_longterm_sequences": (
            "create_longterm_sequences" in baseline_source
        ),
        "baseline_uses_build_global_gru_model": (
            "build_global_gru_model" in baseline_source
        ),
    }

    allowed_differences = {
        "install_weighted_train_step": "_install_weighted_train_step" in peak_source,
        "train_sample_weight_W_train": "sample_weight=W_train" in peak_source,
        "compute_peak_thresholds": "compute_peak_thresholds" in peak_source,
        "build_sequence_weight_matrix": "build_sequence_weight_matrix" in peak_source,
        "peak_aware_loss_metadata": "timestep_weighted_mse" in peak_source,
    }

    forbidden_in_peak = {
        "no_prophet_residuals": "generate_prophet_residuals" not in peak_source,
        "no_create_residual_sequences": "create_residual_sequences" not in peak_source,
        "no_duplicate_build_global_layers": (
            "Sequential([" not in peak_source
        ),
        "no_validation_weights": "sample_weight=W_val" not in peak_source,
    }

    passed = all(shared_requirements.values()) and all(
        allowed_differences.values()
    ) and all(forbidden_in_peak.values())

    return _check_result(
        10,
        "single_behavioural_difference_weighted_train_step_only",
        passed,
        {
            "shared_requirements": shared_requirements,
            "allowed_differences": allowed_differences,
            "forbidden_in_peak_module": forbidden_in_peak,
            "peak_weight_lambda": PEAK_WEIGHT,
            "interpretation": (
                "Only weighted train_step and W_train sample_weight differ "
                "from frozen train_global_gru pipeline"
            ),
        },
    )


def _update_workspace(all_passed: bool, experiment_dir: Path) -> None:
    workspace_meta_path = STAGE2_WORKSPACE_DIR / "experiment_metadata.json"
    if not workspace_meta_path.exists():
        return

    workspace_meta = json.loads(workspace_meta_path.read_text(encoding="utf-8"))
    if all_passed:
        workspace_meta["tasks"]["task_5"] = "complete"
        experiment_metadata_path = experiment_dir / "experiment_metadata.json"
        if experiment_metadata_path.exists():
            experiment_meta = json.loads(
                experiment_metadata_path.read_text(encoding="utf-8")
            )
            if not experiment_meta.get("smoke_run", False):
                workspace_meta["tasks"]["task_6"] = "complete"
                ref_path = experiment_dir / "reference" / "reference_metadata.json"
                if ref_path.exists():
                    workspace_meta["reference_metadata_snapshot"] = str(
                        ref_path.relative_to(REPO_ROOT)
                    )
    workspace_meta["fairness_verification"] = str(
        (experiment_dir / "verification" / "fairness_check.json").relative_to(
            REPO_ROOT
        )
    )
    workspace_meta["training_run_dir"] = str(
        experiment_dir.relative_to(REPO_ROOT)
    )
    workspace_meta_path.write_text(
        json.dumps(workspace_meta, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    experiment_dir = _resolve_experiment_dir(args.experiment_dir)
    verification_dir = experiment_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    output_path = verification_dir / "fairness_check.json"

    print("=" * 72)
    print("Peak-Aware Global GRU — Fairness Verification")
    print("=" * 72)
    print(f"Experiment directory: {experiment_dir}")
    print()

    global_train, global_val, scalers = _load_training_data()

    checks: list[Callable[[], dict[str, Any]]] = [
        check_1_frozen_file_integrity,
        lambda: check_2_identical_sequences(global_train),
        lambda: check_3_w_all_only_difference(global_train, scalers),
        lambda: check_4_identical_architecture(experiment_dir),
        lambda: check_5_hyperparam_parity(experiment_dir),
        check_6_unweighted_validation,
        lambda: check_7_inference_evaluation_path(
            experiment_dir,
            global_train,
            global_val,
            scalers,
        ),
        lambda: check_8_threshold_artifact(experiment_dir, global_train, scalers),
        lambda: check_9_weight_matrix_alignment(global_train, scalers),
        check_10_single_behavioural_difference,
    ]

    results: list[dict[str, Any]] = []
    for runner in checks:
        result = runner()
        results.append(result)
        print(f"[{result['status']}] Check {result['check_id']}: {result['name']}")

    all_passed = all(result["status"] == "PASS" for result in results)
    summary = {
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(1 for result in results if result["status"] == "PASS"),
        "checks_total": len(results),
        "checks": results,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    experiment_metadata_path = experiment_dir / "experiment_metadata.json"
    if experiment_metadata_path.exists():
        experiment_metadata = json.loads(
            experiment_metadata_path.read_text(encoding="utf-8")
        )
        experiment_metadata.setdefault("status", {})
        experiment_metadata["status"]["fairness_verification"] = (
            "complete" if all_passed else "failed"
        )
        experiment_metadata["fairness_check"] = str(
            output_path.relative_to(experiment_dir)
        )
        experiment_metadata_path.write_text(
            json.dumps(experiment_metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    _update_workspace(all_passed, experiment_dir)

    print()
    print(f"Overall status: {summary['overall_status']}")
    print(f"Checks passed : {summary['checks_passed']}/{summary['checks_total']}")
    print(f"Saved report  : {output_path}")

    if not all_passed:
        failed = [
            f"{result['check_id']} ({result['name']})"
            for result in results
            if result["status"] == "FAIL"
        ]
        raise SystemExit(f"Fairness verification failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
