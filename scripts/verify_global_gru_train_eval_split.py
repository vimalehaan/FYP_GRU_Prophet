#!/usr/bin/env python3
"""Verify saved Global GRU artifacts produce identical predictions to in-memory model."""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_artifacts import (  # noqa: E402
    load_global_artifacts,
    save_global_artifacts,
)
from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEMO_CONTAINER_ID,
    FORECAST_HORIZON,
    INPUT_WINDOW,
)
from utils.global_evaluation import evaluate_selected_containers  # noqa: E402
from utils.global_inference import run_global_inference  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify loaded Global GRU model matches in-memory production model."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained Global GRU model",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to Global GRU metadata JSON",
    )
    return parser.parse_args()


def _load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()
    return global_train, global_val, selected, scalers


def _compare_inference(left, right, label: str) -> None:
    arrays = [
        ("day1_pred_scaled", left.day1_pred_scaled, right.day1_pred_scaled),
        ("day1_pred_real", left.day1_pred_real, right.day1_pred_real),
        ("actual_day1_real", left.actual_day1_real, right.actual_day1_real),
    ]
    for name, a, b in arrays:
        if a.shape != b.shape:
            raise AssertionError(
                f"{label} {name}: shape {a.shape} vs {b.shape}"
            )
        if not np.allclose(a, b, rtol=0.0, atol=0.0):
            diff = float(np.max(np.abs(a - b)))
            raise AssertionError(f"{label} {name}: max diff {diff}")
    print(f"  OK  {label}")


def main() -> None:
    args = _parse_args()
    model_path = args.model_path.resolve()
    metadata_path = args.metadata_path.resolve()

    print("=== Global GRU Train/Eval Split Verification ===")
    print(f"Model source: {model_path}")
    print()

    global_train, global_val, selected, scalers = _load_cohort()

    print("Loading production Global GRU model...")
    production_model, production_metadata = load_global_artifacts(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_model_path = Path(tmp) / "global_gru.keras"
        tmp_metadata_path = Path(tmp) / "global_gru_metadata.json"
        save_global_artifacts(
            production_model,
            production_metadata,
            model_path=tmp_model_path,
            metadata_path=tmp_metadata_path,
        )
        loaded_model, loaded_metadata = load_global_artifacts(
            model_path=tmp_model_path,
            metadata_path=tmp_metadata_path,
        )

    assert loaded_metadata.get("model_variant") == production_metadata.get(
        "model_variant"
    )
    print("  OK  metadata round-trip")

    print("Comparing demo container inference (production vs loaded)...")
    production_result = run_global_inference(
        DEMO_CONTAINER_ID,
        global_train,
        global_val,
        production_model,
        scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )
    loaded_result = run_global_inference(
        DEMO_CONTAINER_ID,
        global_train,
        global_val,
        loaded_model,
        scalers,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )
    _compare_inference(
        production_result,
        loaded_result,
        DEMO_CONTAINER_ID,
    )

    print("Comparing full multi-container evaluation (production vs loaded)...")
    production_eval, _, production_failures, production_skipped = (
        evaluate_selected_containers(
            selected,
            global_train,
            global_val,
            production_model,
            scalers,
            input_window=INPUT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
    )
    loaded_eval, _, loaded_failures, loaded_skipped = (
        evaluate_selected_containers(
            selected,
            global_train,
            global_val,
            loaded_model,
            scalers,
            input_window=INPUT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
    )

    assert len(production_failures) == 0 == len(loaded_failures)
    assert production_skipped == loaded_skipped
    assert len(production_eval) == len(loaded_eval)

    merged = production_eval.merge(
        loaded_eval,
        on="container_id",
        suffixes=("_production", "_loaded"),
    )
    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        if not np.allclose(
            merged[f"{metric}_production"],
            merged[f"{metric}_loaded"],
            rtol=0.0,
            atol=0.0,
        ):
            diff = float(
                np.max(
                    np.abs(
                        merged[f"{metric}_production"]
                        - merged[f"{metric}_loaded"]
                    )
                )
            )
            raise AssertionError(f"{metric} mismatch max diff {diff}")

    print(f"  OK  {len(merged)} containers — evaluation metrics identical")
    print()
    print("All Global GRU train/eval split checks passed.")


if __name__ == "__main__":
    main()
