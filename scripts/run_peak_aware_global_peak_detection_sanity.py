#!/usr/bin/env python3
"""Run Peak-Aware Global peak-detection sanity checks (Stage 2 Task 2)."""

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

from utils.global_peak_aware_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    INPUT_WINDOW,
    PEAK_PERCENTILE,
    SEQUENCE_VAL_SPLIT,
    STAGE2_WORKSPACE_DIR,
)
from utils.peak_detection import (  # noqa: E402
    align_with_longterm_sequences,
    assert_binary_weight_values,
    assert_expected_sequence_counts,
    build_sequence_weight_matrix,
    compute_peak_thresholds,
    summarize_weight_matrix,
    verify_binary_weight_values,
)
from utils.sequence_utils import create_longterm_sequences  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak-Aware Global peak detection sanity checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE2_WORKSPACE_DIR / "peak_detection_sanity.json",
        help="Path for sanity check JSON output",
    )
    return parser.parse_args()


def _load_global_train() -> pd.DataFrame:
    """Load the same ``global_train`` cohort used by frozen Global GRU training."""
    train_df = pd.read_parquet(DATA_TRAIN)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    return train_df[train_df["container_id"].isin(selected)].copy()


def run_sanity_checks() -> dict:
    """Compute thresholds and weight matrix; verify Global sequence alignment."""
    global_train = _load_global_train()
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    n_containers_in_train = int(global_train["container_id"].nunique())
    thresholds = compute_peak_thresholds(global_train, scalers)
    n_thresholds = len(thresholds)

    feature_list = list(GLOBAL_FEATURES)
    _, y_all, _ = create_longterm_sequences(
        global_train,
        features=feature_list,
        target=GLOBAL_TARGET,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )
    W_all, _ = build_sequence_weight_matrix(
        global_train=global_train,
        scalers=scalers,
        thresholds=thresholds,
    )

    assert_expected_sequence_counts(len(y_all), W_all)
    assert_binary_weight_values(W_all)

    split_idx = int(len(y_all) * SEQUENCE_VAL_SPLIT)
    cid_alignment = align_with_longterm_sequences(
        global_train=global_train,
        scalers=scalers,
        thresholds=thresholds,
        features=feature_list,
        target=GLOBAL_TARGET,
    )

    if not cid_alignment["cid_arrays_equal"]:
        raise AssertionError(
            "Weight matrix container order does not match "
            "create_longterm_sequences()"
        )

    weight_summary = summarize_weight_matrix(W_all)
    binary_verification = verify_binary_weight_values(W_all)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "task": "task_2_peak_detection_sanity",
        "sequence_builder": "create_longterm_sequences",
        "peak_percentile": PEAK_PERCENTILE,
        "n_thresholds": n_thresholds,
        "n_containers_in_train": n_containers_in_train,
        "thresholds_match_containers": n_thresholds == n_containers_in_train,
        "W_all_shape": list(W_all.shape),
        "y_all_shape": list(y_all.shape),
        "shapes_match": W_all.shape == y_all.shape,
        "n_sequences": int(len(y_all)),
        "expected_n_sequences": EXPECTED_N_SEQUENCES,
        "split_idx": split_idx,
        "n_train": split_idx,
        "expected_n_train": EXPECTED_N_TRAIN,
        "n_val": int(len(y_all) - split_idx),
        "expected_n_val": EXPECTED_N_VAL,
        "cid_alignment": cid_alignment,
        "weight_summary": weight_summary,
        "phase2_reference_peak_timestep_pct": 17.36,
        "binary_weight_verification": binary_verification,
        "status": "PASS",
    }


def main() -> None:
    args = _parse_args()
    result = run_sanity_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Peak detection sanity: PASS")
    print(f"  thresholds: {result['n_thresholds']}")
    print(f"  W_all shape: {result['W_all_shape']}")
    print(f"  peak weight fraction: {result['weight_summary']['peak_weight_fraction']:.4f}")
    print(f"  cid alignment: {result['cid_alignment']['cid_arrays_equal']}")
    print(f"  saved: {args.output}")


if __name__ == "__main__":
    main()
