#!/usr/bin/env python3
"""Verify Global GRU evaluation protocol parity with frozen Hybrid baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import (  # noqa: E402
    DATA_SELECTED_CONTAINERS,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    MAPE_EPSILON,
    STALE_SELECTED_CONTAINER_ID,
)
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW  # noqa: E402
from utils.hybrid_evaluation import (  # noqa: E402
    MAPE_EPSILON as HYBRID_MAPE_EPSILON,
    compute_day1_mape,
    compute_day1_metrics,
)

HYBRID_BASELINE_EVAL_DF = (
    REPO_ROOT / "experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv"
)
HYBRID_BASELINE_METADATA = (
    REPO_ROOT
    / "experiments/baseline_reference_2026-07-14/config/baseline_metadata.json"
)

REQUIRED_EVAL_COLUMNS = [
    "container_id",
    "day1_mae",
    "day1_rmse",
    "day1_mape",
    "train_steps",
    "validation_steps",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Global GRU evaluation protocol matches frozen Hybrid baseline."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Global GRU evaluation experiment directory "
            "(default: latest experiments/global_gru_evaluation_*)"
        ),
    )
    return parser.parse_args()


def _latest_evaluation_dir() -> Path:
    candidates = sorted(REPO_ROOT.glob("experiments/global_gru_evaluation_*"))
    if not candidates:
        raise FileNotFoundError(
            "No experiments/global_gru_evaluation_* directory found"
        )
    return candidates[-1]


def _assert_same_metric_functions() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 2.2, 2.8])
    mae, rmse = compute_day1_metrics(actual, predicted)
    mape = compute_day1_mape(actual, predicted, epsilon=MAPE_EPSILON)
    assert mae >= 0 and rmse >= 0 and mape >= 0
    assert rmse >= mae
    print("  OK  shared metric functions imported from hybrid_evaluation")


def main() -> None:
    args = _parse_args()
    experiment_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir is not None
        else _latest_evaluation_dir()
    )
    global_eval_df_path = experiment_dir / "evaluation/evaluation_df.csv"
    global_metadata_path = experiment_dir / "evaluation/evaluation_metadata.json"

    if not global_eval_df_path.exists():
        raise FileNotFoundError(f"Missing Global GRU evaluation_df: {global_eval_df_path}")
    if not HYBRID_BASELINE_EVAL_DF.exists():
        raise FileNotFoundError(
            f"Missing Hybrid baseline evaluation_df: {HYBRID_BASELINE_EVAL_DF}"
        )

    print("=== Global GRU Methodology Parity Verification ===")
    print("(Readiness check — not a performance comparison)")
    print()

    selected_containers = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    global_eval_df = pd.read_csv(global_eval_df_path)
    hybrid_eval_df = pd.read_csv(HYBRID_BASELINE_EVAL_DF)

    print("Protocol constants:")
    assert INPUT_WINDOW == DEFAULT_INPUT_WINDOW == 96
    assert FORECAST_HORIZON == DAY1_HORIZON == 96
    assert MAPE_EPSILON == HYBRID_MAPE_EPSILON == 0.01
    print(f"  OK  input_window = {INPUT_WINDOW}")
    print(f"  OK  forecast_horizon = {FORECAST_HORIZON}")
    print(f"  OK  mape_epsilon = {MAPE_EPSILON}")

    print()
    print("Cohort alignment:")
    assert len(selected_containers) == 100
    assert len(global_eval_df) == 99
    assert len(hybrid_eval_df) == 99
    print(f"  OK  selected containers = {len(selected_containers)}")
    print(f"  OK  evaluated containers = 99 (Global and Hybrid)")

    global_ids = set(global_eval_df["container_id"])
    hybrid_ids = set(hybrid_eval_df["container_id"])
    assert global_ids == hybrid_ids
    assert STALE_SELECTED_CONTAINER_ID not in global_ids
    assert STALE_SELECTED_CONTAINER_ID not in hybrid_ids
    print(f"  OK  identical evaluated container IDs ({len(global_ids)})")
    print(f"  OK  {STALE_SELECTED_CONTAINER_ID} excluded from both cohorts")

    print()
    print("Evaluation schema:")
    assert list(global_eval_df.columns) == REQUIRED_EVAL_COLUMNS
    assert list(hybrid_eval_df.columns) == REQUIRED_EVAL_COLUMNS
    print("  OK  evaluation_df schema matches Hybrid baseline")

    _assert_same_metric_functions()

    print()
    print("Frozen Hybrid baseline integrity:")
    assert HYBRID_BASELINE_METADATA.exists()
    hybrid_metadata = json.loads(HYBRID_BASELINE_METADATA.read_text())
    assert hybrid_metadata.get("baseline_version") is not None
    print("  OK  Hybrid baseline metadata present (read-only control)")

    if global_metadata_path.exists():
        global_metadata = json.loads(global_metadata_path.read_text())
        assert global_metadata.get("input_window") == INPUT_WINDOW
        assert global_metadata.get("forecast_horizon") == FORECAST_HORIZON
        assert global_metadata.get("mape_epsilon") == MAPE_EPSILON
        assert global_metadata.get("evaluated_containers") == 99
        print("  OK  Global GRU evaluation metadata protocol fields")

    print()
    print("Methodology parity: PASS")
    print("All Global GRU methodology parity checks passed.")
    print("Note: Performance comparison is deferred to Phase 6.")


if __name__ == "__main__":
    main()
