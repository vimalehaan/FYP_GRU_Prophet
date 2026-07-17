#!/usr/bin/env python3
"""Verify Global GRU training sequences have no validation-period target leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import (  # noqa: E402
    EXPECTED_N_SEQUENCES,
    EXPECTED_N_TRAIN,
    EXPECTED_N_VAL,
    FORECAST_HORIZON,
    GLOBAL_FEATURES,
    GLOBAL_TARGET,
    INPUT_WINDOW,
    SEQUENCE_VAL_SPLIT,
)
from utils.sequence_utils import create_longterm_sequences  # noqa: E402

NOTEBOOK_PATH = REPO_ROOT / "notebooks/global_gru_model.ipynb"


def _load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load frozen preprocessing artifacts and build train/val cohort slices."""
    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(
        REPO_ROOT / "data/selected_containers.npy",
        allow_pickle=True,
    )

    global_train = train_df[
        train_df["container_id"].isin(selected)
    ].copy()
    global_val = val_df[
        val_df["container_id"].isin(selected)
    ].copy()
    return global_train, global_val, selected


def _train_period_end_by_container(
    global_train: pd.DataFrame,
) -> dict[str, pd.Timestamp]:
    """Return the last train-period timestamp per container."""
    return (
        global_train.groupby("container_id")["time_stamp"]
        .max()
        .to_dict()
    )


def count_sequences_with_val_period_targets(
    df: pd.DataFrame,
    train_period_end: dict[str, pd.Timestamp],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> int:
    """
    Count sequences whose target window includes any validation-period timestamp.

    A sequence leaks when any target timestep is strictly after the container's
    train-period end (preprocessing val period).
    """
    leak_count = 0

    for container_id, group in df.groupby("container_id"):
        group = group.sort_values("time_stamp")
        train_end = train_period_end.get(container_id)
        if train_end is None:
            continue

        timestamps = group["time_stamp"].values
        max_start = len(group) - input_window - forecast_horizon

        for start_idx in range(max_start):
            target_times = timestamps[
                start_idx + input_window : start_idx + input_window + forecast_horizon
            ]
            if np.any(target_times > train_end):
                leak_count += 1

    return leak_count


def count_val_leaks_in_sequence_prefix(
    df: pd.DataFrame,
    train_period_end: dict[str, pd.Timestamp],
    max_sequences: int,
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> int:
    """Count val-period target leaks within the first ``max_sequences`` sequences."""
    leak_count = 0
    seq_idx = 0

    for container_id, group in df.groupby("container_id"):
        group = group.sort_values("time_stamp")
        train_end = train_period_end.get(container_id)
        if train_end is None:
            continue

        timestamps = group["time_stamp"].values
        max_start = len(group) - input_window - forecast_horizon

        for start_idx in range(max_start):
            if seq_idx >= max_sequences:
                return leak_count

            target_times = timestamps[
                start_idx + input_window : start_idx + input_window + forecast_horizon
            ]
            if np.any(target_times > train_end):
                leak_count += 1
            seq_idx += 1

    return leak_count


def _assert_notebook_has_no_training_concat() -> None:
    """Ensure the refactored notebook does not rebuild leaky train+val sequences."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    leaky_patterns = (
        'pd.concat([global_train, global_val])',
        "pd.concat([global_train, global_val],",
    )
    for pattern in leaky_patterns:
        if pattern in notebook_text:
            raise AssertionError(
                f"Leaky training concat found in {NOTEBOOK_PATH}: {pattern}"
            )


def main() -> None:
    print("Global GRU data-split verification")
    print("=" * 40)

    global_train, global_val, _ = _load_cohort()
    train_period_end = _train_period_end_by_container(global_train)
    feature_list = list(GLOBAL_FEATURES)

    print("Building sequences from global_train only...")
    X_train_only, y_train_only, _ = create_longterm_sequences(
        global_train,
        features=feature_list,
        target=GLOBAL_TARGET,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    print("Building legacy leaky sequences (concat train+val) for contrast...")
    leaky_df = pd.concat([global_train, global_val], ignore_index=True)
    X_leaky, y_leaky, _ = create_longterm_sequences(
        leaky_df,
        features=feature_list,
        target=GLOBAL_TARGET,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    split_idx = int(len(X_train_only) * SEQUENCE_VAL_SPLIT)
    y_train_split = y_train_only[:split_idx]
    y_val_split = y_train_only[split_idx:]

    train_only_leaks = count_sequences_with_val_period_targets(
        global_train,
        train_period_end,
    )
    leaky_total_leaks = count_sequences_with_val_period_targets(
        leaky_df,
        train_period_end,
    )
    leaky_in_train_split = count_val_leaks_in_sequence_prefix(
        global_train,
        train_period_end,
        max_sequences=split_idx,
    )

    print(f"  Sequences (global_train only) : {len(X_train_only)}")
    print(f"  Sequences (legacy concat)     : {len(X_leaky)}")
    print(f"  Val-period target leaks (train only) : {train_only_leaks}")
    print(f"  Val-period target leaks (legacy concat): {leaky_total_leaks}")
    print(f"  Val-period target leaks (train 80% split): {leaky_in_train_split}")

    assert len(X_train_only) == EXPECTED_N_SEQUENCES, (
        f"Expected {EXPECTED_N_SEQUENCES} sequences from global_train, "
        f"got {len(X_train_only)}"
    )
    assert split_idx == EXPECTED_N_TRAIN
    assert len(X_train_only) - split_idx == EXPECTED_N_VAL
    assert len(y_train_split) == EXPECTED_N_TRAIN
    assert len(y_val_split) == EXPECTED_N_VAL

    assert train_only_leaks == 0, (
        "global_train sequences must not target preprocessing validation period"
    )
    assert leaky_in_train_split == 0, (
        "Early-stopping train split must not include validation-period targets"
    )
    assert leaky_total_leaks > 0, (
        "Legacy concat path should demonstrate validation-period target leakage"
    )
    assert len(X_leaky) > len(X_train_only), (
        "Concatenating global_val must produce more sequences than global_train only"
    )

    _assert_notebook_has_no_training_concat()

    print("  OK  Sequence source is global_train only")
    print("  OK  Zero validation-period targets in training sequences")
    print("  OK  Legacy concat path shows expected leakage (contrast check)")
    print("  OK  Notebook has no pd.concat([global_train, global_val])")
    print("\nAll Global GRU data-split checks passed.")


if __name__ == "__main__":
    main()
