#!/usr/bin/env python3
"""Verify Hybrid vs Global GRU methodology comparison output integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import STALE_SELECTED_CONTAINER_ID  # noqa: E402
from utils.methodology_comparison import (  # noqa: E402
    DELTA_SIGN_CONVENTION,
    EXPECTED_EVALUATED,
    GLOBAL_GRU_REFERENCE_DIR,
    HYBRID_BASELINE_REFERENCE_DIR,
    METRIC_COLUMNS,
    METHODOLOGY_GLOBAL,
    METHODOLOGY_HYBRID,
    assert_cohort_alignment,
    build_comparison_table,
    build_per_container_delta,
    load_frozen_evaluation_dfs,
    load_frozen_evaluation_summaries,
    summarize_deltas,
)

TOLERANCE = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Hybrid vs Global GRU comparison artifacts.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Comparison experiment directory "
            "(default: latest experiments/hybrid_vs_global_*)"
        ),
    )
    return parser.parse_args()


def _latest_comparison_dir() -> Path:
    candidates = sorted(REPO_ROOT.glob("experiments/hybrid_vs_global_*"))
    if not candidates:
        raise FileNotFoundError(
            "No experiments/hybrid_vs_global_* directory found"
        )
    return candidates[-1]


def _check_cohort_ids(hybrid_df: pd.DataFrame, global_df: pd.DataFrame) -> None:
    print("Cohort alignment:")
    assert len(hybrid_df) == EXPECTED_EVALUATED
    assert len(global_df) == EXPECTED_EVALUATED
    print(f"  OK  both datasets contain {EXPECTED_EVALUATED} containers")

    hybrid_ids = set(hybrid_df["container_id"])
    global_ids = set(global_df["container_id"])
    assert hybrid_ids == global_ids
    print(f"  OK  identical container IDs ({len(hybrid_ids)})")

    assert STALE_SELECTED_CONTAINER_ID not in hybrid_ids
    assert STALE_SELECTED_CONTAINER_ID not in global_ids
    print(f"  OK  {STALE_SELECTED_CONTAINER_ID} absent from both datasets")


def _check_merge_integrity(per_container_delta: pd.DataFrame) -> None:
    print("Merge integrity:")
    assert len(per_container_delta) == EXPECTED_EVALUATED
    assert not per_container_delta["container_id"].duplicated().any()
    print("  OK  99 delta rows with no duplicate container_id")


def _check_delta_convention(per_container_delta: pd.DataFrame) -> None:
    print("Delta convention (Global − Hybrid):")
    assert DELTA_SIGN_CONVENTION == "global_minus_hybrid"
    for metric in METRIC_COLUMNS:
        hybrid_col = f"hybrid_{metric}"
        global_col = f"global_{metric}"
        delta_col = f"delta_{metric}"
        recomputed = per_container_delta[global_col] - per_container_delta[hybrid_col]
        if not np.allclose(per_container_delta[delta_col], recomputed, atol=1e-12):
            diff = float(
                np.max(np.abs(per_container_delta[delta_col] - recomputed)),
            )
            raise AssertionError(f"{delta_col} mismatch (max diff = {diff})")
        print(f"  OK  {delta_col}")


def _check_no_nan_outputs(
    comparison_table: pd.DataFrame,
    per_container_delta: pd.DataFrame,
) -> None:
    print("Missing values:")
    assert not comparison_table.isna().any().any()
    delta_metric_cols = [
        col
        for col in per_container_delta.columns
        if col.startswith("delta_") or col.startswith("hybrid_") or col.startswith("global_")
    ]
    assert not per_container_delta[delta_metric_cols].isna().any().any()
    print("  OK  no NaN in comparison outputs")


def _check_aggregate_match(comparison_table: pd.DataFrame) -> None:
    print("Aggregate match vs frozen evaluation_summary.csv:")
    hybrid_summary, global_summary = load_frozen_evaluation_summaries()

    hybrid_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_HYBRID
    ].iloc[0]
    global_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_GLOBAL
    ].iloc[0]

    for metric in METRIC_COLUMNS:
        table_hybrid_mean = float(hybrid_row[f"{metric}_mean"])
        table_global_mean = float(global_row[f"{metric}_mean"])
        frozen_hybrid_mean = float(hybrid_summary.loc["mean", metric])
        frozen_global_mean = float(global_summary.loc["mean", metric])

        if not np.isclose(table_hybrid_mean, frozen_hybrid_mean, atol=TOLERANCE):
            raise AssertionError(
                f"Hybrid {metric} mean mismatch: table={table_hybrid_mean}, "
                f"frozen={frozen_hybrid_mean}"
            )
        if not np.isclose(table_global_mean, frozen_global_mean, atol=TOLERANCE):
            raise AssertionError(
                f"Global {metric} mean mismatch: table={table_global_mean}, "
                f"frozen={frozen_global_mean}"
            )
        print(f"  OK  {metric} means match frozen summaries")


def _check_delta_summary_match(
    per_container_delta: pd.DataFrame,
    phase6_comparison: dict,
) -> None:
    print("Delta summary match:")
    recomputed = summarize_deltas(per_container_delta)
    saved = phase6_comparison["delta_summary"]

    assert saved["delta_sign_convention"] == DELTA_SIGN_CONVENTION
    assert saved["n_containers"] == EXPECTED_EVALUATED
    assert recomputed["n_containers"] == saved["n_containers"]

    for metric in METRIC_COLUMNS:
        for key in ("mean_delta", "std_delta", "improved_count", "worsened_count", "tied_count"):
            if not np.isclose(
                recomputed[metric][key],
                saved[metric][key],
                rtol=0.0,
                atol=TOLERANCE,
            ):
                raise AssertionError(
                    f"delta_summary mismatch for {metric}.{key}: "
                    f"recomputed={recomputed[metric][key]}, saved={saved[metric][key]}"
                )
        print(f"  OK  {metric} delta summary")


def main() -> None:
    args = _parse_args()
    experiment_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir is not None
        else _latest_comparison_dir()
    )

    comparison_table_path = experiment_dir / "comparison_table.csv"
    per_container_delta_path = experiment_dir / "per_container_delta.csv"
    phase6_comparison_path = experiment_dir / "phase6_comparison.json"
    comparison_metadata_path = experiment_dir / "comparison_metadata.json"

    for path in (
        comparison_table_path,
        per_container_delta_path,
        phase6_comparison_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison artifact: {path}")

    print("=== Hybrid vs Global GRU Comparison Verification ===")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Hybrid reference  : {HYBRID_BASELINE_REFERENCE_DIR}")
    print(f"Global reference  : {GLOBAL_GRU_REFERENCE_DIR}")
    print()

    hybrid_df, global_df = load_frozen_evaluation_dfs()
    assert_cohort_alignment(hybrid_df, global_df)

    comparison_table = pd.read_csv(comparison_table_path)
    per_container_delta = pd.read_csv(per_container_delta_path)
    phase6_comparison = json.loads(phase6_comparison_path.read_text())

    _check_cohort_ids(hybrid_df, global_df)
    print()
    _check_merge_integrity(per_container_delta)
    print()
    _check_delta_convention(per_container_delta)
    print()
    _check_no_nan_outputs(comparison_table, per_container_delta)
    print()
    _check_aggregate_match(comparison_table)
    print()
    _check_delta_summary_match(per_container_delta, phase6_comparison)

    # Recompute from frozen sources and compare saved artifacts.
    print()
    print("Recompute check from frozen evaluation_df.csv:")
    rebuilt_table = build_comparison_table(hybrid_df, global_df)
    rebuilt_delta = build_per_container_delta(hybrid_df, global_df)
    for metric in METRIC_COLUMNS:
        saved_hybrid = comparison_table.loc[
            comparison_table["methodology"] == METHODOLOGY_HYBRID,
            f"{metric}_mean",
        ].iloc[0]
        rebuilt_hybrid = rebuilt_table.loc[
            rebuilt_table["methodology"] == METHODOLOGY_HYBRID,
            f"{metric}_mean",
        ].iloc[0]
        assert np.isclose(saved_hybrid, rebuilt_hybrid, atol=TOLERANCE)
    merged = per_container_delta.merge(
        rebuilt_delta,
        on="container_id",
        suffixes=("_saved", "_rebuilt"),
    )
    assert len(merged) == EXPECTED_EVALUATED
    for metric in METRIC_COLUMNS:
        delta_col = f"delta_{metric}"
        if not np.allclose(
            merged[f"{delta_col}_saved"],
            merged[f"{delta_col}_rebuilt"],
            atol=TOLERANCE,
        ):
            raise AssertionError(f"Saved vs rebuilt mismatch for {delta_col}")
    print("  OK  saved artifacts match frozen-source recompute")

    if comparison_metadata_path.exists():
        metadata = json.loads(comparison_metadata_path.read_text())
        assert metadata.get("delta_sign_convention") == DELTA_SIGN_CONVENTION
        assert metadata.get("evaluated_containers") == EXPECTED_EVALUATED
        print()
        print("  OK  comparison_metadata.json provenance fields")

    print()
    print("Comparison integrity: PASS")
    print("All Hybrid vs Global GRU comparison checks passed.")


if __name__ == "__main__":
    main()
