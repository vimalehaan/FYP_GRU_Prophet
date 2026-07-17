#!/usr/bin/env python3
"""Build Hybrid vs Global GRU methodology comparison artifacts (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.methodology_comparison import (  # noqa: E402
    DELTA_SIGN_CONVENTION,
    EXPECTED_EVALUATED,
    METRIC_COLUMNS,
    METHODOLOGY_GLOBAL,
    METHODOLOGY_HYBRID,
    assert_cohort_alignment,
    build_comparison_metadata,
    build_comparison_table,
    build_experiment_config,
    build_per_container_delta,
    load_frozen_evaluation_dfs,
    summarize_deltas,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only Hybrid vs Global GRU methodology comparison outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override comparison experiment directory "
            "(default: experiments/hybrid_vs_global_<timestamp>)"
        ),
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _comparison_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return REPO_ROOT / f"experiments/hybrid_vs_global_{stamp}"


def _build_phase6_comparison(
    comparison_table,
    delta_summary: dict,
    experiment_config: dict,
    started_at: str,
    completed_at: str,
) -> dict:
    """Assemble locked comparison summary JSON."""
    hybrid_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_HYBRID
    ].iloc[0]
    global_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_GLOBAL
    ].iloc[0]

    primary_results: dict[str, dict[str, float]] = {}
    mean_deltas: dict[str, float] = {}

    for metric in METRIC_COLUMNS:
        hybrid_mean = float(hybrid_row[f"{metric}_mean"])
        global_mean = float(global_row[f"{metric}_mean"])
        primary_results[metric] = {
            "hybrid_mean": hybrid_mean,
            "global_mean": global_mean,
            "hybrid_std": float(hybrid_row[f"{metric}_std"]),
            "global_std": float(global_row[f"{metric}_std"]),
            "mean_delta_global_minus_hybrid": float(
                delta_summary[metric]["mean_delta"],
            ),
        }
        mean_deltas[metric] = float(delta_summary[metric]["mean_delta"])

    return {
        "comparison_type": "forecasting_methodology",
        "phase": 6,
        "task": 3,
        "delta_sign_convention": DELTA_SIGN_CONVENTION,
        "evaluated_containers": EXPECTED_EVALUATED,
        "started_at": started_at,
        "completed_at": completed_at,
        "primary_results": primary_results,
        "mean_delta_global_minus_hybrid": mean_deltas,
        "delta_summary": delta_summary,
        "experiment_config": experiment_config,
    }


def _assert_task3_checks(
    comparison_table,
    per_container_delta,
    delta_summary: dict,
) -> None:
    """Raise if Task 3 build checks fail."""
    if len(comparison_table) != 2:
        raise RuntimeError(
            f"Expected 2 methodology rows in comparison_table, got {len(comparison_table)}"
        )
    if len(per_container_delta) != EXPECTED_EVALUATED:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED} delta rows, got {len(per_container_delta)}"
        )
    if per_container_delta["container_id"].duplicated().any():
        raise RuntimeError("Duplicate container_id in per_container_delta")

    for metric in METRIC_COLUMNS:
        delta_col = f"delta_{metric}"
        hybrid_col = f"hybrid_{metric}"
        global_col = f"global_{metric}"
        if per_container_delta[delta_col].isna().any():
            raise RuntimeError(f"NaN values in {delta_col}")
        recomputed = (
            per_container_delta[global_col] - per_container_delta[hybrid_col]
        )
        if not np.allclose(per_container_delta[delta_col], recomputed, atol=1e-12):
            raise RuntimeError(f"Delta convention mismatch for {metric}")

    if delta_summary["n_containers"] != EXPECTED_EVALUATED:
        raise RuntimeError("Delta summary container count mismatch")


def main() -> None:
    args = _parse_args()
    experiment_dir = _comparison_dir(
        args.output_dir.resolve() if args.output_dir else None,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    comparison_table_path = experiment_dir / "comparison_table.csv"
    per_container_delta_path = experiment_dir / "per_container_delta.csv"
    experiment_config_path = experiment_dir / "experiment_config.json"
    phase6_comparison_path = experiment_dir / "phase6_comparison.json"
    comparison_metadata_path = experiment_dir / "comparison_metadata.json"

    print("=" * 72)
    print("Hybrid vs Global GRU — Methodology Comparison Build (Task 3)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Delta convention     : {DELTA_SIGN_CONVENTION}")
    print()

    started_at = _iso_timestamp()

    hybrid_df, global_df = load_frozen_evaluation_dfs()
    assert_cohort_alignment(hybrid_df, global_df)

    comparison_table = build_comparison_table(hybrid_df, global_df)
    per_container_delta = build_per_container_delta(hybrid_df, global_df)
    delta_summary = summarize_deltas(per_container_delta)
    experiment_config = build_experiment_config()

    _assert_task3_checks(comparison_table, per_container_delta, delta_summary)

    completed_at = _iso_timestamp()
    phase6_comparison = _build_phase6_comparison(
        comparison_table=comparison_table,
        delta_summary=delta_summary,
        experiment_config=experiment_config,
        started_at=started_at,
        completed_at=completed_at,
    )
    comparison_metadata = build_comparison_metadata(
        experiment_dir=experiment_dir,
        comparison_table=comparison_table,
        per_container_delta=per_container_delta,
        delta_summary=delta_summary,
        experiment_config=experiment_config,
        comparison_timestamp=completed_at,
    )

    comparison_table.to_csv(comparison_table_path, index=False)
    per_container_delta.to_csv(per_container_delta_path, index=False)
    experiment_config_path.write_text(
        json.dumps(experiment_config, indent=2) + "\n",
    )
    phase6_comparison_path.write_text(
        json.dumps(phase6_comparison, indent=2) + "\n",
    )
    comparison_metadata_path.write_text(
        json.dumps(comparison_metadata, indent=2) + "\n",
    )

    hybrid_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_HYBRID
    ].iloc[0]
    global_row = comparison_table.loc[
        comparison_table["methodology"] == METHODOLOGY_GLOBAL
    ].iloc[0]

    print(f"Evaluated containers : {EXPECTED_EVALUATED}")
    print(f"Delta rows           : {len(per_container_delta)}")
    print()
    print("Primary results (mean, real CPU %):")
    for metric in METRIC_COLUMNS:
        print(
            f"  {metric:12s}: hybrid={hybrid_row[f'{metric}_mean']:.4f}  "
            f"global={global_row[f'{metric}_mean']:.4f}  "
            f"delta={delta_summary[metric]['mean_delta']:+.4f}"
        )
    print()
    print(
        "MAE improved / worsened (Global better / worse): "
        f"{delta_summary['day1_mae']['improved_count']} / "
        f"{delta_summary['day1_mae']['worsened_count']}"
    )
    print()
    print("Saved outputs:")
    print(f"  comparison_table.csv      : {comparison_table_path}")
    print(f"  per_container_delta.csv   : {per_container_delta_path}")
    print(f"  experiment_config.json    : {experiment_config_path}")
    print(f"  phase6_comparison.json    : {phase6_comparison_path}")
    print(f"  comparison_metadata.json  : {comparison_metadata_path}")
    print()
    print("Task 3 build checks: PASS")


if __name__ == "__main__":
    main()
