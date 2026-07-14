#!/usr/bin/env python3
"""Phase 2 supplementary — quantify near-idle zero-threshold container impact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments/peak_exploration_2026-07-14"
PERCENTILES = (85, 90, 95)


def _peak_timesteps(row: pd.Series, percentile: int) -> int:
    pct_col = f"peak_timestep_pct_p{percentile}"
    return int(round(row["train_timesteps"] * row[pct_col] / 100.0))


def main() -> None:
    per_container = pd.read_csv(
        EXPERIMENT_DIR / "per_container_peak_stats.csv"
    )
    summary = pd.read_csv(EXPERIMENT_DIR / "timestep_summary.csv")

    # Primary focus: P90 threshold == 0 (user observation)
    zero_p90 = per_container[per_container["threshold_p90"] == 0.0].copy()
    zero_p90_ids = zero_p90["container_id"].tolist()

    total_timesteps = int(per_container["train_timesteps"].sum())
    idle_timesteps = int(zero_p90["train_timesteps"].sum())
    idle_timestep_pct = idle_timesteps / total_timesteps * 100.0

    container_rows = []
    for _, row in zero_p90.iterrows():
        container_rows.append({
            "container_id": row["container_id"],
            "pattern_type": row["pattern_type"],
            "cv": row["cv"],
            "train_timesteps": int(row["train_timesteps"]),
            "threshold_p85": row["threshold_p85"],
            "threshold_p90": row["threshold_p90"],
            "threshold_p95": row["threshold_p95"],
            "mean_cpu_real": row["mean_cpu_real"],
            "max_cpu_real": row["max_cpu_real"],
            "peak_timesteps_p85": _peak_timesteps(row, 85),
            "peak_timesteps_p90": _peak_timesteps(row, 90),
            "peak_timesteps_p95": _peak_timesteps(row, 95),
        })

    idle_containers_df = pd.DataFrame(container_rows)

    peak_contribution = {}
    adjusted_global = {}
    original_global = {
        int(r["percentile"]): float(r["global_peak_timestep_pct"])
        for _, r in summary.iterrows()
    }

    for percentile in PERCENTILES:
        peak_col = f"peak_timesteps_p{percentile}"
        idle_peak_count = int(idle_containers_df[peak_col].sum())
        all_peak_count = int(
            round(
                total_timesteps
                * original_global[percentile]
                / 100.0
            )
        )
        non_idle_timesteps = total_timesteps - idle_timesteps
        non_idle_peak_count = all_peak_count - idle_peak_count
        adjusted_pct = (
            non_idle_peak_count / non_idle_timesteps * 100.0
            if non_idle_timesteps > 0
            else 0.0
        )

        peak_contribution[percentile] = {
            "idle_container_peak_timesteps": idle_peak_count,
            "idle_share_of_all_peak_timesteps_pct": (
                idle_peak_count / all_peak_count * 100.0
                if all_peak_count > 0
                else 0.0
            ),
            "original_global_peak_timestep_pct": original_global[percentile],
            "adjusted_global_peak_timestep_pct_excluding_idle": adjusted_pct,
            "absolute_change_percentage_points": (
                adjusted_pct - original_global[percentile]
            ),
        }
        adjusted_global[percentile] = adjusted_pct

    impact_summary = pd.DataFrame([
        {
            "percentile": p,
            "original_global_peak_pct": original_global[p],
            "adjusted_global_peak_pct_excluding_idle": adjusted_global[p],
            "change_percentage_points": adjusted_global[p] - original_global[p],
            "idle_peak_timesteps": peak_contribution[p][
                "idle_container_peak_timesteps"
            ],
            "idle_share_of_all_peaks_pct": peak_contribution[p][
                "idle_share_of_all_peak_timesteps_pct"
            ],
        }
        for p in PERCENTILES
    ])

    idle_containers_df.to_csv(
        EXPERIMENT_DIR / "near_idle_zero_threshold_containers.csv",
        index=False,
    )
    impact_summary.to_csv(
        EXPERIMENT_DIR / "near_idle_impact_summary.csv",
        index=False,
    )

    results = {
        "analysis": "near_idle_zero_threshold_impact",
        "criterion": "threshold_p90 == 0.0",
        "containers_affected": zero_p90_ids,
        "container_count": len(zero_p90_ids),
        "idle_train_timesteps": idle_timesteps,
        "total_train_timesteps": total_timesteps,
        "idle_timestep_pct_of_total": idle_timestep_pct,
        "per_container": container_rows,
        "peak_contribution_by_percentile": peak_contribution,
    }
    with open(EXPERIMENT_DIR / "near_idle_impact.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print("=== Near-Idle Zero-Threshold Impact Analysis ===")
    print(f"Containers (P90 threshold = 0): {len(zero_p90_ids)}")
    print(f"Container IDs: {', '.join(zero_p90_ids)}")
    print(f"Idle train timesteps: {idle_timesteps} / {total_timesteps} "
          f"({idle_timestep_pct:.2f}%)")
    print()
    print(idle_containers_df.to_string(index=False))
    print()
    print(impact_summary.to_string(index=False))


if __name__ == "__main__":
    main()
