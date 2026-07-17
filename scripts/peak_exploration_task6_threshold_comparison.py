#!/usr/bin/env python3
"""Phase 2 Task 6 — Consolidated P85 / P90 / P95 threshold comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments/peak_exploration_2026-07-14"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

PERCENTILES: tuple[int, ...] = (85, 90, 95)
PERCENTILE_COLORS = {85: "#4C72B0", 90: "#DD8452", 95: "#55A868"}


def _build_comparison_table() -> pd.DataFrame:
    timestep = pd.read_csv(EXPERIMENT_DIR / "timestep_summary.csv")
    sequence = pd.read_csv(EXPERIMENT_DIR / "sequence_peak_stats.csv")
    sequence_ref = pd.read_csv(
        EXPERIMENT_DIR / "sequence_peak_stats_reference.csv"
    )
    workload = pd.read_csv(EXPERIMENT_DIR / "workload_stratification.csv")
    val_sanity = pd.read_csv(EXPERIMENT_DIR / "val_sanity_check.csv")
    near_idle = pd.read_csv(EXPERIMENT_DIR / "near_idle_impact_summary.csv")

    workload_off = workload[
        (workload["cohort"] == "official_all_containers")
        & (workload["analysis_level"] == "timestep")
    ]
    val_off = val_sanity[val_sanity["cohort"] == "official_all_containers"]

    rows: list[dict] = []
    for percentile in PERCENTILES:
        ts_row = timestep[timestep["percentile"] == percentile].iloc[0]
        seq_row = sequence[sequence["percentile"] == percentile].iloc[0]
        seq_ref_row = sequence_ref[
            sequence_ref["percentile"] == percentile
        ].iloc[0]
        val_row = val_off[val_off["percentile"] == percentile].iloc[0]
        idle_row = near_idle[near_idle["percentile"] == percentile].iloc[0]

        stable_pct = workload_off[
            (workload_off["percentile"] == percentile)
            & (workload_off["pattern_type"] == "stable")
        ]["weighted_peak_timestep_pct"].iloc[0]
        medium_pct = workload_off[
            (workload_off["percentile"] == percentile)
            & (workload_off["pattern_type"] == "medium")
        ]["weighted_peak_timestep_pct"].iloc[0]
        spiky_pct = workload_off[
            (workload_off["percentile"] == percentile)
            & (workload_off["pattern_type"] == "spiky")
        ]["weighted_peak_timestep_pct"].iloc[0]

        zero_peak_seq_pct = (
            seq_row["zero_peak_sequences"] / seq_row["total_sequences"] * 100.0
        )

        rows.append({
            "percentile": percentile,
            "train_peak_timestep_pct": float(
                ts_row["global_peak_timestep_pct"]
            ),
            "train_peak_timestep_pct_reference_excl_idle": float(
                idle_row["adjusted_global_peak_pct_excluding_idle"]
            ),
            "train_peak_sequence_pct": float(seq_row["peak_sequence_pct"]),
            "train_zero_peak_sequence_pct": float(zero_peak_seq_pct),
            "train_mean_peak_steps_per_sequence": float(
                seq_row["mean_peak_timesteps_per_sequence"]
            ),
            "train_mean_peak_steps_reference_excl_idle": float(
                seq_ref_row["mean_peak_timesteps_per_sequence"]
            ),
            "mean_peak_pct_stable": float(stable_pct),
            "mean_peak_pct_medium": float(medium_pct),
            "mean_peak_pct_spiky": float(spiky_pct),
            "val_peak_timestep_pct": float(val_row["val_peak_timestep_pct"]),
            "val_day1_peak_timestep_pct": float(
                val_row["val_day1_peak_timestep_pct"]
            ),
            "val_day1_containers_with_peak_pct": float(
                val_row["val_day1_containers_with_peak_pct"]
            ),
            "val_peak_evaluation_feasible": int(
                val_row["val_peak_evaluation_feasible"]
            ),
            "near_idle_share_of_train_peaks_pct": float(
                idle_row["idle_share_of_all_peaks_pct"]
            ),
        })

    comparison = pd.DataFrame(rows)

    notes = {
        85: (
            "Loosest threshold; highest peak rates; limited discrimination "
            "between peak/non-peak sequences (~3.4% zero-peak); near-idle "
            "inflates rates by ~4.2 pp."
        ),
        90: (
            "Balanced candidate; ~17% train / ~28% val peak timesteps; "
            "~3.8% zero-peak sequences; strong validation feasibility; "
            "moderate near-idle sensitivity."
        ),
        95: (
            "Strictest threshold; lowest peak rates (~11% train); best "
            "zero-peak sequence discrimination (~7.7%); highest near-idle "
            "share of peak labels (37%); still validation-feasible."
        ),
    }
    comparison["summary_notes"] = comparison["percentile"].map(notes)
    return comparison


def _plot_threshold_comparison(comparison: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(PERCENTILES))
    labels = [f"P{p}" for p in PERCENTILES]
    colors = [PERCENTILE_COLORS[p] for p in PERCENTILES]

    # Train vs val timestep rates
    ax = axes[0, 0]
    width = 0.35
    ax.bar(
        x - width / 2,
        comparison["train_peak_timestep_pct"],
        width,
        label="Train",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        comparison["val_peak_timestep_pct"],
        width,
        label="Validation",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Peak Timestep Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Peak sequence rate and zero-peak sequence rate
    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        comparison["train_peak_sequence_pct"],
        width,
        label="Peak sequences",
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        comparison["train_zero_peak_sequence_pct"],
        width,
        label="Zero-peak sequences",
        color="#AAAAAA",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Sequence-Level Rates (Train, %)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Workload stratification (spiky timestep %)
    ax = axes[1, 0]
    ax.plot(
        PERCENTILES,
        comparison["mean_peak_pct_stable"],
        marker="o",
        label="Stable",
        color="#55A868",
        linewidth=2,
    )
    ax.plot(
        PERCENTILES,
        comparison["mean_peak_pct_medium"],
        marker="s",
        label="Medium",
        color="#4C72B0",
        linewidth=2,
    )
    ax.plot(
        PERCENTILES,
        comparison["mean_peak_pct_spiky"],
        marker="^",
        label="Spiky",
        color="#DD8452",
        linewidth=2,
    )
    ax.set_title("Weighted Peak Timestep % by Workload", fontsize=11)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Peak timesteps (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Reference-adjusted train timestep rate
    ax = axes[1, 1]
    ax.bar(
        x - width / 2,
        comparison["train_peak_timestep_pct"],
        width,
        label="Official",
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        comparison["train_peak_timestep_pct_reference_excl_idle"],
        width,
        label="Excl. near-idle",
        color="#C44E52",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Near-Idle Sensitivity (Train Timestep %)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 2 Threshold Comparison — P85 vs P90 vs P95",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comparison = _build_comparison_table()
    comparison.to_csv(
        EXPERIMENT_DIR / "threshold_comparison.csv",
        index=False,
    )

    advantages = {
        85: [
            "Highest peak label coverage; strongest emphasis on upper-tail timesteps",
            "Largest number of peak timesteps for peak-aware training signal",
            "All validation feasibility criteria met",
        ],
        90: [
            "Balanced peak frequency (~17% train, ~28% val timesteps)",
            "Aligns with common statistical practice (90th percentile)",
            "Better zero-peak sequence discrimination than P85 (~3.8% vs ~3.4%)",
            "Strong validation and Day-1 evaluation support",
            "Leading candidate from Tasks 2–5",
        ],
        95: [
            "Strictest peak definition; closest to 'rare event' semantics",
            "Best zero-peak sequence discrimination (~7.7% zero-peak sequences)",
            "Lowest sensitivity to near-idle threshold-zero artefact on label share",
            "Still validation-feasible with 97.98% containers having Day-1 peaks",
        ],
    }
    disadvantages = {
        85: [
            "May be too permissive — ~22% of timesteps labelled peaks",
            "Minimal zero-peak sequence discrimination (sequence weighting ineffective)",
            "Less interpretable as 'spike' in thesis language",
        ],
        90: [
            "Still high peak-sequence rate (~96%); sequence-level weighting remains ineffective",
            "Moderate near-idle distortion (~29% of peak labels from 5 containers)",
            "Spiky workload group appears elevated (artefact-driven)",
        ],
        95: [
            "Fewer peak timesteps (~11% train) — weaker training emphasis on peaks",
            "Higher mean peak-step variance across containers",
            "May under-emphasize moderate utilization surges",
        ],
    }

    task6_results = {
        "task": 6,
        "comparison": comparison.to_dict(orient="records"),
        "advantages": {str(k): v for k, v in advantages.items()},
        "disadvantages": {str(k): v for k, v in disadvantages.items()},
        "preliminary_recommendation": (
            "P90 per-container on real CPU % remains the leading candidate "
            "for Task 7, pending formal decision."
        ),
    }
    with open(EXPERIMENT_DIR / "task6_results.json", "w") as handle:
        json.dump(task6_results, handle, indent=2)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_threshold_comparison(
        comparison,
        FIGURES_DIR / "task6_threshold_comparison",
    )

    print("=== Phase 2 Task 6 — Threshold Comparison ===")
    print()
    print(comparison.drop(columns=["summary_notes"]).to_string(index=False))
    print()
    print("Saved:", EXPERIMENT_DIR / "threshold_comparison.csv")


if __name__ == "__main__":
    main()
