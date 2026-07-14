#!/usr/bin/env python3
"""Phase 2 Task 4 — Workload stratification (stable / medium / spiky)."""

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
PATTERN_ORDER = ("stable", "medium", "spiky")
PATTERN_COLORS = {"stable": "#55A868", "medium": "#4C72B0", "spiky": "#DD8452"}

NEAR_IDLE_CONTAINERS: tuple[str, ...] = (
    "c_13308",
    "c_14106",
    "c_15035",
    "c_15446",
    "c_15794",
)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    if total == 0:
        return 0.0
    return float((values * weights).sum() / total)


def _timestep_stratification(
    per_container: pd.DataFrame,
    cohort_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for percentile in PERCENTILES:
        pct_col = f"peak_timestep_pct_p{percentile}"
        thr_col = f"threshold_p{percentile}"
        for pattern in PATTERN_ORDER:
            subset = per_container[per_container["pattern_type"] == pattern]
            rows.append({
                "cohort": cohort_label,
                "analysis_level": "timestep",
                "percentile": percentile,
                "pattern_type": pattern,
                "container_count": int(len(subset)),
                "train_timesteps": int(subset["train_timesteps"].sum()),
                "weighted_peak_timestep_pct": _weighted_mean(
                    subset[pct_col], subset["train_timesteps"]
                ),
                "mean_peak_timestep_pct": float(subset[pct_col].mean()),
                "median_peak_timestep_pct": float(subset[pct_col].median()),
                "mean_threshold": float(subset[thr_col].mean()),
                "mean_cpu_real": float(subset["mean_cpu_real"].mean()),
                "mean_cv": float(subset["cv"].mean()),
            })
    return pd.DataFrame(rows)


def _sequence_stratification(
    per_container_seq: pd.DataFrame,
    cohort_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for percentile in PERCENTILES:
        seq_pct_col = f"peak_sequence_pct_p{percentile}"
        mean_steps_col = f"mean_peak_steps_p{percentile}"
        for pattern in PATTERN_ORDER:
            subset = per_container_seq[
                per_container_seq["pattern_type"] == pattern
            ]
            rows.append({
                "cohort": cohort_label,
                "analysis_level": "sequence",
                "percentile": percentile,
                "pattern_type": pattern,
                "container_count": int(len(subset)),
                "total_sequences": int(subset["total_sequences"].sum()),
                "weighted_peak_sequence_pct": _weighted_mean(
                    subset[seq_pct_col], subset["total_sequences"]
                ),
                "mean_peak_sequence_pct": float(subset[seq_pct_col].mean()),
                "weighted_mean_peak_steps_per_sequence": _weighted_mean(
                    subset[mean_steps_col], subset["total_sequences"]
                ),
                "mean_peak_steps_per_sequence": float(
                    subset[mean_steps_col].mean()
                ),
            })
    return pd.DataFrame(rows)


def _attach_pattern_type(
    per_container: pd.DataFrame,
    per_container_seq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure pattern_type is present on sequence stats."""
    if "pattern_type" not in per_container_seq.columns:
        meta = per_container[["container_id", "pattern_type", "cv"]]
        per_container_seq = per_container_seq.merge(
            meta, on="container_id", how="left"
        )
    return per_container, per_container_seq


def _plot_timestep_by_workload(
    strat: pd.DataFrame,
    cohort_label: str,
    out_base: Path,
) -> None:
    subset = strat[
        (strat["cohort"] == cohort_label)
        & (strat["analysis_level"] == "timestep")
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PERCENTILES))
    width = 0.25
    for i, pattern in enumerate(PATTERN_ORDER):
        pcts = [
            subset[
                (subset["percentile"] == p)
                & (subset["pattern_type"] == pattern)
            ]["weighted_peak_timestep_pct"].iloc[0]
            for p in PERCENTILES
        ]
        ax.bar(
            x + (i - 1) * width,
            pcts,
            width,
            label=pattern.capitalize(),
            color=PATTERN_COLORS[pattern],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in PERCENTILES])
    title_suffix = (
        "Official (99 containers)"
        if cohort_label == "official_all_containers"
        else "Reference (excl. near-idle)"
    )
    ax.set_title(
        f"Peak Timestep Rate by Workload Type\n{title_suffix}",
        fontsize=13,
    )
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel("Weighted Peak Timesteps (%)", fontsize=12)
    ax.legend(title="Workload", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_sequence_by_workload(
    strat: pd.DataFrame,
    cohort_label: str,
    out_base: Path,
) -> None:
    subset = strat[
        (strat["cohort"] == cohort_label)
        & (strat["analysis_level"] == "sequence")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(PERCENTILES))
    width = 0.25
    for i, pattern in enumerate(PATTERN_ORDER):
        seq_pcts = [
            subset[
                (subset["percentile"] == p)
                & (subset["pattern_type"] == pattern)
            ]["weighted_peak_sequence_pct"].iloc[0]
            for p in PERCENTILES
        ]
        mean_steps = [
            subset[
                (subset["percentile"] == p)
                & (subset["pattern_type"] == pattern)
            ]["weighted_mean_peak_steps_per_sequence"].iloc[0]
            for p in PERCENTILES
        ]
        axes[0].bar(
            x + (i - 1) * width,
            seq_pcts,
            width,
            label=pattern.capitalize(),
            color=PATTERN_COLORS[pattern],
            edgecolor="black",
            linewidth=0.4,
        )
        axes[1].bar(
            x + (i - 1) * width,
            mean_steps,
            width,
            label=pattern.capitalize(),
            color=PATTERN_COLORS[pattern],
            edgecolor="black",
            linewidth=0.4,
        )

    title_suffix = (
        "Official"
        if cohort_label == "official_all_containers"
        else "Reference"
    )
    axes[0].set_title(f"Peak Sequence Rate ({title_suffix})", fontsize=12)
    axes[1].set_title(f"Mean Peak Steps / Sequence ({title_suffix})", fontsize=12)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in PERCENTILES])
        ax.set_xlabel("Threshold Percentile", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Weighted Peak Sequences (%)", fontsize=11)
    axes[1].set_ylabel("Mean Peak Timesteps (of 96)", fontsize=11)
    axes[0].legend(title="Workload", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_cv_vs_peak_scatter(
    per_container: pd.DataFrame,
    percentile: int,
    out_base: Path,
) -> None:
    col = f"peak_timestep_pct_p{percentile}"
    fig, ax = plt.subplots(figsize=(8, 5))
    for pattern in PATTERN_ORDER:
        subset = per_container[per_container["pattern_type"] == pattern]
        ax.scatter(
            subset["cv"],
            subset[col],
            label=pattern.capitalize(),
            color=PATTERN_COLORS[pattern],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.3,
            s=50,
        )
    ax.set_xscale("log")
    ax.set_title(
        f"Coefficient of Variation vs Peak Timestep Rate (P{percentile})",
        fontsize=13,
    )
    ax.set_xlabel("CV (log scale)", fontsize=12)
    ax.set_ylabel("Peak Timesteps (%)", fontsize=12)
    ax.legend(title="Workload", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    per_container = pd.read_csv(
        EXPERIMENT_DIR / "per_container_peak_stats.csv"
    )
    per_container_seq = pd.read_csv(
        EXPERIMENT_DIR / "per_container_sequence_stats.csv"
    )
    per_container_seq = per_container_seq[
        per_container_seq["cohort"] == "official_all_containers"
    ].copy()

    per_container, per_container_seq = _attach_pattern_type(
        per_container, per_container_seq
    )

    # Official cohort
    timestep_official = _timestep_stratification(
        per_container, "official_all_containers"
    )
    sequence_official = _sequence_stratification(
        per_container_seq, "official_all_containers"
    )

    # Reference cohort (exclude near-idle)
    per_container_ref = per_container[
        ~per_container["container_id"].isin(NEAR_IDLE_CONTAINERS)
    ].copy()
    per_container_seq_ref = per_container_seq[
        ~per_container_seq["container_id"].isin(NEAR_IDLE_CONTAINERS)
    ].copy()

    timestep_ref = _timestep_stratification(
        per_container_ref, "reference_excl_near_idle"
    )
    sequence_ref = _sequence_stratification(
        per_container_seq_ref, "reference_excl_near_idle"
    )

    stratification = pd.concat(
        [timestep_official, sequence_official, timestep_ref, sequence_ref],
        ignore_index=True,
    )
    stratification.to_csv(
        EXPERIMENT_DIR / "workload_stratification.csv",
        index=False,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_timestep_by_workload(
        stratification,
        "official_all_containers",
        FIGURES_DIR / "task4_peak_timestep_by_workload_official",
    )
    _plot_timestep_by_workload(
        stratification,
        "reference_excl_near_idle",
        FIGURES_DIR / "task4_peak_timestep_by_workload_reference",
    )
    _plot_sequence_by_workload(
        stratification,
        "official_all_containers",
        FIGURES_DIR / "task4_peak_sequence_by_workload_official",
    )
    _plot_sequence_by_workload(
        stratification,
        "reference_excl_near_idle",
        FIGURES_DIR / "task4_peak_sequence_by_workload_reference",
    )
    _plot_cv_vs_peak_scatter(
        per_container,
        90,
        FIGURES_DIR / "task4_cv_vs_peak_timestep_p90",
    )

    task4_results = {
        "task": 4,
        "official_timestep": timestep_official.to_dict(orient="records"),
        "official_sequence": sequence_official.to_dict(orient="records"),
        "reference_timestep": timestep_ref.to_dict(orient="records"),
        "reference_sequence": sequence_ref.to_dict(orient="records"),
        "near_idle_in_spiky": {
            "spiky_total": int(
                (per_container["pattern_type"] == "spiky").sum()
            ),
            "near_idle_count": len(NEAR_IDLE_CONTAINERS),
            "near_idle_ids": list(NEAR_IDLE_CONTAINERS),
        },
    }
    with open(EXPERIMENT_DIR / "task4_results.json", "w") as handle:
        json.dump(task4_results, handle, indent=2)

    print("=== Phase 2 Task 4 — Workload Stratification ===")
    print()
    print("--- Official: Timestep level ---")
    print(
        timestep_official.to_string(index=False)
    )
    print()
    print("--- Official: Sequence level ---")
    print(
        sequence_official.to_string(index=False)
    )
    print()
    print("--- Reference (excl. near-idle): Timestep level ---")
    print(
        timestep_ref.to_string(index=False)
    )


if __name__ == "__main__":
    main()
