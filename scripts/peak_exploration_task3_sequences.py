#!/usr/bin/env python3
"""Phase 2 Task 3 — Peak sequence analysis (96→96 sliding windows)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments/peak_exploration_2026-07-14"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

INPUT_WINDOW = 96
FORECAST_HORIZON = 96
PERCENTILES: tuple[int, ...] = (85, 90, 95)
PERCENTILE_COLORS = {85: "#4C72B0", 90: "#DD8452", 95: "#55A868"}

NEAR_IDLE_CONTAINERS: tuple[str, ...] = (
    "c_13308",
    "c_14106",
    "c_15035",
    "c_15446",
    "c_15794",
)


def _cpu_real(series_scaled: pd.Series, scaler) -> np.ndarray:
    values = series_scaled.values.reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def _evaluable_containers(selected: np.ndarray, train_df: pd.DataFrame) -> list[str]:
    train_ids = set(train_df["container_id"].unique())
    return [cid for cid in selected if cid in train_ids]


def _container_thresholds(
    global_train: pd.DataFrame,
    scalers: dict,
) -> dict[str, dict[int, float]]:
    thresholds: dict[str, dict[int, float]] = {}
    for container_id, group in global_train.groupby("container_id"):
        cpu_real = _cpu_real(group.sort_values("time_stamp")["cpu_scaled"], scalers[container_id])
        thresholds[container_id] = {
            p: float(np.percentile(cpu_real, p)) for p in PERCENTILES
        }
    return thresholds


def _peak_counts_per_sequence(
    global_train: pd.DataFrame,
    scalers: dict,
    container_thresholds: dict[str, dict[int, float]],
    container_filter: set[str] | None = None,
) -> dict[int, list[int]]:
    """
    Return peak timestep counts in each forecast horizon per percentile.

    Each list entry is the number of peak timesteps (0–96) in one sequence.
    """
    counts: dict[int, list[int]] = {p: [] for p in PERCENTILES}

    for container_id, group in global_train.groupby("container_id"):
        if container_filter is not None and container_id not in container_filter:
            continue

        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])
        max_start = len(group) - INPUT_WINDOW - FORECAST_HORIZON

        for start in range(max_start):
            horizon = cpu_real[
                start + INPUT_WINDOW : start + INPUT_WINDOW + FORECAST_HORIZON
            ]
            for percentile in PERCENTILES:
                threshold = container_thresholds[container_id][percentile]
                peak_count = int(np.sum(horizon >= threshold))
                counts[percentile].append(peak_count)

    return counts


def _summarize_counts(
    counts: list[int],
    percentile: int,
    cohort_label: str,
) -> dict[str, float | int | str]:
    arr = np.array(counts, dtype=int)
    total = int(len(arr))
    peak_sequences = int(np.sum(arr >= 1))
    return {
        "cohort": cohort_label,
        "percentile": percentile,
        "total_sequences": total,
        "peak_sequences": peak_sequences,
        "peak_sequence_pct": float(peak_sequences / total * 100.0) if total else 0.0,
        "mean_peak_timesteps_per_sequence": float(arr.mean()) if total else 0.0,
        "median_peak_timesteps_per_sequence": float(np.median(arr)) if total else 0.0,
        "std_peak_timesteps_per_sequence": float(arr.std()) if total else 0.0,
        "max_peak_timesteps_per_sequence": int(arr.max()) if total else 0,
        "zero_peak_sequences": int(np.sum(arr == 0)),
        "full_peak_sequences": int(np.sum(arr == FORECAST_HORIZON)),
    }


def _distribution_df(
    counts: list[int],
    percentile: int,
    cohort_label: str,
) -> pd.DataFrame:
    arr = np.array(counts, dtype=int)
    total = len(arr)
    rows = []
    for peak_steps in range(FORECAST_HORIZON + 1):
        seq_count = int(np.sum(arr == peak_steps))
        if seq_count == 0:
            continue
        rows.append({
            "cohort": cohort_label,
            "percentile": percentile,
            "peak_timesteps_in_horizon": peak_steps,
            "sequence_count": seq_count,
            "sequence_pct": seq_count / total * 100.0 if total else 0.0,
        })
    return pd.DataFrame(rows)


def _per_container_sequence_df(
    global_train: pd.DataFrame,
    scalers: dict,
    container_thresholds: dict[str, dict[int, float]],
    cohort_label: str,
    container_filter: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    for container_id, group in global_train.groupby("container_id"):
        if container_filter is not None and container_id not in container_filter:
            continue

        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])
        max_start = len(group) - INPUT_WINDOW - FORECAST_HORIZON
        n_sequences = max_start

        row: dict = {
            "cohort": cohort_label,
            "container_id": container_id,
            "total_sequences": n_sequences,
        }
        for percentile in PERCENTILES:
            threshold = container_thresholds[container_id][percentile]
            peak_counts: list[int] = []
            for start in range(max_start):
                horizon = cpu_real[
                    start + INPUT_WINDOW : start + INPUT_WINDOW + FORECAST_HORIZON
                ]
                peak_counts.append(int(np.sum(horizon >= threshold)))
            arr = np.array(peak_counts, dtype=int)
            row[f"peak_sequences_p{percentile}"] = int(np.sum(arr >= 1))
            row[f"peak_sequence_pct_p{percentile}"] = (
                float(np.sum(arr >= 1) / n_sequences * 100.0)
                if n_sequences else 0.0
            )
            row[f"mean_peak_steps_p{percentile}"] = float(arr.mean()) if n_sequences else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("container_id").reset_index(drop=True)


def _plot_peak_sequence_rates(
    official: pd.DataFrame,
    reference: pd.DataFrame,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PERCENTILES))
    width = 0.35
    official_pcts = [
        official.loc[official["percentile"] == p, "peak_sequence_pct"].iloc[0]
        for p in PERCENTILES
    ]
    reference_pcts = [
        reference.loc[reference["percentile"] == p, "peak_sequence_pct"].iloc[0]
        for p in PERCENTILES
    ]
    ax.bar(
        x - width / 2,
        official_pcts,
        width,
        label="Official (99 containers)",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        reference_pcts,
        width,
        label="Reference (excl. 5 near-idle)",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in PERCENTILES])
    ax.set_title(
        "Peak Sequence Rate by Percentile Threshold\n"
        f"({INPUT_WINDOW}→{FORECAST_HORIZON} Sliding Windows, Train Period)",
        fontsize=13,
    )
    ax.set_xlabel("Per-Container Threshold Percentile", fontsize=12)
    ax.set_ylabel("Sequences with ≥1 Peak Timestep (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_peak_step_distribution(
    distribution: pd.DataFrame,
    percentile: int,
    cohort_label: str,
    out_base: Path,
) -> None:
    subset = distribution[
        (distribution["percentile"] == percentile)
        & (distribution["cohort"] == cohort_label)
    ].sort_values("peak_timesteps_in_horizon")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        subset["peak_timesteps_in_horizon"],
        subset["sequence_pct"],
        color=PERCENTILE_COLORS[percentile],
        edgecolor="black",
        linewidth=0.3,
        width=1.0,
    )
    ax.set_title(
        f"Distribution of Peak Timesteps per Sequence (P{percentile})\n"
        f"{cohort_label} — {INPUT_WINDOW}→{FORECAST_HORIZON} Forecast Horizon",
        fontsize=13,
    )
    ax.set_xlabel("Peak Timesteps in Forecast Horizon (of 96)", fontsize=12)
    ax.set_ylabel("Sequences (%)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_mean_peak_steps(
    official: pd.DataFrame,
    reference: pd.DataFrame,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PERCENTILES))
    width = 0.35
    official_means = [
        official.loc[official["percentile"] == p, "mean_peak_timesteps_per_sequence"].iloc[0]
        for p in PERCENTILES
    ]
    reference_means = [
        reference.loc[reference["percentile"] == p, "mean_peak_timesteps_per_sequence"].iloc[0]
        for p in PERCENTILES
    ]
    ax.bar(x - width / 2, official_means, width, label="Official", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, reference_means, width, label="Reference", color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in PERCENTILES])
    ax.set_title(
        "Mean Peak Timesteps per Sequence\n"
        f"({INPUT_WINDOW}→{FORECAST_HORIZON} Sliding Windows)",
        fontsize=13,
    )
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel("Mean Peak Timesteps (of 96)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    selected = np.load(
        REPO_ROOT / "data/selected_containers.npy",
        allow_pickle=True,
    )
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as handle:
        scalers = pickle.load(handle)

    evaluable = _evaluable_containers(selected, train_df)
    global_train = train_df[train_df["container_id"].isin(evaluable)].copy()
    container_thresholds = _container_thresholds(global_train, scalers)

    all_ids = set(evaluable)
    reference_ids = all_ids - set(NEAR_IDLE_CONTAINERS)

    official_counts = _peak_counts_per_sequence(
        global_train, scalers, container_thresholds, container_filter=all_ids
    )
    reference_counts = _peak_counts_per_sequence(
        global_train, scalers, container_thresholds, container_filter=reference_ids
    )

    official_rows = [
        _summarize_counts(official_counts[p], p, "official_all_containers")
        for p in PERCENTILES
    ]
    reference_rows = [
        _summarize_counts(reference_counts[p], p, "reference_excl_near_idle")
        for p in PERCENTILES
    ]

    official_summary = pd.DataFrame(official_rows)
    reference_summary = pd.DataFrame(reference_rows)

    official_summary.to_csv(EXPERIMENT_DIR / "sequence_peak_stats.csv", index=False)
    reference_summary.to_csv(
        EXPERIMENT_DIR / "sequence_peak_stats_reference.csv",
        index=False,
    )

    dist_frames = []
    for cohort_label, counts in [
        ("official_all_containers", official_counts),
        ("reference_excl_near_idle", reference_counts),
    ]:
        for percentile in PERCENTILES:
            dist_frames.append(
                _distribution_df(counts[percentile], percentile, cohort_label)
            )
    distribution = pd.concat(dist_frames, ignore_index=True)
    distribution.to_csv(
        EXPERIMENT_DIR / "sequence_peak_distribution.csv",
        index=False,
    )

    per_container_official = _per_container_sequence_df(
        global_train,
        scalers,
        container_thresholds,
        cohort_label="official_all_containers",
        container_filter=all_ids,
    )
    per_container_reference = _per_container_sequence_df(
        global_train,
        scalers,
        container_thresholds,
        cohort_label="reference_excl_near_idle",
        container_filter=reference_ids,
    )
    per_container = pd.concat(
        [per_container_official, per_container_reference],
        ignore_index=True,
    )
    per_container.to_csv(
        EXPERIMENT_DIR / "per_container_sequence_stats.csv",
        index=False,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_peak_sequence_rates(
        official_summary,
        reference_summary,
        FIGURES_DIR / "task3_peak_sequence_rate",
    )
    _plot_mean_peak_steps(
        official_summary,
        reference_summary,
        FIGURES_DIR / "task3_mean_peak_timesteps_per_sequence",
    )
    for percentile in PERCENTILES:
        _plot_peak_step_distribution(
            distribution,
            percentile,
            "official_all_containers",
            FIGURES_DIR / f"task3_peak_step_distribution_p{percentile}_official",
        )
        _plot_peak_step_distribution(
            distribution,
            percentile,
            "reference_excl_near_idle",
            FIGURES_DIR / f"task3_peak_step_distribution_p{percentile}_reference",
        )

    task3_results = {
        "task": 3,
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "near_idle_excluded_reference": list(NEAR_IDLE_CONTAINERS),
        "official": official_rows,
        "reference_excl_near_idle": reference_rows,
    }
    with open(EXPERIMENT_DIR / "task3_results.json", "w") as handle:
        json.dump(task3_results, handle, indent=2)

    print("=== Phase 2 Task 3 — Peak Sequence Analysis ===")
    print()
    print("--- Official (99 containers) ---")
    print(official_summary.to_string(index=False))
    print()
    print("--- Reference (94 containers, excl. near-idle) ---")
    print(reference_summary.to_string(index=False))


if __name__ == "__main__":
    main()
