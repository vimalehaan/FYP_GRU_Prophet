#!/usr/bin/env python3
"""Phase 2 Task 5 — Validation sanity check using train-fitted thresholds."""

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
DAY1_HORIZON = 96
PERCENTILES: tuple[int, ...] = (85, 90, 95)
PERCENTILE_COLORS = {85: "#4C72B0", 90: "#DD8452", 95: "#55A868"}

NEAR_IDLE_CONTAINERS: tuple[str, ...] = (
    "c_13308",
    "c_14106",
    "c_15035",
    "c_15446",
    "c_15794",
)

# Minimum rates for labelling validation peak evaluation as feasible.
MIN_VAL_PEAK_TIMESTEP_PCT = 2.0
MIN_VAL_DAY1_CONTAINER_PEAK_PCT = 50.0


def _cpu_real(series_scaled: pd.Series, scaler) -> np.ndarray:
    values = series_scaled.values.reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def _evaluable_containers(
    selected: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[str]:
    train_ids = set(train_df["container_id"].unique())
    val_ids = set(val_df["container_id"].unique())
    return [cid for cid in selected if cid in train_ids and cid in val_ids]


def _train_thresholds(
    global_train: pd.DataFrame,
    scalers: dict,
) -> dict[str, dict[int, float]]:
    thresholds: dict[str, dict[int, float]] = {}
    for container_id, group in global_train.groupby("container_id"):
        cpu_real = _cpu_real(
            group.sort_values("time_stamp")["cpu_scaled"],
            scalers[container_id],
        )
        thresholds[container_id] = {
            p: float(np.percentile(cpu_real, p)) for p in PERCENTILES
        }
    return thresholds


def _val_timestep_stats(
    global_val: pd.DataFrame,
    scalers: dict,
    thresholds: dict[str, dict[int, float]],
    container_filter: set[str],
) -> dict[int, dict[str, float | int]]:
    results: dict[int, dict[str, float | int]] = {}
    total_timesteps = 0
    peak_counts = {p: 0 for p in PERCENTILES}

    for container_id, group in global_val.groupby("container_id"):
        if container_id not in container_filter:
            continue
        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])
        total_timesteps += len(cpu_real)
        for percentile in PERCENTILES:
            threshold = thresholds[container_id][percentile]
            peak_counts[percentile] += int(np.sum(cpu_real >= threshold))

    for percentile in PERCENTILES:
        results[percentile] = {
            "val_timesteps": total_timesteps,
            "val_peak_timesteps": peak_counts[percentile],
            "val_peak_timestep_pct": (
                peak_counts[percentile] / total_timesteps * 100.0
                if total_timesteps else 0.0
            ),
        }
    return results


def _val_sliding_sequences(
    global_val: pd.DataFrame,
    scalers: dict,
    thresholds: dict[str, dict[int, float]],
    container_filter: set[str],
) -> dict[int, dict[str, float | int]]:
    """96→96 sliding windows on validation only (requires len >= 192)."""
    results: dict[int, dict[str, float | int]] = {
        p: {
            "val_total_sequences": 0,
            "val_peak_sequences": 0,
            "val_peak_sequence_pct": 0.0,
        }
        for p in PERCENTILES
    }

    for container_id, group in global_val.groupby("container_id"):
        if container_id not in container_filter:
            continue
        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])
        max_start = len(cpu_real) - INPUT_WINDOW - FORECAST_HORIZON
        if max_start < 0:
            continue

        for start in range(max_start):
            horizon = cpu_real[
                start + INPUT_WINDOW : start + INPUT_WINDOW + FORECAST_HORIZON
            ]
            for percentile in PERCENTILES:
                threshold = thresholds[container_id][percentile]
                peak_count = int(np.sum(horizon >= threshold))
                results[percentile]["val_total_sequences"] += 1
                if peak_count >= 1:
                    results[percentile]["val_peak_sequences"] += 1

    for percentile in PERCENTILES:
        total = int(results[percentile]["val_total_sequences"])
        peaks = int(results[percentile]["val_peak_sequences"])
        results[percentile]["val_peak_sequence_pct"] = (
            peaks / total * 100.0 if total else 0.0
        )
    return results


def _val_day1_horizon_stats(
    global_val: pd.DataFrame,
    scalers: dict,
    thresholds: dict[str, dict[int, float]],
    container_filter: set[str],
) -> tuple[dict[int, dict[str, float | int]], pd.DataFrame]:
    """
    Evaluation-aligned sanity check: first 96 validation timesteps per container.

    Mirrors the baseline Day-1 evaluation horizon (one horizon per container).
    """
    rows: list[dict] = []
    horizon_peak_counts = {p: 0 for p in PERCENTILES}
    containers_with_peak = {p: 0 for p in PERCENTILES}
    total_peak_steps = {p: 0 for p in PERCENTILES}
    n_containers = 0

    for container_id, group in global_val.groupby("container_id"):
        if container_id not in container_filter:
            continue
        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])
        horizon = cpu_real[:DAY1_HORIZON]
        if len(horizon) == 0:
            continue
        n_containers += 1
        row: dict = {
            "container_id": container_id,
            "val_timesteps_in_horizon": len(horizon),
        }
        for percentile in PERCENTILES:
            threshold = thresholds[container_id][percentile]
            peak_steps = int(np.sum(horizon >= threshold))
            has_peak = peak_steps >= 1
            row[f"day1_peak_steps_p{percentile}"] = peak_steps
            row[f"day1_has_peak_p{percentile}"] = int(has_peak)
            total_peak_steps[percentile] += peak_steps
            if has_peak:
                containers_with_peak[percentile] += 1
                horizon_peak_counts[percentile] += 1
        rows.append(row)

    per_container = pd.DataFrame(rows).sort_values("container_id")
    summary = {}
    for percentile in PERCENTILES:
        total_steps = n_containers * DAY1_HORIZON
        summary[percentile] = {
            "val_day1_containers": n_containers,
            "val_day1_containers_with_peak": containers_with_peak[percentile],
            "val_day1_container_peak_pct": (
                containers_with_peak[percentile] / n_containers * 100.0
                if n_containers else 0.0
            ),
            "val_day1_peak_timestep_pct": (
                total_peak_steps[percentile] / total_steps * 100.0
                if total_steps else 0.0
            ),
            "val_day1_horizons_with_peak": horizon_peak_counts[percentile],
        }
    return summary, per_container


def _plot_train_vs_val_timestep(
    sanity: pd.DataFrame,
    cohort_label: str,
    out_base: Path,
) -> None:
    subset = sanity[sanity["cohort"] == cohort_label]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERCENTILES))
    width = 0.35
    ax.bar(
        x - width / 2,
        subset["train_peak_timestep_pct"],
        width,
        label="Train",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        subset["val_peak_timestep_pct"],
        width,
        label="Validation",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in PERCENTILES])
    title = (
        "Official (99 containers)"
        if cohort_label == "official_all_containers"
        else "Reference (excl. near-idle)"
    )
    ax.set_title(
        f"Train vs Validation Peak Timestep Rate\n{title}",
        fontsize=13,
    )
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel("Peak Timesteps (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(
        REPO_ROOT / "data/selected_containers.npy",
        allow_pickle=True,
    )
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as handle:
        scalers = pickle.load(handle)

    train_summary = pd.read_csv(EXPERIMENT_DIR / "timestep_summary.csv")
    seq_summary = pd.read_csv(EXPERIMENT_DIR / "sequence_peak_stats.csv")

    evaluable = _evaluable_containers(selected, train_df, val_df)
    all_ids = set(evaluable)
    reference_ids = all_ids - set(NEAR_IDLE_CONTAINERS)

    global_train = train_df[train_df["container_id"].isin(evaluable)].copy()
    global_val = val_df[val_df["container_id"].isin(evaluable)].copy()
    thresholds = _train_thresholds(global_train, scalers)

    sanity_rows: list[dict] = []
    day1_frames: list[pd.DataFrame] = []

    for cohort_label, container_filter in [
        ("official_all_containers", all_ids),
        ("reference_excl_near_idle", reference_ids),
    ]:
        val_ts = _val_timestep_stats(
            global_val, scalers, thresholds, container_filter
        )
        val_sliding = _val_sliding_sequences(
            global_val, scalers, thresholds, container_filter
        )
        day1_summary, day1_per_container = _val_day1_horizon_stats(
            global_val, scalers, thresholds, container_filter
        )
        day1_per_container["cohort"] = cohort_label
        day1_frames.append(day1_per_container)

        for percentile in PERCENTILES:
            train_row = train_summary[
                train_summary["percentile"] == percentile
            ].iloc[0]
            train_seq_row = seq_summary[
                (seq_summary["cohort"] == "official_all_containers")
                & (seq_summary["percentile"] == percentile)
            ].iloc[0]

            val_peak_pct = float(val_ts[percentile]["val_peak_timestep_pct"])
            train_peak_pct = float(train_row["global_peak_timestep_pct"])
            ratio = (
                val_peak_pct / train_peak_pct
                if train_peak_pct > 0
                else np.nan
            )

            day1_container_peak_pct = float(
                day1_summary[percentile]["val_day1_container_peak_pct"]
            )
            feasible = int(
                val_peak_pct >= MIN_VAL_PEAK_TIMESTEP_PCT
                and day1_container_peak_pct >= MIN_VAL_DAY1_CONTAINER_PEAK_PCT
            )

            sanity_rows.append({
                "cohort": cohort_label,
                "percentile": percentile,
                "train_peak_timestep_pct": train_peak_pct,
                "val_peak_timestep_pct": val_peak_pct,
                "train_val_timestep_ratio": ratio,
                "train_total_sequences": int(
                    train_seq_row["total_sequences"]
                ),
                "val_sliding_sequences": int(
                    val_sliding[percentile]["val_total_sequences"]
                ),
                "train_peak_sequence_pct": float(
                    train_seq_row["peak_sequence_pct"]
                ),
                "val_sliding_peak_sequence_pct": float(
                    val_sliding[percentile]["val_peak_sequence_pct"]
                ),
                "val_day1_containers_with_peak_pct": day1_container_peak_pct,
                "val_day1_peak_timestep_pct": float(
                    day1_summary[percentile]["val_day1_peak_timestep_pct"]
                ),
                "val_sliding_sequences_feasible": int(
                    val_sliding[percentile]["val_total_sequences"] > 0
                ),
                "val_peak_evaluation_feasible": feasible,
            })

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(EXPERIMENT_DIR / "val_sanity_check.csv", index=False)

    day1_df = pd.concat(day1_frames, ignore_index=True)
    day1_df.to_csv(
        EXPERIMENT_DIR / "val_day1_horizon_sanity.csv",
        index=False,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_train_vs_val_timestep(
        sanity_df,
        "official_all_containers",
        FIGURES_DIR / "task5_train_vs_val_peak_timestep_official",
    )
    _plot_train_vs_val_timestep(
        sanity_df,
        "reference_excl_near_idle",
        FIGURES_DIR / "task5_train_vs_val_peak_timestep_reference",
    )

    task5_results = {
        "task": 5,
        "val_period_length_note": (
            "Validation period ~154 steps/container; insufficient for "
            "96+96=192 sliding-window sequences on val data alone."
        ),
        "evaluation_aligned_check": (
            "Day-1 horizon sanity uses first 96 val timesteps per container, "
            "matching baseline evaluation."
        ),
        "feasibility_criteria": {
            "min_val_peak_timestep_pct": MIN_VAL_PEAK_TIMESTEP_PCT,
            "min_val_day1_container_peak_pct": MIN_VAL_DAY1_CONTAINER_PEAK_PCT,
        },
        "sanity": sanity_rows,
    }
    with open(EXPERIMENT_DIR / "task5_results.json", "w") as handle:
        json.dump(task5_results, handle, indent=2)

    print("=== Phase 2 Task 5 — Validation Sanity Check ===")
    print()
    print(sanity_df.to_string(index=False))
    print()
    print(
        "Note: val_sliding_sequences = 0 because val length (~154) < "
        "input_window + forecast_horizon (192)."
    )


if __name__ == "__main__":
    main()
