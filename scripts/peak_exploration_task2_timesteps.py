#!/usr/bin/env python3
"""Phase 2 Task 2 — Peak timestep analysis on frozen training data."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments/peak_exploration_2026-07-14"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

PERCENTILES: tuple[int, ...] = (85, 90, 95)
PERCENTILE_COLORS = {
    85: "#4C72B0",
    90: "#DD8452",
    95: "#55A868",
}


def _cpu_real(series_scaled: pd.Series, scaler) -> np.ndarray:
    """Inverse-transform scaled CPU to real utilization percent."""
    values = series_scaled.values.reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def _evaluable_containers(
    selected: np.ndarray,
    train_df: pd.DataFrame,
) -> list[str]:
    train_ids = set(train_df["container_id"].unique())
    return [cid for cid in selected if cid in train_ids]


def _per_container_rows(
    global_train: pd.DataFrame,
    scalers: dict,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for container_id, group in global_train.groupby("container_id"):
        group = group.sort_values("time_stamp")
        cpu_real = _cpu_real(group["cpu_scaled"], scalers[container_id])

        meta_row = metadata.loc[
            metadata["container_id"] == container_id
        ]
        pattern_type = (
            meta_row["pattern_type"].iloc[0]
            if not meta_row.empty
            else "unknown"
        )
        cv = (
            float(meta_row["cv"].iloc[0])
            if not meta_row.empty
            else np.nan
        )

        thresholds = {
            p: float(np.percentile(cpu_real, p)) for p in PERCENTILES
        }
        peak_pcts = {
            p: float(np.mean(cpu_real >= thresholds[p]) * 100.0)
            for p in PERCENTILES
        }

        rows.append({
            "container_id": container_id,
            "pattern_type": pattern_type,
            "cv": cv,
            "train_timesteps": len(group),
            "threshold_p85": thresholds[85],
            "threshold_p90": thresholds[90],
            "threshold_p95": thresholds[95],
            "peak_timestep_pct_p85": peak_pcts[85],
            "peak_timestep_pct_p90": peak_pcts[90],
            "peak_timestep_pct_p95": peak_pcts[95],
            "mean_cpu_real": float(np.mean(cpu_real)),
            "max_cpu_real": float(np.max(cpu_real)),
        })

    return pd.DataFrame(rows).sort_values("container_id").reset_index(
        drop=True
    )


def _timestep_summary(per_container: pd.DataFrame) -> pd.DataFrame:
    total_timesteps = int(per_container["train_timesteps"].sum())
    rows: list[dict] = []

    for percentile in PERCENTILES:
        col = f"peak_timestep_pct_p{percentile}"
        container_pcts = per_container[col]
        peak_timesteps = float(
            np.sum(
                per_container["train_timesteps"] * container_pcts / 100.0
            )
        )
        rows.append({
            "percentile": percentile,
            "global_peak_timestep_pct": peak_timesteps / total_timesteps * 100.0,
            "mean_container_peak_pct": float(container_pcts.mean()),
            "median_container_peak_pct": float(container_pcts.median()),
            "std_container_peak_pct": float(container_pcts.std()),
            "min_container_peak_pct": float(container_pcts.min()),
            "max_container_peak_pct": float(container_pcts.max()),
            "containers_with_zero_peaks": int((container_pcts == 0).sum()),
        })

    return pd.DataFrame(rows)


def _plot_global_peak_rates(summary: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PERCENTILE_COLORS[p] for p in summary["percentile"]]
    bars = ax.bar(
        [f"P{int(p)}" for p in summary["percentile"]],
        summary["global_peak_timestep_pct"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=11)
    ax.set_title(
        "Global Peak Timestep Rate by Percentile Threshold\n"
        "(Train Period, 99 Containers)",
        fontsize=13,
    )
    ax.set_xlabel("Per-Container Threshold Percentile", fontsize=12)
    ax.set_ylabel("Peak Timesteps (% of all train timesteps)", fontsize=12)
    ax.set_ylim(0, max(summary["global_peak_timestep_pct"]) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_container_distribution(
    per_container: pd.DataFrame,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [
        per_container["peak_timestep_pct_p85"],
        per_container["peak_timestep_pct_p90"],
        per_container["peak_timestep_pct_p95"],
    ]
    parts = ax.boxplot(
        data,
        tick_labels=["P85", "P90", "P95"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, percentile in zip(parts["boxes"], PERCENTILES):
        patch.set_facecolor(PERCENTILE_COLORS[percentile])
        patch.set_alpha(0.75)
    ax.set_title(
        "Per-Container Peak Timestep Rate Distribution\n"
        "(Train Period)",
        fontsize=13,
    )
    ax.set_xlabel("Threshold Percentile", fontsize=12)
    ax.set_ylabel("Peak Timesteps (% per container)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_top_containers(
    per_container: pd.DataFrame,
    percentile: int,
    out_base: Path,
    top_n: int = 15,
) -> None:
    col = f"peak_timestep_pct_p{percentile}"
    top = per_container.nlargest(top_n, col)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        PERCENTILE_COLORS.get(p, "#888888")
        for p in top["pattern_type"].map({
            "stable": 95,
            "medium": 90,
            "spiky": 85,
        })
    ]
    ax.barh(
        top["container_id"],
        top[col],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.invert_yaxis()
    ax.set_title(
        f"Top {top_n} Containers by Peak Timestep Rate (P{percentile})",
        fontsize=13,
    )
    ax.set_xlabel("Peak Timesteps (% of container train timesteps)", fontsize=12)
    ax.set_ylabel("Container ID", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
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
    metadata = pd.read_parquet(
        REPO_ROOT / "data/container_metadata.parquet"
    )

    evaluable = _evaluable_containers(selected, train_df)
    global_train = train_df[
        train_df["container_id"].isin(evaluable)
    ].copy()

    per_container = _per_container_rows(global_train, scalers, metadata)
    summary = _timestep_summary(per_container)

    per_container.to_csv(
        EXPERIMENT_DIR / "per_container_peak_stats.csv",
        index=False,
    )
    summary.to_csv(EXPERIMENT_DIR / "timestep_summary.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_global_peak_rates(
        summary,
        FIGURES_DIR / "task2_global_peak_timestep_rate",
    )
    _plot_container_distribution(
        per_container,
        FIGURES_DIR / "task2_per_container_peak_distribution",
    )
    _plot_top_containers(
        per_container,
        90,
        FIGURES_DIR / "task2_top_containers_p90",
    )

    task2_summary = {
        "task": 2,
        "name": "peak_timestep_analysis",
        "containers_analyzed": int(len(per_container)),
        "total_train_timesteps": int(per_container["train_timesteps"].sum()),
        "timestep_summary": summary.to_dict(orient="records"),
    }
    with open(EXPERIMENT_DIR / "task2_results.json", "w") as handle:
        json.dump(task2_summary, handle, indent=2)

    print("=== Phase 2 Task 2 — Peak Timestep Analysis ===")
    print(f"Containers analyzed : {len(per_container)}")
    print(f"Total train steps   : {per_container['train_timesteps'].sum()}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {EXPERIMENT_DIR / 'per_container_peak_stats.csv'}")
    print(f"Saved: {EXPERIMENT_DIR / 'timestep_summary.csv'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
