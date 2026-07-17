#!/usr/bin/env python3
"""Build Phase 5 Task 4 comparison tables and plots from saved evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(REPO_ROOT))

from utils.peak_config import BASELINE_REFERENCE_DIR  # noqa: E402

COLOR_BASELINE = "#4C72B0"
COLOR_PEAK_AWARE = "#C44E52"
COLOR_ACTUAL = "#2CA02C"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build comparison tables and plots for peak-aware evaluation.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "experiments/peak_aware_2026-07-14_164518",
        help="Peak-aware experiment directory.",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_summary_mean(path: Path) -> dict[str, float]:
    summary = pd.read_csv(path, index_col=0)
    return {
        "day1_mae": float(summary.loc["mean", "day1_mae"]),
        "day1_rmse": float(summary.loc["mean", "day1_rmse"]),
        "day1_mape": float(summary.loc["mean", "day1_mape"]),
    }


def build_comparison_table(
    frozen_baseline_summary: dict[str, float],
    treatment_summary: dict[str, float],
    peak_subset_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Overall + peak-subset metrics with treatment − control deltas."""
    rows: list[dict[str, object]] = []

    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        baseline = frozen_baseline_summary[metric]
        treatment = treatment_summary[metric]
        rows.append(
            {
                "scope": "overall",
                "metric": metric,
                "baseline": baseline,
                "peak_aware": treatment,
                "delta_treatment_minus_baseline": treatment - baseline,
            }
        )

    baseline_peak = peak_subset_summary.loc[
        peak_subset_summary["model"] == "baseline"
    ].iloc[0]
    treatment_peak = peak_subset_summary.loc[
        peak_subset_summary["model"] == "peak_aware"
    ].iloc[0]

    for metric in ("peak_mae", "peak_rmse", "non_peak_mae", "non_peak_rmse"):
        baseline = float(baseline_peak[metric])
        treatment = float(treatment_peak[metric])
        rows.append(
            {
                "scope": "peak_subset",
                "metric": metric,
                "baseline": baseline,
                "peak_aware": treatment,
                "delta_treatment_minus_baseline": treatment - baseline,
            }
        )

    rows.append(
        {
            "scope": "peak_subset",
            "metric": "peak_step_fraction",
            "baseline": float(baseline_peak["peak_step_fraction"]),
            "peak_aware": float(treatment_peak["peak_step_fraction"]),
            "delta_treatment_minus_baseline": 0.0,
        }
    )

    return pd.DataFrame(rows)


def build_per_container_delta(
    treatment_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Paired per-container overall deltas (treatment − baseline re-run)."""
    merged = treatment_df.merge(
        baseline_df,
        on="container_id",
        suffixes=("_treatment", "_baseline"),
    )
    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_treatment"] - merged[f"{metric}_baseline"]
        )
    return merged.sort_values("container_id").reset_index(drop=True)


def build_evaluation_summary_json(
    comparison_table: pd.DataFrame,
    per_container_delta: pd.DataFrame,
    experiment_dir: Path,
) -> dict[str, object]:
    """Machine-readable comparison record for Phase 5 lock."""
    overall = comparison_table[comparison_table["scope"] == "overall"]
    peak_subset = comparison_table[comparison_table["scope"] == "peak_subset"]

    return {
        "generated_at": _iso_timestamp(),
        "phase": 5,
        "task": 4,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "baseline_reference": str(BASELINE_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "containers_evaluated": int(len(per_container_delta)),
        "overall_metrics": {
            row["metric"]: {
                "baseline": row["baseline"],
                "peak_aware": row["peak_aware"],
                "delta_treatment_minus_baseline": row[
                    "delta_treatment_minus_baseline"
                ],
            }
            for _, row in overall.iterrows()
        },
        "peak_subset_metrics": {
            row["metric"]: {
                "baseline": row["baseline"],
                "peak_aware": row["peak_aware"],
                "delta_treatment_minus_baseline": row[
                    "delta_treatment_minus_baseline"
                ],
            }
            for _, row in peak_subset.iterrows()
        },
        "per_container_delta_summary": {
            "delta_day1_mae": {
                "mean": float(per_container_delta["delta_day1_mae"].mean()),
                "std": float(per_container_delta["delta_day1_mae"].std()),
                "min": float(per_container_delta["delta_day1_mae"].min()),
                "max": float(per_container_delta["delta_day1_mae"].max()),
                "containers_improved": int(
                    (per_container_delta["delta_day1_mae"] < 0).sum()
                ),
                "containers_worsened": int(
                    (per_container_delta["delta_day1_mae"] > 0).sum()
                ),
            },
            "delta_day1_rmse": {
                "mean": float(per_container_delta["delta_day1_rmse"].mean()),
                "std": float(per_container_delta["delta_day1_rmse"].std()),
                "min": float(per_container_delta["delta_day1_rmse"].min()),
                "max": float(per_container_delta["delta_day1_rmse"].max()),
            },
        },
        "outputs": {
            "comparison_table": "evaluation/comparison_table.csv",
            "per_container_delta": "evaluation/per_container_delta.csv",
            "plots_dir": "evaluation/plots/",
        },
    }


def _load_inference_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _select_plot_containers(per_container_delta: pd.DataFrame) -> list[str]:
    """Pick representative containers for actual-vs-predicted panels."""
    valid = per_container_delta.dropna(subset=["delta_day1_mae"])
    median_idx = (valid["delta_day1_mae"] - valid["delta_day1_mae"].median()).abs()
    median_container = valid.loc[median_idx.idxmin(), "container_id"]

    best = valid.loc[valid["delta_day1_mae"].idxmin(), "container_id"]
    worst = valid.loc[valid["delta_day1_mae"].idxmax(), "container_id"]

    ordered = []
    for container_id in (median_container, best, worst):
        if container_id not in ordered:
            ordered.append(container_id)
    return ordered[:3]


def plot_actual_vs_predicted(
    container_ids: list[str],
    treatment_cache: dict[str, dict[str, np.ndarray]],
    baseline_cache: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> list[str]:
    """Day-1 actual vs predicted for sample containers (both models)."""
    saved: list[str] = []
    n_containers = len(container_ids)
    fig, axes = plt.subplots(n_containers, 1, figsize=(10, 3.2 * n_containers))
    if n_containers == 1:
        axes = [axes]

    for ax, container_id in zip(axes, container_ids):
        actual = treatment_cache[container_id]["actual_day1_real"]
        treatment = treatment_cache[container_id]["day1_final_real"]
        baseline = baseline_cache[container_id]["day1_final_real"]
        steps = np.arange(1, len(actual) + 1)

        ax.plot(steps, actual, color=COLOR_ACTUAL, linewidth=2, label="Actual")
        ax.plot(
            steps,
            baseline,
            color=COLOR_BASELINE,
            linewidth=1.5,
            linestyle="--",
            label="Baseline predicted",
        )
        ax.plot(
            steps,
            treatment,
            color=COLOR_PEAK_AWARE,
            linewidth=1.5,
            linestyle="-.",
            label="Peak-aware predicted",
        )
        ax.set_title(f"Day-1 Forecast — {container_id}", fontsize=11)
        ax.set_xlabel("Timestep (15-min)")
        ax.set_ylabel("CPU utilization")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Actual vs Predicted — Sample Containers",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_base = out_dir / "actual_vs_predicted_samples"
    fig.savefig(out_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    saved.extend([str(out_base.with_suffix(ext)) for ext in (".png", ".pdf")])
    return saved


def plot_error_distribution(
    per_container_delta: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    """Histogram of per-container ΔMAE (treatment − baseline)."""
    deltas = per_container_delta["delta_day1_mae"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=20, color=COLOR_BASELINE, edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="No change")
    ax.axvline(
        deltas.mean(),
        color=COLOR_PEAK_AWARE,
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ΔMAE = {deltas.mean():.4f}",
    )
    ax.set_title(
        "Per-Container Day-1 MAE Delta (Peak-Aware − Baseline)",
        fontsize=12,
    )
    ax.set_xlabel("ΔMAE (treatment − baseline)")
    ax.set_ylabel("Container count")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out_base = out_dir / "error_distribution"
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(out_base.with_suffix(ext)) for ext in (".png", ".pdf")]


def plot_peak_subset_comparison(
    comparison_table: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    """Grouped bar chart for pooled peak vs non-peak MAE/RMSE."""
    subset = comparison_table[
        (comparison_table["scope"] == "peak_subset")
        & comparison_table["metric"].isin(
            ["peak_mae", "peak_rmse", "non_peak_mae", "non_peak_rmse"]
        )
    ].copy()
    labels = ["Peak MAE", "Peak RMSE", "Non-peak MAE", "Non-peak RMSE"]
    metric_keys = ["peak_mae", "peak_rmse", "non_peak_mae", "non_peak_rmse"]
    baseline_vals = [float(subset.loc[subset["metric"] == m, "baseline"].iloc[0]) for m in metric_keys]
    treatment_vals = [
        float(subset.loc[subset["metric"] == m, "peak_aware"].iloc[0]) for m in metric_keys
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color=COLOR_BASELINE,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        treatment_vals,
        width,
        label="Peak-aware",
        color=COLOR_PEAK_AWARE,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Pooled Peak-Subset Metrics — Baseline vs Peak-Aware", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out_base = out_dir / "peak_subset_comparison"
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(out_base.with_suffix(ext)) for ext in (".png", ".pdf")]


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    evaluation_dir = experiment_dir / "evaluation"
    plots_dir = evaluation_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frozen_baseline_path = (
        BASELINE_REFERENCE_DIR / "evaluation" / "evaluation_summary.csv"
    )
    treatment_summary_path = evaluation_dir / "evaluation_summary.csv"
    baseline_rerun_summary_path = evaluation_dir / "baseline_evaluation_summary.csv"
    treatment_df_path = evaluation_dir / "evaluation_df.csv"
    baseline_df_path = evaluation_dir / "baseline_evaluation_df.csv"
    peak_subset_summary_path = evaluation_dir / "peak_subset_summary.csv"
    treatment_cache_path = evaluation_dir / "treatment_inference_cache.pkl"
    baseline_cache_path = evaluation_dir / "baseline_inference_cache.pkl"

    print("=" * 72)
    print("Peak-Aware Hybrid — Build Comparison Tables + Plots (Task 4)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")

    frozen_baseline_summary = _read_summary_mean(frozen_baseline_path)
    treatment_summary = _read_summary_mean(treatment_summary_path)
    baseline_rerun_summary = _read_summary_mean(baseline_rerun_summary_path)
    peak_subset_summary = pd.read_csv(peak_subset_summary_path)
    treatment_df = pd.read_csv(treatment_df_path)
    baseline_df = pd.read_csv(baseline_df_path)

    comparison_table = build_comparison_table(
        frozen_baseline_summary,
        treatment_summary,
        peak_subset_summary,
    )
    per_container_delta = build_per_container_delta(treatment_df, baseline_df)
    evaluation_summary = build_evaluation_summary_json(
        comparison_table,
        per_container_delta,
        experiment_dir,
    )

    comparison_path = evaluation_dir / "comparison_table.csv"
    delta_path = evaluation_dir / "per_container_delta.csv"
    summary_json_path = evaluation_dir / "evaluation_summary.json"

    comparison_table.to_csv(comparison_path, index=False)
    per_container_delta.to_csv(delta_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation_summary, handle, indent=2)

    treatment_cache = _load_inference_cache(treatment_cache_path)
    baseline_cache = _load_inference_cache(baseline_cache_path)
    sample_containers = _select_plot_containers(per_container_delta)

    plot_paths: list[str] = []
    plot_paths.extend(
        plot_actual_vs_predicted(
            sample_containers,
            treatment_cache,
            baseline_cache,
            plots_dir,
        )
    )
    plot_paths.extend(plot_error_distribution(per_container_delta, plots_dir))
    plot_paths.extend(plot_peak_subset_comparison(comparison_table, plots_dir))

    evaluation_summary["plot_outputs"] = [
        str(Path(path).relative_to(REPO_ROOT)) for path in plot_paths
    ]
    evaluation_summary["sample_plot_containers"] = sample_containers
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation_summary, handle, indent=2)

    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        with open(workspace_path, encoding="utf-8") as handle:
            workspace = json.load(handle)
        workspace["tasks"]["task_4"] = "complete"
        workspace["comparison_outputs"] = evaluation_summary["outputs"]
        workspace["comparison_generated_at"] = evaluation_summary["generated_at"]
        with open(workspace_path, "w", encoding="utf-8") as handle:
            json.dump(workspace, handle, indent=2)

    print()
    print("Overall (frozen baseline reference vs peak-aware):")
    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        row = comparison_table[
            (comparison_table["scope"] == "overall")
            & (comparison_table["metric"] == metric)
        ].iloc[0]
        print(
            f"  {metric:12s}: baseline={row['baseline']:.6f}  "
            f"peak-aware={row['peak_aware']:.6f}  "
            f"delta={row['delta_treatment_minus_baseline']:+.6f}"
        )

    print()
    print("Baseline re-run MAE (paired control):", f"{baseline_rerun_summary['day1_mae']:.6f}")
    print("Per-container ΔMAE: mean =", f"{per_container_delta['delta_day1_mae'].mean():+.6f}")
    print()
    print("Saved:")
    print(f"  {comparison_path}")
    print(f"  {delta_path}")
    print(f"  {summary_json_path}")
    print(f"  {plots_dir}/ ({len(plot_paths)} plot files)")


if __name__ == "__main__":
    main()
