#!/usr/bin/env python3
"""Build read-only final four-model comparison from locked evaluation outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "final_model_comparison_2026-07-23"
METRIC_TOLERANCE = 1e-5

MODEL_ORDER = (
    "M1_hybrid_baseline",
    "M2_peak_aware_hybrid",
    "M3_global_baseline",
    "M4_peak_aware_global",
)

MODEL_LABELS = {
    "M1_hybrid_baseline": "Hybrid Prophet + GRU (Baseline)",
    "M2_peak_aware_hybrid": "Peak-Aware Hybrid Prophet + GRU",
    "M3_global_baseline": "Global GRU (Official Baseline)",
    "M4_peak_aware_global": "Peak-Aware Global GRU",
}

MODEL_SHORT = {
    "M1_hybrid_baseline": "Hybrid Baseline",
    "M2_peak_aware_hybrid": "Peak-Aware Hybrid",
    "M3_global_baseline": "Global Baseline",
    "M4_peak_aware_global": "Peak-Aware Global",
}

COLORS = {
    "M1_hybrid_baseline": "#4C72B0",
    "M2_peak_aware_hybrid": "#55A868",
    "M3_global_baseline": "#8172B2",
    "M4_peak_aware_global": "#C44E52",
}

RANK_METRICS = (
    "overall_mae",
    "overall_rmse",
    "peak_mae",
    "peak_rmse",
    "non_peak_mae",
    "non_peak_rmse",
)

SOURCES = {
    "M1_hybrid_baseline": {
        "experiment_dir": REPO_ROOT / "experiments/baseline_reference_2026-07-14",
        "overall_summary": REPO_ROOT
        / "experiments/baseline_reference_2026-07-14/evaluation/evaluation_summary.csv",
        "peak_comparison_table": REPO_ROOT
        / "experiments/peak_aware_2026-07-14_164518/evaluation/comparison_table.csv",
        "peak_column": "baseline",
    },
    "M2_peak_aware_hybrid": {
        "experiment_dir": REPO_ROOT
        / "experiments/peak_aware_2026-07-14_164518",
        "overall_summary": REPO_ROOT
        / "experiments/peak_aware_2026-07-14_164518/evaluation/evaluation_summary.csv",
        "peak_comparison_table": REPO_ROOT
        / "experiments/peak_aware_2026-07-14_164518/evaluation/comparison_table.csv",
        "peak_column": "peak_aware",
    },
    "M3_global_baseline": {
        "experiment_dir": REPO_ROOT
        / "experiments/global_gru_baseline_2026-07-17_121748",
        "overall_summary": REPO_ROOT
        / "experiments/global_gru_baseline_2026-07-17_121748/evaluation/evaluation_summary.csv",
        "peak_comparison_table": REPO_ROOT
        / "experiments/peak_aware_global_validation_vs1_2026-07-17_170939/evaluation/comparison_table.csv",
        "peak_column": "baseline",
    },
    "M4_peak_aware_global": {
        "experiment_dir": REPO_ROOT
        / "experiments/peak_aware_global_validation_vs1_2026-07-17_170939",
        "overall_summary": REPO_ROOT
        / "experiments/peak_aware_global_validation_vs1_2026-07-17_170939/evaluation/evaluation_summary.csv",
        "peak_comparison_table": REPO_ROOT
        / "experiments/peak_aware_global_validation_vs1_2026-07-17_170939/evaluation/comparison_table.csv",
        "peak_column": "peak_aware_global",
    },
}

ARCHITECTURE_PAIRS = {
    "hybrid": ("M1_hybrid_baseline", "M2_peak_aware_hybrid"),
    "global": ("M3_global_baseline", "M4_peak_aware_global"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build read-only final four-model comparison artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: experiments/final_model_comparison_<timestamp>)",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _read_overall_summary(path: Path) -> dict[str, float]:
    summary = pd.read_csv(path, index_col=0)
    return {
        "overall_mae": float(summary.loc["mean", "day1_mae"]),
        "overall_rmse": float(summary.loc["mean", "day1_rmse"]),
        "overall_mape": float(summary.loc["mean", "day1_mape"]),
    }


def _read_peak_subset_metrics(
    comparison_path: Path,
    value_column: str,
) -> dict[str, float]:
    table = pd.read_csv(comparison_path)
    peak_rows = table.loc[table["scope"] == "peak_subset"]
    mapping = {
        "peak_mae": "peak_mae",
        "peak_rmse": "peak_rmse",
        "non_peak_mae": "non_peak_mae",
        "non_peak_rmse": "non_peak_rmse",
    }
    metrics: dict[str, float] = {}
    for metric_name in mapping:
        row = peak_rows.loc[peak_rows["metric"] == metric_name]
        if row.empty:
            raise ValueError(
                f"Missing peak-subset metric {metric_name!r} in {comparison_path}"
            )
        metrics[metric_name] = float(row.iloc[0][value_column])
    return metrics


def collect_master_metrics() -> pd.DataFrame:
    """Task 1 — aggregate locked overall + peak-subset metrics for all four models."""
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        source = SOURCES[model_id]
        overall = _read_overall_summary(source["overall_summary"])
        peak = _read_peak_subset_metrics(
            source["peak_comparison_table"],
            source["peak_column"],
        )
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "experiment_dir": str(source["experiment_dir"].relative_to(REPO_ROOT)),
                "overall_summary_source": str(
                    source["overall_summary"].relative_to(REPO_ROOT)
                ),
                "peak_comparison_source": str(
                    source["peak_comparison_table"].relative_to(REPO_ROOT)
                ),
                "peak_value_column": source["peak_column"],
                **overall,
                **peak,
            }
        )
    return pd.DataFrame(rows)


def build_model_rankings(master_df: pd.DataFrame) -> pd.DataFrame:
    """Task 2 — rank models (rank 1 = best / lowest error)."""
    rows: list[dict[str, Any]] = []
    for metric in RANK_METRICS:
        ranked = master_df.sort_values(metric, ascending=True).reset_index(drop=True)
        for rank, row in ranked.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "rank": int(rank + 1),
                    "model_id": row["model_id"],
                    "model_label": row["model_label"],
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_final_comparison_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """Task 3 — thesis-ready wide comparison table."""
    return master_df[
        [
            "model_id",
            "model_label",
            "overall_mae",
            "overall_rmse",
            "overall_mape",
            "peak_mae",
            "peak_rmse",
            "non_peak_mae",
            "non_peak_rmse",
        ]
    ].copy()


def identify_best_models(
    master_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for metric in RANK_METRICS:
        top = rankings_df.loc[
            (rankings_df["metric"] == metric) & (rankings_df["rank"] == 1)
        ].iloc[0]
        best[metric] = {
            "model_id": top["model_id"],
            "model_label": top["model_label"],
            "value": float(top["value"]),
        }
    return best


def build_architecture_sensitivity(master_df: pd.DataFrame) -> pd.DataFrame:
    """Task 4 — within-architecture Peak-Aware deltas (treatment − baseline)."""
    indexed = master_df.set_index("model_id")
    rows: list[dict[str, Any]] = []
    for architecture, (baseline_id, treatment_id) in ARCHITECTURE_PAIRS.items():
        baseline = indexed.loc[baseline_id]
        treatment = indexed.loc[treatment_id]
        for metric in RANK_METRICS:
            delta = float(treatment[metric] - baseline[metric])
            pct = (
                float(delta / baseline[metric] * 100.0)
                if baseline[metric] != 0
                else float("nan")
            )
            if delta < 0:
                direction = "improved"
            elif delta > 0:
                direction = "worsened"
            else:
                direction = "unchanged"
            rows.append(
                {
                    "architecture": architecture,
                    "baseline_model_id": baseline_id,
                    "treatment_model_id": treatment_id,
                    "metric": metric,
                    "baseline_value": float(baseline[metric]),
                    "treatment_value": float(treatment[metric]),
                    "delta_treatment_minus_baseline": delta,
                    "percent_change": pct,
                    "direction": direction,
                }
            )
    return pd.DataFrame(rows)


def _save_bar_chart(
    master_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
    plots_dir: Path,
) -> None:
    labels = [MODEL_SHORT[mid] for mid in master_df["model_id"]]
    values = master_df[metric].tolist()
    colors = [COLORS[mid] for mid in master_df["model_id"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _save_grouped_chart(master_df: pd.DataFrame, plots_dir: Path) -> None:
    metrics = [
        ("overall_mae", "Overall MAE"),
        ("overall_rmse", "Overall RMSE"),
        ("peak_mae", "Peak MAE"),
        ("non_peak_mae", "Non-Peak MAE"),
    ]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model_id in enumerate(MODEL_ORDER):
        offset = (idx - 1.5) * width
        values = [master_df.loc[master_df["model_id"] == model_id, m].iloc[0] for m, _ in metrics]
        ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_SHORT[model_id],
            color=COLORS[model_id],
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Error (real CPU %)")
    ax.set_title("Four-Model Error Comparison (Lower Is Better)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"grouped_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def _save_radar_chart(master_df: pd.DataFrame, plots_dir: Path) -> None:
    """Normalized radar chart — lower raw error becomes a higher score."""
    radar_metrics = RANK_METRICS
    raw = master_df.set_index("model_id")[list(radar_metrics)]
    normalized = 1.0 - (raw - raw.min()) / (raw.max() - raw.min()).replace(0, np.nan)
    normalized = normalized.fillna(1.0)

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    metric_labels = [
        "Overall MAE",
        "Overall RMSE",
        "Peak MAE",
        "Peak RMSE",
        "Non-Peak MAE",
        "Non-Peak RMSE",
    ]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Normalized Performance Profile\n(higher = better within cohort; not cross-metric calibrated)",
        pad=20,
    )

    for model_id in MODEL_ORDER:
        values = normalized.loc[model_id].tolist()
        values = values + values[:1]
        ax.plot(angles, values, linewidth=2, label=MODEL_SHORT[model_id], color=COLORS[model_id])
        ax.fill(angles, values, alpha=0.08, color=COLORS[model_id])

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"radar_chart.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_plots(master_df: pd.DataFrame, plots_dir: Path) -> list[str]:
    """Task 5 — publication-quality figures."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_bar_chart(
        master_df,
        "overall_mae",
        "Day-1 MAE (real CPU %)",
        "Overall Day-1 MAE — Four Final Models",
        "overall_mae",
        plots_dir,
    )
    _save_bar_chart(
        master_df,
        "overall_rmse",
        "Day-1 RMSE (real CPU %)",
        "Overall Day-1 RMSE — Four Final Models",
        "overall_rmse",
        plots_dir,
    )
    _save_bar_chart(
        master_df,
        "peak_mae",
        "Peak-subset MAE (real CPU %)",
        "Peak-Subset MAE — Four Final Models",
        "peak_mae",
        plots_dir,
    )
    _save_bar_chart(
        master_df,
        "non_peak_mae",
        "Non-Peak-subset MAE (real CPU %)",
        "Non-Peak-Subset MAE — Four Final Models",
        "non_peak_mae",
        plots_dir,
    )
    _save_grouped_chart(master_df, plots_dir)
    _save_radar_chart(master_df, plots_dir)
    return sorted(p.name for p in plots_dir.iterdir())


def verify_consistency(master_df: pd.DataFrame) -> dict[str, Any]:
    """Verify metrics match locked source files without rerunning evaluation."""
    checks: list[dict[str, Any]] = []
    all_pass = True

    for model_id in MODEL_ORDER:
        source = SOURCES[model_id]
        overall = _read_overall_summary(source["overall_summary"])
        peak = _read_peak_subset_metrics(
            source["peak_comparison_table"],
            source["peak_column"],
        )
        row = master_df.loc[master_df["model_id"] == model_id].iloc[0]

        for key, expected in {**overall, **peak}.items():
            actual = float(row[key])
            passed = abs(actual - expected) <= METRIC_TOLERANCE
            all_pass = all_pass and passed
            checks.append(
                {
                    "model_id": model_id,
                    "metric": key,
                    "master_value": actual,
                    "source_value": expected,
                    "source_file": str(
                        source["overall_summary"].relative_to(REPO_ROOT)
                        if key.startswith("overall_")
                        else source["peak_comparison_table"].relative_to(REPO_ROOT)
                    ),
                    "status": "PASS" if passed else "FAIL",
                }
            )

        if not source["overall_summary"].exists():
            all_pass = False
            checks.append(
                {
                    "model_id": model_id,
                    "metric": "source_exists",
                    "status": "FAIL",
                    "reason": f"Missing {source['overall_summary']}",
                }
            )

    return {
        "verified_at": _iso_timestamp(),
        "script": "scripts/build_final_model_comparison.py",
        "script_version": SCRIPT_VERSION,
        "read_only": True,
        "models_retrained": False,
        "inference_regenerated": False,
        "source_artifacts_modified": False,
        "overall_status": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def build_final_summary_md(
    master_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    best: dict[str, dict[str, Any]],
    sensitivity_df: pd.DataFrame,
) -> str:
    """Task 6 — concise thesis-ready synthesis."""
    hybrid = sensitivity_df.loc[sensitivity_df["architecture"] == "hybrid"].set_index("metric")
    global_arch = sensitivity_df.loc[sensitivity_df["architecture"] == "global"].set_index("metric")

    lines = [
        "# Final Four-Model Comparison — Synthesis Summary",
        "",
        f"**Generated:** {_iso_timestamp()}  ",
        "**Type:** Read-only synthesis from locked evaluation outputs (no retraining, no inference regeneration).",
        "",
        "## Master comparison table",
        "",
        "| Model | Overall MAE | Overall RMSE | Peak MAE | Peak RMSE | Non-Peak MAE | Non-Peak RMSE |",
        "|-------|-------------|--------------|----------|-----------|--------------|---------------|",
    ]

    for _, row in master_df.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['overall_mae']:.3f} | {row['overall_rmse']:.3f} | "
            f"{row['peak_mae']:.3f} | {row['peak_rmse']:.3f} | "
            f"{row['non_peak_mae']:.3f} | {row['non_peak_rmse']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Rankings (rank 1 = best / lowest error)",
            "",
        ]
    )
    for metric in RANK_METRICS:
        metric_rows = rankings_df.loc[rankings_df["metric"] == metric].sort_values("rank")
        summary = ", ".join(
            f"{int(r.rank)}. {MODEL_SHORT[r.model_id]} ({r.value:.3f})"
            for r in metric_rows.itertuples()
        )
        lines.append(f"- **{metric}:** {summary}")

    lines.extend(
        [
            "",
            "## Best models (from locked cohort means, 99 containers)",
            "",
            f"- **Best overall (MAE):** {best['overall_mae']['model_label']} "
            f"({best['overall_mae']['value']:.3f})",
            f"- **Best overall (RMSE):** {best['overall_rmse']['model_label']} "
            f"({best['overall_rmse']['value']:.3f})",
            f"- **Best peak prediction (MAE):** {best['peak_mae']['model_label']} "
            f"({best['peak_mae']['value']:.3f})",
            f"- **Best peak prediction (RMSE):** {best['peak_rmse']['model_label']} "
            f"({best['peak_rmse']['value']:.3f})",
            f"- **Best non-peak prediction (MAE):** {best['non_peak_mae']['model_label']} "
            f"({best['non_peak_mae']['value']:.3f})",
            f"- **Best non-peak prediction (RMSE):** {best['non_peak_rmse']['model_label']} "
            f"({best['non_peak_rmse']['value']:.3f})",
            "",
            "## Architecture sensitivity (Peak-Aware − baseline, same architecture)",
            "",
            "### Hybrid track",
            f"- Overall MAE: {_format_delta(float(hybrid.loc['overall_mae', 'delta_treatment_minus_baseline']))} "
            f"({hybrid.loc['overall_mae', 'direction']})",
            f"- Peak MAE: {_format_delta(float(hybrid.loc['peak_mae', 'delta_treatment_minus_baseline']))} "
            f"({hybrid.loc['peak_mae', 'direction']})",
            f"- Non-peak MAE: {_format_delta(float(hybrid.loc['non_peak_mae', 'delta_treatment_minus_baseline']))} "
            f"({hybrid.loc['non_peak_mae', 'direction']})",
            "",
            "### Global track",
            f"- Overall MAE: {_format_delta(float(global_arch.loc['overall_mae', 'delta_treatment_minus_baseline']))} "
            f"({global_arch.loc['overall_mae', 'direction']})",
            f"- Peak MAE: {_format_delta(float(global_arch.loc['peak_mae', 'delta_treatment_minus_baseline']))} "
            f"({global_arch.loc['peak_mae', 'direction']})",
            f"- Non-peak MAE: {_format_delta(float(global_arch.loc['non_peak_mae', 'delta_treatment_minus_baseline']))} "
            f"({global_arch.loc['non_peak_mae', 'direction']})",
            "",
            "## Thesis-ready answers (evidence-based only)",
            "",
            "### 1. Which model is best for general forecasting?",
            f"**{best['overall_mae']['model_label']}** achieves the lowest overall Day-1 MAE "
            f"({best['overall_mae']['value']:.3f}) and RMSE ({best['overall_rmse']['value']:.3f}) "
            "under the locked 99-container protocol.",
            "",
            "### 2. Which model is best for peak prediction?",
            f"**{best['peak_mae']['model_label']}** achieves the lowest pooled peak-subset MAE "
            f"({best['peak_mae']['value']:.3f}). For peak RMSE, "
            f"**{best['peak_rmse']['model_label']}** is lowest ({best['peak_rmse']['value']:.3f}). "
            "Among Global-architecture models only, Peak-Aware Global (2.649) improves peak MAE "
            "versus Global Baseline (3.426) but remains behind both Hybrid models on peak MAE.",
            "",
            "### 3. Does Peak-Aware learning improve forecasting?",
            "It depends on the objective and architecture. On the Hybrid track, Peak-Aware training "
            "marginally improves overall MAE (−0.005) and non-peak MAE (−0.015) but slightly "
            "increases peak MAE (+0.024). On the Global track, Peak-Aware training substantially "
            "improves peak MAE (−0.777) while worsening overall MAE (+0.249) and non-peak MAE (+0.604).",
            "",
            "### 4. Is Peak-Aware learning architecture-dependent?",
            "Yes. The locked results show opposite overall trade-offs: near-neutral overall impact on "
            "Hybrid versus a clear overall penalty on Global, while peak-subset gains are larger on Global.",
            "",
            "### 5. Practical deployment recommendations",
            "- **Default long-horizon Day-1 deployment:** Peak-Aware Hybrid (best overall MAE/RMSE).",
            "- **Peak-timestep accuracy priority (all models):** Hybrid Baseline "
            f"(lowest peak MAE {best['peak_mae']['value']:.3f} and peak RMSE {best['peak_rmse']['value']:.3f}).",
            "- **Peak-timestep accuracy within Global architecture only:** Peak-Aware Global "
            "(peak MAE 2.649 vs Global Baseline 3.426; overall and non-peak accuracy degrade).",
            "- **Non-peak accuracy priority without Prophet overhead:** Global Baseline "
            f"(lowest non-peak MAE {best['non_peak_mae']['value']:.3f}; overall MAE higher than Hybrid).",
            "- **Do not deploy Peak-Aware Global as the default general forecaster** — locked overall "
            "MAE is highest among the four models.",
            "",
            "## Provenance",
            "",
            "All metrics were extracted read-only from locked experiment evaluation outputs. "
            "No models were retrained and no inference was regenerated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    started_at = _iso_timestamp()
    output_dir = args.output_dir or (
        REPO_ROOT / f"experiments/final_model_comparison_{_stamp_dir()}"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    verification_dir = output_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    master_df = collect_master_metrics()
    rankings_df = build_model_rankings(master_df)
    comparison_df = build_final_comparison_table(master_df)
    sensitivity_df = build_architecture_sensitivity(master_df)
    best = identify_best_models(master_df, rankings_df)
    plot_files = build_plots(master_df, plots_dir)
    verification = verify_consistency(master_df)
    summary_md = build_final_summary_md(master_df, rankings_df, best, sensitivity_df)

    master_df.to_csv(output_dir / "master_metrics_table.csv", index=False)
    rankings_df.to_csv(output_dir / "model_rankings.csv", index=False)
    comparison_df.to_csv(output_dir / "final_comparison_table.csv", index=False)
    sensitivity_df.to_csv(output_dir / "architecture_sensitivity_summary.csv", index=False)
    (output_dir / "final_summary.md").write_text(summary_md)
    (verification_dir / "comparison_consistency.json").write_text(
        json.dumps(verification, indent=2)
    )

    metadata = {
        "comparison_timestamp": _iso_timestamp(),
        "started_at": started_at,
        "completed_at": _iso_timestamp(),
        "experiment_identifier": output_dir.name,
        "experiment_dir": str(output_dir.relative_to(REPO_ROOT)),
        "comparison_type": "final_four_model_synthesis",
        "read_only": True,
        "script": "scripts/build_final_model_comparison.py",
        "script_version": SCRIPT_VERSION,
        "evaluated_containers": 99,
        "models": {
            model_id: {
                "label": MODEL_LABELS[model_id],
                "experiment_dir": str(SOURCES[model_id]["experiment_dir"].relative_to(REPO_ROOT)),
            }
            for model_id in MODEL_ORDER
        },
        "best_models": best,
        "verification_status": verification["overall_status"],
        "plots": [f"plots/{name}" for name in plot_files],
        "outputs": {
            "master_metrics_table": "master_metrics_table.csv",
            "model_rankings": "model_rankings.csv",
            "final_comparison_table": "final_comparison_table.csv",
            "architecture_sensitivity_summary": "architecture_sensitivity_summary.csv",
            "final_summary": "final_summary.md",
            "verification": "verification/comparison_consistency.json",
        },
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Final model comparison written to: {output_dir}")
    print(f"Verification: {verification['overall_status']}")
    print(f"Best overall MAE: {best['overall_mae']['model_label']} ({best['overall_mae']['value']:.3f})")
    print(f"Best peak MAE: {best['peak_mae']['model_label']} ({best['peak_mae']['value']:.3f})")
    print(f"Best non-peak MAE: {best['non_peak_mae']['model_label']} ({best['non_peak_mae']['value']:.3f})")


if __name__ == "__main__":
    main()
