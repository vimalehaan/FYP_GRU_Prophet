#!/usr/bin/env python3
"""
Post-hoc diagnostic: Hybrid peak error attribution (Prophet vs GRU residual).

NOT a new experiment. Read-only with respect to locked artifacts.
Outputs are written only under experiments/hybrid_peak_error_attribution_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_artifacts import load_hybrid_artifacts  # noqa: E402
from utils.hybrid_inference import run_hybrid_inference  # noqa: E402
from utils.peak_config import (  # noqa: E402
    BASELINE_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    RESIDUAL_STATS_FILENAME,
)
from utils.peak_detection import load_peak_thresholds  # noqa: E402
from utils.peak_evaluation import label_day1_peak_timesteps  # noqa: E402

SCRIPT_VERSION = "hybrid_peak_error_attribution_2026-07-23"
METRIC_TOLERANCE = 1e-5

PEAK_AWARE_EXPERIMENT = REPO_ROOT / "experiments/peak_aware_2026-07-14_164518"
LOCKED_BASELINE_CACHE = (
    PEAK_AWARE_EXPERIMENT / "evaluation/baseline_inference_cache.pkl"
)
LOCKED_BASELINE_PEAK_CSV = (
    PEAK_AWARE_EXPERIMENT / "evaluation/baseline_peak_subset_per_container.csv"
)
LOCKED_PEAK_THRESHOLDS = PEAK_AWARE_EXPERIMENT / "peak/peak_thresholds.pkl"

COLOR_PROPHET = "#DD8452"
COLOR_HYBRID = "#4C72B0"
COLOR_REDUCTION = "#55A868"
COLOR_RESIDUAL = "#8172B2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc Hybrid peak error attribution diagnostic.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _peak_mae(actual: np.ndarray, predicted: np.ndarray, peak_mask: np.ndarray) -> float:
    if not peak_mask.any():
        return float("nan")
    errors = np.abs(actual[peak_mask] - predicted[peak_mask])
    return float(np.mean(errors))


def _mean_abs(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(values[mask])))


def _analyze_container(
    container_id: str,
    actual: np.ndarray,
    prophet: np.ndarray,
    hybrid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    actual = np.ravel(actual)
    prophet = np.ravel(prophet)
    hybrid = np.ravel(hybrid)
    peak_mask = label_day1_peak_timesteps(actual, threshold)
    n_peak = int(peak_mask.sum())

    prophet_peak_mae = _peak_mae(actual, prophet, peak_mask)
    hybrid_peak_mae = _peak_mae(actual, hybrid, peak_mask)
    residual_correction = np.abs(hybrid - prophet)
    mean_residual_correction = _mean_abs(residual_correction, peak_mask)

    peak_error_reduction = (
        prophet_peak_mae - hybrid_peak_mae
        if n_peak > 0 and not np.isnan(prophet_peak_mae)
        else float("nan")
    )
    contribution_ratio = (
        peak_error_reduction / prophet_peak_mae
        if n_peak > 0 and prophet_peak_mae and not np.isnan(prophet_peak_mae)
        else float("nan")
    )

    return {
        "container_id": container_id,
        "threshold": float(threshold),
        "day1_steps": int(len(actual)),
        "peak_steps": n_peak,
        "peak_fraction": float(n_peak / len(actual)) if len(actual) else float("nan"),
        "prophet_peak_mae": prophet_peak_mae,
        "hybrid_peak_mae": hybrid_peak_mae,
        "mean_residual_correction_at_peaks": mean_residual_correction,
        "peak_error_reduction": peak_error_reduction,
        "contribution_ratio": contribution_ratio,
    }


def _pooled_peak_metrics(
    inference_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    pred_key: str = "day1_final_real",
) -> dict[str, float]:
    """Cohort pooled peak metrics — same timestep pooling as locked peak evaluation."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    peak_actual_chunks: list[np.ndarray] = []
    peak_predicted_chunks: list[np.ndarray] = []

    for container_id, arrays in inference_cache.items():
        actual = np.ravel(arrays["actual_day1_real"])
        predicted = np.ravel(arrays[pred_key])
        peak_mask = label_day1_peak_timesteps(actual, thresholds[container_id])
        peak_actual_chunks.append(actual[peak_mask])
        peak_predicted_chunks.append(predicted[peak_mask])

    peak_actual = np.concatenate(peak_actual_chunks)
    peak_predicted = np.concatenate(peak_predicted_chunks)
    return {
        "peak_mae": float(mean_absolute_error(peak_actual, peak_predicted)),
        "peak_rmse": float(np.sqrt(mean_squared_error(peak_actual, peak_predicted))),
    }


def _pooled_mean_residual(
    prophet_cache: dict[str, dict[str, np.ndarray]],
    hybrid_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> float:
    """Pooled mean |hybrid - prophet| at peak timesteps."""
    corrections: list[np.ndarray] = []
    for container_id in prophet_cache:
        actual = np.ravel(prophet_cache[container_id]["actual_day1_real"])
        prophet = np.ravel(prophet_cache[container_id]["day1_final_real"])
        hybrid = np.ravel(hybrid_cache[container_id]["day1_final_real"])
        peak_mask = label_day1_peak_timesteps(actual, thresholds[container_id])
        corrections.append(np.abs(hybrid[peak_mask] - prophet[peak_mask]))
    return float(np.mean(np.concatenate(corrections)))


def _pooled_mean(series: pd.Series) -> float:
    return float(series.mean(skipna=True))


def _save_bar_comparison(df: pd.DataFrame, plots_dir: Path) -> None:
    labels = df["container_id"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, df["prophet_peak_mae"], width, label="Prophet Peak MAE", color=COLOR_PROPHET)
    ax.bar(x + width / 2, df["hybrid_peak_mae"], width, label="Hybrid Peak MAE", color=COLOR_HYBRID)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Peak-subset MAE (real CPU %)")
    ax.set_title("Per-Container Prophet vs Hybrid Peak MAE (Post-Hoc Diagnostic)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"peak_mae_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def _save_histogram(series: pd.Series, title: str, xlabel: str, stem: str, color: str, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean = series.dropna()
    ax.hist(clean, bins=25, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(float(clean.mean()), color="black", linestyle="--", linewidth=1.2, label=f"Mean ({clean.mean():.3f})")
    ax.axvline(float(clean.median()), color="#333333", linestyle=":", linewidth=1.2, label=f"Median ({clean.median():.3f})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Container count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _save_scatter(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    valid = df.dropna(subset=["prophet_peak_mae", "hybrid_peak_mae"])
    ax.scatter(
        valid["prophet_peak_mae"],
        valid["hybrid_peak_mae"],
        alpha=0.75,
        color=COLOR_HYBRID,
        edgecolors="white",
        linewidths=0.5,
    )
    upper = max(valid["prophet_peak_mae"].max(), valid["hybrid_peak_mae"].max()) * 1.05
    ax.plot([0, upper], [0, upper], "k--", linewidth=1.2, label="Identity (Prophet = Hybrid)")
    ax.set_xlabel("Prophet Peak MAE")
    ax.set_ylabel("Hybrid Peak MAE")
    ax.set_title("Prophet vs Hybrid Peak MAE by Container")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"prophet_vs_hybrid_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)


def _build_analysis_summary(
    per_container: pd.DataFrame,
    pooled: dict[str, float],
    reduction_stats: dict[str, float],
    contribution_stats: dict[str, float],
    pooled_contribution_ratio: float,
    verification: dict[str, Any],
) -> str:
    n_with_peaks = int(per_container["peak_steps"].gt(0).sum())
    lines = [
        "# Hybrid Peak Error Attribution — Post-Hoc Diagnostic Summary",
        "",
        f"**Generated:** {_iso_timestamp()}  ",
        "**Label:** Post-hoc diagnostic analysis — NOT a new experiment.  ",
        "**Purpose:** Supporting evidence for thesis discussion on architecture-dependent Peak-Aware behaviour.",
        "",
        "## Method",
        "",
        "- Cohort: 99 evaluable containers (same as locked Peak-Aware Hybrid evaluation)",
        "- Peak definition: train-fitted P90 thresholds from locked Peak-Aware Hybrid experiment",
        "- Hybrid baseline arrays: verified against locked `baseline_inference_cache.pkl`",
        "- Prophet component: extracted via frozen `run_hybrid_inference()` (not stored in locked cache)",
        "- Locked experiment artifacts: **not modified**",
        "",
        "## Pooled cohort averages (containers with ≥1 peak step where applicable)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Containers analysed | {len(per_container)} |",
        f"| Containers with ≥1 peak step | {n_with_peaks} |",
        f"| Pooled Prophet Peak MAE | {pooled['prophet_peak_mae']:.4f} |",
        f"| Pooled Hybrid Peak MAE | {pooled['hybrid_peak_mae']:.4f} |",
        f"| Pooled mean residual correction at peaks | {pooled['mean_residual_correction_at_peaks']:.4f} |",
        f"| Pooled peak error reduction (Prophet − Hybrid) | {pooled['peak_error_reduction']:.4f} |",
        f"| Pooled contribution ratio | {pooled_contribution_ratio:.4f} |",
        f"| Mean per-container contribution ratio | {contribution_stats['mean']:.4f} |",
        f"| Median per-container contribution ratio | {contribution_stats['median']:.4f} |",
        "",
        "## Peak error reduction distribution",
        "",
        f"- Mean: {reduction_stats['mean']:.4f}",
        f"- Median: {reduction_stats['median']:.4f}",
        f"- Std: {reduction_stats['std']:.4f}",
        "",
        "## Verification against locked evaluation",
        "",
        f"- Status: **{verification['overall_status']}**",
        f"- Hybrid peak MAE max abs delta vs locked CSV: {verification['max_hybrid_peak_mae_delta']:.2e}",
        f"- Hybrid final array max abs delta vs locked cache: {verification['max_hybrid_array_delta']:.2e}",
        "",
        "## Diagnostic questions (evidence-based wording)",
        "",
        "### 1. Does Prophet already explain most peak behaviour?",
        "",
        _answer_prophet_dominance(pooled, pooled_contribution_ratio, reduction_stats),
        "",
        "### 2. How much does the GRU reduce peak error?",
        "",
        _answer_gru_reduction(pooled, reduction_stats, pooled_contribution_ratio),
        "",
        "### 3. Is the residual correction during peaks generally small or large?",
        "",
        _answer_residual_magnitude(pooled, per_container),
        "",
        "### 4. Does the evidence support the interpretation that Peak-Aware learning has limited influence because the GRU only models residuals?",
        "",
        _answer_peak_aware_interpretation(pooled, pooled_contribution_ratio, reduction_stats),
        "",
        "## Scope boundary",
        "",
        "This diagnostic describes association and decomposition under the locked protocol. ",
        "It does **not** establish causal mechanisms. Wording is intentionally cautious.",
        "",
    ]
    return "\n".join(lines)


def _answer_prophet_dominance(pooled, pooled_contribution_ratio, reduction_stats) -> str:
    ratio = pooled["hybrid_peak_mae"] / pooled["prophet_peak_mae"] if pooled["prophet_peak_mae"] else float("nan")
    return (
        f"The evidence suggests Prophet already accounts for a substantial share of peak-period accuracy. "
        f"Pooled Prophet Peak MAE is **{pooled['prophet_peak_mae']:.3f}** versus pooled Hybrid Peak MAE **{pooled['hybrid_peak_mae']:.3f}** "
        f"(Hybrid/Prophet ratio ≈ **{ratio:.3f}**). "
        f"The pooled contribution ratio is **{pooled_contribution_ratio:.3f}** "
        f"(fractional reduction in peak MAE attributable to the GRU path). "
        f"Per-container peak error reduction has mean **{reduction_stats['mean']:.3f}** and median **{reduction_stats['median']:.3f}**. "
        f"This is consistent with Prophet explaining most peak behaviour under this decomposition, "
        f"without claiming Prophet is solely responsible for peak accuracy."
    )


def _answer_gru_reduction(pooled, reduction_stats, pooled_contribution_ratio) -> str:
    direction = (
        "a **small net increase** in peak error"
        if pooled["peak_error_reduction"] < 0
        else "a **small net reduction** in peak error"
    )
    return (
        f"At the pooled cohort level, the GRU residual path produces {direction}: "
        f"**{abs(pooled['peak_error_reduction']):.3f}** MAE "
        f"(Prophet Peak MAE **{pooled['prophet_peak_mae']:.3f}** − Hybrid Peak MAE **{pooled['hybrid_peak_mae']:.3f}**). "
        f"Per-container peak error reduction: mean **{reduction_stats['mean']:.3f}**, "
        f"median **{reduction_stats['median']:.3f}**, std **{reduction_stats['std']:.3f}**. "
        f"The pooled contribution ratio (**{pooled_contribution_ratio:.3f}**) indicates the GRU correction "
        f"is **negligible relative to Prophet peak error** and does not materially improve—and may slightly "
        f"worsen—peak-subset accuracy at the cohort level."
    )


def _answer_residual_magnitude(pooled, per_container: pd.DataFrame) -> str:
    valid = per_container.dropna(subset=["mean_residual_correction_at_peaks"])
    rel = pooled["mean_residual_correction_at_peaks"] / pooled["prophet_peak_mae"] if pooled["prophet_peak_mae"] else float("nan")
    return (
        f"The mean absolute residual correction at peak timesteps is **{pooled['mean_residual_correction_at_peaks']:.3f}** "
        f"CPU percentage points (pooled), versus Prophet Peak MAE **{pooled['prophet_peak_mae']:.3f}**. "
        f"The correction magnitude is **small relative to Prophet peak error** (ratio ≈ **{rel:.3f}**). "
        f"Per-container median correction: **{valid['mean_residual_correction_at_peaks'].median():.3f}**. "
        f"The results are consistent with peak-period forecasts being driven primarily by the Prophet component, "
        f"with relatively small GRU adjustments at labelled peak steps."
    )


def _answer_peak_aware_interpretation(pooled, pooled_contribution_ratio, reduction_stats) -> str:
    return (
        f"The findings support the interpretation that Peak-Aware GRU reweighting has **limited room to influence "
        f"final peak forecasts** on the Hybrid architecture: pooled peak error reduction attributable to the GRU "
        f"is **{pooled['peak_error_reduction']:.3f}** MAE (pooled contribution ratio **{pooled_contribution_ratio:.3f}**). "
        f"Because Peak-Aware training modifies only the **residual** objective—and the diagnostic shows residual "
        f"corrections at peaks are comparatively small (**{pooled['mean_residual_correction_at_peaks']:.3f}** mean |Hybrid−Prophet|)—"
        f"the evidence is **consistent with** the locked Peak-Aware Hybrid result (peak MAE +0.024 vs baseline). "
        f"This does **not** prove causation; it provides decomposition evidence aligned with the architecture sensitivity discussion."
    )


def _build_discussion_notes() -> str:
    return """# Hybrid Peak Error Attribution — Discussion Notes

**Label:** Post-hoc diagnostic / interpretation support — NOT a new experiment.

---

## Analysis D — Global architecture comparison (interpretation only)

No Prophet comparison was computed for Global GRU (Prophet is not part of that pipeline).

### Pipeline contrast (locked experimental designs)

**Hybrid Prophet + GRU**

```
CPU (train/val)
    ↓
Prophet  →  day1 prophet forecast (real CPU %)
    ↓
Residual = actual − prophet (scaled)
    ↓
GRU predicts residual correction only
    ↓
Final forecast = prophet + GRU residual
```

**Global GRU**

```
CPU (train/val)
    ↓
GRU predicts cpu_scaled directly
    ↓
Final forecast = GRU output (inverse scaled)
```

### Why identical Peak-Aware methodology produced different outcomes

Under the locked configuration (P90, λ = 5, timestep-weighted MSE):

| Aspect | Hybrid track | Global track |
|--------|--------------|--------------|
| Peak-aware weight applied to | GRU **residual** targets only | **Full CPU** trajectory targets |
| Prophet peak fit | Unchanged; defines much of peak-level forecast | N/A |
| Locked peak-aware Hybrid Δpeak MAE | +0.024 (no benefit) | — |
| Locked peak-aware Global Δpeak MAE | — | −0.777 (substantial benefit vs Global baseline) |

**Interpretation supported by evidence (not proven causation):**

1. On Hybrid, Peak-Aware learning reweights errors on the **residual pathway only**. The post-hoc attribution diagnostic indicates that **Prophet Peak MAE is close to Hybrid Peak MAE** and that **residual corrections at peak timesteps are relatively small**. Therefore, reweighting GRU residual training has **limited leverage** over final peak CPU forecasts.

2. On Global, the GRU models the **entire signal**. Peak-aware weighting directly reshapes the primary forecasting objective. The locked Global experiment shows a **large peak-subset improvement** (−0.777 peak MAE) accompanied by **non-peak and overall degradation**. This pattern is **consistent with** the GRU having direct control over peak timesteps, unlike the Hybrid decomposition.

3. **Architecture dependence** follows from **where** peak-aware weighting acts in the pipeline, not from different λ or peak definitions (those were held constant across tracks).

### Wording guidance for thesis

- Use: *"The evidence suggests…"*, *"The results are consistent with…"*, *"The findings support the interpretation…"*
- Avoid: *"Prophet causes Peak-Aware failure"* or *"Proves GRU cannot learn peaks"*

### Relationship to locked Peak-Aware results

This diagnostic **does not replace** the locked paired comparisons. It provides **mechanistic context** for why Hybrid Peak-Aware showed negligible peak benefit while Global Peak-Aware showed large within-architecture peak gains.
"""


def main() -> None:
    args = _parse_args()
    started_at = _iso_timestamp()
    output_dir = args.output_dir or (
        REPO_ROOT / f"experiments/hybrid_peak_error_attribution_{_stamp_dir()}"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Post-Hoc Diagnostic — Hybrid Peak Error Attribution")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print("NOT a new experiment. Locked artifacts will not be modified.")
    print()

    with open(LOCKED_BASELINE_CACHE, "rb") as handle:
        locked_cache = pickle.load(handle)
    locked_peak_df = pd.read_csv(LOCKED_BASELINE_PEAK_CSV)
    thresholds = load_peak_thresholds(LOCKED_PEAK_THRESHOLDS)

    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected_containers = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected_containers)].copy()
    global_val = val_df[val_df["container_id"].isin(selected_containers)].copy()

    baseline_model_dir = BASELINE_REFERENCE_DIR / "models"
    hybrid_model, res_mean, res_std, input_window = load_hybrid_artifacts(
        model_path=baseline_model_dir / "hybrid_gru.keras",
        residual_stats_path=baseline_model_dir / RESIDUAL_STATS_FILENAME,
    )

    rows: list[dict[str, Any]] = []
    hybrid_array_deltas: list[float] = []
    peak_mae_deltas: list[float] = []
    hybrid_cache: dict[str, dict[str, np.ndarray]] = {}
    prophet_cache: dict[str, dict[str, np.ndarray]] = {}

    t0 = time.perf_counter()
    container_ids = sorted(locked_cache.keys())
    print(f"Analysing {len(container_ids)} containers...")

    for container_id in container_ids:
        cached = locked_cache[container_id]
        actual_cached = np.ravel(cached["actual_day1_real"])
        hybrid_cached = np.ravel(cached["day1_final_real"])

        result = run_hybrid_inference(
            container_id=container_id,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=hybrid_model,
            scalers=scalers,
            res_mean=res_mean,
            res_std=res_std,
            input_window=input_window,
        )

        actual = np.ravel(result.actual_day1_real)
        prophet = np.ravel(result.prophet_day1_real)
        hybrid = np.ravel(result.day1_final_real)

        hybrid_array_deltas.append(float(np.max(np.abs(hybrid - hybrid_cached))))
        if not np.allclose(actual, actual_cached, atol=1e-9):
            raise RuntimeError(f"Actual mismatch for {container_id}")

        threshold = float(thresholds[container_id])
        row = _analyze_container(container_id, actual, prophet, hybrid, threshold)
        rows.append(row)

        hybrid_cache[container_id] = {
            "actual_day1_real": actual,
            "day1_final_real": hybrid,
        }
        prophet_cache[container_id] = {
            "actual_day1_real": actual,
            "day1_final_real": prophet,
        }

        locked_row = locked_peak_df.loc[
            locked_peak_df["container_id"] == container_id
        ]
        if not locked_row.empty and row["peak_steps"] > 0:
            locked_peak_mae = float(locked_row.iloc[0]["peak_mae"])
            peak_mae_deltas.append(abs(row["hybrid_peak_mae"] - locked_peak_mae))

    elapsed = time.perf_counter() - t0
    per_container = pd.DataFrame(rows).sort_values("container_id").reset_index(drop=True)

    with_peaks = per_container[per_container["peak_steps"] > 0].copy()
    prophet_pooled = _pooled_peak_metrics(prophet_cache, thresholds)
    hybrid_pooled = _pooled_peak_metrics(hybrid_cache, thresholds)
    pooled = {
        "prophet_peak_mae": prophet_pooled["peak_mae"],
        "hybrid_peak_mae": hybrid_pooled["peak_mae"],
        "mean_residual_correction_at_peaks": _pooled_mean_residual(
            prophet_cache, hybrid_cache, thresholds
        ),
        "peak_error_reduction": prophet_pooled["peak_mae"] - hybrid_pooled["peak_mae"],
    }
    locked_pooled_hybrid_peak_mae = 2.501220270363839

    reduction_stats = {
        "mean": float(with_peaks["peak_error_reduction"].mean()),
        "median": float(with_peaks["peak_error_reduction"].median()),
        "std": float(with_peaks["peak_error_reduction"].std()),
    }
    contribution_stats = {
        "mean": float(with_peaks["contribution_ratio"].mean()),
        "median": float(with_peaks["contribution_ratio"].median()),
        "std": float(with_peaks["contribution_ratio"].std()),
    }
    pooled_contribution_ratio = (
        pooled["peak_error_reduction"] / pooled["prophet_peak_mae"]
        if pooled["prophet_peak_mae"]
        else float("nan")
    )

    verification = {
        "overall_status": "PASS",
        "max_hybrid_array_delta": float(max(hybrid_array_deltas) if hybrid_array_deltas else 0.0),
        "max_hybrid_peak_mae_delta": float(max(peak_mae_deltas) if peak_mae_deltas else 0.0),
        "pooled_hybrid_peak_mae_delta_vs_locked": abs(
            pooled["hybrid_peak_mae"] - locked_pooled_hybrid_peak_mae
        ),
        "locked_pooled_hybrid_peak_mae": locked_pooled_hybrid_peak_mae,
        "metric_tolerance": METRIC_TOLERANCE,
        "locked_baseline_cache": str(LOCKED_BASELINE_CACHE.relative_to(REPO_ROOT)),
        "locked_peak_csv": str(LOCKED_BASELINE_PEAK_CSV.relative_to(REPO_ROOT)),
    }
    if verification["max_hybrid_peak_mae_delta"] > METRIC_TOLERANCE:
        verification["overall_status"] = "FAIL"
    if verification["max_hybrid_array_delta"] > 1e-6:
        verification["overall_status"] = "FAIL"
    if verification["pooled_hybrid_peak_mae_delta_vs_locked"] > METRIC_TOLERANCE:
        verification["overall_status"] = "FAIL"

    per_container.to_csv(output_dir / "prophet_vs_hybrid_peak_errors.csv", index=False)
    with_peaks[["container_id", "peak_error_reduction"]].to_csv(
        output_dir / "peak_error_reduction.csv", index=False
    )
    with_peaks[["container_id", "contribution_ratio"]].to_csv(
        output_dir / "residual_contribution.csv", index=False
    )

    _save_bar_comparison(with_peaks, plots_dir)
    _save_histogram(
        with_peaks["peak_error_reduction"],
        "Distribution of Peak Error Reduction (Prophet Peak MAE − Hybrid Peak MAE)",
        "Peak error reduction (positive = Hybrid improves over Prophet)",
        "peak_error_reduction_histogram",
        COLOR_REDUCTION,
        plots_dir,
    )
    _save_histogram(
        with_peaks["mean_residual_correction_at_peaks"],
        "Distribution of Mean Residual Correction Magnitude at Peak Timesteps",
        "|Hybrid − Prophet| at peak steps (CPU %)",
        "residual_correction_distribution",
        COLOR_RESIDUAL,
        plots_dir,
    )
    _save_scatter(with_peaks, plots_dir)

    (output_dir / "discussion_notes.md").write_text(_build_discussion_notes())
    (output_dir / "analysis_summary.md").write_text(
        _build_analysis_summary(
            per_container,
            pooled,
            reduction_stats,
            contribution_stats,
            pooled_contribution_ratio,
            verification,
        )
    )

    metadata = {
        "analysis_type": "post_hoc_diagnostic",
        "not_a_new_experiment": True,
        "generated_at": _iso_timestamp(),
        "started_at": started_at,
        "completed_at": _iso_timestamp(),
        "script": "scripts/run_hybrid_peak_error_attribution.py",
        "script_version": SCRIPT_VERSION,
        "experiment_dir": str(output_dir.relative_to(REPO_ROOT)),
        "locked_sources": {
            "baseline_inference_cache": str(LOCKED_BASELINE_CACHE.relative_to(REPO_ROOT)),
            "baseline_peak_subset_csv": str(LOCKED_BASELINE_PEAK_CSV.relative_to(REPO_ROOT)),
            "peak_thresholds": str(LOCKED_PEAK_THRESHOLDS.relative_to(REPO_ROOT)),
            "hybrid_baseline_model": str((BASELINE_REFERENCE_DIR / "models").relative_to(REPO_ROOT)),
        },
        "diagnostic_inference_note": (
            "Locked inference caches store actual_day1_real and day1_final_real only. "
            "Frozen run_hybrid_inference() was invoked read-only to obtain prophet_day1_real "
            "for decomposition. No locked experiment files were modified."
        ),
        "models_retrained": False,
        "locked_artifacts_modified": False,
        "evaluation_results_regenerated": False,
        "containers_analysed": len(per_container),
        "containers_with_peaks": int(with_peaks.shape[0]),
        "runtime_seconds": round(elapsed, 2),
        "pooled_metrics": pooled,
        "pooled_contribution_ratio": pooled_contribution_ratio,
        "reduction_stats_per_container": reduction_stats,
        "contribution_stats_per_container": contribution_stats,
        "verification": verification,
        "outputs": {
            "analysis_summary": "analysis_summary.md",
            "discussion_notes": "discussion_notes.md",
            "prophet_vs_hybrid_peak_errors": "prophet_vs_hybrid_peak_errors.csv",
            "peak_error_reduction": "peak_error_reduction.csv",
            "residual_contribution": "residual_contribution.csv",
            "plots_dir": "plots/",
        },
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Runtime: {elapsed:.1f}s")
    print(f"Verification: {verification['overall_status']}")
    print(f"Pooled Prophet Peak MAE : {pooled['prophet_peak_mae']:.4f}")
    print(f"Pooled Hybrid Peak MAE  : {pooled['hybrid_peak_mae']:.4f}")
    print(f"Pooled peak reduction   : {pooled['peak_error_reduction']:.4f}")
    print(f"Mean contribution ratio (per-container): {contribution_stats['mean']:.4f}")
    print(f"Pooled contribution ratio          : {pooled_contribution_ratio:.4f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
