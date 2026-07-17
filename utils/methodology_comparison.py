"""Read-only Hybrid vs Global GRU methodology comparison utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.global_config import (
    FORECAST_HORIZON,
    GLOBAL_GRU_REFERENCE_DIR,
    HYBRID_BASELINE_REFERENCE_DIR,
    INPUT_WINDOW,
    MAPE_EPSILON,
    MODEL_VARIANT,
    RANDOM_SEED,
    REPO_ROOT,
    STALE_SELECTED_CONTAINER_ID,
)

METHODOLOGY_HYBRID: str = "hybrid_prophet_gru"
METHODOLOGY_GLOBAL: str = "global_gru_v1"
DELTA_SIGN_CONVENTION: str = "global_minus_hybrid"
COMPARISON_SCRIPT: str = "scripts/build_hybrid_vs_global_comparison.py"
VERIFICATION_SCRIPT: str = "scripts/verify_hybrid_vs_global_comparison.py"
COMPARISON_SCRIPT_VERSION: str = "phase_6_task_2_2026-07-17"

METRIC_COLUMNS: tuple[str, ...] = ("day1_mae", "day1_rmse", "day1_mape")
EXPECTED_EVALUATED: int = 99
EVALUATION_DF_FILENAME: str = "evaluation_df.csv"
EVALUATION_SUMMARY_FILENAME: str = "evaluation_summary.csv"

# Authoritative Phase 5 evaluation run (immutable record).
GLOBAL_GRU_EVALUATION_RUN_DIR = (
    REPO_ROOT / "experiments/global_gru_evaluation_2026-07-17_092120"
)

__all__ = [
    "DELTA_SIGN_CONVENTION",
    "EXPECTED_EVALUATED",
    "GLOBAL_GRU_EVALUATION_RUN_DIR",
    "METHODOLOGY_GLOBAL",
    "METHODOLOGY_HYBRID",
    "assert_cohort_alignment",
    "build_comparison_metadata",
    "build_comparison_table",
    "build_experiment_config",
    "build_per_container_delta",
    "load_frozen_evaluation_dfs",
    "load_frozen_evaluation_summaries",
    "summarize_deltas",
]


def _evaluation_df_path(reference_dir: Path) -> Path:
    return reference_dir / "evaluation" / EVALUATION_DF_FILENAME


def _evaluation_summary_path(reference_dir: Path) -> Path:
    return reference_dir / "evaluation" / EVALUATION_SUMMARY_FILENAME


def load_frozen_evaluation_dfs(
    hybrid_reference_dir: Path = HYBRID_BASELINE_REFERENCE_DIR,
    global_gru_reference_dir: Path = GLOBAL_GRU_REFERENCE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frozen per-container evaluation tables from both references."""
    hybrid_path = _evaluation_df_path(hybrid_reference_dir)
    global_path = _evaluation_df_path(global_gru_reference_dir)

    if not hybrid_path.exists():
        raise FileNotFoundError(f"Hybrid evaluation table not found: {hybrid_path}")
    if not global_path.exists():
        raise FileNotFoundError(f"Global GRU evaluation table not found: {global_path}")

    hybrid_df = pd.read_csv(hybrid_path)
    global_df = pd.read_csv(global_path)
    return hybrid_df, global_df


def load_frozen_evaluation_summaries(
    hybrid_reference_dir: Path = HYBRID_BASELINE_REFERENCE_DIR,
    global_gru_reference_dir: Path = GLOBAL_GRU_REFERENCE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frozen aggregate summaries from both references."""
    hybrid_summary = pd.read_csv(
        _evaluation_summary_path(hybrid_reference_dir),
        index_col=0,
    )
    global_summary = pd.read_csv(
        _evaluation_summary_path(global_gru_reference_dir),
        index_col=0,
    )
    return hybrid_summary, global_summary


def assert_cohort_alignment(
    hybrid_df: pd.DataFrame,
    global_df: pd.DataFrame,
    stale_container_id: str = STALE_SELECTED_CONTAINER_ID,
    expected_evaluated: int = EXPECTED_EVALUATED,
) -> None:
    """Assert both methodologies evaluated the same container cohort."""
    if len(hybrid_df) != expected_evaluated:
        raise AssertionError(
            f"Hybrid evaluation has {len(hybrid_df)} rows; "
            f"expected {expected_evaluated}"
        )
    if len(global_df) != expected_evaluated:
        raise AssertionError(
            f"Global GRU evaluation has {len(global_df)} rows; "
            f"expected {expected_evaluated}"
        )

    if hybrid_df["container_id"].duplicated().any():
        raise AssertionError("Duplicate container_id values in Hybrid evaluation_df")
    if global_df["container_id"].duplicated().any():
        raise AssertionError("Duplicate container_id values in Global evaluation_df")

    hybrid_ids = set(hybrid_df["container_id"])
    global_ids = set(global_df["container_id"])
    if hybrid_ids != global_ids:
        only_hybrid = sorted(hybrid_ids - global_ids)
        only_global = sorted(global_ids - hybrid_ids)
        raise AssertionError(
            "Container ID mismatch between frozen evaluations: "
            f"hybrid_only={only_hybrid[:5]}, global_only={only_global[:5]}"
        )

    if stale_container_id in hybrid_ids or stale_container_id in global_ids:
        raise AssertionError(
            f"Stale container {stale_container_id!r} must not appear in evaluation_df"
        )


def _methodology_summary_row(
    methodology: str,
    evaluation_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute aggregate statistics for one methodology."""
    row: dict[str, Any] = {
        "methodology": methodology,
        "n_containers": int(len(evaluation_df)),
    }
    for metric in METRIC_COLUMNS:
        values = evaluation_df[metric]
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std())
        row[f"{metric}_min"] = float(values.min())
        row[f"{metric}_max"] = float(values.max())
    return row


def build_comparison_table(
    hybrid_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-methodology aggregate comparison table."""
    rows = [
        _methodology_summary_row(METHODOLOGY_HYBRID, hybrid_df),
        _methodology_summary_row(METHODOLOGY_GLOBAL, global_df),
    ]
    return pd.DataFrame(rows)


def build_per_container_delta(
    hybrid_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build paired per-container deltas.

    Delta convention (locked): Global − Hybrid for MAE, RMSE, and MAPE.
    """
    merged = hybrid_df.merge(
        global_df,
        on="container_id",
        suffixes=("_hybrid", "_global"),
        validate="one_to_one",
    )

    delta_df = pd.DataFrame({
        "container_id": merged["container_id"],
        "hybrid_day1_mae": merged["day1_mae_hybrid"],
        "global_day1_mae": merged["day1_mae_global"],
        "delta_day1_mae": merged["day1_mae_global"] - merged["day1_mae_hybrid"],
        "hybrid_day1_rmse": merged["day1_rmse_hybrid"],
        "global_day1_rmse": merged["day1_rmse_global"],
        "delta_day1_rmse": merged["day1_rmse_global"] - merged["day1_rmse_hybrid"],
        "hybrid_day1_mape": merged["day1_mape_hybrid"],
        "global_day1_mape": merged["day1_mape_global"],
        "delta_day1_mape": merged["day1_mape_global"] - merged["day1_mape_hybrid"],
        "hybrid_train_steps": merged["train_steps_hybrid"],
        "global_train_steps": merged["train_steps_global"],
        "hybrid_validation_steps": merged["validation_steps_hybrid"],
        "global_validation_steps": merged["validation_steps_global"],
    })
    return delta_df.sort_values("container_id").reset_index(drop=True)


def summarize_deltas(
    per_container_delta: pd.DataFrame,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Summarize per-container delta statistics."""
    summary: dict[str, Any] = {
        "delta_sign_convention": DELTA_SIGN_CONVENTION,
        "n_containers": int(len(per_container_delta)),
    }

    for metric in METRIC_COLUMNS:
        delta_col = f"delta_{metric}"
        deltas = per_container_delta[delta_col]
        improved = int((deltas < -tie_tolerance).sum())
        worsened = int((deltas > tie_tolerance).sum())
        tied = int(len(deltas) - improved - worsened)

        summary[metric] = {
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std()),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "improved_count": improved,
            "worsened_count": worsened,
            "tied_count": tied,
        }

    return summary


def build_experiment_config(
    hybrid_reference_dir: Path = HYBRID_BASELINE_REFERENCE_DIR,
    global_gru_reference_dir: Path = GLOBAL_GRU_REFERENCE_DIR,
) -> dict[str, Any]:
    """Return shared protocol snapshot for the methodology comparison."""
    return {
        "comparison_type": "forecasting_methodology",
        "dataset": "Alibaba Cluster Trace",
        "target": "cpu_util_percent",
        "resampling": "15min",
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "evaluation_period": "preprocessing validation temporal holdout",
        "metric_unit": "real CPU %",
        "mape_epsilon": MAPE_EPSILON,
        "selected_containers": 100,
        "evaluable_containers": EXPECTED_EVALUATED,
        "skipped_container_id": STALE_SELECTED_CONTAINER_ID,
        "random_seed": RANDOM_SEED,
        "delta_sign_convention": DELTA_SIGN_CONVENTION,
        "hybrid_methodology": METHODOLOGY_HYBRID,
        "global_methodology": MODEL_VARIANT,
        "hybrid_frozen_reference": str(
            hybrid_reference_dir.relative_to(REPO_ROOT),
        ),
        "global_gru_frozen_reference": str(
            global_gru_reference_dir.relative_to(REPO_ROOT),
        ),
        "hybrid_evaluation_source": str(
            _evaluation_df_path(hybrid_reference_dir).relative_to(REPO_ROOT),
        ),
        "global_evaluation_source": str(
            _evaluation_df_path(global_gru_reference_dir).relative_to(REPO_ROOT),
        ),
        "global_evaluation_run": str(
            GLOBAL_GRU_EVALUATION_RUN_DIR.relative_to(REPO_ROOT),
        ),
    }


def build_comparison_metadata(
    experiment_dir: Path,
    comparison_table: pd.DataFrame,
    per_container_delta: pd.DataFrame,
    delta_summary: dict[str, Any],
    experiment_config: dict[str, Any],
    hybrid_reference_dir: Path = HYBRID_BASELINE_REFERENCE_DIR,
    global_gru_reference_dir: Path = GLOBAL_GRU_REFERENCE_DIR,
    comparison_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build provenance metadata for a methodology comparison experiment."""
    if comparison_timestamp is None:
        comparison_timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        )

    experiment_dir = Path(experiment_dir)
    return {
        "comparison_timestamp": comparison_timestamp,
        "experiment_identifier": experiment_dir.name,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "hybrid_frozen_reference": str(
            hybrid_reference_dir.relative_to(REPO_ROOT),
        ),
        "global_gru_frozen_reference": str(
            global_gru_reference_dir.relative_to(REPO_ROOT),
        ),
        "hybrid_evaluation_source": str(
            _evaluation_df_path(hybrid_reference_dir).relative_to(REPO_ROOT),
        ),
        "global_evaluation_source": str(
            _evaluation_df_path(global_gru_reference_dir).relative_to(REPO_ROOT),
        ),
        "global_evaluation_run": str(
            GLOBAL_GRU_EVALUATION_RUN_DIR.relative_to(REPO_ROOT),
        ),
        "comparison_script": COMPARISON_SCRIPT,
        "comparison_script_version": COMPARISON_SCRIPT_VERSION,
        "verification_script": VERIFICATION_SCRIPT,
        "delta_sign_convention": DELTA_SIGN_CONVENTION,
        "protocol_summary": {
            "input_window": INPUT_WINDOW,
            "forecast_horizon": FORECAST_HORIZON,
            "evaluated_containers": EXPECTED_EVALUATED,
            "skipped_container_id": STALE_SELECTED_CONTAINER_ID,
            "mape_epsilon": MAPE_EPSILON,
            "metric_unit": "real CPU %",
        },
        "evaluated_containers": EXPECTED_EVALUATED,
        "skipped_containers": [
            {
                "container_id": STALE_SELECTED_CONTAINER_ID,
                "reason": "missing from frozen train/val data",
            },
        ],
        "comparison_table": comparison_table.to_dict(orient="records"),
        "delta_summary": delta_summary,
        "experiment_config": experiment_config,
        "outputs": {
            "comparison_table": "comparison_table.csv",
            "per_container_delta": "per_container_delta.csv",
            "experiment_config": "experiment_config.json",
            "phase6_comparison": "phase6_comparison.json",
            "comparison_metadata": "comparison_metadata.json",
        },
        "per_container_delta_rows": int(len(per_container_delta)),
    }
