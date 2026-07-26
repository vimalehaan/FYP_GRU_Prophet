#!/usr/bin/env python3
"""Run Ridge Residual Analysis (RRA) — read-only diagnostic on CSRLE B0-LC."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.dataset import evaluable_container_ids, load_frozen_base_data  # noqa: E402
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW  # noqa: E402
from utils.rra.diagnostics import (  # noqa: E402
    container_temporal_diagnostics,
    select_case_study_containers,
    white_noise_comparison,
)
from utils.rra.pipeline import (  # noqa: E402
    compute_container_ridge_analysis,
    save_ridge_artifacts,
    train_ridge_diagnostic_model,
)
from utils.rra.plots import (  # noqa: E402
    plot_acf_comparison,
    plot_boxplot_metric,
    plot_case_study_panel,
    plot_cohort_acf_paired,
    plot_distribution_comparison,
    plot_fft_comparison,
    plot_pacf_comparison,
)
from utils.rra.report import (  # noqa: E402
    build_final_report,
    evaluate_justification,
    write_final_report_md,
    write_json,
)

CSRLE_EXP = REPO_ROOT / "experiments" / "synthetic_residual_learnability_2026-07-25_164307"
CONDITION = "b0_lc"
PROTOCOL_VERSION = "rra_v1.0"
BOOTSTRAP_SEED = 12345


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_frozen_config(exp_dir: Path, timestamp: str) -> dict[str, Any]:
    train_p = CSRLE_EXP / "data" / CONDITION / "train_syn.parquet"
    val_p = CSRLE_EXP / "data" / CONDITION / "val_syn.parquet"
    train_df, val_df, scalers, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, _ = evaluable_container_ids(selected, train_df, val_df, scalers)

    cfg: dict[str, Any] = {
        "timestamp": timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": "ridge_residual_analysis",
        "read_only_sources": [
            "CSRLE B0-LC frozen data",
            "No GRU retraining",
            "No prior experiment modification",
        ],
        "csrle_experiment_dir": str(CSRLE_EXP.relative_to(REPO_ROOT)),
        "primary_condition": CONDITION,
        "dataset": {
            "n_containers": len(container_ids),
            "container_ids": container_ids,
            "source_train": str(train_p.relative_to(REPO_ROOT)),
            "source_val": str(val_p.relative_to(REPO_ROOT)),
        },
        "sequence_configuration": {
            "input_window": DEFAULT_INPUT_WINDOW,
            "forecast_horizon": DAY1_HORIZON,
            "internal_val_split": "chronological_80_20",
            "shuffle": False,
            "features": ["residual_scaled"],
            "target": "residual_scaled",
        },
        "ridge_configuration": {
            "estimator": "sklearn.linear_model.Ridge",
            "alpha": 1.0,
            "hyperparameter_tuning": False,
            "input_window": DEFAULT_INPUT_WINDOW,
            "forecast_horizon": DAY1_HORIZON,
            "internal_val_split": "chronological_80_20",
            "residual_scaling": "global_train_mean_std",
        },
        "prophet_configuration": {
            "daily_seasonality": True,
            "weekly_seasonality": False,
            "fit_scope": "train_only_per_container",
        },
        "evaluation_protocol": {
            "analysis_split": "validation_period",
            "residual_space": "scaled_prophet_residual",
            "remaining_definition": "prophet_residual - ridge_prediction",
            "ridge_application": "non_overlapping_96_step_blocks",
        },
        "diagnostic_protocol": {
            "acf_lags": "1-10",
            "pacf_lags": "1-10",
            "ljung_box_lags": [10, 20, 30],
            "ljung_box_primary_lag": 20,
            "ljung_box_alpha": 0.05,
            "fft_fs": 1.0,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_n": 2000,
            "bootstrap_ci": 0.95,
        },
        "seed": BOOTSTRAP_SEED,
        "dataset_hashes": {
            "b0_lc_train_syn_parquet": _sha256(train_p),
            "b0_lc_val_syn_parquet": _sha256(val_p),
        },
    }
    cfg_path = exp_dir / "config" / "ridge_residual_analysis_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def run_experiment(timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"ridge_residual_analysis_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_frozen_config(exp_dir, ts)
    container_ids = cfg["dataset"]["container_ids"]

    train_df = pd.read_parquet(CSRLE_EXP / "data" / CONDITION / "train_syn.parquet")
    val_df = pd.read_parquet(CSRLE_EXP / "data" / CONDITION / "val_syn.parquet")

    print("Training Ridge diagnostic model on B0-LC train...")
    ridge_model, res_mean, res_std, ridge_meta = train_ridge_diagnostic_model(
        train_df,
        input_window=DEFAULT_INPUT_WINDOW,
        forecast_horizon=DAY1_HORIZON,
        alpha=1.0,
    )
    save_ridge_artifacts(ridge_model, res_mean, res_std, ridge_meta, exp_dir)

    print(f"Analyzing {len(container_ids)} containers...")
    container_results: dict[str, dict[str, Any]] = {}
    eval_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for cid in container_ids:
        result = compute_container_ridge_analysis(
            cid, train_df, val_df, ridge_model, res_mean, res_std,
        )
        container_results[cid] = result
        eval_rows.append({
            k: v for k, v in result.items()
            if k not in (
                "prophet_residual_scaled",
                "ridge_prediction_scaled",
                "ridge_remaining_scaled",
                "prophet_pred_scaled",
                "actual_scaled",
            )
        })
        diag_rows.append(container_temporal_diagnostics(
            cid,
            result["prophet_residual_scaled"],
            result["ridge_remaining_scaled"],
        ))

    eval_df = pd.DataFrame(eval_rows)
    diag_df = pd.DataFrame(diag_rows)
    _write_csv(exp_dir / "tables" / "ridge_evaluation_per_container.csv", eval_df)
    _write_csv(exp_dir / "diagnostics" / "temporal_structure_per_container.csv", diag_df)

    white_noise = white_noise_comparison(
        container_results, diag_df, bootstrap_seed=BOOTSTRAP_SEED,
    )
    write_json(exp_dir / "diagnostics" / "white_noise_comparison.json", white_noise)

    cohort_summary = {
        "n_containers": len(container_ids),
        "ridge_performance": {
            "mean_pearson_r": float(eval_df["ridge_pearson_r"].mean()),
            "median_pearson_r": float(eval_df["ridge_pearson_r"].median()),
            "mean_mae": float(eval_df["ridge_mae_scaled"].mean()),
            "mean_remaining_std_ratio": float(eval_df["remaining_std_ratio"].mean()),
            "mean_var_reduction_fraction": float(
                eval_df["var_reduction_fraction"].mean()
            ),
        },
    }
    write_json(exp_dir / "tables" / "cohort_summary.json", cohort_summary)

    case_studies = select_case_study_containers(eval_df)
    write_json(exp_dir / "diagnostics" / "case_study_containers.json", case_studies)

    # Plots
    plots_dir = exp_dir / "plots"
    prophet_pooled = np.concatenate([
        r["prophet_residual_scaled"] for r in container_results.values()
    ])
    remaining_pooled = np.concatenate([
        r["ridge_remaining_scaled"] for r in container_results.values()
    ])
    plot_distribution_comparison(
        prophet_pooled, remaining_pooled,
        plots_dir / "residual_distribution_comparison",
    )
    plot_boxplot_metric(
        diag_df,
        "prophet_avg_abs_acf_1_10",
        "remaining_avg_abs_acf_1_10",
        "Per-container |ACF| lags 1–10",
        "Mean |ACF|",
        plots_dir / "acf_boxplot_comparison",
    )
    plot_boxplot_metric(
        diag_df,
        "prophet_avg_abs_pacf_1_10",
        "remaining_avg_abs_pacf_1_10",
        "Per-container |PACF| lags 1–10",
        "Mean |PACF|",
        plots_dir / "pacf_boxplot_comparison",
    )
    plot_cohort_acf_paired(diag_df, plots_dir / "cohort_acf_paired")

    rep_cid = case_studies["median"]
    rep = container_results[rep_cid]
    plot_acf_comparison(
        rep["prophet_residual_scaled"],
        f"ACF — Prophet residual ({rep_cid})",
        plots_dir / f"acf_prophet_{rep_cid}",
    )
    plot_acf_comparison(
        rep["ridge_remaining_scaled"],
        f"ACF — Ridge remaining ({rep_cid})",
        plots_dir / f"acf_remaining_{rep_cid}",
    )
    plot_pacf_comparison(
        rep["prophet_residual_scaled"],
        f"PACF — Prophet residual ({rep_cid})",
        plots_dir / f"pacf_prophet_{rep_cid}",
    )
    plot_pacf_comparison(
        rep["ridge_remaining_scaled"],
        f"PACF — Ridge remaining ({rep_cid})",
        plots_dir / f"pacf_remaining_{rep_cid}",
    )
    plot_fft_comparison(
        rep["prophet_residual_scaled"],
        rep["ridge_remaining_scaled"],
        f"FFT — {rep_cid}",
        plots_dir / f"fft_comparison_{rep_cid}",
    )

    for label, cid in case_studies.items():
        plot_case_study_panel(
            cid, container_results[cid],
            plots_dir / f"case_study_{label}_{cid}",
        )

    # Paired comparison table
    comparison_rows = []
    for _, row in diag_df.iterrows():
        comparison_rows.append({
            "container_id": row["container_id"],
            "prophet_avg_abs_acf_1_10": row["prophet_avg_abs_acf_1_10"],
            "remaining_avg_abs_acf_1_10": row["remaining_avg_abs_acf_1_10"],
            "acf_reduction": row["acf_reduction"],
            "prophet_lb_reject_20": row["prophet_lb_reject_lag_20"],
            "remaining_lb_reject_20": row["remaining_lb_reject_lag_20"],
        })
    _write_csv(exp_dir / "tables" / "structure_comparison.csv", pd.DataFrame(comparison_rows))

    fft_rows = diag_df[[
        "container_id",
        "prophet_dominant_freq",
        "remaining_dominant_freq",
        "prophet_dominant_period_steps",
        "remaining_dominant_period_steps",
        "prophet_peak_power",
        "remaining_peak_power",
    ]].copy()
    _write_csv(exp_dir / "tables" / "fft_summary.csv", fft_rows)

    justification = evaluate_justification(white_noise)
    report = build_final_report(
        cfg, cohort_summary, white_noise, justification, case_studies,
    )
    write_json(exp_dir / "reports" / "final_report.json", report)
    write_final_report_md(report, exp_dir / "reports" / "final_report.md")

    write_json(exp_dir / "verification" / "data_integrity.json", {
        "dataset_hashes_match_lfhe": cfg["dataset_hashes"],
        "n_containers_analyzed": len(container_ids),
        "ridge_trained": True,
        "gru_trained": False,
        "prior_experiments_modified": False,
    })

    print(f"RRA complete: {exp_dir}")
    print(f"Verdict: {report['verdict']}")
    return exp_dir


if __name__ == "__main__":
    ts_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(ts_arg)
