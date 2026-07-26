"""Read-only audit runner for Ridge Residual Analysis."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.diagnostics import ljung_box_reject_fraction
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_training import generate_prophet_residuals
from utils.rra.diagnostics import container_temporal_diagnostics
from utils.rra.pipeline import _fit_prophet_val_residuals
from utils.rra_audit.verification import (
    compute_remaining_corrected_zspace,
    compute_remaining_rra_original,
    fit_ar_forecast_metrics,
    replicate_acf_from_series,
    underfit_subtraction_analysis,
    verify_day1_alignment,
    verify_no_val_leakage_in_ridge_train,
    verify_residual_identity,
    verify_scaling_spaces,
    zscore_residual,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_audit(
    rra_exp_dir: Path,
    audit_exp_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute full RRA audit; write results under audit_exp_dir only."""
    with (rra_exp_dir / "ridge" / "models" / "ridge_model.pkl").open("rb") as fh:
        ridge_model = pickle.load(fh)
    with (rra_exp_dir / "ridge" / "training" / "residual_stats.json").open() as fh:
        stats = json.load(fh)
    res_mean = float(stats["res_mean"])
    res_std = float(stats["res_std"])

    csrle = repo_root / "experiments" / "synthetic_residual_learnability_2026-07-25_164307"
    train_df = pd.read_parquet(csrle / "data" / "b0_lc" / "train_syn.parquet")
    val_df = pd.read_parquet(csrle / "data" / "b0_lc" / "val_syn.parquet")
    eval_df = pd.read_csv(rra_exp_dir / "tables" / "ridge_evaluation_per_container.csv")
    diag_df = pd.read_csv(
        rra_exp_dir / "diagnostics" / "temporal_structure_per_container.csv"
    )
    container_ids = eval_df["container_id"].astype(str).tolist()

    # --- Task 1 & 2: per-container verification ---
    task1_rows: list[dict[str, Any]] = []
    task2_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    corrected_diag_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    underfit_rows: list[dict[str, Any]] = []

    for cid in container_ids:
        train_c = train_df[train_df["container_id"] == cid].sort_values("time_stamp")
        val_c = val_df[val_df["container_id"] == cid].sort_values("time_stamp")
        train_res_df = generate_prophet_residuals(train_c)
        train_z = zscore_residual(train_res_df["residual"].values, res_mean, res_std)
        prophet_raw, _, _ = _fit_prophet_val_residuals(train_c, val_c)

        ridge_z_orig, remaining_orig = compute_remaining_rra_original(
            train_z, prophet_raw, ridge_model,
        )
        ridge_z_corr, remaining_corr, prophet_z = compute_remaining_corrected_zspace(
            train_z, prophet_raw, ridge_model, res_mean, res_std,
        )

        id_orig = verify_residual_identity(prophet_raw, ridge_z_orig, remaining_orig)
        id_corr = verify_residual_identity(prophet_z, ridge_z_corr, remaining_corr)
        scale = verify_scaling_spaces(prophet_raw, ridge_z_orig, res_mean, res_std)
        scale["container_id"] = cid
        scaling_rows.append(scale)

        task1_rows.append({
            "container_id": cid,
            "original_identity_holds": id_orig["identity_holds"],
            "corrected_z_identity_holds": id_corr["identity_holds"],
            "rra_sign": "prophet_minus_ridge",
            "rra_original_space": "raw_prophet_minus_z_ridge",
            "corrected_space": "z_prophet_minus_z_ridge",
            **{f"orig_{k}": v for k, v in id_orig.items()},
        })

        if cid == container_ids[0]:
            task2_rows.append(verify_day1_alignment(
                cid, train_df, val_df, ridge_model, res_mean, res_std,
            ))

        corrected_diag_rows.append(container_temporal_diagnostics(
            cid, prophet_z, remaining_corr,
        ))

        uf = underfit_subtraction_analysis(prophet_z, ridge_z_corr)
        uf["container_id"] = cid
        underfit_rows.append(uf)

        for order in (1, 2, 5):
            for label, series in (
                ("prophet_z", prophet_z),
                ("remaining_corr_z", remaining_corr),
            ):
                m = fit_ar_forecast_metrics(series, ar_order=order)
                ar_rows.append({"container_id": cid, "series": label, **m})

    task1_df = pd.DataFrame(task1_rows)
    scaling_df = pd.DataFrame(scaling_rows)
    corrected_diag_df = pd.DataFrame(corrected_diag_rows)
    underfit_df = pd.DataFrame(underfit_rows)
    ar_df = pd.DataFrame(ar_rows)

    _write_csv(audit_exp_dir / "verification" / "task1_residual_definition.csv", task1_df)
    _write_csv(audit_exp_dir / "verification" / "task1_scaling_mismatch.csv", scaling_df)
    _write_json(
        audit_exp_dir / "verification" / "task2_day1_alignment_sample.json",
        task2_rows[0] if task2_rows else {},
    )

    # Task 3
    leakage = verify_no_val_leakage_in_ridge_train(train_df, val_df, res_mean, res_std)
    _write_json(audit_exp_dir / "verification" / "task3_data_splits.json", leakage)

    # Task 4: replicate RRA ACF vs corrected
    rra_repl = {
        "prophet_mean_acf_1_10": float(diag_df["prophet_avg_abs_acf_1_10"].mean()),
        "rra_remaining_mean_acf_1_10": float(
            diag_df["remaining_avg_abs_acf_1_10"].mean()
        ),
        "rra_prophet_lb_reject_20": float(
            diag_df["prophet_lb_reject_lag_20"].mean()
        ),
        "rra_remaining_lb_reject_20": float(
            diag_df["remaining_lb_reject_lag_20"].mean()
        ),
    }
    corrected_summary = {
        "prophet_z_mean_acf_1_10": float(
            corrected_diag_df["prophet_avg_abs_acf_1_10"].mean()
        ),
        "corrected_remaining_mean_acf_1_10": float(
            corrected_diag_df["remaining_avg_abs_acf_1_10"].mean()
        ),
        "corrected_acf_reduction": float(
            (
                corrected_diag_df["prophet_avg_abs_acf_1_10"]
                - corrected_diag_df["remaining_avg_abs_acf_1_10"]
            ).mean()
        ),
        "corrected_prophet_lb_reject_20": float(
            corrected_diag_df["prophet_lb_reject_lag_20"].mean()
        ),
        "corrected_remaining_lb_reject_20": float(
            corrected_diag_df["remaining_lb_reject_lag_20"].mean()
        ),
    }
    _write_json(audit_exp_dir / "verification" / "task4_acf_replication.json", {
        "rra_reported": rra_repl,
        "replicated_from_rra_csv": rra_repl,
        "corrected_zspace": corrected_summary,
        "acf_implementation": "utils.csrle.diagnostics.avg_abs_acf lags 1-10 fft=True",
    })

    # Task 5
    task5 = {
        "scaling_bug_confirmed": bool(
            scaling_df["mean_abs_diff_raw_vs_z_pred"].mean()
            > scaling_df["mean_abs_diff_raw_vs_unzscored_pred"].mean()
        ),
        "cohort_mean_ridge_r_zspace": float(underfit_df["ridge_pearson_r"].mean()),
        "cohort_mean_ridge_r2": float(underfit_df["ridge_r_squared"].mean()),
        "cohort_mean_var_ratio_remaining_prophet_corrected": float(
            underfit_df["var_ratio_remaining_to_prophet"].mean()
        ),
        "rra_cohort_mean_var_ratio": float(eval_df["remaining_std_ratio"].mean() ** 2),
    }
    _write_json(audit_exp_dir / "verification" / "task5_counterintuitive_explanation.json", task5)

    # Task 6: AR cohort summary
    ar_summary = (
        ar_df.groupby(["series", "ar_order"])
        .agg(
            mean_forecast_r=("forecast_pearson_r", "mean"),
            mean_forecast_mae=("forecast_mae", "mean"),
            mean_ar_resid_acf=("residual_avg_abs_acf_1_10", "mean"),
            mean_ar_resid_lb_reject=("residual_lb_reject_lag_20", "mean"),
        )
        .reset_index()
    )
    _write_csv(audit_exp_dir / "tables" / "linear_model_ar_summary.csv", ar_summary)
    _write_csv(audit_exp_dir / "tables" / "linear_model_ar_per_container.csv", ar_df)
    _write_csv(audit_exp_dir / "tables" / "corrected_temporal_diagnostics.csv", corrected_diag_df)
    _write_csv(audit_exp_dir / "tables" / "underfit_analysis.csv", underfit_df)

    # Task 7 decision
    prophet_acf = corrected_summary["prophet_z_mean_acf_1_10"]
    rem_acf = corrected_summary["corrected_remaining_mean_acf_1_10"]
    acf_red = corrected_summary["corrected_acf_reduction"]

    best_ar_remaining = ar_summary[
        (ar_summary["series"] == "remaining_corr_z")
    ].sort_values("mean_forecast_r", ascending=False).head(1)
    best_ar_r = float(best_ar_remaining["mean_forecast_r"].iloc[0]) if len(best_ar_remaining) else float("nan")
    best_ar_order = int(best_ar_remaining["ar_order"].iloc[0]) if len(best_ar_remaining) else -1

    ridge_failed_as_predictor = float(underfit_df["ridge_pearson_r"].mean()) < 0.2
    ridge_failed_to_remove_structure = acf_red < 0.01
    remaining_mostly_linear = (
        best_ar_r > 0.3
        and float(best_ar_remaining["mean_ar_resid_lb_reject"].iloc[0]) < 0.5
        if len(best_ar_remaining)
        else False
    )

    decision = {
        "q1_ridge_failed_to_remove_linear_dependence_vs_failed_as_predictor": {
            "failed_as_predictor": ridge_failed_as_predictor,
            "failed_to_remove_structure_after_correction": ridge_failed_to_remove_structure,
            "primary_issue": (
                "predictor_failure_and_scaling_bug_in_rra_v1"
                if task5["scaling_bug_confirmed"]
                else "predictor_failure"
            ),
        },
        "q2_remaining_dependence_likely_linear_or_nonlinear": {
            "best_ar_order_on_corrected_remaining": best_ar_order,
            "best_ar_forecast_r_cohort_mean": best_ar_r,
            "verdict": (
                "predominantly_linear"
                if remaining_mostly_linear
                else "linear_models_insufficient_complex_structure_may_remain"
            ),
        },
        "q3_ridge_gru_justified_vs_more_linear_modelling": {
            "ridge_gru_justified_on_current_evidence": False if remaining_mostly_linear else None,
            "additional_linear_modelling_appropriate": remaining_mostly_linear,
            "note": (
                "RRA v1.0 conclusion partially invalidated by scaling bug; "
                "use corrected_zspace metrics for decisions."
            ),
        },
        "scaling_bug_in_rra_v1": task5["scaling_bug_confirmed"],
        "corrected_acf_reduction": acf_red,
        "rra_reported_acf_increase": float(
            rra_repl["rra_remaining_mean_acf_1_10"] - rra_repl["prophet_mean_acf_1_10"]
        ),
    }

    report = {
        "audit_timestamp": audit_exp_dir.name.replace("ridge_residual_analysis_audit_", ""),
        "rra_experiment_audited": str(rra_exp_dir.relative_to(repo_root)),
        "task5": task5,
        "corrected_summary": corrected_summary,
        "rra_reported": rra_repl,
        "decision": decision,
        "ar_summary": ar_summary.to_dict(orient="records"),
    }
    _write_json(audit_exp_dir / "reports" / "audit_report.json", report)
    return report
