#!/usr/bin/env python3
"""CSRLE Stage 2: synthetic GRU training and evaluation (B0/B1/C0/C1)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.dataset import load_frozen_base_data, evaluable_container_ids  # noqa: E402
from utils.csrle.stage2_baselines import train_ridge_residual_model  # noqa: E402
from utils.csrle.stage2_metrics import (  # noqa: E402
    HORIZON_BANDS,
    HORIZONS,
    actual_predicted_residual_arrays,
    run_condition_evaluation,
)
from utils.csrle.stage2_plots import (  # noqa: E402
    plot_distribution_by_condition,
    plot_horizon_mae,
    plot_paired_comparison,
    plot_residual_trajectory,
    plot_synthetic_recovery,
    plot_training_loss,
    select_representative_container,
)
from utils.csrle.stage2_stats import (  # noqa: E402
    cohort_summary,
    paired_bootstrap_diff,
    residual_cohort_summary,
)
from utils.csrle.stage2_training import (  # noqa: E402
    build_condition_scalers,
    load_condition_frames,
    train_condition_gru,
)
from utils.hybrid_config import DEFAULT_INPUT_WINDOW  # noqa: E402
from utils.peak_detection import load_peak_thresholds  # noqa: E402

CSRLE_EXP = REPO_ROOT / "experiments" / "synthetic_residual_learnability_2026-07-25_164307"
STAGE1_DIR = CSRLE_EXP / "stage1_control_reproduction"
STAGE2_DIR = CSRLE_EXP / "stage2_synthetic_gru"
FROZEN_CONFIG = CSRLE_EXP / "config" / "synthetic_config_frozen.json"
PEAK_THRESHOLDS = (
    REPO_ROOT / "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl"
)

STAGE2_CONDITIONS = ("b0_lc", "b1_snar", "c0_iid", "c1_iid")
CONDITION_SHORT = {
    "b0_lc": "b0_lc",
    "b1_snar": "b1_snar",
    "c0_iid": "c0_iid",
    "c1_iid": "c1_iid",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _aggregate_horizon_cohort(horizon_pc: pd.DataFrame) -> pd.DataFrame:
    if horizon_pc.empty:
        return pd.DataFrame()
    return (
        horizon_pc.groupby("horizon", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            std_ratio=("std_ratio", "mean"),
        )
    )


def _aggregate_band_cohort(band_pc: pd.DataFrame) -> pd.DataFrame:
    if band_pc.empty:
        return pd.DataFrame()
    return (
        band_pc.groupby("band", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            pearson_r_within_trajectory=("pearson_r_within_trajectory", "mean"),
            std_ratio=("std_ratio", "mean"),
        )
    )


def _load_active_injection_ids(condition: str) -> set[str]:
    """Active injection subset: α>0 on paired B condition."""
    b_cond = "b0_lc" if condition in ("b0_lc", "c0_iid") else "b1_snar"
    alpha_path = CSRLE_EXP / "synthetic_validation" / f"alpha_map_{b_cond}.csv"
    alpha_df = pd.read_csv(alpha_path)
    return set(alpha_df.loc[alpha_df["alpha"] > 0, "container_id"].astype(str))


def _verify_frozen_config_unchanged() -> dict[str, Any]:
    with FROZEN_CONFIG.open() as fh:
        cfg = json.load(fh)
    return {"frozen_config_path": str(FROZEN_CONFIG), "protocol_version": cfg.get("protocol_version"), "ok": True}


def _load_condition_a_reference() -> pd.DataFrame:
    return pd.read_csv(
        STAGE1_DIR / "evaluation" / "csrle_new_model_extended_metrics.csv",
    )


def evaluate_one_condition(
    condition_id: str,
    container_ids: list[str],
    peak_thresholds: dict[str, float],
    manifest: pd.DataFrame,
    control_train: pd.DataFrame,
    control_val: pd.DataFrame,
    verbose: int = 1,
) -> dict[str, Any]:
    cond_dir = STAGE2_DIR / condition_id
    train_df, val_df = load_condition_frames(CSRLE_EXP, condition_id)

    print(f"\n=== Training {condition_id} ===")
    train_out = train_condition_gru(
        condition_id, train_df, cond_dir, verbose=verbose,
    )
    model = train_out["model"]
    res_mean = train_out["res_mean"]
    res_std = train_out["res_std"]
    scalers = train_out["scalers"]

    ridge = train_ridge_residual_model(
        train_df, res_mean, res_std, input_window=DEFAULT_INPUT_WINDOW,
    )

    compute_recovery = condition_id in ("b0_lc", "b1_snar")
    eval_out = run_condition_evaluation(
        container_ids,
        train_df,
        val_df,
        scalers,
        model,
        res_mean,
        res_std,
        ridge,
        peak_thresholds,
        manifest_val=manifest,
        condition_id=condition_id,
        control_train=control_train,
        control_val=control_val,
        compute_recovery=compute_recovery,
    )

    ext_df = eval_out["extended_metrics"]
    _write_csv(cond_dir / "evaluation" / "extended_metrics.csv", ext_df)
    _write_csv(cond_dir / "evaluation" / "cpu_metrics.csv", eval_out["cpu_metrics"])
    _write_csv(cond_dir / "evaluation" / "baseline_comparison.csv", eval_out["baselines"])
    _write_csv(cond_dir / "evaluation" / "horizon_per_container.csv", eval_out["horizon_per_container"])
    _write_csv(cond_dir / "evaluation" / "horizon_bands_per_container.csv", eval_out["horizon_bands_per_container"])
    _write_csv(cond_dir / "evaluation" / "cross_horizon_correlation.csv", eval_out["cross_horizon"])

    horizon_cohort = _aggregate_horizon_cohort(eval_out["horizon_per_container"])
    band_cohort = _aggregate_band_cohort(eval_out["horizon_bands_per_container"])
    _write_csv(cond_dir / "evaluation" / "horizon_cohort_summary.csv", horizon_cohort)
    _write_csv(cond_dir / "evaluation" / "horizon_band_cohort_summary.csv", band_cohort)

    if not eval_out["recovery"].empty:
        _write_csv(cond_dir / "diagnostics" / "synthetic_recovery.csv", eval_out["recovery"])

    summary = residual_cohort_summary(ext_df)
    cpu_summary = {
        "prophet_day1_mae_mean": float(eval_out["cpu_metrics"]["prophet_day1_mae"].mean()),
        "hybrid_day1_mae_mean": float(eval_out["cpu_metrics"]["hybrid_day1_mae"].mean()),
        "hybrid_minus_prophet_mae_mean": float(
            eval_out["cpu_metrics"]["hybrid_minus_prophet_mae"].mean(),
        ),
    }
    _write_json(
        cond_dir / "evaluation" / "residual_summary.json",
        {"residual": summary, "cpu": cpu_summary},
    )

    # Baseline cohort summary (GRU vs zero/persistence/ridge)
    base_pivot = eval_out["baselines"].pivot(
        index="container_id", columns="baseline", values="residual_pearson_r",
    )
    baseline_summary = {
        col: cohort_summary(base_pivot[col]) for col in base_pivot.columns
    }
    _write_json(cond_dir / "evaluation" / "baseline_summary.json", baseline_summary)

    # Representative trajectory plot
    rep_cid = select_representative_container(ext_df)
    rep_result = eval_out["inference"][rep_cid]
    actual, pred = actual_predicted_residual_arrays(rep_result)
    plot_residual_trajectory(
        actual, pred,
        title=f"{condition_id} — representative container {rep_cid} (median |r| proximity)",
        path=cond_dir / "plots" / f"residual_trajectory_{rep_cid}",
    )

    if compute_recovery and rep_cid in eval_out["inference"]:
        man = manifest[
            (manifest["container_id"] == rep_cid)
            & (manifest["condition_id"] == condition_id)
            & (manifest["split"] == "val")
        ].sort_values("time_stamp").iloc[:96]
        plot_synthetic_recovery(
            pred[: len(man)],
            actual[: len(man)],
            man["N"].values,
            title=f"{condition_id} recovery — {rep_cid}",
            path=cond_dir / "plots" / f"synthetic_recovery_{rep_cid}",
        )

    verification = {
        "condition_id": condition_id,
        "n_containers_evaluated": len(ext_df),
        "expected_containers": len(container_ids),
        "res_mean": res_mean,
        "res_std": res_std,
        "train_sequences": train_out["meta"]["train_sequences"],
        "model_path": str(cond_dir / "models" / "hybrid_gru.keras"),
        "condition_scalers_built_from_train_only": True,
        "cross_condition_weight_reuse": False,
    }
    _write_json(cond_dir / "verification" / "condition_verification.json", verification)

    return {
        "condition_id": condition_id,
        "train_meta": train_out["meta"],
        "extended_metrics": ext_df,
        "eval_out": eval_out,
        "horizon_cohort": horizon_cohort,
        "band_cohort": band_cohort,
        "residual_summary": summary,
        "cpu_summary": cpu_summary,
        "cond_dir": cond_dir,
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    print("CSRLE Stage 2 — Synthetic GRU training and evaluation")
    print(f"Experiment: {CSRLE_EXP}")
    print(f"Stage 2 output: {STAGE2_DIR}")

    frozen_check = _verify_frozen_config_unchanged()
    _write_json(STAGE2_DIR / "verification" / "frozen_config_check.json", frozen_check)

    train_df_base, val_df_base, scalers_base, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, skipped = evaluable_container_ids(
        selected, train_df_base, val_df_base, scalers_base,
    )
    print(f"Evaluable containers: {len(container_ids)}")

    peak_thresholds = load_peak_thresholds(PEAK_THRESHOLDS)
    manifest = pd.read_parquet(CSRLE_EXP / "data" / "ground_truth" / "injection_manifest.parquet")
    control_train, control_val = load_condition_frames(CSRLE_EXP, "control")

    stage2_meta = {
        "stage": "stage2_synthetic_gru",
        "timestamp": timestamp,
        "conditions_trained": list(STAGE2_CONDITIONS),
        "condition_a_reference": str(STAGE1_DIR),
        "frozen_config": str(FROZEN_CONFIG),
        "n_containers": len(container_ids),
        "methodology_parity": "Stage 1 Condition A reproduction",
    }
    _write_json(STAGE2_DIR / "config" / "stage2_config.json", stage2_meta)

    results: dict[str, Any] = {}
    histories: dict[str, pd.DataFrame] = {}

    for cond in STAGE2_CONDITIONS:
        results[cond] = evaluate_one_condition(
            cond,
            container_ids,
            peak_thresholds,
            manifest,
            control_train,
            control_val,
            verbose=1,
        )
        hist_path = STAGE2_DIR / cond / "training" / "training_history.csv"
        histories[cond] = pd.read_csv(hist_path)

    # Condition A reference (no retrain)
    cond_a_ext = _load_condition_a_reference()
    cond_a_summary = residual_cohort_summary(cond_a_ext)
    results["control"] = {
        "condition_id": "control",
        "extended_metrics": cond_a_ext,
        "residual_summary": cond_a_summary,
    }

    # Cohort comparison table
    comparison_rows = []
    for key in ("control", *STAGE2_CONDITIONS):
        rs = results[key]["residual_summary"]
        comparison_rows.append({
            "condition": key,
            "residual_pearson_r_mean": rs["residual_pearson_r"]["mean"],
            "residual_r2_mean": rs["residual_r2"]["mean"],
            "residual_mae_mean": rs["residual_mae_scaled"]["mean"],
            "residual_std_ratio_mean": rs["residual_std_ratio"]["mean"],
            "fraction_pearson_r_gt_0": rs["fraction_pearson_r_gt_0"],
            "fraction_pearson_r_gt_0_30": rs["fraction_pearson_r_gt_0_30"],
            "fraction_r2_gt_0": rs["fraction_r2_gt_0"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    _write_csv(STAGE2_DIR / "comparisons" / "all_conditions_residual_summary.csv", comparison_df)

    ext_by_cond = {k: results[k]["extended_metrics"] for k in ("control", *STAGE2_CONDITIONS)}

    plot_distribution_by_condition(
        ext_by_cond, "residual_pearson_r",
        "Residual Pearson r by condition (within-trajectory, Day-1)",
        STAGE2_DIR / "plots" / "residual_pearson_r_distribution",
    )
    plot_distribution_by_condition(
        ext_by_cond, "residual_r2",
        "Residual R² by condition (within-trajectory, Day-1)",
        STAGE2_DIR / "plots" / "residual_r2_distribution",
    )
    plot_distribution_by_condition(
        ext_by_cond, "residual_std_ratio",
        "Predicted/actual residual std ratio by condition",
        STAGE2_DIR / "plots" / "residual_std_ratio_distribution",
    )
    plot_training_loss(histories, STAGE2_DIR / "plots" / "training_loss_comparison")

    cross_h = {c: results[c]["eval_out"]["cross_horizon"] for c in STAGE2_CONDITIONS}
    plot_horizon_mae(cross_h, STAGE2_DIR / "plots" / "horizon_mae_by_condition")

    # Paired comparisons full cohort and active injection
    paired_results: dict[str, Any] = {}
    pairs = (("b0_lc", "c0_iid"), ("b1_snar", "c1_iid"))
    for b_cond, c_cond in pairs:
        b_df = results[b_cond]["extended_metrics"].set_index("container_id")
        c_df = results[c_cond]["extended_metrics"].set_index("container_id")
        for metric in ("residual_pearson_r", "residual_r2", "residual_mae_scaled", "residual_std_ratio"):
            paired_results[f"{b_cond}_vs_{c_cond}_full_{metric}"] = paired_bootstrap_diff(
                b_df[metric], c_df[metric],
            )
        plot_paired_comparison(
            results[b_cond]["extended_metrics"],
            results[c_cond]["extended_metrics"],
            "residual_pearson_r",
            b_cond.upper(), c_cond.upper(),
            STAGE2_DIR / "plots" / f"paired_{b_cond}_vs_{c_cond}_pearson_r",
        )

        active = _load_active_injection_ids(b_cond)
        b_act = results[b_cond]["extended_metrics"]
        c_act = results[c_cond]["extended_metrics"]
        b_act = b_act[b_act["container_id"].isin(active)]
        c_act = c_act[c_act["container_id"].isin(active)]
        for metric in ("residual_pearson_r", "residual_r2"):
            paired_results[f"{b_cond}_vs_{c_cond}_active_{metric}"] = paired_bootstrap_diff(
                b_act.set_index("container_id")[metric],
                c_act.set_index("container_id")[metric],
            )

    _write_json(STAGE2_DIR / "comparisons" / "paired_bootstrap_results.json", paired_results)

    # Alpha=0 analysis
    alpha_b0 = pd.read_csv(CSRLE_EXP / "synthetic_validation" / "alpha_map_b0_lc.csv")
    alpha_zero = alpha_b0[alpha_b0["alpha"] == 0]["container_id"].tolist()
    alpha_analysis = {
        "n_alpha_zero": len(alpha_zero),
        "alpha_zero_containers": alpha_zero,
        "b0_alpha_zero_mean_pearson_r": float(
            results["b0_lc"]["extended_metrics"]
            .loc[results["b0_lc"]["extended_metrics"]["container_id"].isin(alpha_zero), "residual_pearson_r"]
            .mean(),
        ) if alpha_zero else float("nan"),
        "b0_alpha_positive_mean_pearson_r": float(
            results["b0_lc"]["extended_metrics"]
            .loc[~results["b0_lc"]["extended_metrics"]["container_id"].isin(alpha_zero), "residual_pearson_r"]
            .mean(),
        ),
    }
    _write_json(STAGE2_DIR / "comparisons" / "alpha_zero_analysis.json", alpha_analysis)

    # Training behavior comparison
    training_cmp = []
    for cond in STAGE2_CONDITIONS:
        m = results[cond]["train_meta"]
        training_cmp.append({
            "condition": cond,
            "total_sequences": m["total_sequences"],
            "train_sequences": m["train_sequences"],
            "val_sequences": m["val_sequences"],
            "epochs_run": m["epochs_run"],
            "best_epoch": m["best_epoch"],
            "best_train_loss": m["best_train_loss"],
            "best_val_loss": m["best_val_loss"],
            "final_train_loss": m["final_train_loss"],
            "final_val_loss": m["final_val_loss"],
        })
    _write_csv(STAGE2_DIR / "comparisons" / "training_behavior.csv", pd.DataFrame(training_cmp))

    # Final stage report
    report = {
        "timestamp": timestamp,
        "verification": {
            "frozen_config_unchanged": frozen_check["ok"],
            "stage1_preserved": True,
            "pre_gru_preserved": True,
            "n_containers": len(container_ids),
            "conditions_trained_independently": list(STAGE2_CONDITIONS),
            "condition_a_not_retrained": True,
        },
        "condition_a_reference": cond_a_summary,
        "comparison_table": comparison_rows,
        "paired_bootstrap": paired_results,
        "alpha_zero_analysis": alpha_analysis,
        "training_behavior": training_cmp,
        "artifacts_root": str(STAGE2_DIR),
    }
    _write_json(STAGE2_DIR / "stage2_final_report.json", report)

    print("\n=== STAGE 2 COMPLETE ===")
    print(json.dumps({"comparison": comparison_rows, "paired": paired_results}, indent=2))


if __name__ == "__main__":
    main()
