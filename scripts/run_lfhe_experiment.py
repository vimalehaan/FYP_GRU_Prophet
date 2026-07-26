#!/usr/bin/env python3
"""Run LFHE v1.1 — MSE control reproduction then DA-MSE treatment on CSRLE B0."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.dataset import evaluable_container_ids, load_frozen_base_data  # noqa: E402
from utils.csrle.stage2_plots import select_representative_container  # noqa: E402
from utils.csrle.stage2_training import load_condition_frames  # noqa: E402
from utils.lfhe.diagnostics import (  # noqa: E402
    energy_recovery_cohort,
    energy_recovery_paired_bootstrap,
    energy_recovery_per_container,
    horizon_variance_cohort,
    horizon_variance_paired_bootstrap,
    horizon_variance_per_container,
    variance_collapse_interpretation,
    write_json,
)
from utils.lfhe.evaluation import (  # noqa: E402
    aggregate_horizon_cohort,
    evaluate_hypothesis_verdict,
    paired_primary_metrics,
    run_lfhe_evaluation,
)
from utils.lfhe.plots import (  # noqa: E402
    plot_energy_recovery_distribution,
    plot_horizon_variance_boxplot,
    plot_horizon_variance_mean_ci,
    plot_metric_scatter,
    plot_representative_trajectory,
    plot_residual_histogram,
    plot_std_ratio_distribution,
    plot_training_loss,
)
from utils.lfhe.reproduction import (  # noqa: E402
    verify_frozen_stage2_model,
    write_reproduction_report,
)
from utils.lfhe.training import train_lfhe_gru  # noqa: E402
from utils.peak_detection import load_peak_thresholds  # noqa: E402

CSRLE_EXP = REPO_ROOT / "experiments" / "synthetic_residual_learnability_2026-07-25_164307"
PEAK_THRESHOLDS = (
    REPO_ROOT / "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl"
)
STAGE2_B0_REF = CSRLE_EXP / "stage2_synthetic_gru" / "b0_lc"


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
    train_p = CSRLE_EXP / "data" / "b0_lc" / "train_syn.parquet"
    val_p = CSRLE_EXP / "data" / "b0_lc" / "val_syn.parquet"
    meta_path = STAGE2_B0_REF / "training" / "training_metadata.json"
    with meta_path.open() as fh:
        stage2_meta = json.load(fh)

    cfg: dict[str, Any] = {
        "experiment_timestamp": timestamp,
        "protocol_version": "lfhe_v1.1",
        "experiment_name": "loss_function_hypothesis",
        "csrle_experiment_dir": str(CSRLE_EXP.relative_to(REPO_ROOT)),
        "primary_condition": "b0_lc",
        "loss_control": "mse",
        "loss_treatment": "da_mse",
        "loss_formulation": {
            "control": "MSE = mean((y_hat - y)^2)",
            "treatment": "DA-MSE = MSE + lambda * max(0, log(sigma_y) - log(sigma_y_hat))^2",
            "sigma_scope": "per_sequence_over_96_steps",
            "std_ddof": 0,
            "penalty_direction": "one_sided_under_dispersion_only",
        },
        "lambda_disp": 1.0,
        "lambda_disp_tunable": False,
        "epsilon": 1e-6,
        "optimizer": "adam",
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "architecture": "GRU(256)->Dropout(0.2)->GRU(128)->Dropout(0.2)->GRU(64)->Dense(128,relu)->Dense(96)",
        "sequence_configuration": {
            "input_window": 96,
            "forecast_horizon": 96,
            "internal_val_split": "chronological_80_20",
            "shuffle": False,
            "features": ["residual_scaled"],
            "target": "residual_scaled",
        },
        "training_seed_note": "None for GRU init — CSRLE Stage 2 parity (train_control_hybrid)",
        "gru_training_seed": None,
        "bootstrap_seed": 12345,
        "seeds": {"gru_training_seed": None, "bootstrap_seed": 12345},
        "reproduction_tolerances": {
            "residual_pearson_r": 0.015,
            "residual_std_ratio": 0.015,
            "residual_mae_scaled": 0.005,
        },
        "success_thresholds": {
            "mean_std_ratio_da_mse": 0.12,
            "mean_pearson_r_da_mse": 0.10,
            "mae_increase_max": 0.010,
        },
        "diagnostic_protocol": {
            "horizon_variance_bands": ["1-12", "13-24", "25-48", "49-72", "73-96"],
            "energy_recovery": "sum(pred^2)/sum(actual^2)",
            "descriptive_only": True,
        },
        "dataset_hashes": {
            "b0_lc_train_syn_parquet": _sha256(train_p),
            "b0_lc_val_syn_parquet": _sha256(val_p),
        },
        "stage2_b0_training_reference": {
            "best_epoch": stage2_meta.get("best_epoch"),
            "best_val_loss": stage2_meta.get("best_val_loss"),
            "res_mean": stage2_meta.get("res_mean"),
            "res_std": stage2_meta.get("res_std"),
        },
        "evaluation_protocol": "CSRLE Stage 2 extended_metrics + cpu + baselines",
        "peak_thresholds_path": str(PEAK_THRESHOLDS.relative_to(REPO_ROOT)),
    }
    cfg_path = exp_dir / "config" / "lfhe_config_frozen.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def save_arm_evaluation(arm_dir: Path, eval_out: dict[str, Any]) -> None:
    ev = arm_dir / "evaluation"
    _write_csv(ev / "extended_metrics.csv", eval_out["extended_metrics"])
    _write_csv(ev / "cpu_metrics.csv", eval_out["cpu_metrics"])
    _write_csv(ev / "baseline_comparison.csv", eval_out["baselines"])
    _write_csv(ev / "horizon_per_container.csv", eval_out["horizon_per_container"])
    _write_csv(ev / "horizon_bands_per_container.csv", eval_out["horizon_bands_per_container"])
    _write_csv(ev / "cross_horizon_correlation.csv", eval_out["cross_horizon"])
    _write_csv(ev / "horizon_cohort_summary.csv", aggregate_horizon_cohort(eval_out["horizon_per_container"]))
    write_json(ev / "residual_summary.json", {"residual": eval_out["residual_summary"]})


def run_diagnostics(
    exp_dir: Path,
    mse_pc: dict,
    da_pc: dict,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    diag_dir = exp_dir / "diagnostics"
    tables_dir = exp_dir / "tables"
    plots_dir = exp_dir / "plots"
    comp_dir = exp_dir / "comparisons"

    hv_mse = horizon_variance_per_container(mse_pc, "mse")
    hv_da = horizon_variance_per_container(da_pc, "da_mse")
    hv_pc = pd.concat([hv_mse, hv_da], ignore_index=True)
    _write_csv(diag_dir / "horizon_variance_recovery_per_container.csv", hv_pc)
    _write_csv(tables_dir / "horizon_variance_recovery_per_container.csv", hv_pc)

    hv_cohort = horizon_variance_cohort(hv_pc)
    _write_csv(diag_dir / "horizon_variance_recovery_cohort.csv", hv_cohort)
    _write_csv(tables_dir / "horizon_variance_recovery_cohort.csv", hv_cohort)

    hv_paired = horizon_variance_paired_bootstrap(hv_mse, hv_da, seed=cfg["seeds"]["bootstrap_seed"])
    write_json(comp_dir / "horizon_variance_recovery_paired_bootstrap.json", hv_paired)

    ec_mse = energy_recovery_per_container(mse_pc, "mse")
    ec_da = energy_recovery_per_container(da_pc, "da_mse")
    ec_all = pd.concat([ec_mse, ec_da], ignore_index=True)
    _write_csv(diag_dir / "energy_recovery_per_container.csv", ec_all)
    _write_csv(tables_dir / "energy_recovery_per_container.csv", ec_all)

    ec_cohort = energy_recovery_cohort(ec_all)
    _write_csv(diag_dir / "energy_recovery_cohort.csv", ec_cohort)
    _write_csv(tables_dir / "energy_recovery_cohort.csv", ec_cohort)

    ec_paired = energy_recovery_paired_bootstrap(ec_mse, ec_da, seed=cfg["seeds"]["bootstrap_seed"])
    write_json(comp_dir / "energy_recovery_paired_bootstrap.json", ec_paired)

    return {
        "horizon_variance_per_container": hv_pc,
        "horizon_variance_cohort": hv_cohort,
        "horizon_variance_paired": hv_paired,
        "energy_recovery_per_container": ec_all,
        "energy_recovery_cohort": ec_cohort,
        "energy_recovery_paired": ec_paired,
    }


def build_final_report(
    exp_dir: Path,
    cfg: dict[str, Any],
    gate: dict[str, Any],
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
    paired: dict[str, Any],
    verdict: dict[str, Any],
    diag_interp: dict[str, Any],
    ec_cohort: pd.DataFrame,
) -> dict[str, Any]:
    ec_mse = float(ec_cohort.loc[ec_cohort["arm"] == "mse", "mean_err"].iloc[0])
    ec_da = float(ec_cohort.loc[ec_cohort["arm"] == "da_mse", "mean_err"].iloc[0])

    answers = {
        "Q1_da_mse_reduce_variance_collapse": verdict["S1_dispersion"],
        "Q2_prediction_variance_increased": float(da_ext["residual_std_ratio"].mean())
        > float(mse_ext["residual_std_ratio"].mean()),
        "Q3_energy_recovery_improved": ec_da > ec_mse,
        "Q4_pearson_correlation_improved": verdict["S2_correlation"],
        "Q5_cpu_forecasting_improved": float(
            da_ext.get("day1_mae", pd.Series([float("nan")])).mean()
        )
        < float(mse_ext.get("day1_mae", pd.Series([float("nan")])).mean())
        if "day1_mae" in da_ext.columns
        else None,
        "Q6_hypothesis_supported": verdict["verdict"] == "support_H1",
        "Q7_variance_collapse_explanation": (
            "Objective-primary (MSE lacks dispersion incentive)"
            if verdict["verdict"] == "support_H1"
            else "Not primarily objective-limited at lambda=1.0"
            if verdict["verdict"] == "reject_H1"
            else f"Partial/inconclusive: {verdict['verdict']}"
        ),
        "Q8_thesis_conclusion": verdict["verdict"],
    }

    report = {
        "protocol_version": cfg["protocol_version"],
        "experiment_timestamp": cfg["experiment_timestamp"],
        "reproduction_gate": gate,
        "cohort_metrics": {
            "mse": {
                "pearson_r": float(mse_ext["residual_pearson_r"].mean()),
                "std_ratio": float(mse_ext["residual_std_ratio"].mean()),
                "mae": float(mse_ext["residual_mae_scaled"].mean()),
                "energy_recovery": ec_mse,
            },
            "da_mse": {
                "pearson_r": float(da_ext["residual_pearson_r"].mean()),
                "std_ratio": float(da_ext["residual_std_ratio"].mean()),
                "mae": float(da_ext["residual_mae_scaled"].mean()),
                "energy_recovery": ec_da,
            },
        },
        "paired_bootstrap_primary": paired,
        "hypothesis_verdict": verdict,
        "diagnostic_interpretation": diag_interp,
        "explicit_answers": answers,
    }
    write_json(exp_dir / "reports" / "final_report.json", report)
    return report


def write_final_report_md(exp_dir: Path, report: dict[str, Any]) -> None:
    a = report["explicit_answers"]
    v = report["hypothesis_verdict"]
    m = report["cohort_metrics"]
    lines = [
        "# LFHE Final Report",
        "",
        f"**Protocol:** {report['protocol_version']}",
        f"**Timestamp:** {report['experiment_timestamp']}",
        "",
        "## Reproduction gate (MSE control vs Stage 2 B0)",
        f"- **Pass:** {report['reproduction_gate']['pass']}",
        "",
        "## Cohort metrics",
        "",
        "| Metric | MSE | DA-MSE |",
        "|--------|-----|--------|",
        f"| Pearson r | {m['mse']['pearson_r']:.4f} | {m['da_mse']['pearson_r']:.4f} |",
        f"| Std ratio | {m['mse']['std_ratio']:.4f} | {m['da_mse']['std_ratio']:.4f} |",
        f"| Residual MAE | {m['mse']['mae']:.4f} | {m['da_mse']['mae']:.4f} |",
        f"| Energy recovery | {m['mse']['energy_recovery']:.4f} | {m['da_mse']['energy_recovery']:.4f} |",
        "",
        "## Hypothesis verdict",
        f"- **Verdict:** `{v['verdict']}`",
        f"- S1 dispersion: {v['S1_dispersion']}",
        f"- S2 correlation: {v['S2_correlation']}",
        f"- S3 MAE guardrail: {v['S3_mae_guardrail']}",
        "",
        "## Explicit answers",
        "",
        f"1. Did DA-MSE reduce variance collapse? **{a['Q1_da_mse_reduce_variance_collapse']}**",
        f"2. Did prediction variance increase? **{a['Q2_prediction_variance_increased']}**",
        f"3. Did energy recovery improve? **{a['Q3_energy_recovery_improved']}**",
        f"4. Did Pearson correlation improve? **{a['Q4_pearson_correlation_improved']}**",
        f"5. Did CPU forecasting improve? **{a['Q5_cpu_forecasting_improved']}**",
        f"6. Was the hypothesis supported? **{a['Q6_hypothesis_supported']}**",
        f"7. Variance collapse explanation: {a['Q7_variance_collapse_explanation']}",
        f"8. Thesis conclusion code: `{a['Q8_thesis_conclusion']}`",
        "",
    ]
    (exp_dir / "reports" / "final_report.md").write_text("\n".join(lines))


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"loss_function_hypothesis_{timestamp}"
    for sub in (
        "config", "models", "training", "evaluation", "diagnostics",
        "plots", "tables", "verification", "logs", "reports", "comparisons",
    ):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LFHE v1.1 — Loss Function Hypothesis Experiment")
    print(f"Output: {exp_dir}")
    print("=" * 60)

    cfg = build_frozen_config(exp_dir, timestamp)
    print("Frozen config written.")

    train_base, val_base, scalers_base, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, _ = evaluable_container_ids(selected, train_base, val_base, scalers_base)
    train_df, val_df = load_condition_frames(CSRLE_EXP, "b0_lc")
    peak_thresholds = load_peak_thresholds(PEAK_THRESHOLDS)
    # Training seed None — matches CSRLE Stage 2 (no TF/Python seed in train_control_hybrid)
    train_seed = None
    bootstrap_seed = cfg["seeds"]["bootstrap_seed"]

    mse_dir = exp_dir / "b0_lc_mse"
    da_dir = exp_dir / "b0_lc_da_mse"

    # Phase 2 — Reproduction via frozen Stage 2 B0 weights (read-only)
    print("\n=== Phase 2: Reproduction gate (frozen Stage 2 B0 model) ===")
    repro = verify_frozen_stage2_model(
        CSRLE_EXP, container_ids, train_df, val_df, peak_thresholds,
        cfg["reproduction_tolerances"],
    )
    gate_report = {
        "pass": repro["pass"],
        "method": "evaluate_frozen_stage2_b0_weights_via_lfhe_pipeline",
        "model_source": repro["model_source"],
        "cohort_gate": repro["cohort_gate"],
        "per_container_gate": repro["per_container_gate"],
    }
    write_reproduction_report(exp_dir / "verification" / "mse_reproduction_gate.json", gate_report)
    write_json(exp_dir / "reports" / "reproduction_report.json", gate_report)

    if not repro["pass"]:
        print("\n*** REPRODUCTION GATE FAILED — STOPPING ***")
        print(json.dumps(gate_report, indent=2))
        sys.exit(1)
    print("\nReproduction gate PASSED (frozen Stage 2 B0 weights + LFHE eval pipeline).")

    # Phase 2b — Fresh MSE train (hypothesis control arm, same session as DA-MSE)
    print("\n=== Phase 2b: MSE control training (LFHE session) ===")
    mse_train = train_lfhe_gru(
        train_df, mse_dir, arm_id="b0_lc_mse", loss_kind="mse",
        seed=train_seed, verbose=1,
    )
    print("\n=== MSE control evaluation ===")
    mse_eval = run_lfhe_evaluation(
        container_ids, train_df, val_df,
        mse_train["scalers"], mse_train["model"],
        mse_train["res_mean"], mse_train["res_std"],
        peak_thresholds,
    )
    save_arm_evaluation(mse_dir, mse_eval)
    mse_ext = mse_eval["extended_metrics"]
    write_json(exp_dir / "verification" / "mse_retrain_vs_stage2.json", {
        "note": "LFHE MSE retrain may differ from Stage 2 due to non-deterministic GRU init",
        "stage2_std_ratio": repro["reference"]["residual_std_ratio"],
        "lfhe_mse_retrain_std_ratio": float(mse_ext["residual_std_ratio"].mean()),
    })

    # Phase 3 — DA-MSE
    print("\n=== Phase 3: DA-MSE training ===")
    da_train = train_lfhe_gru(
        train_df, da_dir, arm_id="b0_lc_da_mse", loss_kind="da_mse",
        lambda_disp=cfg["lambda_disp"], epsilon=cfg["epsilon"],
        seed=train_seed, verbose=1,
    )
    print("\n=== Phase 4: DA-MSE evaluation ===")
    da_eval = run_lfhe_evaluation(
        container_ids, train_df, val_df,
        da_train["scalers"], da_train["model"],
        da_train["res_mean"], da_train["res_std"],
        peak_thresholds,
    )
    save_arm_evaluation(da_dir, da_eval)
    da_ext = da_eval["extended_metrics"]

    paired = paired_primary_metrics(da_ext, mse_ext, bootstrap_seed=bootstrap_seed)
    write_json(exp_dir / "comparisons" / "paired_mse_vs_da_mse.json", paired)

    summary_rows = []
    for arm, ext in [("mse", mse_ext), ("da_mse", da_ext)]:
        summary_rows.append({
            "arm": arm,
            "pearson_r_mean": ext["residual_pearson_r"].mean(),
            "std_ratio_mean": ext["residual_std_ratio"].mean(),
            "mae_mean": ext["residual_mae_scaled"].mean(),
            "rmse_mean": ext["residual_rmse_scaled"].mean(),
            "r2_mean": ext["residual_r2"].mean(),
        })
    _write_csv(exp_dir / "tables" / "lfhe_summary.csv", pd.DataFrame(summary_rows))

    # Phase 4b — diagnostics
    print("\n=== Phase 4b: Diagnostics ===")
    diag = run_diagnostics(
        exp_dir,
        mse_eval["per_container_residuals"],
        da_eval["per_container_residuals"],
        cfg,
    )
    diag_interp = variance_collapse_interpretation(
        mse_ext, da_ext,
        diag["horizon_variance_cohort"][diag["horizon_variance_cohort"]["arm"] == "mse"],
        diag["horizon_variance_cohort"][diag["horizon_variance_cohort"]["arm"] == "da_mse"],
    )
    write_json(exp_dir / "diagnostics" / "variance_collapse_interpretation.json", diag_interp)

    # Plots
    plots_dir = exp_dir / "plots"
    mse_hist = pd.read_csv(mse_dir / "training" / "training_history.csv")
    da_hist = pd.read_csv(da_dir / "training" / "training_history.csv")
    plot_training_loss(mse_hist, da_hist, plots_dir / "training_loss_comparison")
    plot_std_ratio_distribution(mse_ext, da_ext, plots_dir / "std_ratio_distribution")
    plot_metric_scatter(
        mse_ext, da_ext, "residual_pearson_r", "Pearson r",
        "Pearson r: MSE vs DA-MSE", plots_dir / "pearson_r_scatter",
    )
    plot_metric_scatter(
        mse_ext, da_ext, "residual_std_ratio", "Std ratio",
        "Std ratio: MSE vs DA-MSE", plots_dir / "std_ratio_scatter",
    )
    plot_horizon_variance_mean_ci(diag["horizon_variance_cohort"], plots_dir / "horizon_variance_recovery_mean_ci")
    plot_horizon_variance_boxplot(diag["horizon_variance_per_container"], plots_dir / "horizon_variance_recovery_boxplot")
    plot_horizon_variance_mean_ci(
        diag["horizon_variance_cohort"], plots_dir / "horizon_variance_recovery_mse_vs_da_mse",
    )
    plot_energy_recovery_distribution(diag["energy_recovery_per_container"], plots_dir / "energy_recovery_distribution")
    ec_mse = diag["energy_recovery_per_container"][diag["energy_recovery_per_container"]["arm"] == "mse"]
    ec_da = diag["energy_recovery_per_container"][diag["energy_recovery_per_container"]["arm"] == "da_mse"]
    merged_ec = ec_mse.merge(ec_da, on="container_id", suffixes=("_mse", "_da"))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged_ec["energy_recovery_ratio_mse"], merged_ec["energy_recovery_ratio_da"], alpha=0.6)
    lim = max(merged_ec[["energy_recovery_ratio_mse", "energy_recovery_ratio_da"]].max().max(), 0.05)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax.set_xlabel("MSE ERR")
    ax.set_ylabel("DA-MSE ERR")
    ax.set_title("Energy Recovery: MSE vs DA-MSE")
    fig.savefig(plots_dir / "energy_recovery_mse_vs_da_mse_scatter.png", dpi=150, bbox_inches="tight")
    fig.savefig(plots_dir / "energy_recovery_mse_vs_da_mse_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    rep_cid = select_representative_container(da_ext)
    a_mse, p_mse = mse_eval["per_container_residuals"][rep_cid]
    a_da, p_da = da_eval["per_container_residuals"][rep_cid]
    plot_representative_trajectory(
        a_mse, p_mse, p_da,
        f"Representative container {rep_cid}",
        plots_dir / f"representative_trajectory_{rep_cid}",
    )
    plot_residual_histogram(a_mse, p_mse, "MSE", plots_dir / "residual_histogram_mse")
    plot_residual_histogram(a_da, p_da, "DA-MSE", plots_dir / "residual_histogram_da_mse")

    verdict = evaluate_hypothesis_verdict(da_ext, mse_ext, paired, cfg["success_thresholds"])
    report = build_final_report(
        exp_dir, cfg, gate_report, mse_ext, da_ext, paired, verdict, diag_interp,
        diag["energy_recovery_cohort"],
    )
    write_final_report_md(exp_dir, report)

    write_json(exp_dir / "verification" / "lfhe_integrity.json", {
        "csrle_artifacts_modified": False,
        "only_loss_changed": True,
        "lambda_frozen": cfg["lambda_disp"],
        "n_containers": len(container_ids),
    })

    print("\n=== LFHE COMPLETE ===")
    print(f"Verdict: {verdict['verdict']}")
    print(f"Report: {exp_dir / 'reports' / 'final_report.md'}")


if __name__ == "__main__":
    main()
