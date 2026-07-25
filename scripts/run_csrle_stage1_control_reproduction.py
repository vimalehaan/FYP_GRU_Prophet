#!/usr/bin/env python3
"""CSRLE Stage 1: Condition A Hybrid methodology reproduction."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.config import CONTROL_REPRODUCTION_GATES  # noqa: E402
from utils.csrle.control_reproduction import (  # noqa: E402
    build_methodology_record,
    compare_to_frozen_baseline,
    compute_container_extended_metrics,
    frozen_eval_path_sanity,
    train_control_hybrid,
)
from utils.csrle.dataset import evaluable_container_ids, load_frozen_base_data  # noqa: E402
from utils.hybrid_artifacts import load_hybrid_artifacts, save_hybrid_artifacts  # noqa: E402
from utils.hybrid_config import DEFAULT_INPUT_WINDOW  # noqa: E402
from utils.hybrid_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.peak_detection import load_peak_thresholds  # noqa: E402

CSRLE_EXP = REPO_ROOT / "experiments" / "synthetic_residual_learnability_2026-07-25_164307"
STAGE_DIR = CSRLE_EXP / "stage1_control_reproduction"
FROZEN_BASELINE_DIR = REPO_ROOT / "experiments" / "baseline_reference_2026-07-14"
FROZEN_EVAL_CSV = FROZEN_BASELINE_DIR / "evaluation" / "evaluation_df.csv"
PEAK_THRESHOLDS = (
    REPO_ROOT / "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stage_dir = STAGE_DIR
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSRLE Stage 1 — Condition A Hybrid reproduction")
    print(f"Stage directory: {stage_dir}")
    print(f"Pre-GRU artifacts preserved at: {CSRLE_EXP}")

    methodology = build_methodology_record()
    _write_json(stage_dir / "config" / "methodology_reconstruction.json", methodology)

    gate_def = {
        "purpose": "Methodology reproduction gate (aggregate + rank stability)",
        "rationale": (
            "Frozen baseline trained via train_hybrid_gru without enforced TF seeds; "
            "exact weight replication is not expected. Pre-specified tolerances from "
            "utils/csrle/config.py CONTROL_REPRODUCTION_GATES mirror CSRLE design."
        ),
        "nondeterminism_sources": [
            "TensorFlow/Keras weight initialization",
            "GPU/CPU backend floating-point ordering",
            "Prophet/Stan MCMC (per-container refit at inference only)",
            "Early stopping epoch selection sensitivity",
        ],
        "gates": CONTROL_REPRODUCTION_GATES,
        "frozen_eval_path_sanity": {
            "purpose": "Isolate pipeline/evaluation differences from retraining",
            "method": "Load frozen baseline model; evaluate via CSRLE evaluation path",
            "expected": "Exact match to frozen evaluation_df.csv (same model)",
            "tolerance": 1e-9,
        },
    }
    _write_json(stage_dir / "config" / "reproduction_gate_definition.json", gate_def)

    train_df, val_df, scalers, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, skipped = evaluable_container_ids(
        selected, train_df, val_df, scalers
    )
    print(f"Evaluable containers: {len(container_ids)}; skipped: {skipped}")

    global_train = train_df[train_df["container_id"].isin(container_ids)].copy()
    global_val = val_df[val_df["container_id"].isin(container_ids)].copy()

    peak_thresholds = load_peak_thresholds(PEAK_THRESHOLDS)

    # --- Frozen model through CSRLE evaluation path (read-only) ---
    print("Frozen-model CSRLE evaluation-path sanity check...")
    frozen_model, frozen_res_mean, frozen_res_std, frozen_iw = load_hybrid_artifacts(
        FROZEN_BASELINE_DIR / "models" / "hybrid_gru.keras",
        FROZEN_BASELINE_DIR / "models" / "residual_stats.pkl",
    )
    assert frozen_iw == DEFAULT_INPUT_WINDOW

    frozen_eval_df, frozen_inf, frozen_fail, frozen_skip = evaluate_selected_containers(
        container_ids,
        global_train,
        global_val,
        frozen_model,
        scalers,
        frozen_res_mean,
        frozen_res_std,
        input_window=DEFAULT_INPUT_WINDOW,
    )
    frozen_baseline_csv = pd.read_csv(FROZEN_EVAL_CSV)
    sanity = frozen_eval_path_sanity(frozen_eval_df, frozen_baseline_csv)
    _write_json(stage_dir / "verification" / "frozen_eval_path_sanity.json", sanity)
    _write_csv(
        stage_dir / "evaluation" / "frozen_model_csrle_path_evaluation_df.csv",
        frozen_eval_df,
    )
    print(f"  Frozen eval path sanity: {'PASS' if sanity['pass'] else 'FAIL'}")
    print(f"  Max diffs: {sanity['max_abs_diff']}")

    # --- Train NEW Condition A model ---
    print("Training new Condition A Hybrid GRU (methodology parity)...")
    model, res_mean, res_std, train_residual_df, train_meta = train_control_hybrid(
        global_train,
        input_window=DEFAULT_INPUT_WINDOW,
        verbose=1,
    )

    model_path = stage_dir / "models" / "hybrid_gru_control_reproduction.keras"
    stats_path = stage_dir / "models" / "residual_stats.pkl"
    save_hybrid_artifacts(
        model, res_mean, res_std,
        model_path=model_path,
        residual_stats_path=stats_path,
        input_window=DEFAULT_INPUT_WINDOW,
    )

    hist = train_meta.pop("history")
    _write_csv(stage_dir / "training" / "training_history.csv", pd.DataFrame(hist))
    train_meta["trained_at"] = timestamp
    train_meta["model_path"] = str(model_path)
    train_meta["frozen_baseline_comparison_model"] = str(
        FROZEN_BASELINE_DIR / "models/hybrid_gru.keras"
    )
    _write_json(stage_dir / "training" / "training_metadata.json", train_meta)

    print(
        f"  Sequences: total={train_meta['total_sequences']} "
        f"train={train_meta['train_sequences']} val={train_meta['val_sequences']}"
    )
    print(
        f"  Epochs: {train_meta['epochs_run']} best={train_meta['best_epoch']} "
        f"best_val_loss={train_meta['best_val_loss']:.6f}"
    )

    # --- Evaluate new model ---
    print("Evaluating new Condition A model...")
    new_eval_df, new_inf, new_fail, new_skip = evaluate_selected_containers(
        container_ids,
        global_train,
        global_val,
        model,
        scalers,
        res_mean,
        res_std,
        input_window=DEFAULT_INPUT_WINDOW,
    )
    new_summary = summarize_evaluation_metrics(new_eval_df)
    _write_csv(
        stage_dir / "evaluation" / "csrle_new_model_evaluation_df.csv",
        new_eval_df,
    )
    _write_csv(stage_dir / "evaluation" / "csrle_new_model_summary.csv", new_summary)

    ext_rows = []
    for cid in container_ids:
        if cid in new_inf:
            ext_rows.append(
                compute_container_extended_metrics(
                    new_inf[cid], peak_thresholds[cid],
                )
            )
    ext_df = pd.DataFrame(ext_rows)
    _write_csv(
        stage_dir / "evaluation" / "csrle_new_model_extended_metrics.csv",
        ext_df,
    )

    # --- Reproduction comparison ---
    print("Comparing new Condition A vs frozen baseline...")
    repro = compare_to_frozen_baseline(
        new_eval_df, frozen_baseline_csv, CONTROL_REPRODUCTION_GATES,
    )
    _write_csv(
        stage_dir / "evaluation" / "reproduction_comparison.csv",
        repro["merged_per_container"],
    )
    repro_summary = {
        "reproduction_gate_pass": repro["pass"],
        "summary": repro["summary"],
        "checks": repro["checks"],
        "extended_metrics_pooled": {
            "peak_mae_mean": float(ext_df["peak_mae"].mean()),
            "peak_rmse_mean": float(ext_df["peak_rmse"].mean()),
            "non_peak_mae_mean": float(ext_df["non_peak_mae"].mean()),
            "non_peak_rmse_mean": float(ext_df["non_peak_rmse"].mean()),
            "residual_mae_scaled_mean": float(ext_df["residual_mae_scaled"].mean()),
            "residual_rmse_scaled_mean": float(ext_df["residual_rmse_scaled"].mean()),
            "residual_pearson_r_mean": float(ext_df["residual_pearson_r"].mean()),
            "residual_r2_mean": float(ext_df["residual_r2"].mean()),
            "residual_std_ratio_mean": float(ext_df["residual_std_ratio"].mean()),
        },
        "new_model_day1": {
            "mae_mean": float(new_eval_df["day1_mae"].mean()),
            "rmse_mean": float(new_eval_df["day1_rmse"].mean()),
        },
        "frozen_baseline_day1": {
            "mae_mean": CONTROL_REPRODUCTION_GATES["frozen_mae_mean"],
            "rmse_mean": CONTROL_REPRODUCTION_GATES["frozen_rmse_mean"],
        },
        "frozen_eval_path_sanity_pass": sanity["pass"],
    }
    _write_json(
        stage_dir / "evaluation" / "reproduction_summary.json",
        repro_summary,
    )
    _write_json(
        stage_dir / "verification" / "reproduction_gate_result.json",
        {
            "pass": repro["pass"] and sanity["pass"],
            "reproduction": repro["checks"],
            "frozen_eval_path_sanity": sanity,
        },
    )

    verification = {
        "stage": "stage1_control_reproduction",
        "timestamp": timestamp,
        "pre_gru_run_preserved": str(CSRLE_EXP),
        "synthetic_conditions_trained": [],
        "b0_b1_c0_c1_trained": False,
        "gru_training_occurred": True,
        "gru_training_scope": "control_only",
        "frozen_baseline_unmodified": True,
        "reproduction_gate_pass": repro["pass"],
        "frozen_eval_path_sanity_pass": sanity["pass"],
        "recommendation": (
            "READY FOR SYNTHETIC GRU STAGE"
            if repro["pass"] and sanity["pass"]
            else "NOT READY"
        ),
    }
    _write_json(stage_dir / "verification" / "stage1_verification.json", verification)

    print("\n=== STAGE 1 SUMMARY ===")
    print(json.dumps(repro_summary, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
