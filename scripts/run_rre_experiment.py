#!/usr/bin/env python3
"""Run RRE v1.0 — Residual Scaling Experiment (R0 vs R3)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.peak_detection import load_peak_thresholds  # noqa: E402
from utils.rre.comparison import compare_variants  # noqa: E402
from utils.rre.config import (  # noqa: E402
    EXPERIMENT_NAME,
    PROTOCOL_VERSION,
    SEED_POLICY,
    STAGE0_REFERENCE,
    TRAINING_CONFIG,
    VARIANT_ORDER,
    VARIANTS,
)
from utils.rre.evaluation import (  # noqa: E402
    evaluate_rre_variant,
    optimization_summary,
    save_inference_cache,
    summarize_variant_cohort,
)
from utils.rre.report import (  # noqa: E402
    answer_research_questions,
    build_final_report_md,
    write_json,
)
from utils.rre.training import save_variant_artifacts, train_rre_variant  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_frozen_config(exp_dir: Path, timestamp: str) -> dict[str, Any]:
    train_p = REPO_ROOT / "data/train_df.parquet"
    val_p = REPO_ROOT / "data/val_df.parquet"
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_timestamp": timestamp,
        "experiment_dir": str(exp_dir.relative_to(REPO_ROOT)),
        "research_question": (
            "Can a more optimisation-friendly scaling of the SAME Prophet residual "
            "improve Hybrid GRU learning without changing the underlying temporal "
            "information?"
        ),
        "stage0_reference": STAGE0_REFERENCE,
        "distinction": (
            "NOT a temporal representation experiment. Stage 0 showed R3 preserves "
            "identical temporal structure to R0. Only numerical conditioning differs."
        ),
        "variants": VARIANTS,
        "variant_order": list(VARIANT_ORDER),
        "training_config": TRAINING_CONFIG,
        "seed_policy": SEED_POLICY,
        "frozen_baseline_ref": "experiments/baseline_reference_2026-07-14",
        "data_hashes": {
            "train_df.parquet": _sha256(train_p),
            "val_df.parquet": _sha256(val_p),
        },
        "cohort": {
            "selected_containers": "data/selected_containers.npy",
            "expected_evaluable": 99,
        },
        "only_change": "residual scaling method (R0 global z-score vs R3 median/MAD)",
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"rre_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("config", "variants", "comparison", "reports", "verification"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    config = build_frozen_config(exp_dir, timestamp)
    write_json(exp_dir / "config" / "rre_config_frozen.json", config)

    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(REPO_ROOT / "data/selected_containers.npy", allow_pickle=True)
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)

    peak_thresholds = load_peak_thresholds(
        REPO_ROOT / "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl",
    )

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    eval_dfs: dict[str, pd.DataFrame] = {}
    cohort_summaries: dict[str, Any] = {}
    opt_summaries: dict[str, Any] = {}
    training_histories: dict[str, pd.DataFrame] = {}

    print("=" * 72)
    print("RRE v1.0 — Residual Scaling Experiment (R0 vs R3)")
    print("=" * 72)
    print(f"Experiment directory: {exp_dir}")
    print()

    for variant_id in VARIANT_ORDER:
        variant_dir = exp_dir / "variants" / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 72)
        print(f"Training {variant_id}: {VARIANTS[variant_id]['label']}")
        print("-" * 72)

        model, stats, _, train_meta = train_rre_variant(
            global_train=global_train,
            variant_id=variant_id,
            verbose=1,
        )

        save_variant_artifacts(model, stats, variant_dir)

        hist = train_meta.pop("history")
        history_df = pd.DataFrame({
            "epoch": np.arange(1, len(hist["loss"]) + 1),
            "loss": hist["loss"],
            "val_loss": hist["val_loss"],
            "mae": hist.get("mae", [np.nan] * len(hist["loss"])),
            "val_mae": hist.get("val_mae", [np.nan] * len(hist["loss"])),
        })
        training_histories[variant_id] = history_df
        _write_csv(variant_dir / "training" / "training_history.csv", history_df)

        opt = optimization_summary({**train_meta, "history": hist})
        opt_summaries[variant_id] = opt
        write_json(variant_dir / "training" / "training_metadata.json", {**train_meta, "history": hist})
        write_json(variant_dir / "training" / "optimization_summary.json", opt)

        eval_df, inference_results, failures, skipped = evaluate_rre_variant(
            selected_containers=selected,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=model,
            scalers=scalers,
            scaling_stats=stats,
            peak_thresholds=peak_thresholds,
            input_window=TRAINING_CONFIG["input_window"],
            variant_id=variant_id,
        )

        _write_csv(variant_dir / "evaluation" / "evaluation_extended.csv", eval_df)
        save_inference_cache(
            inference_results,
            variant_dir / "evaluation" / "inference_cache.pkl",
        )
        summary = summarize_variant_cohort(eval_df)
        cohort_summaries[variant_id] = summary
        write_json(variant_dir / "evaluation" / "cohort_summary.json", summary)
        write_json(
            variant_dir / "verification" / "failures.json",
            {"failures": failures, "skipped": skipped},
        )

        eval_dfs[variant_id] = eval_df
        print(f"  Evaluated {len(eval_df)} containers; failures={len(failures)}")

    comparison = compare_variants(
        eval_dfs["R0"],
        eval_dfs["R3"],
        opt_summaries["R0"],
        opt_summaries["R3"],
    )
    write_json(exp_dir / "comparison" / "paired_tests.json", comparison)

    merged = eval_dfs["R0"].merge(
        eval_dfs["R3"],
        on="container_id",
        suffixes=("_R0", "_R3"),
    )
    _write_csv(exp_dir / "comparison" / "per_container_paired.csv", merged)

    answers = answer_research_questions(
        comparison,
        cohort_summaries["R0"],
        cohort_summaries["R3"],
        opt_summaries["R0"],
        opt_summaries["R3"],
    )

    final_report = {
        "config": config,
        "cohort_summaries": cohort_summaries,
        "optimization_summaries": opt_summaries,
        "comparison": comparison,
        "answers": answers,
    }
    write_json(exp_dir / "reports" / "final_report.json", final_report)

    md = build_final_report_md(
        config,
        comparison,
        answers,
        cohort_summaries["R0"],
        cohort_summaries["R3"],
    )
    (exp_dir / "reports" / "final_report.md").write_text(md)

    print()
    print("=" * 72)
    print("RRE complete")
    print("=" * 72)
    print(f"R0 mean day1 MAE: {cohort_summaries['R0']['day1_mae']['mean']:.4f}")
    print(f"R3 mean day1 MAE: {cohort_summaries['R3']['day1_mae']['mean']:.4f}")
    print(f"Primary answer (accuracy): {answers['q1_accuracy_improved']['answer']}")
    print(f"Replace baseline scaling: {answers['q7_replace_baseline_scaling']['answer']}")
    print(f"Report: {exp_dir / 'reports' / 'final_report.md'}")


if __name__ == "__main__":
    main()
