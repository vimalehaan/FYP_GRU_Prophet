#!/usr/bin/env python3
"""Run HCERL — Hybrid v2 Context-Enriched Residual Learning ablation."""

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

from utils.hcerl.ablation import (  # noqa: E402
    build_cohort_comparison_table,
    run_full_ablation_analysis,
)
from utils.hcerl.case_studies import (  # noqa: E402
    build_case_study_narratives,
    case_studies_for_json,
    select_case_study_containers,
)
from utils.hcerl.config import (  # noqa: E402
    EXPERIMENT_NAME,
    HCERL_VARIANTS,
    PROTOCOL_VERSION,
    SEED_POLICY,
    TRAINING_CONFIG,
    VARIANT_FEATURES,
    VARIANT_ORDER,
)
from utils.hcerl.evaluation import (  # noqa: E402
    evaluate_hcerl_variant,
    save_inference_cache,
    summarize_variant_cohort,
)
from utils.hcerl.features import validate_feature_leakage  # noqa: E402
from utils.hcerl.plots import generate_all_plots  # noqa: E402
from utils.hcerl.report import write_final_report  # noqa: E402
from utils.hcerl.training import (  # noqa: E402
    save_variant_artifacts,
    train_hcerl_variant,
)
from utils.peak_detection import load_peak_thresholds  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


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
            "Which contextual information actually improves Hybrid residual "
            "learning beyond the baseline residual-only input?"
        ),
        "variants": HCERL_VARIANTS,
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
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"hcerl_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for sub in (
        "config",
        "variants",
        "ablation",
        "plots",
        "reports",
        "case_studies",
        "verification",
    ):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    config = build_frozen_config(exp_dir, timestamp)
    _write_json(exp_dir / "config" / "hcerl_config_frozen.json", config)

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

    variant_eval_dfs: dict[str, pd.DataFrame] = {}
    variant_summaries: dict[str, Any] = {}
    inference_by_variant: dict[str, dict] = {}
    training_histories: dict[str, pd.DataFrame] = {}

    print("=" * 72)
    print("HCERL — Hybrid v2 Context-Enriched Residual Learning")
    print("=" * 72)
    print(f"Experiment directory: {exp_dir}")
    print(f"Variants: {', '.join(VARIANT_ORDER)}")
    print()

    for variant_id in VARIANT_ORDER:
        features = VARIANT_FEATURES[variant_id]
        variant_dir = exp_dir / "variants" / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 72)
        print(f"Training {variant_id} — features={features}")
        print("-" * 72)

        model, stats, enriched_df, train_meta = train_hcerl_variant(
            global_train=global_train,
            variant_id=variant_id,
            features=features,
            verbose=1,
        )

        leakage_checks = validate_feature_leakage(enriched_df, features)
        _write_json(
            variant_dir / "verification" / "feature_leakage_checks.json",
            leakage_checks,
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
        _write_json(variant_dir / "training" / "training_metadata.json", train_meta)

        eval_df, inference_results, _, failures, skipped = evaluate_hcerl_variant(
            selected_containers=selected,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=model,
            scalers=scalers,
            stats=stats,
            features=features,
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
        variant_summaries[variant_id] = summary
        _write_json(
            variant_dir / "evaluation" / "cohort_summary.json",
            summary,
        )

        variant_eval_dfs[variant_id] = eval_df
        inference_by_variant[variant_id] = inference_results

        _write_json(
            variant_dir / "evaluation" / "failures.json",
            {"failures": failures, "skipped": skipped},
        )

        print(
            f"  Evaluated {len(eval_df)} containers | "
            f"mean MAE={eval_df['day1_mae'].mean():.3f} | "
            f"mean r={eval_df['residual_pearson_r'].mean():.3f}"
        )
        print()

    # Ablation analysis
    print("Running ablation analysis...")
    ablation = run_full_ablation_analysis(variant_eval_dfs)
    cohort_table = build_cohort_comparison_table(variant_eval_dfs, VARIANT_ORDER)

    _write_csv(exp_dir / "ablation" / "cohort_comparison.csv", cohort_table)
    _write_json(
        exp_dir / "ablation" / "paired_tests.json",
        {
            "transitions": [
                {
                    "transition": t["transition"],
                    "metrics": t["metrics"],
                    "n_containers": t["n_containers"],
                }
                for t in ablation["transitions"]
            ],
            "feature_contribution_ranking": ablation["feature_contribution_ranking"],
        },
    )

    for t in ablation["transitions"]:
        safe_name = t["transition"].replace(" ", "_").replace("->", "_to_")
        _write_csv(
            exp_dir / "ablation" / f"deltas_{safe_name}.csv",
            t["per_container_deltas"],
        )

    ranking_df = pd.DataFrame(ablation["feature_contribution_ranking"])
    _write_csv(exp_dir / "ablation" / "feature_contribution_ranking.csv", ranking_df)

    # Case studies
    case_ids = select_case_study_containers(
        variant_eval_dfs["v0_baseline"],
        variant_eval_dfs["v4_full"],
    )
    case_studies = build_case_study_narratives(
        case_ids,
        variant_eval_dfs,
        inference_by_variant,
        VARIANT_ORDER,
    )
    _write_json(
        exp_dir / "case_studies" / "case_studies.json",
        case_studies_for_json(case_studies),
    )

    # Plots
    print("Generating plots...")
    generate_all_plots(
        cohort_table=cohort_table,
        variant_eval_dfs=variant_eval_dfs,
        ablation_result=ablation,
        training_histories=training_histories,
        case_studies=case_studies,
        out_dir=exp_dir / "plots",
    )

    # Final report
    write_final_report(
        exp_dir=exp_dir,
        cohort_table=cohort_table,
        ablation=ablation,
        case_studies=case_studies,
        config=config,
    )

    print("=" * 72)
    print("HCERL complete.")
    print(f"Results: {exp_dir / 'reports' / 'final_report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
