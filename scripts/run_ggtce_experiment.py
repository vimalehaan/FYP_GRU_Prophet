#!/usr/bin/env python3
"""Run GGTCE — Global GRU Temporal Context Experiment (G96/G192/G288)."""

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

from utils.ggtce.ablation import (  # noqa: E402
    build_cohort_comparison_table,
    run_window_comparison,
    sequence_reduction_table,
)
from utils.ggtce.case_studies import (  # noqa: E402
    build_case_study_narratives,
    case_studies_for_json,
    select_case_study_containers,
)
from utils.ggtce.config import (  # noqa: E402
    EXPERIMENT_NAME,
    FROZEN_GLOBAL_BASELINE,
    GGTCE_VARIANTS,
    PROTOCOL_VERSION,
    SEED_POLICY,
    TMA_EXPERIMENT,
    TRAINING_CONFIG,
    VARIANT_ORDER,
    VARIANT_WINDOWS,
)
from utils.ggtce.evaluation import (  # noqa: E402
    evaluate_ggtce_variant,
    save_inference_cache,
    summarize_variant_cohort,
)
from utils.ggtce.memory_analysis import run_memory_analysis  # noqa: E402
from utils.ggtce.plots import generate_all_plots  # noqa: E402
from utils.ggtce.report import write_final_report, _df_to_md_table  # noqa: E402
from utils.ggtce.training import (  # noqa: E402
    load_variant_artifacts,
    save_variant_artifacts,
    train_ggtce_variant,
)
from utils.ggtce.verification import (  # noqa: E402
    verify_frozen_parity,
    verify_g96_replication,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


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
            "Does increasing the historical input window of the Global GRU improve "
            "long-term CPU forecasting because additional temporal memory exists "
            "beyond the current 96-step window?"
        ),
        "co_primary_question": (
            "Do workloads with stronger long-range temporal dependence benefit more "
            "from longer input windows?"
        ),
        "variants": GGTCE_VARIANTS,
        "variant_order": list(VARIANT_ORDER),
        "training_config": TRAINING_CONFIG,
        "seed_policy": SEED_POLICY,
        "frozen_baseline_ref": FROZEN_GLOBAL_BASELINE,
        "tma_ref": TMA_EXPERIMENT,
        "data_hashes": {
            "train_df.parquet": _sha256(train_p),
            "val_df.parquet": _sha256(val_p),
        },
        "cohort": {
            "selected_containers": "data/selected_containers.npy",
            "expected_evaluable": 99,
        },
        "frozen_note": "Only input_window varies (96, 192, 288). All else frozen.",
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"ggtce_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for sub in (
        "config",
        "variants",
        "window_comparison",
        "memory_analysis",
        "plots",
        "reports",
        "case_studies",
        "verification",
        "sequence_analysis",
    ):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    config = build_frozen_config(exp_dir, timestamp)
    _write_json(exp_dir / "config" / "ggtce_config_frozen.json", config)

    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(REPO_ROOT / "data/selected_containers.npy", allow_pickle=True)
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    print("=" * 72)
    print("GGTCE — Global GRU Temporal Context Experiment")
    print("=" * 72)
    print(f"Experiment directory: {exp_dir}")
    print(f"Variants: {', '.join(VARIANT_ORDER)}")
    print()

    # Pre-training parity verification
    parity = verify_frozen_parity(REPO_ROOT, global_train, global_val, selected)
    _write_json(exp_dir / "verification" / "frozen_parity.json", parity)
    print("Frozen parity verification:")
    for check in parity["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['check']}: {check['detail']}")
    if not parity["all_passed"]:
        print("WARNING: Some parity checks failed — review verification/frozen_parity.json")
    print()

    variant_eval_dfs: dict[str, pd.DataFrame] = {}
    variant_summaries: dict[str, Any] = {}
    inference_by_variant: dict[str, dict] = {}
    training_meta: dict[str, dict[str, Any]] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}

    for variant_id in VARIANT_ORDER:
        input_window = VARIANT_WINDOWS[variant_id]
        variant_dir = exp_dir / "variants" / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 72)
        print(f"Training {variant_id} — input_window={input_window}")
        print("-" * 72)

        model, train_meta = train_ggtce_variant(
            global_train=global_train,
            variant_id=variant_id,
            input_window=input_window,
            verbose=1,
        )

        save_variant_artifacts(model, train_meta, variant_dir, input_window)

        hist = train_meta.pop("history")
        training_histories[variant_id] = hist
        history_df = pd.DataFrame({
            "epoch": np.arange(1, len(hist["loss"]) + 1),
            "loss": hist["loss"],
            "val_loss": hist["val_loss"],
            "mae": hist.get("mae", [np.nan] * len(hist["loss"])),
            "val_mae": hist.get("val_mae", [np.nan] * len(hist["loss"])),
        })
        _write_csv(variant_dir / "training" / "training_history.csv", history_df)
        _write_json(variant_dir / "training" / "training_metadata.json", train_meta)
        training_meta[variant_id] = train_meta

        eval_df, inference_results, failures, skipped = evaluate_ggtce_variant(
            selected_containers=selected,
            global_train=global_train,
            global_val=global_val,
            global_gru_model=model,
            scalers=scalers,
            input_window=input_window,
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
            f"  Sequences={train_meta['n_sequences']} | "
            f"Evaluated {len(eval_df)} containers | "
            f"mean MAE={eval_df['day1_mae'].mean():.3f} | "
            f"mean r={eval_df['forecast_pearson_r'].mean():.3f} | "
            f"train time={train_meta['training_seconds']:.1f}s"
        )
        print()

    # G96 replication gate
    g96_replication = verify_g96_replication(variant_eval_dfs["g96"], REPO_ROOT)
    _write_json(exp_dir / "verification" / "g96_replication.json", g96_replication)
    print(
        f"G96 replication: {'PASSED' if g96_replication['passed'] else 'FAILED'} "
        f"(Δ mean MAE={g96_replication['mean_mae_abs_delta']:.4f})"
    )
    print()

    # Window comparison
    print("Running window comparison analysis...")
    window_comparison = run_window_comparison(variant_eval_dfs)
    cohort_table = build_cohort_comparison_table(variant_eval_dfs)
    seq_table = sequence_reduction_table(training_meta)
    seq_table = seq_table.merge(
        cohort_table[["variant_id", "mean_day1_mae", "mean_forecast_pearson_r"]],
        on="variant_id",
        how="left",
    )

    _write_csv(exp_dir / "window_comparison" / "cohort_comparison.csv", cohort_table)
    _write_csv(exp_dir / "sequence_analysis" / "sequence_reduction.csv", seq_table)
    _write_json(
        exp_dir / "window_comparison" / "paired_tests.json",
        {
            "transitions": [
                {
                    "transition": t["transition"],
                    "metrics": t["metrics"],
                    "n_containers": t["n_containers"],
                }
                for t in window_comparison["transitions"]
            ],
        },
    )
    for key, deltas_df in window_comparison["delta_tables"].items():
        _write_csv(exp_dir / "window_comparison" / f"deltas_{key}.csv", deltas_df)

    # Memory-aware co-primary analysis
    print("Running memory-aware analysis (TMA-linked)...")
    memory_result = run_memory_analysis(
        REPO_ROOT,
        global_train,
        global_val,
        window_comparison["delta_tables"],
        primary_transition=("g96", "g288"),
    )
    _write_csv(
        exp_dir / "memory_analysis" / "memory_table.csv",
        memory_result["memory_table"],
    )
    _write_csv(
        exp_dir / "memory_analysis" / "merged_deltas_g96_g288.csv",
        memory_result["merged_deltas"],
    )
    _write_json(
        exp_dir / "memory_analysis" / "correlation_and_subgroups.json",
        {
            "correlation_mae": memory_result["correlation_mae"],
            "subgroup_comparison": memory_result["subgroup_comparison"],
            "answers": memory_result["answers"],
            "examples_by_class": memory_result["examples_by_class"],
        },
    )

    # Case studies
    case_ids = select_case_study_containers(
        variant_eval_dfs["g96"],
        variant_eval_dfs["g288"],
    )
    case_studies = build_case_study_narratives(
        case_ids,
        variant_eval_dfs,
        inference_by_variant,
        memory_result["merged_deltas"],
        VARIANT_ORDER,
        baseline_id="g96",
        compare_id="g288",
    )
    _write_json(
        exp_dir / "case_studies" / "case_studies.json",
        case_studies_for_json(case_studies),
    )
    with (exp_dir / "case_studies" / "case_studies_full.pkl").open("wb") as fh:
        pickle.dump(case_studies, fh)

    # Plots
    print("Generating plots...")
    generate_all_plots(
        cohort_table=cohort_table,
        window_comparison=window_comparison,
        training_histories=training_histories,
        memory_analysis=memory_result,
        seq_table=seq_table,
        case_narratives=case_studies,
        variant_eval_dfs=variant_eval_dfs,
        out_dir=exp_dir / "plots",
    )

    # Final report
    write_final_report(
        exp_dir=exp_dir,
        cohort_table=cohort_table,
        window_comparison=window_comparison,
        memory_analysis=memory_result,
        seq_table=seq_table,
        g96_replication=g96_replication,
        training_meta=training_meta,
    )

    # Sync docs stub pointers
    _sync_docs(exp_dir, cohort_table, g96_replication, memory_result, training_meta)

    print("=" * 72)
    print("GGTCE complete.")
    print(f"Results: {exp_dir / 'reports' / 'final_report.md'}")
    print("=" * 72)


def _sync_docs(
    exp_dir: Path,
    cohort_table: pd.DataFrame,
    g96_replication: dict[str, Any],
    memory_result: dict[str, Any],
    training_meta: dict[str, Any],
) -> None:
    """Update docs/global_window_experiment/ with run results."""
    docs = REPO_ROOT / "docs/global_window_experiment"
    docs.mkdir(parents=True, exist_ok=True)

    rel_exp = exp_dir.relative_to(REPO_ROOT)
    summary_lines = [
        "# GGTCE Summary",
        "",
        f"**Authoritative run:** `{rel_exp}`",
        "",
        "## Cohort means",
        "",
        _df_to_md_table(cohort_table),
        "",
        f"G96 replication: {'PASSED' if g96_replication.get('passed') else 'FAILED'}",
        "",
        f"High-memory benefits more: {memory_result.get('answers', {}).get('high_memory_benefits_more')}",
        "",
        "See `reports/final_report.md` in the experiment directory for full answers.",
    ]
    (docs / "summary.md").write_text("\n".join(summary_lines))

    training_lines = [
        "# GGTCE Training",
        "",
        f"Run: `{rel_exp}`",
        "",
    ]
    for vid, meta in training_meta.items():
        training_lines.append(
            f"- **{vid}**: {meta['n_sequences']} sequences, "
            f"{meta['epochs_run']} epochs (best={meta['best_epoch']}), "
            f"{meta['training_seconds']:.1f}s"
        )
    (docs / "training.md").write_text("\n".join(training_lines))

    results_lines = [
        "# GGTCE Results",
        "",
        f"Run: `{rel_exp}`",
        "",
        _df_to_md_table(cohort_table),
    ]
    (docs / "results.md").write_text("\n".join(results_lines))

    (docs / "implementation_log.md").write_text(
        f"# Implementation Log\n\n"
        f"GGTCE implemented {datetime.now().strftime('%Y-%m-%d')}.\n\n"
        f"Authoritative run: `{rel_exp}`\n"
        f"Module: `utils/ggtce/`\n"
        f"Runner: `scripts/run_ggtce_experiment.py`\n"
    )


if __name__ == "__main__":
    main()
