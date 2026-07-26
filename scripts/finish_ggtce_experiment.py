#!/usr/bin/env python3
"""Resume/finish GGTCE after partial run — skip trained variants, complete analysis."""

from __future__ import annotations

import json
import pickle
import sys
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
from utils.ggtce.config import VARIANT_ORDER, VARIANT_WINDOWS  # noqa: E402
from utils.ggtce.evaluation import (  # noqa: E402
    evaluate_ggtce_variant,
    save_inference_cache,
    summarize_variant_cohort,
)
from utils.ggtce.memory_analysis import run_memory_analysis  # noqa: E402
from utils.ggtce.plots import generate_all_plots  # noqa: E402
from utils.ggtce.report import write_final_report  # noqa: E402
from utils.ggtce.training import (  # noqa: E402
    load_variant_artifacts,
    save_variant_artifacts,
    train_ggtce_variant,
)
from utils.ggtce.verification import verify_g96_replication  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: finish_ggtce_experiment.py <experiments/ggtce_* dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.is_absolute():
        exp_dir = REPO_ROOT / exp_dir

    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(REPO_ROOT / "data/selected_containers.npy", allow_pickle=True)
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    variant_eval_dfs: dict[str, pd.DataFrame] = {}
    inference_by_variant: dict[str, dict] = {}
    training_meta: dict[str, dict[str, Any]] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}

    for variant_id in VARIANT_ORDER:
        input_window = VARIANT_WINDOWS[variant_id]
        variant_dir = exp_dir / "variants" / variant_id
        eval_path = variant_dir / "evaluation" / "evaluation_extended.csv"

        if eval_path.exists():
            print(f"Skipping {variant_id} — evaluation exists")
            eval_df = pd.read_csv(eval_path)
            variant_eval_dfs[variant_id] = eval_df
            with (variant_dir / "evaluation" / "inference_cache.pkl").open("rb") as fh:
                cache = pickle.load(fh)
            from utils.global_inference import GlobalInferenceResult
            inference_by_variant[variant_id] = {
                cid: GlobalInferenceResult(
                    day1_pred_real=np.array(v["day1_pred_real"]),
                    actual_day1_real=np.array(v["actual_day1_real"]),
                    day1_pred_scaled=np.array(v["day1_pred_scaled"]),
                    input_window=v["input_window"],
                    n_train_steps=0,
                    n_val_steps=0,
                )
                for cid, v in cache.items()
            }
            meta_path = variant_dir / "training" / "training_metadata.json"
            with meta_path.open() as fh:
                training_meta[variant_id] = json.load(fh)
            hist_df = pd.read_csv(variant_dir / "training" / "training_history.csv")
            training_histories[variant_id] = {
                "loss": hist_df["loss"].tolist(),
                "val_loss": hist_df["val_loss"].tolist(),
                "mae": hist_df["mae"].tolist(),
                "val_mae": hist_df["val_mae"].tolist(),
            }
            summary = summarize_variant_cohort(eval_df)
            _write_json(
                variant_dir / "evaluation" / "cohort_summary.json",
                summary,
            )
            continue

        print(f"Training {variant_id}...")
        model, train_meta = train_ggtce_variant(
            global_train=global_train,
            variant_id=variant_id,
            input_window=input_window,
            verbose=1,
        )
        save_variant_artifacts(model, train_meta, variant_dir, input_window)
        hist = train_meta.pop("history")
        training_histories[variant_id] = hist
        _write_csv(
            variant_dir / "training" / "training_history.csv",
            pd.DataFrame({
                "epoch": np.arange(1, len(hist["loss"]) + 1),
                "loss": hist["loss"],
                "val_loss": hist["val_loss"],
                "mae": hist.get("mae", [np.nan] * len(hist["loss"])),
                "val_mae": hist.get("val_mae", [np.nan] * len(hist["loss"])),
            }),
        )
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
        _write_csv(eval_path, eval_df)
        save_inference_cache(
            inference_results,
            variant_dir / "evaluation" / "inference_cache.pkl",
        )
        summary = summarize_variant_cohort(eval_df)
        _write_json(variant_dir / "evaluation" / "cohort_summary.json", summary)
        _write_json(
            variant_dir / "evaluation" / "failures.json",
            {"failures": failures, "skipped": skipped},
        )
        variant_eval_dfs[variant_id] = eval_df
        inference_by_variant[variant_id] = inference_results

    g96_replication = verify_g96_replication(variant_eval_dfs["g96"], REPO_ROOT)
    _write_json(exp_dir / "verification" / "g96_replication.json", g96_replication)

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

    memory_result = run_memory_analysis(
        REPO_ROOT,
        global_train,
        global_val,
        window_comparison["delta_tables"],
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
        },
    )

    case_ids = select_case_study_containers(
        variant_eval_dfs["g96"], variant_eval_dfs["g288"],
    )
    case_studies = build_case_study_narratives(
        case_ids,
        variant_eval_dfs,
        inference_by_variant,
        memory_result["merged_deltas"],
        VARIANT_ORDER,
    )
    _write_json(
        exp_dir / "case_studies" / "case_studies.json",
        case_studies_for_json(case_studies),
    )

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

    write_final_report(
        exp_dir=exp_dir,
        cohort_table=cohort_table,
        window_comparison=window_comparison,
        memory_analysis=memory_result,
        seq_table=seq_table,
        g96_replication=g96_replication,
        training_meta=training_meta,
    )

    print(f"GGTCE finished: {exp_dir / 'reports' / 'final_report.md'}")


if __name__ == "__main__":
    main()
