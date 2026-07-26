#!/usr/bin/env python3
"""Finish HCERL post-processing (plots + reports) from a completed training run."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

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
from utils.hcerl.config import VARIANT_ORDER  # noqa: E402
from utils.hcerl.plots import generate_all_plots  # noqa: E402
from utils.hcerl.report import write_final_report  # noqa: E402
from utils.hcerl.training import load_variant_artifacts  # noqa: E402
from utils.hybrid_inference import HybridInferenceResult  # noqa: E402


def _load_inference_cache(path: Path) -> dict[str, dict]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _rebuild_inference_results(
    cache: dict[str, dict],
    variant_id: str,
) -> dict[str, HybridInferenceResult]:
    """Minimal HybridInferenceResult stubs for case-study trajectories."""
    out: dict[str, HybridInferenceResult] = {}
    for cid, arrays in cache.items():
        out[cid] = HybridInferenceResult(
            container_id=cid,
            train_container=pd.DataFrame(),
            val_container=pd.DataFrame(),
            prophet_train=pd.DataFrame(),
            prophet_model=None,
            forecast=pd.DataFrame(),
            train_forecast=pd.DataFrame(),
            residual_input=arrays["actual_day1_real"][:0],
            X_input=arrays["actual_day1_real"][:0],
            day1_residual_scaled=arrays["actual_day1_real"][:0],
            day1_residual=arrays["actual_day1_real"][:0],
            day1_prophet=arrays["actual_day1_real"][:0],
            day1_final_scaled=arrays["day1_final_real"],
            day2_input_residuals=arrays["actual_day1_real"][:0],
            X_day2=arrays["actual_day1_real"][:0],
            day2_residual_scaled=arrays["actual_day1_real"][:0],
            day2_residual=arrays["actual_day1_real"][:0],
            day2_prophet=arrays["actual_day1_real"][:0],
            available_len=0,
            day2_final_scaled=arrays["actual_day1_real"][:0],
            actual_day1_scaled=arrays["actual_day1_real"][:0],
            actual_day2_scaled=arrays["actual_day1_real"][:0],
            scaler=None,
            day1_final_real=arrays["day1_final_real"],
            actual_day1_real=arrays["actual_day1_real"],
            prophet_day1_real=arrays["actual_day1_real"][:0],
            day2_final_real=arrays["actual_day1_real"][:0],
            actual_day2_real=arrays["actual_day1_real"][:0],
            prophet_day2_real=arrays["actual_day1_real"][:0],
            input_window=96,
            day1_horizon=96,
            n_train_steps=0,
            n_val_steps=0,
        )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=str(sorted((REPO_ROOT / "experiments").glob("hcerl_*"))[-1]),
    )
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    with (exp_dir / "config" / "hcerl_config_frozen.json").open() as fh:
        config = json.load(fh)

    variant_eval_dfs: dict[str, pd.DataFrame] = {}
    training_histories: dict[str, pd.DataFrame] = {}
    inference_by_variant: dict[str, dict] = {}

    for vid in VARIANT_ORDER:
        vdir = exp_dir / "variants" / vid
        variant_eval_dfs[vid] = pd.read_csv(
            vdir / "evaluation" / "evaluation_extended.csv",
        )
        training_histories[vid] = pd.read_csv(
            vdir / "training" / "training_history.csv",
        )
        inference_by_variant[vid] = _rebuild_inference_results(
            _load_inference_cache(vdir / "evaluation" / "inference_cache.pkl"),
            vid,
        )

    ablation = run_full_ablation_analysis(variant_eval_dfs)
    cohort_table = build_cohort_comparison_table(variant_eval_dfs, VARIANT_ORDER)

    ablation_dir = exp_dir / "ablation"
    cohort_table.to_csv(ablation_dir / "cohort_comparison.csv", index=False)
    with (ablation_dir / "paired_tests.json").open("w") as fh:
        json.dump(
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
            fh,
            indent=2,
        )

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
    with (exp_dir / "case_studies" / "case_studies.json").open("w") as fh:
        json.dump(case_studies_for_json(case_studies), fh, indent=2)

    generate_all_plots(
        cohort_table=cohort_table,
        variant_eval_dfs=variant_eval_dfs,
        ablation_result=ablation,
        training_histories=training_histories,
        case_studies=case_studies,
        out_dir=exp_dir / "plots",
    )

    write_final_report(
        exp_dir=exp_dir,
        cohort_table=cohort_table,
        ablation=ablation,
        case_studies=case_studies,
        config=config,
    )

    print(f"Post-processing complete: {exp_dir}")


if __name__ == "__main__":
    main()
