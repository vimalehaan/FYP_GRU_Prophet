#!/usr/bin/env python3
"""
Re-run production evaluation using saved model and cache (no retraining).

Usage:
    tf_metal_env/bin/python production/scripts/evaluate_production_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
from datetime import datetime, timezone

import pandas as pd

from production.utils.artifacts import (
    load_production_model,
    load_production_scalers,
    load_residual_stats,
)
from production.utils.config import (
    EVALUATION_RESULTS_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TRAIN_DF_PATH,
    UNSEEN_CONTAINER_IDS_PATH,
    UNSEEN_RAW_PATH,
    VAL_DF_PATH,
)
from production.utils.evaluation import (
    evaluate_known_containers,
    evaluate_unseen_containers,
    save_prediction_records,
    summarize_metrics_by_split,
)


def main() -> None:
    print("=" * 70)
    print("Production Hybrid Evaluation (no retraining)")
    print("=" * 70)

    model = load_production_model()
    scalers = load_production_scalers()
    res_mean, res_std, input_window, _ = load_residual_stats()

    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    unseen_raw = pd.read_parquet(UNSEEN_RAW_PATH)

    with UNSEEN_CONTAINER_IDS_PATH.open() as fh:
        unseen_ids = json.load(fh)["container_ids"]

    retained_train_ids = sorted(train_df["container_id"].unique().tolist())
    known_eval, known_results, known_failures = evaluate_known_containers(
        container_ids=retained_train_ids,
        global_train=train_df,
        global_val=val_df,
        hybrid_gru_model=model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
        input_window=input_window,
    )
    unseen_eval, unseen_results, unseen_failures = evaluate_unseen_containers(
        container_ids=unseen_ids,
        unseen_raw=unseen_raw,
        hybrid_gru_model=model,
        res_mean=res_mean,
        res_std=res_std,
        input_window=input_window,
    )

    evaluation_df = pd.concat([known_eval, unseen_eval], ignore_index=True)
    evaluation_df.to_csv(EVALUATION_RESULTS_PATH, index=False)

    save_prediction_records(known_results, "known", PREDICTIONS_DIR)
    save_prediction_records(unseen_results, "unseen", PREDICTIONS_DIR)

    summary = summarize_metrics_by_split(evaluation_df)
    summary.to_csv(METRICS_DIR / "summary_by_split.csv")

    run_log = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": "evaluate_only",
        "n_known_evaluated": int(len(known_eval)),
        "n_unseen_evaluated": int(len(unseen_eval)),
        "known_failures": known_failures,
        "unseen_failures": unseen_failures,
    }
    with (METRICS_DIR / "run_summary.json").open("w") as fh:
        json.dump(run_log, fh, indent=2)

    print(summary)
    print(f"Saved: {EVALUATION_RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
