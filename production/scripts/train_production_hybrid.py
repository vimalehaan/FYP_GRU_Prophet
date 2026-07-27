#!/usr/bin/env python3
"""
Train the production Hybrid Prophet + GRU model.

Usage:
    tf_metal_env/bin/python production/scripts/train_production_hybrid.py
    tf_metal_env/bin/python production/scripts/train_production_hybrid.py --force-recreate-split
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from production.utils.artifacts import ensure_production_dirs, save_production_artifacts
from production.utils.config import (
    DATA_RESAMPLED,
    EVALUATION_RESULTS_PATH,
    HYBRID_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TRAIN_DF_PATH,
    VAL_DF_PATH,
)
from production.utils.evaluation import (
    evaluate_known_containers,
    evaluate_unseen_containers,
    save_prediction_records,
    summarize_metrics_by_split,
)
from production.utils.preprocess import extract_unseen_resampled, preprocess_train_containers
from production.utils.split import load_or_create_container_split
from production.utils.training import train_production_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train production Hybrid Prophet + GRU model."
    )
    parser.add_argument(
        "--force-recreate-split",
        action="store_true",
        help="Recreate container split (debugging only).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Keras training verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_production_dirs()

    print("=" * 70)
    print("Production Hybrid Training Pipeline")
    print("=" * 70)

    if not DATA_RESAMPLED.exists():
        raise FileNotFoundError(f"Missing source data: {DATA_RESAMPLED}")

    df_resampled = pd.read_parquet(DATA_RESAMPLED)
    all_containers = sorted(df_resampled["container_id"].unique().tolist())
    print(f"Loaded {len(df_resampled):,} rows, {len(all_containers)} containers")

    train_ids, unseen_ids, split_meta = load_or_create_container_split(
        all_containers,
        force_recreate=args.force_recreate_split,
    )
    print(
        f"Container split: {len(train_ids)} train, "
        f"{len(unseen_ids)} unseen (seed={split_meta['random_seed']})"
    )

    train_df, val_df, scalers, preprocess_meta = preprocess_train_containers(
        df_resampled,
        train_ids,
    )
    TRAIN_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_DF_PATH, index=False)
    val_df.to_parquet(VAL_DF_PATH, index=False)
    print(
        f"Preprocessed train containers: {preprocess_meta['n_train_containers_retained']} "
        f"({preprocess_meta['train_rows']:,} train rows, "
        f"{preprocess_meta['val_rows']:,} val rows)"
    )

    unseen_raw = extract_unseen_resampled(df_resampled, unseen_ids)
    print(f"Unseen resampled rows: {len(unseen_raw):,}")

    print("\nTraining production Hybrid GRU (frozen baseline methodology)...")
    model, res_mean, res_std, _, training_meta = train_production_hybrid(
        train_df,
        verbose=args.verbose,
    )
    print(
        f"Training complete — best epoch {training_meta['best_epoch']}, "
        f"best val_loss {training_meta['best_val_loss']:.6f}"
    )

    save_production_artifacts(
        model=model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
        training_metadata=training_meta,
        container_split_meta=split_meta,
        preprocess_meta=preprocess_meta,
    )
    print(f"Artifacts saved to {HYBRID_DIR}")

    retained_train_ids = sorted(train_df["container_id"].unique().tolist())
    known_eval, known_results, known_failures = evaluate_known_containers(
        container_ids=retained_train_ids,
        global_train=train_df,
        global_val=val_df,
        hybrid_gru_model=model,
        scalers=scalers,
        res_mean=res_mean,
        res_std=res_std,
    )
    unseen_eval, unseen_results, unseen_failures = evaluate_unseen_containers(
        container_ids=unseen_ids,
        unseen_raw=unseen_raw,
        hybrid_gru_model=model,
        res_mean=res_mean,
        res_std=res_std,
    )

    evaluation_df = pd.concat([known_eval, unseen_eval], ignore_index=True)
    evaluation_df.to_csv(EVALUATION_RESULTS_PATH, index=False)

    save_prediction_records(known_results, "known", PREDICTIONS_DIR)
    save_prediction_records(unseen_results, "unseen", PREDICTIONS_DIR)

    summary = summarize_metrics_by_split(evaluation_df)
    summary.to_csv(METRICS_DIR / "summary_by_split.csv")

    run_log = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_known_evaluated": int(len(known_eval)),
        "n_unseen_evaluated": int(len(unseen_eval)),
        "known_failures": known_failures,
        "unseen_failures": unseen_failures,
        "mean_day1_mae_known": float(known_eval["day1_mae"].mean())
        if not known_eval.empty
        else None,
        "mean_day1_mae_unseen": float(unseen_eval["day1_mae"].mean())
        if not unseen_eval.empty
        else None,
    }
    with (METRICS_DIR / "run_summary.json").open("w") as fh:
        json.dump(run_log, fh, indent=2)

    print("\nEvaluation summary:")
    print(summary)
    print(f"\nResults saved: {EVALUATION_RESULTS_PATH}")
    if known_failures:
        print(f"Known failures: {len(known_failures)}")
    if unseen_failures:
        print(f"Unseen failures: {len(unseen_failures)}")
    print("=" * 70)
    print("Production training complete.")


if __name__ == "__main__":
    main()
