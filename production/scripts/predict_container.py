#!/usr/bin/env python3
"""
Run Day-1 Hybrid inference for a single container.

Usage:
    tf_metal_env/bin/python production/scripts/predict_container.py c_1016 --split known
    tf_metal_env/bin/python production/scripts/predict_container.py c_1181 --split unseen
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from production.utils.artifacts import (
    load_production_model,
    load_production_scalers,
    load_residual_stats,
)
from production.utils.config import METRICS_DIR, TRAIN_DF_PATH, UNSEEN_RAW_PATH, VAL_DF_PATH
from production.utils.evaluation import compute_extended_metrics
from production.utils.inference import (
    run_known_container_inference,
    run_unseen_container_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Hybrid single-container inference.")
    parser.add_argument("container_id", help="Container ID, e.g. c_1016")
    parser.add_argument(
        "--split",
        choices=["known", "unseen"],
        default="known",
        help="Whether the container is in the training cohort or unseen holdout.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write metrics JSON to production/hybrid/metrics/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container_id = args.container_id.strip()

    model = load_production_model()
    res_mean, res_std, input_window, _ = load_residual_stats()

    if args.split == "known":
        train_df = pd.read_parquet(TRAIN_DF_PATH)
        val_df = pd.read_parquet(VAL_DF_PATH)
        scalers = load_production_scalers()
        result = run_known_container_inference(
            container_id,
            train_df,
            val_df,
            model,
            scalers,
            res_mean,
            res_std,
            input_window,
        )
    else:
        unseen_raw = pd.read_parquet(UNSEEN_RAW_PATH)
        result = run_unseen_container_inference(
            container_id,
            unseen_raw,
            model,
            res_mean,
            res_std,
            input_window,
        )

    metrics = compute_extended_metrics(
        result.actual_day1_real,
        result.day1_final_real,
    )
    payload = {
        "container_id": container_id,
        "split": args.split,
        "train_steps": result.n_train_steps,
        "validation_steps": result.n_val_steps,
        **{f"day1_{k}": v for k, v in metrics.items()},
        "forecast_horizon_steps": int(len(np.ravel(result.day1_final_real))),
    }

    print(json.dumps(payload, indent=2))

    if args.save_json:
        out = METRICS_DIR / f"predict_{container_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
