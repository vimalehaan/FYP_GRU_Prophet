#!/usr/bin/env python3
"""Generate publication-quality plots for a Global GRU evaluation run."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import DATA_SCALERS, DATA_TRAIN  # noqa: E402
from utils.global_plotting import generate_evaluation_plots  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Global GRU evaluation plots (PNG + PDF).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation experiment directory "
            "(default: latest experiments/global_gru_evaluation_*)"
        ),
    )
    return parser.parse_args()


def _latest_evaluation_dir() -> Path:
    candidates = sorted(REPO_ROOT.glob("experiments/global_gru_evaluation_*"))
    if not candidates:
        raise FileNotFoundError(
            "No experiments/global_gru_evaluation_* directory found"
        )
    return candidates[-1]


def main() -> None:
    args = _parse_args()
    experiment_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir is not None
        else _latest_evaluation_dir()
    )

    evaluation_dir = experiment_dir / "evaluation"
    plots_dir = experiment_dir / "plots"
    eval_df_path = evaluation_dir / "evaluation_df.csv"
    inference_cache_path = evaluation_dir / "inference_cache.pkl"
    plot_metadata_path = plots_dir / "plot_metadata.json"

    if not eval_df_path.exists():
        raise FileNotFoundError(f"Missing evaluation_df: {eval_df_path}")
    if not inference_cache_path.exists():
        raise FileNotFoundError(f"Missing inference cache: {inference_cache_path}")

    print("=" * 72)
    print("Global GRU v1 — Evaluation Plot Generation")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Plots directory      : {plots_dir}")
    print()

    evaluation_df = pd.read_csv(eval_df_path)
    with open(inference_cache_path, "rb") as handle:
        inference_cache = pickle.load(handle)

    train_df = pd.read_parquet(DATA_TRAIN)
    selected_ids = set(evaluation_df["container_id"])
    global_train = train_df[train_df["container_id"].isin(selected_ids)].copy()

    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    saved_paths = generate_evaluation_plots(
        evaluation_df=evaluation_df,
        inference_cache=inference_cache,
        global_train=global_train,
        scalers=scalers,
        plots_dir=plots_dir,
    )

    plot_metadata = {
        "phase": 5,
        "task": 5,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "plot_module": "utils.global_plotting",
        "saved_paths": saved_paths,
        "plot_counts": {
            category: len(paths) for category, paths in saved_paths.items()
        },
    }
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metadata_path.write_text(json.dumps(plot_metadata, indent=2) + "\n")

    print("Saved plots:")
    for category, paths in saved_paths.items():
        print(f"  {category}:")
        for path in paths:
            print(f"    - {path}")

    print()
    print(f"Plot metadata: {plot_metadata_path}")
    print("Task 5 plot generation: PASS")


if __name__ == "__main__":
    main()
