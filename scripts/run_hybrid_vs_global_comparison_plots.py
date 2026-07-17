#!/usr/bin/env python3
"""Generate cross-methodology comparison plots for Hybrid vs Global GRU."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_comparison_plotting import (  # noqa: E402
    generate_comparison_plots,
    select_comparison_sample_container_ids,
)
from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_TRAIN,
    DATA_VAL,
    DEMO_CONTAINER_ID,
)
from utils.hybrid_artifacts import load_hybrid_artifacts  # noqa: E402
from utils.hybrid_inference import run_hybrid_inference  # noqa: E402
from utils.methodology_comparison import (  # noqa: E402
    GLOBAL_GRU_EVALUATION_RUN_DIR,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Hybrid vs Global GRU comparison plots (PNG + PDF). "
            "Hybrid inference arrays are generated for plot containers only."
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=REPO_ROOT / "experiments/hybrid_vs_global_2026-07-17_095147",
        help="Verified comparison experiment directory",
    )
    parser.add_argument(
        "--global-evaluation-dir",
        type=Path,
        default=GLOBAL_GRU_EVALUATION_RUN_DIR,
        help="Phase 5 Global GRU evaluation run (inference cache source)",
    )
    parser.add_argument(
        "--rebuild-hybrid-cache",
        action="store_true",
        help="Regenerate hybrid inference cache even if it already exists",
    )
    return parser.parse_args()


def _load_global_inference_cache(global_evaluation_dir: Path) -> dict:
    cache_path = global_evaluation_dir / "evaluation/inference_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing global inference cache: {cache_path}")
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def _build_hybrid_inference_cache(
    container_ids: list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    scalers: dict,
) -> dict[str, dict]:
    """Run frozen Hybrid inference for plot containers (visualization only)."""
    hybrid_gru_model, res_mean, res_std, input_window = load_hybrid_artifacts()
    cache: dict[str, dict] = {}

    for container_id in container_ids:
        result = run_hybrid_inference(
            container_id=container_id,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=hybrid_gru_model,
            scalers=scalers,
            res_mean=res_mean,
            res_std=res_std,
            input_window=input_window,
        )
        cache[container_id] = {
            "actual_day1_real": result.actual_day1_real,
            "day1_pred_real": result.day1_final_real,
        }

    return cache


def main() -> None:
    args = _parse_args()
    comparison_dir = args.comparison_dir.resolve()
    global_evaluation_dir = args.global_evaluation_dir.resolve()
    plots_dir = comparison_dir / "plots"
    inference_cache_dir = comparison_dir / "inference_cache"
    hybrid_cache_path = inference_cache_dir / "hybrid_inference_cache.pkl"
    plot_metadata_path = plots_dir / "plot_metadata.json"
    delta_path = comparison_dir / "per_container_delta.csv"

    if not delta_path.exists():
        raise FileNotFoundError(f"Missing per_container_delta: {delta_path}")

    print("=" * 72)
    print("Hybrid vs Global GRU — Comparison Plot Generation (Phase 6 Task 5)")
    print("=" * 72)
    print(f"Comparison directory      : {comparison_dir}")
    print(f"Global evaluation source  : {global_evaluation_dir}")
    print(f"Plots directory           : {plots_dir}")
    print()

    per_container_delta = pd.read_csv(delta_path)
    global_inference_cache = _load_global_inference_cache(global_evaluation_dir)

    sample_container_ids = select_comparison_sample_container_ids(
        per_container_delta,
        demo_container_id=DEMO_CONTAINER_ID,
        n_samples=4,
    )
    plot_container_ids = sorted(set(sample_container_ids))

    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    plot_train = train_df[train_df["container_id"].isin(plot_container_ids)].copy()
    plot_val = val_df[val_df["container_id"].isin(plot_container_ids)].copy()

    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    if hybrid_cache_path.exists() and not args.rebuild_hybrid_cache:
        print(f"Loading existing hybrid inference cache: {hybrid_cache_path}")
        with open(hybrid_cache_path, "rb") as handle:
            hybrid_inference_cache = pickle.load(handle)
    else:
        print(
            "Building hybrid inference cache for plot containers "
            f"({len(plot_container_ids)}): {plot_container_ids}"
        )
        hybrid_inference_cache = _build_hybrid_inference_cache(
            container_ids=plot_container_ids,
            global_train=plot_train,
            global_val=plot_val,
            scalers=scalers,
        )
        inference_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(hybrid_cache_path, "wb") as handle:
            pickle.dump(hybrid_inference_cache, handle)
        print(f"Saved hybrid inference cache: {hybrid_cache_path}")

    missing_global = [
        container_id
        for container_id in plot_container_ids
        if container_id not in global_inference_cache
    ]
    if missing_global:
        raise KeyError(
            "Plot containers missing from global inference cache: "
            f"{missing_global}"
        )

    plot_global_cache = {
        container_id: global_inference_cache[container_id]
        for container_id in plot_container_ids
    }

    saved_paths = generate_comparison_plots(
        per_container_delta=per_container_delta,
        hybrid_inference_cache=hybrid_inference_cache,
        global_inference_cache=plot_global_cache,
        global_train=plot_train,
        scalers=scalers,
        plots_dir=plots_dir,
        sample_container_ids=sample_container_ids,
    )

    plot_metadata = {
        "phase": 6,
        "task": 5,
        "comparison_dir": str(comparison_dir.relative_to(REPO_ROOT)),
        "global_evaluation_dir": str(global_evaluation_dir.relative_to(REPO_ROOT)),
        "plot_module": "utils.global_comparison_plotting",
        "plot_containers": plot_container_ids,
        "sample_container_ids": sample_container_ids,
        "hybrid_inference_cache": str(hybrid_cache_path.relative_to(REPO_ROOT)),
        "global_inference_cache": str(
            (global_evaluation_dir / "evaluation/inference_cache.pkl").relative_to(
                REPO_ROOT,
            ),
        ),
        "saved_paths": saved_paths,
        "plot_counts": {
            category: len(paths) for category, paths in saved_paths.items()
        },
        "note": (
            "Hybrid inference arrays are visualization-only; locked comparison "
            "metrics come from frozen evaluation CSVs."
        ),
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
    print("Phase 6 Task 5 plot generation: PASS")


if __name__ == "__main__":
    main()
