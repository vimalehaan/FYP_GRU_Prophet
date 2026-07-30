#!/usr/bin/env python3
"""Run an isolated AFMLF experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.adaptive_lifecycle.config import AFMLFConfig, PageHinkleyMode
from utils.adaptive_lifecycle.pipeline import run_afmlf_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AFMLF isolated experiment.")
    parser.add_argument("--max-containers", type=int, default=None)
    parser.add_argument("--max-origins-per-container", type=int, default=None)
    parser.add_argument(
        "--skip-candidate-retraining",
        action="store_true",
        help="Run monitoring/drift/decisions only.",
    )
    parser.add_argument(
        "--page-hinkley-mode",
        choices=["off", "confirmatory", "mandatory"],
        default="confirmatory",
    )
    parser.add_argument("--pilot", action="store_true", help="3 containers, 2 origins.")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Official thesis-scale 99-container evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AFMLFConfig(
        page_hinkley_mode=PageHinkleyMode(args.page_hinkley_mode),
        skip_candidate_retraining=args.skip_candidate_retraining,
    )
    official = args.official or (not args.pilot and args.max_containers is None)

    if args.pilot:
        config.max_containers = 3
        config.max_origins_per_container = 2
        config.candidate_training_verbose = 0
        official = False
    if args.max_containers is not None:
        config.max_containers = args.max_containers
        official = False
    if args.max_origins_per_container is not None:
        config.max_origins_per_container = args.max_origins_per_container

    exp_dir = run_afmlf_experiment(_REPO_ROOT, config, official=official)
    print(f"\nExperiment directory: {exp_dir}")
    if official:
        print(f"Executive summary: {exp_dir / 'reports' / 'AFMLF_FINAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
