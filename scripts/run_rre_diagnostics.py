#!/usr/bin/env python3
"""Run Stage 0 — Residual Representation Diagnostic Study (read-only, no GRU)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.rre_diagnostics.comparison import (  # noqa: E402
    build_comparison_row,
    build_recommendation,
    information_assessment,
)
from utils.rre_diagnostics.config import (  # noqa: E402
    BASELINE_REFERENCE,
    BASELINE_RES_MEAN,
    BASELINE_RES_STD,
    ENTROPY_BINS,
    EXPERIMENT_NAME,
    FFT_FS,
    MAX_LAG,
    PROTOCOL_VERSION,
    REPRESENTATIONS,
    STEPS_PER_DAY,
)
from utils.rre_diagnostics.data import (  # noqa: E402
    build_representation_series,
    load_cohort_train_residuals,
)
from utils.rre_diagnostics.distribution import distribution_profile  # noqa: E402
from utils.rre_diagnostics.frequency import frequency_profile  # noqa: E402
from utils.rre_diagnostics.report import build_final_report_md, write_json  # noqa: E402
from utils.rre_diagnostics.stationarity import stationarity_profile  # noqa: E402
from utils.rre_diagnostics.temporal import temporal_profile  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "sklearn")
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def build_config(
    timestamp: str,
    container_ids: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "read_only": True,
        "read_only_constraints": [
            "No Hybrid / Global GRU / production pipeline modification",
            "No neural network training",
            "No GRU implementation",
            "No modification of previous experiments or artifacts",
            "Prophet residuals computed via frozen generate_prophet_residuals()",
        ],
        "stage": "Stage 0 — Residual Representation Diagnostic Study",
        "representations": list(REPRESENTATIONS),
        "selected_containers": container_ids,
        "n_containers": len(container_ids),
        "n_skipped": len(meta.get("skipped", [])),
        "n_train_timesteps": meta.get("n_timesteps"),
        "maximum_lag": MAX_LAG,
        "steps_per_day": STEPS_PER_DAY,
        "entropy_bins": ENTROPY_BINS,
        "fft_fs": FFT_FS,
        "baseline_reference": BASELINE_REFERENCE,
        "baseline_residual_mean": BASELINE_RES_MEAN,
        "baseline_residual_std": BASELINE_RES_STD,
        "software_versions": _software_versions(),
        "rationale": (
            "Representation selection for RRE v1.0 must be evidence-driven. "
            "Velocity (R1) is one candidate among R0–R3, not a predetermined winner."
        ),
    }


def verify_r0_alignment(
    residual_df: pd.DataFrame,
    exp_dir: Path,
) -> dict[str, Any]:
    """Check raw train level residuals against frozen baseline reference."""
    raw = residual_df["residual"].astype(float)
    mean_val = float(raw.mean())
    std_val = float(raw.std())
    mean_diff = abs(mean_val - BASELINE_RES_MEAN)
    std_diff = abs(std_val - BASELINE_RES_STD)
    ok = mean_diff < 1e-3 and std_diff < 0.01
    payload = {
        "raw_level_mean": mean_val,
        "raw_level_std": std_val,
        "baseline_mean": BASELINE_RES_MEAN,
        "baseline_std": BASELINE_RES_STD,
        "mean_abs_diff": mean_diff,
        "std_abs_diff": std_diff,
        "aligned_with_baseline_reference": ok,
        "note": "R0 z-scored distribution has mean≈0, std≈1 by construction",
    }
    write_json(exp_dir / "verification" / "r0_baseline_alignment.json", payload)
    return payload


def run_diagnostics(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or REPO_ROOT
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = repo_root / "experiments" / f"rre_diagnostics_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    container_ids, residual_df, meta = load_cohort_train_residuals(repo_root)
    stats = meta["stats"]
    config = build_config(timestamp, container_ids, meta)
    write_json(exp_dir / "config" / "rre_diagnostics_config.json", config)

    profiles: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for rep in REPRESENTATIONS:
        rep_id = rep["id"]
        print(f"Analyzing {rep_id}: {rep['name']}...")
        series = build_representation_series(
            residual_df, container_ids, stats, rep_id
        )

        dist = distribution_profile(series, bins=ENTROPY_BINS)
        temp = temporal_profile(series, max_lag=MAX_LAG)
        freq = frequency_profile(series, fs=FFT_FS)
        stat = stationarity_profile(series)
        info = information_assessment(rep_id, temp, dist, stat, freq)

        profiles[rep_id] = {
            "representation": rep,
            "distribution": dist,
            "temporal": temp,
            "frequency": freq,
            "stationarity": stat,
            "information": info,
        }

        rep_dir = exp_dir / "representations" / rep_id
        write_json(rep_dir / "distribution.json", dist)
        write_json(rep_dir / "temporal.json", temp)
        write_json(rep_dir / "frequency.json", freq)
        write_json(rep_dir / "stationarity.json", stat)
        write_json(rep_dir / "information.json", info)

        pd.DataFrame(
            {
                "lag": temp["acf_lags"],
                "acf_mean": temp["acf_mean"],
                "acf_ci_lower": temp["acf_ci_lower"],
                "acf_ci_upper": temp["acf_ci_upper"],
            }
        ).to_csv(rep_dir / "acf_cohort.csv", index=False)

        pd.DataFrame(
            {
                "lag": temp["pacf_lags"],
                "pacf_mean": temp["pacf_mean"],
            }
        ).to_csv(rep_dir / "pacf_cohort.csv", index=False)

        if dist["histogram_bins"]:
            pd.DataFrame(
                {
                    "bin_center": dist["histogram_bins"],
                    "count": dist["histogram_counts"],
                }
            ).to_csv(rep_dir / "histogram.csv", index=False)

        if dist["qq_sample"]:
            pd.DataFrame(
                {
                    "sample_quantile": dist["qq_sample"],
                    "theoretical_quantile": dist["qq_theoretical"],
                }
            ).to_csv(rep_dir / "qq_plot.csv", index=False)

        if freq["psd_freqs"]:
            pd.DataFrame(
                {"freq": freq["psd_freqs"], "psd_mean": freq["psd_mean"]}
            ).to_csv(rep_dir / "psd.csv", index=False)

        comparison_rows.append(
            build_comparison_row(
                rep_id, rep["name"], dist, temp, stat, info
            )
        )

    if "R0" in profiles:
        verify_r0_alignment(residual_df, exp_dir)

    comparison_df = pd.DataFrame(comparison_rows)
    (exp_dir / "comparison").mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(exp_dir / "comparison" / "comparison_table.csv", index=False)

    recommendation = build_recommendation(comparison_rows)
    write_json(exp_dir / "comparison" / "recommendation.json", recommendation)

    final_report = {
        "config": config,
        "comparison": comparison_rows,
        "recommendation": recommendation,
        "verification": {
            "data_paths": {
                "train_df": str(repo_root / "data" / "train_df.parquet"),
                "selected_containers": str(
                    repo_root / "data" / "selected_containers.npy"
                ),
            },
            "train_df_sha256": _sha256(repo_root / "data" / "train_df.parquet"),
        },
    }
    write_json(exp_dir / "reports" / "final_report.json", final_report)

    md = build_final_report_md(config, comparison_rows, recommendation)
    (exp_dir / "reports" / "final_report.md").write_text(md)

    print(f"\nStage 0 complete: {exp_dir}")
    print(f"Primary recommendation: {recommendation['primary_recommendation']}")
    print(recommendation["rationale"])
    return exp_dir


def main() -> None:
    run_diagnostics()


if __name__ == "__main__":
    main()
