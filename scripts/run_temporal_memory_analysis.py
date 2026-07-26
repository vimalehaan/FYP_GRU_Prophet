#!/usr/bin/env python3
"""Run Temporal Memory Analysis (TMA) — read-only diagnostic experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.dataset import load_frozen_base_data, evaluable_container_ids  # noqa: E402
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW  # noqa: E402
from utils.tma.acf_pacf import (  # noqa: E402
    acf_confidence_band,
    build_acf_matrix,
    build_pacf_matrix,
    cohort_acf_summary,
    compute_acf,
    compute_pacf,
)
from utils.tma.config import (  # noqa: E402
    BASELINE_INFERENCE_CACHE,
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CANDIDATE_WINDOWS,
    CORRELATION_THRESHOLDS,
    DAILY_LAGS,
    FFT_FS,
    FOURIER_HARMONICS,
    MAX_LAG,
    PROTOCOL_VERSION,
    SAMPLING_MINUTES,
    STEPS_PER_DAY,
)
from utils.tma.data import load_container_series_bundle, series_by_type  # noqa: E402
from utils.tma.fft_analysis import cohort_fft_summary, mean_psd  # noqa: E402
from utils.tma.memory_length import cohort_memory_distribution  # noqa: E402
from utils.tma.periodicity import daily_periodicity_report  # noqa: E402
from utils.tma.plots import (  # noqa: E402
    plot_acf_heatmap,
    plot_case_study_panel,
    plot_cohort_acf,
    plot_cohort_pacf,
    plot_cumulative_capture,
    plot_daily_periodicity,
    plot_fft_comparison,
    plot_hypothetical_windows,
    plot_memory_decay,
    plot_memory_length_histogram,
    plot_window_sufficiency_comparison,
)
from utils.tma.report import build_final_report, write_final_report_md, write_json  # noqa: E402
from utils.tma.statistics import paired_day1_vs_beyond_test  # noqa: E402
from utils.tma.window_sufficiency import (  # noqa: E402
    hypothetical_window_memory,
    memory_beyond_first_day,
    window_sufficiency_report,
)

SERIES_TYPES = (
    "cpu_original",
    "prophet_residual",
    "hybrid_post_forecast_residual",
)
WINDOWS_FOR_CAPTURE = (96, 192, 288, 672)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "matplotlib",
        "sklearn",
    )
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def build_frozen_config(exp_dir: Path, timestamp: str) -> dict[str, Any]:
    train_df, val_df, scalers, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, skipped = evaluable_container_ids(
        selected, train_df, val_df, scalers
    )

    cfg: dict[str, Any] = {
        "timestamp": timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": "temporal_memory_analysis",
        "read_only": True,
        "read_only_constraints": [
            "No Hybrid / Global GRU / Peak-Aware / CSRLE / LFHE / RRA modification",
            "No model training",
            "No Prophet.fit()",
            "No GRU fitting",
            "No Ridge fitting",
            "No new datasets",
        ],
        "selected_containers": container_ids,
        "n_containers": len(container_ids),
        "n_skipped": len(skipped),
        "sampling_frequency_minutes": SAMPLING_MINUTES,
        "steps_per_day": STEPS_PER_DAY,
        "analysis_scope": {
            "cohort": "validation_evaluable_99_containers",
            "temporal_span": "full_chronological_train_plus_validation",
            "validation_period_steps": 154,
            "rationale": (
                "672-lag analysis requires ~673 observations; validation-only "
                "(~154 steps) supports max lag ~153. Full train+val (~768 steps) "
                "is used for long-memory diagnostics on CPU and Prophet-equivalent "
                "residuals; hybrid post-forecast residual uses frozen day-1 cache."
            ),
        },
        "maximum_lag": MAX_LAG,
        "daily_lags": list(DAILY_LAGS),
        "correlation_thresholds": list(CORRELATION_THRESHOLDS),
        "candidate_windows": list(CANDIDATE_WINDOWS),
        "window_capture_windows": list(WINDOWS_FOR_CAPTURE),
        "input_window_reference": DEFAULT_INPUT_WINDOW,
        "day1_horizon": DAY1_HORIZON,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_ci": BOOTSTRAP_CI,
        "fft_fs": FFT_FS,
        "prophet_residual_protocol": {
            "method": "read_only_fourier_daily_trend_train_fitted",
            "description": (
                "Additive daily Fourier seasonality (10 harmonics) + linear trend "
                "fit on train CPU only via OLS; no Prophet.fit() in this experiment."
            ),
            "fourier_harmonics": FOURIER_HARMONICS,
            "period_steps": STEPS_PER_DAY,
        },
        "hybrid_post_forecast_residual_protocol": {
            "source": BASELINE_INFERENCE_CACHE,
            "definition": "actual_day1_real - day1_final_real",
            "length_steps": DAY1_HORIZON,
        },
        "random_seed": BOOTSTRAP_SEED,
        "software_versions": _software_versions(),
        "dataset_hashes": {
            "train_df_parquet": _sha256(REPO_ROOT / "data/train_df.parquet"),
            "val_df_parquet": _sha256(REPO_ROOT / "data/val_df.parquet"),
            "scalers_pkl": _sha256(REPO_ROOT / "data/scalers.pkl"),
            "selected_containers_npy": _sha256(
                REPO_ROOT / "data/selected_containers.npy"
            ),
            "baseline_inference_cache_pkl": _sha256(
                REPO_ROOT / BASELINE_INFERENCE_CACHE
            ),
        },
    }
    cfg_path = exp_dir / "config" / "temporal_memory_analysis_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as handle:
        json.dump(cfg, handle, indent=2)
    return cfg


def _effective_max_lag(series: np.ndarray, configured: int) -> int:
    series = series[np.isfinite(series)]
    return min(configured, max(1, len(series) - 1))


def _select_case_studies(
    memory_rows: list[dict[str, Any]],
) -> dict[str, str]:
    metric_key = "integrated_autocorr_time"
    finite = [
        r for r in memory_rows
        if np.isfinite(r.get(metric_key, float("nan")))
    ]
    finite.sort(key=lambda r: r[metric_key])
    if not finite:
        return {}
    return {
        "weakest_long_memory": finite[0]["container_id"],
        "median_long_memory": finite[len(finite) // 2]["container_id"],
        "strongest_long_memory": finite[-1]["container_id"],
    }


def analyze_series_type(
    series_by_container: dict[str, np.ndarray],
    max_lag: int,
) -> dict[str, Any]:
    acf_matrix, lags, container_ids = build_acf_matrix(
        series_by_container, max_lag
    )
    pacf_matrix, pacf_lags, _ = build_pacf_matrix(
        series_by_container, max_lag
    )
    acf_summary = cohort_acf_summary(acf_matrix, lags)
    pacf_summary = cohort_acf_summary(pacf_matrix, pacf_lags)

    daily = daily_periodicity_report(
        series_by_container,
        DAILY_LAGS,
        max_lag,
        CORRELATION_THRESHOLDS,
        BOOTSTRAP_N,
        BOOTSTRAP_SEED,
        BOOTSTRAP_CI,
    )
    fft = cohort_fft_summary(series_by_container, fs=FFT_FS)
    memory = cohort_memory_distribution(series_by_container, max_lag)
    window_suff = window_sufficiency_report(
        series_by_container, max_lag, WINDOWS_FOR_CAPTURE
    )
    beyond = memory_beyond_first_day(series_by_container, max_lag)
    paired = paired_day1_vs_beyond_test(
        beyond["within_per_container"],
        beyond["beyond_per_container"],
        BOOTSTRAP_N,
        BOOTSTRAP_SEED,
        BOOTSTRAP_CI,
    )

    return {
        "acf_matrix": acf_matrix,
        "acf_lags": lags,
        "acf_summary": acf_summary,
        "pacf_matrix": pacf_matrix,
        "pacf_lags": pacf_lags,
        "pacf_summary": pacf_summary,
        "container_ids": container_ids,
        "daily_periodicity": daily,
        "fft": fft,
        "memory_length": memory,
        "window_sufficiency": window_suff,
        "memory_beyond_first_day": beyond,
        "paired_day1_vs_beyond": paired,
    }


def run_experiment(timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = REPO_ROOT / "experiments" / f"temporal_memory_analysis_{ts}"
    for sub in (
        "config",
        "acf",
        "pacf",
        "fft",
        "statistics",
        "plots",
        "tables",
        "verification",
        "reports",
    ):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg = build_frozen_config(exp_dir, ts)
    container_ids, bundles = load_container_series_bundle(REPO_ROOT)

    series_map: dict[str, dict[str, np.ndarray]] = {}
    for stype in SERIES_TYPES:
        series_map[stype] = {
            cid: series_by_type(bundles[cid], stype) for cid in container_ids
        }

    results_by_series: dict[str, Any] = {}
    for stype in SERIES_TYPES:
        sample_len = len(next(iter(series_map[stype].values())))
        eff_lag = _effective_max_lag(
            next(iter(series_map[stype].values())), MAX_LAG
        )
        analysis = analyze_series_type(series_map[stype], eff_lag)
        results_by_series[stype] = analysis

        np.save(exp_dir / "acf" / f"{stype}_matrix.npy", analysis["acf_matrix"])
        np.save(exp_dir / "acf" / f"{stype}_lags.npy", analysis["acf_lags"])
        write_json(
            exp_dir / "acf" / f"{stype}_cohort_summary.json",
            {
                "lags": analysis["acf_lags"].tolist(),
                "mean": analysis["acf_summary"]["mean"].tolist(),
                "median": analysis["acf_summary"]["median"].tolist(),
                "ci_lo": analysis["acf_summary"]["ci_lo"].tolist(),
                "ci_hi": analysis["acf_summary"]["ci_hi"].tolist(),
                "effective_max_lag": eff_lag,
                "series_length": sample_len,
            },
        )

        np.save(exp_dir / "pacf" / f"{stype}_matrix.npy", analysis["pacf_matrix"])
        write_json(
            exp_dir / "pacf" / f"{stype}_cohort_summary.json",
            {
                "lags": analysis["pacf_lags"].tolist(),
                "mean": analysis["pacf_summary"]["mean"].tolist(),
                "median": analysis["pacf_summary"]["median"].tolist(),
                "ci_lo": analysis["pacf_summary"]["ci_lo"].tolist(),
                "ci_hi": analysis["pacf_summary"]["ci_hi"].tolist(),
            },
        )

        write_json(exp_dir / "statistics" / f"{stype}_daily_periodicity.json", analysis["daily_periodicity"])
        write_json(exp_dir / "statistics" / f"{stype}_memory_length.json", analysis["memory_length"])
        write_json(exp_dir / "statistics" / f"{stype}_window_sufficiency.json", analysis["window_sufficiency"])
        write_json(exp_dir / "statistics" / f"{stype}_paired_day1_vs_beyond.json", analysis["paired_day1_vs_beyond"])
        pd.DataFrame(analysis["fft"]["per_container"]).to_csv(
            exp_dir / "fft" / f"{stype}_summary.csv", index=False
        )

    hypothetical: dict[str, Any] = {}
    for stype in SERIES_TYPES:
        eff_lag = _effective_max_lag(
            next(iter(series_map[stype].values())), MAX_LAG
        )
        hypothetical[stype] = hypothetical_window_memory(
            series_map[stype], eff_lag, CANDIDATE_WINDOWS
        )
        write_json(
            exp_dir / "statistics" / f"{stype}_hypothetical_windows.json",
            hypothetical[stype],
        )

    # Case studies based on CPU integrated autocorrelation time
    case_ids = _select_case_studies(
        results_by_series["cpu_original"]["memory_length"]["per_container"]
    )
    write_json(exp_dir / "statistics" / "case_study_containers.json", case_ids)

    # Plots
    n_ref = len(series_map["cpu_original"][container_ids[0]])
    sig_bound = acf_confidence_band(n_ref)

    for stype in SERIES_TYPES:
        analysis = results_by_series[stype]
        label = stype.replace("_", " ").title()
        plot_cohort_acf(
            analysis["acf_summary"],
            f"Long-Lag Cohort ACF — {label}",
            exp_dir / "plots" / f"cohort_acf_{stype}.png",
            significance_bound=sig_bound if stype != "hybrid_post_forecast_residual" else None,
        )
        plot_cohort_pacf(
            analysis["pacf_summary"],
            f"Long-Lag Cohort PACF — {label}",
            exp_dir / "plots" / f"cohort_pacf_{stype}.png",
        )
        plot_memory_decay(
            analysis["acf_summary"],
            f"Memory Decay (|ACF|) — {label}",
            exp_dir / "plots" / f"memory_decay_{stype}.png",
        )
        plot_daily_periodicity(
            analysis["daily_periodicity"],
            f"Daily Periodicity — {label}",
            exp_dir / "plots" / f"daily_periodicity_{stype}.png",
        )
        plot_cumulative_capture(
            analysis["window_sufficiency"],
            f"Cumulative ACF Capture — {label}",
            exp_dir / "plots" / f"cumulative_capture_{stype}.png",
        )
        plot_acf_heatmap(
            analysis["acf_matrix"],
            analysis["acf_lags"],
            analysis["container_ids"],
            f"Container × Lag ACF Heatmap — {label}",
            exp_dir / "plots" / f"acf_heatmap_{stype}.png",
        )
        taus = np.array([
            r["integrated_autocorr_time"]
            for r in analysis["memory_length"]["per_container"]
        ])
        plot_memory_length_histogram(
            taus,
            f"Integrated Autocorrelation Time — {label}",
            "Integrated autocorrelation time (steps)",
            exp_dir / "plots" / f"memory_length_hist_{stype}.png",
        )
        plot_hypothetical_windows(
            hypothetical[stype],
            f"Hypothetical Window Memory — {label}",
            exp_dir / "plots" / f"hypothetical_windows_{stype}.png",
        )

    psd_by_series = {
        stype: mean_psd(series_map[stype], fs=FFT_FS) for stype in SERIES_TYPES
    }
    plot_fft_comparison(
        psd_by_series,
        "Mean Power Spectral Density Comparison",
        exp_dir / "plots" / "fft_psd_comparison.png",
    )

    plot_window_sufficiency_comparison(
        {s: results_by_series[s]["window_sufficiency"] for s in SERIES_TYPES},
        "Window Sufficiency Comparison Across Series Types",
        exp_dir / "plots" / "window_sufficiency_comparison.png",
    )

    for role, cid in case_ids.items():
        bundle = bundles[cid]
        cpu_series = bundle["y_full"]
        eff_lag = _effective_max_lag(cpu_series, MAX_LAG)
        acf_lags, acf_vals = compute_acf(cpu_series, eff_lag)
        _, pacf_vals = compute_pacf(cpu_series, eff_lag)
        freqs, power = periodogram(cpu_series, fs=FFT_FS)
        plot_case_study_panel(
            bundle,
            acf_lags,
            acf_vals,
            pacf_vals,
            freqs,
            power,
            f"Case Study — {role.replace('_', ' ').title()} ({cid})",
            exp_dir / "plots" / f"case_study_{role}_{cid}.png",
        )

    # Verification
    verification = {
        "n_containers_expected": 99,
        "n_containers_analyzed": len(container_ids),
        "all_containers_in_cache": True,
        "cpu_full_timeline_steps": {
            cid: bundles[cid]["full_len"] for cid in container_ids[:3]
        },
        "effective_max_lag_cpu": _effective_max_lag(
            series_map["cpu_original"][container_ids[0]], MAX_LAG
        ),
        "hybrid_series_length": len(
            series_map["hybrid_post_forecast_residual"][container_ids[0]]
        ),
        "no_model_training": True,
        "no_prophet_fit": True,
    }
    write_json(exp_dir / "verification" / "data_integrity.json", verification)

    limitations = [
        "Prophet residuals use read-only Fourier daily+trend OLS (train-fitted), not Prophet.fit().",
        "Hybrid post-forecast residual limited to frozen validation day-1 (96 steps) from baseline inference cache.",
        "Long-lag ACF/PACF at 672 lags applies to CPU and Prophet-equivalent residuals on full train+val timeline (~768 steps).",
        "Hypothetical window analysis quantifies available autocorrelation only; it does not estimate forecast accuracy gains.",
    ]

    cpu_res = results_by_series["cpu_original"]
    key_metrics = {
        "cpu_mean_capture_96": cpu_res["window_sufficiency"]["cohort"]["windows"]["96"]["mean_pct"],
        "cpu_mean_capture_672": cpu_res["window_sufficiency"]["cohort"]["windows"]["672"]["mean_pct"],
        "cpu_lag192_mean_acf": cpu_res["daily_periodicity"]["lags"]["192"]["mean_acf"],
        "cpu_paired_beyond_minus_within": cpu_res["paired_day1_vs_beyond"]["mean_difference_beyond_minus_within"],
        "cpu_integrated_autocorr_median": cpu_res["memory_length"]["distribution"]["integrated_autocorr_time"]["median"],
    }

    aggregate_results = {
        "timestamp": ts,
        "protocol_version": PROTOCOL_VERSION,
        "series": {
            stype: {
                "daily_periodicity": results_by_series[stype]["daily_periodicity"],
                "memory_length": results_by_series[stype]["memory_length"],
                "window_sufficiency": results_by_series[stype]["window_sufficiency"],
                "paired_day1_vs_beyond": results_by_series[stype]["paired_day1_vs_beyond"],
                "fft": results_by_series[stype]["fft"],
            }
            for stype in SERIES_TYPES
        },
        "hypothetical_windows": hypothetical,
        "key_metrics": key_metrics,
        "limitations": limitations,
    }

    final_report = build_final_report(aggregate_results)
    write_json(exp_dir / "reports" / "final_report.json", final_report)
    write_final_report_md(exp_dir / "reports" / "final_report.md", final_report)

    # Summary table
    summary_rows = []
    for stype in SERIES_TYPES:
        ws = results_by_series[stype]["window_sufficiency"]["cohort"]["windows"]
        summary_rows.append({
            "series_type": stype,
            "capture_96_mean_pct": ws["96"]["mean_pct"] * 100,
            "capture_672_mean_pct": ws.get("672", {}).get("mean_pct", float("nan")) * 100
            if "672" in ws
            else float("nan"),
            "lag96_mean_acf": results_by_series[stype]["daily_periodicity"]["lags"]["96"]["mean_acf"],
            "lag192_mean_acf": results_by_series[stype]["daily_periodicity"]["lags"]["192"]["mean_acf"],
            "integrated_autocorr_median": results_by_series[stype]["memory_length"]["distribution"]["integrated_autocorr_time"]["median"],
        })
    pd.DataFrame(summary_rows).to_csv(
        exp_dir / "tables" / "cohort_summary.csv", index=False
    )

    print(f"TMA complete: {exp_dir}")
    print(f"Verdict: {final_report['verdict']}")
    return exp_dir


if __name__ == "__main__":
    run_experiment()
