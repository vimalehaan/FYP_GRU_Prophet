"""Final report generation for Temporal Memory Analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _answer_q1(results: dict[str, Any]) -> str:
    cpu = results["series"]["cpu_original"]
    mem = cpu["memory_length"]["distribution"]["integrated_autocorr_time"]["median"]
    cap96 = cpu["window_sufficiency"]["cohort"]["windows"]["96"]["mean_pct"]
    lag96 = cpu["daily_periodicity"]["lags"]["96"]["mean_acf"]
    return (
        f"Cohort integrated autocorrelation time (CPU) median = {mem:.2f} steps; "
        f"mean cumulative squared-ACF capture within 96 lags = {cap96 * 100:.1f}%; "
        f"lag-96 mean ACF = {lag96:.3f}."
    )


def _answer_q2(results: dict[str, Any]) -> str:
    cpu = results["series"]["cpu_original"]
    daily = cpu["daily_periodicity"]["lags"]["192"]
    paired = cpu["paired_day1_vs_beyond"]
    return (
        f"At lag 192 (2 days), mean ACF = {daily['mean_acf']:.4f}, "
        f"{daily['pct_abs_acf_ge_0.1'] * 100:.1f}% of containers have |ACF|≥0.1; "
        f"paired beyond-vs-within mean |ACF| difference = "
        f"{paired['mean_difference_beyond_minus_within']:.4f} "
        f"(Wilcoxon p = {paired['wilcoxon_pvalue']:.4g})."
    )


def _answer_q3(results: dict[str, Any]) -> str:
    prophet = results["series"]["prophet_residual"]
    daily96 = prophet["daily_periodicity"]["lags"]["96"]
    fft = prophet["fft"]["per_container"]
    daily_fracs = [r["daily_freq_power_fraction"] for r in fft if r["daily_freq_power_fraction"] == r["daily_freq_power_fraction"]]
    mean_daily = sum(daily_fracs) / len(daily_fracs) if daily_fracs else float("nan")
    return (
        f"After Prophet-equivalent decomposition, lag-96 mean ACF = {daily96['mean_acf']:.4f}; "
        f"mean daily-cycle PSD fraction = {mean_daily:.4f}."
    )


def _answer_q4(results: dict[str, Any]) -> str:
    hybrid = results["series"]["hybrid_post_forecast_residual"]
    mem = hybrid["memory_length"]["distribution"]["decorrelation_lag_0p1"]["median"]
    cap96 = hybrid["window_sufficiency"]["cohort"]["windows"].get("96", {}).get("mean_pct", float("nan"))
    return (
        f"Hybrid day-1 post-forecast residual decorrelation lag (|ACF|<0.1) "
        f"median = {mem:.1f} steps (96-step series); "
        f"within-96 capture = {cap96 * 100:.1f}% when defined."
    )


def _answer_q5(results: dict[str, Any]) -> str:
    cpu = results["series"]["cpu_original"]
    cap96 = cpu["window_sufficiency"]["cohort"]["windows"]["96"]["mean_pct"]
    cap672 = cpu["window_sufficiency"]["cohort"]["windows"]["672"]["mean_pct"]
    return (
        f"On full timeline CPU, mean cumulative squared-ACF capture: "
        f"96 lags = {cap96 * 100:.1f}%, 672 lags = {cap672 * 100:.1f}%; "
        f"gap beyond 96 lags = {(cap672 - cap96) * 100:.1f} percentage points."
    )


def _answer_q6(results: dict[str, Any]) -> str:
    hyp = results["hypothetical_windows"]["cpu_original"]
    w96 = hyp["windows"]["96"]["mean_abs_acf_within_window"]
    w384 = hyp["windows"]["384"]["mean_abs_acf_within_window"]
    return (
        f"Available mean |ACF| within window — 96: {w96:.4f}, 384: {w384:.4f}; "
        f"increment 96→384 = {w384 - w96:.4f} (information proxy only, not forecast gain)."
    )


def _answer_q7(results: dict[str, Any]) -> str:
    cpu = results["series"]["cpu_original"]
    cap96 = cpu["window_sufficiency"]["cohort"]["windows"]["96"]["mean_pct"]
    daily192 = cpu["daily_periodicity"]["lags"]["192"]
    paired = cpu["paired_day1_vs_beyond"]
    beyond_mean = paired["mean_beyond"]
    if cap96 >= 0.85 and abs(daily192["mean_acf"]) < 0.05 and beyond_mean < 0.05:
        rec = (
            "Evidence suggests most autocorrelation structure is captured within 96 lags; "
            "longer windows are not strongly indicated by memory diagnostics alone."
        )
    elif cap96 < 0.75 or abs(daily192["mean_acf"]) >= 0.1 or beyond_mean >= 0.08:
        rec = (
            "Evidence indicates non-trivial temporal dependence beyond 96 lags; "
            "future work may investigate longer input windows (without claiming forecast improvement)."
        )
    else:
        rec = (
            "Evidence is mixed: some memory exists beyond 96 lags but cumulative capture "
            "within 96 lags is substantial; longer-window investigation is optional, not mandatory."
        )
    return rec


def build_final_report(results: dict[str, Any]) -> dict[str, Any]:
    """Build machine-readable final report with explicit answers."""
    answers = {
        "q1_temporal_memory_magnitude": _answer_q1(results),
        "q2_dependence_beyond_one_day": _answer_q2(results),
        "q3_daily_cycles_after_prophet": _answer_q3(results),
        "q4_hybrid_long_range_dependence": _answer_q4(results),
        "q5_was_96_step_window_sufficient": _answer_q5(results),
        "q6_longer_window_information": _answer_q6(results),
        "q7_future_work_recommendation": _answer_q7(results),
    }
    return {
        "experiment_name": "temporal_memory_analysis",
        "protocol_version": results.get("protocol_version", "tma_v1.0"),
        "timestamp": results["timestamp"],
        "verdict": answers["q7_future_work_recommendation"],
        "explicit_answers": answers,
        "key_metrics": results.get("key_metrics", {}),
        "limitations": results.get("limitations", []),
    }


def write_final_report_md(path: Path, report: dict[str, Any]) -> None:
    """Write human-readable final report."""
    lines = [
        "# Temporal Memory Analysis — Final Report",
        "",
        f"**Timestamp:** {report['timestamp']}",
        f"**Protocol:** {report['protocol_version']}",
        "",
        "## Explicit answers",
        "",
    ]
    questions = [
        ("1", "How much temporal memory exists in the selected Alibaba workloads?"),
        ("2", "Does meaningful temporal dependence exist beyond one day?"),
        ("3", "Are daily or multi-day cycles still present after Prophet?"),
        ("4", "Does the Hybrid residual retain long-range temporal dependence?"),
        ("5", "Was the chosen 96-step input window sufficient?"),
        ("6", "Would longer windows (192, 288, 384) provide additional temporal information?"),
        ("7", "Should future work investigate longer input windows?"),
    ]
    keys = list(report["explicit_answers"].keys())
    for (num, qtext), key in zip(questions, keys, strict=True):
        lines.append(f"### Q{num}. {qtext}")
        lines.append("")
        lines.append(report["explicit_answers"][key])
        lines.append("")

    lines.extend([
        "## Limitations",
        "",
    ])
    for lim in report.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
