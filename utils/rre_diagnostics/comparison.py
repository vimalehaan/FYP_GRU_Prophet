"""Cross-representation comparison and evidence-based recommendation."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.rre_diagnostics.representations import reconstruction_notes


INTERPRETABILITY: dict[str, str] = {
    "R0": "High — identical to frozen Hybrid residual target",
    "R1": "Medium — velocity requires integration back to level",
    "R2": "High — per-container scale, same physical units after inverse",
    "R3": "Medium — robust scale less familiar than z-score",
}


def information_assessment(
    representation_id: str,
    temporal: dict[str, Any],
    distribution: dict[str, Any],
    stationarity: dict[str, Any],
    frequency: dict[str, Any],
) -> dict[str, str]:
    """Qualitative assessment of temporal information preservation."""
    acf_l1 = abs(float(temporal.get("acf_lag1", 0.0)))
    mean_abs_acf = float(temporal.get("mean_abs_acf_lag1_96", 0.0))
    integrated_tau = float(temporal.get("integrated_autocorr_time_median", 0.0))
    daily_frac = float(frequency.get("daily_freq_power_fraction_mean", 0.0))
    adf_frac = float(stationarity.get("adf_reject_unit_root_frac", 0.0))

    if representation_id == "R1":
        temporal_verdict = (
            "Amplifies short-lag dynamics; differencing removes slow drift "
            "and can sharpen local temporal structure at the cost of level memory."
        )
    elif representation_id == "R2":
        temporal_verdict = (
            "Preserves relative temporal shape within each container; "
            "cross-container amplitude differences are removed."
        )
    elif representation_id == "R3":
        temporal_verdict = (
            "Down-weights extreme tails; temporal shape largely preserved "
            "but outlier-driven dynamics may be suppressed."
        )
    else:
        temporal_verdict = (
            "Baseline level residual retains full Prophet-complement signal "
            "including slow components and heavy-tail spikes."
        )

    if mean_abs_acf >= 0.08 and integrated_tau >= 5:
        structure = "Strong temporal dependence retained"
    elif mean_abs_acf >= 0.04:
        structure = "Moderate temporal dependence retained"
    else:
        structure = "Weak temporal dependence — signal may appear near white noise"

    if daily_frac >= 0.15:
        periodic = "Daily-cycle energy preserved or emphasized"
    elif daily_frac >= 0.05:
        periodic = "Some daily-cycle energy present"
    else:
        periodic = "Limited daily-cycle energy in spectrum"

    if adf_frac >= 0.7:
        stat_note = "Mostly stationary under ADF — favourable for sequence models"
    elif adf_frac >= 0.4:
        stat_note = "Mixed stationarity — may benefit from shorter memory windows"
    else:
        stat_note = "Often non-stationary — harder for fixed-window GRU"

    return {
        "temporal_structure": structure,
        "periodic_content": periodic,
        "stationarity_note": stat_note,
        "overall": (
            f"{temporal_verdict} Observed lag-1 |ACF|={acf_l1:.3f}, "
            f"mean |ACF| (1–96)={mean_abs_acf:.3f}, "
            f"integrated τ median={integrated_tau:.1f}."
        ),
    }


def build_comparison_row(
    representation_id: str,
    name: str,
    distribution: dict[str, Any],
    temporal: dict[str, Any],
    stationarity: dict[str, Any],
    information: dict[str, str],
) -> dict[str, Any]:
    notes = reconstruction_notes(representation_id)
    lb_p = temporal.get("ljung_box", {}).get("p_value", [])
    lb_sig = bool(lb_p and min(lb_p) < 0.05) if lb_p else False

    return {
        "representation_id": representation_id,
        "name": name,
        "temporal_dependence": float(temporal.get("mean_abs_acf_lag1_96", float("nan"))),
        "acf_lag1": float(temporal.get("acf_lag1", float("nan"))),
        "integrated_autocorr_time_median": float(
            temporal.get("integrated_autocorr_time_median", float("nan"))
        ),
        "decorrelation_lag_0_1": int(temporal.get("decorrelation_lag_0_1", 0)),
        "ljung_box_significant": lb_sig,
        "stationarity_adf_reject_frac": float(
            stationarity.get("adf_reject_unit_root_frac", float("nan"))
        ),
        "stationarity_kpss_pass_frac": float(
            stationarity.get("kpss_stationary_frac", float("nan"))
        ),
        "sparsity_frac_abs_lt_0_05": float(
            distribution.get("sparsity_frac_abs_lt_0_05", float("nan"))
        ),
        "skewness": float(distribution.get("skewness", float("nan"))),
        "kurtosis": float(distribution.get("kurtosis", float("nan"))),
        "variance": float(distribution.get("variance", float("nan"))),
        "entropy_bits": float(distribution.get("entropy_bits", float("nan"))),
        "interpretability": INTERPRETABILITY[representation_id],
        "inference_complexity": notes["inference_complexity"],
        "reconstruction_complexity": notes["reconstruction_complexity"],
        "leakage_risk": notes["leakage_risk"],
        "information_assessment": information["overall"],
    }


def _rank_higher_better(values: dict[str, float]) -> dict[str, float]:
    ids = list(values.keys())
    arr = np.array([values[i] for i in ids], dtype=float)
    order = (-arr).argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ids) + 1)
    return {ids[i]: float(ranks[i]) for i in range(len(ids))}


def _rank_lower_better(values: dict[str, float]) -> dict[str, float]:
    ids = list(values.keys())
    arr = np.array([values[i] for i in ids], dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ids) + 1)
    return {ids[i]: float(ranks[i]) for i in range(len(ids))}


def build_recommendation(
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evidence-based scoring — transparent weights, no pre-assumed winner."""
    ids = [r["representation_id"] for r in comparison_rows]
    by_id = {r["representation_id"]: r for r in comparison_rows}

    temporal_score = _rank_higher_better(
        {i: by_id[i]["temporal_dependence"] for i in ids}
    )
    learnability_score = _rank_higher_better(
        {
            i: 0.5 * by_id[i]["temporal_dependence"]
            + 0.3 * by_id[i]["stationarity_adf_reject_frac"]
            + 0.2 * (1.0 - by_id[i]["sparsity_frac_abs_lt_0_05"])
            for i in ids
        }
    )
    stability_score = _rank_higher_better(
        {
            i: 0.4 * by_id[i]["stationarity_adf_reject_frac"]
            + 0.3 * (1.0 / (1.0 + abs(by_id[i]["skewness"])))
            + 0.3 * (1.0 / (1.0 + max(0.0, by_id[i]["kurtosis"] - 3.0)))
            for i in ids
        }
    )
    balance_score = _rank_higher_better(
        {
            i: 0.4 * temporal_score[i]
            + 0.35 * stability_score[i]
            + 0.25 * (5.0 - learnability_score[i])
            for i in ids
        }
    )

    composite = {
        i: (
            temporal_score[i]
            + learnability_score[i]
            + stability_score[i]
            + balance_score[i]
        )
        / 4.0
        for i in ids
    }
    primary = min(composite, key=composite.get)

    answers = {
        "q1_most_temporal_structure": min(temporal_score, key=temporal_score.get),
        "q2_most_learnable_proxy": min(learnability_score, key=learnability_score.get),
        "q3_best_balance": min(balance_score, key=balance_score.get),
        "q4_primary_candidate": primary,
        "q5_level_baseline_remains": primary == "R0",
    }

    return {
        "scoring_weights": {
            "temporal_dependence": "Rank mean |ACF| lags 1–96 (higher better)",
            "learnability_proxy": (
                "0.5 temporal + 0.3 ADF stationarity + 0.2 non-sparsity"
            ),
            "stability": (
                "0.4 ADF + 0.3 inverse |skew| + 0.3 inverse excess kurtosis"
            ),
            "balance": "0.4 temporal rank + 0.35 stability rank + 0.25 learnability rank",
        },
        "rankings": {
            "temporal_structure": temporal_score,
            "learnability_proxy": learnability_score,
            "stability": stability_score,
            "balance": balance_score,
            "composite_mean_rank": composite,
        },
        "answers": answers,
        "primary_recommendation": primary,
        "primary_recommendation_name": by_id[primary]["name"],
        "rationale": (
            f"Representation {primary} ({by_id[primary]['name']}) achieved the "
            f"lowest mean rank across temporal structure, learnability proxy, "
            f"statistical stability, and their balance. This recommendation is "
            f"derived solely from diagnostic statistics — velocity (R1) was not "
            f"assumed a priori."
        ),
        "note_if_r0_wins": (
            "If R0 is selected, the original level residual should remain both "
            "the RRE baseline and the primary candidate; alternative "
            "representations did not demonstrably improve temporal learnability proxies."
        ),
    }
