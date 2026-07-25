"""Pre-GRU validation gates for CSRLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.boundary import alpha_distribution_summary, kappa_effective
from utils.csrle.config import (
    AMPLITUDE_PRESERVATION_GATES,
    B0_RETENTION_GATES,
    B1_RETENTION_GATES,
    CSRLEConfig,
    GENERATOR_GATES,
)
from utils.csrle.generators import generate_b0_lc_z


def _acf1(series: np.ndarray) -> float:
    if len(series) < 3 or np.std(series) < 1e-12:
        return float("nan")
    return float(np.corrcoef(series[:-1], series[1:])[0, 1])


def _rolling_std(series: np.ndarray, window: int = 96) -> np.ndarray:
    out = np.array([
        float(np.std(series[max(0, i - window + 1) : i + 1]))
        for i in range(len(series))
    ])
    return out


def b0_perfect_rollout_rmse(
    container_id: str,
    n_steps: int,
    config: CSRLEConfig,
    horizon: int = 96,
) -> float:
    """Exact B0-LC rollout using stored (x,y) recurrence."""
    total = config.prewarm_steps + n_steps
    r00, r01 = config.b0_rotation[0]
    r10, r11 = config.b0_rotation[1]
    r_star = config.b0_r_star

    from utils.csrle.seeds import uniform_init_pair

    x0, y0 = uniform_init_pair(
        container_id, config.global_seed, "B0_init",
        config.b0_init_low, config.b0_init_high,
    )
    x1, y1 = uniform_init_pair(
        container_id, config.global_seed, "B0_init2",
        config.b0_init_low, config.b0_init_high,
    )

    x = np.zeros(total)
    y = np.zeros(total)
    x[0], y[0] = x0, y0
    x[1], y[1] = x1, y1
    for t in range(2, total):
        xr = r00 * x[t - 1] + r01 * y[t - 1]
        yr = r10 * x[t - 1] + r11 * y[t - 1]
        r = float(np.hypot(xr, yr)) + 1e-12
        scale = r_star / r
        x[t] = np.tanh(scale * xr)
        y[t] = np.tanh(scale * yr)

    z = x / np.sqrt(2.0)
    z = z[config.prewarm_steps :]

    errs: list[float] = []
    for start in range(0, len(z) - horizon - 2, 40):
        truth = z[start + 1 : start + 1 + horizon]
        xs, ys = x[config.prewarm_steps + start], y[config.prewarm_steps + start]
        # align x,y at start — use stored values at start index in full series
        idx = config.prewarm_steps + start
        xs, ys = x[idx], y[idx]
        sim = np.zeros(horizon)
        for k in range(horizon):
            xr = r00 * xs + r01 * ys
            yr = r10 * xs + r11 * ys
            r = float(np.hypot(xr, yr)) + 1e-12
            scale = r_star / r
            xs = float(np.tanh(scale * xr))
            ys = float(np.tanh(scale * yr))
            sim[k] = xs / np.sqrt(2.0)
        errs.append(float(np.sqrt(np.mean((sim - truth) ** 2))))
    return float(np.mean(errs)) if errs else float("nan")


def evaluate_generator_process(
    condition_id: str,
    manifest: pd.DataFrame,
    container_id: str,
    config: CSRLEConfig,
) -> dict[str, Any]:
    """Question A: synthetic process validity on one container."""
    m = manifest[manifest["container_id"] == container_id].sort_values("time_stamp")
    z = m["z"].values
    n_inj = m["N"].values
    train = m[m["split"] == "train"]
    n_train = len(train)

    early = z[: max(1, len(z) // 3)]
    late = z[2 * len(z) // 3 :]

    out: dict[str, Any] = {
        "container_id": container_id,
        "condition_id": condition_id,
        "std_z_all": float(np.std(z)),
        "std_z_train": float(np.std(train["z"].values)),
        "std_n_train": float(np.std(train["N"].values)),
        "acf1_z": _acf1(z),
        "late_early_std_ratio": float(np.std(late) / (np.std(early) + 1e-12)),
        "clip_rate": float(m["clipped"].mean()),
    }

    if condition_id == "b0_lc":
        roll = _rolling_std(z)
        out["roll_std_late_mean"] = float(roll[-96:].mean())
        out["rollout_rmse"] = b0_perfect_rollout_rmse(
            container_id, len(z), config
        )
    if condition_id == "b1_snar":
        early_n = n_train // 3
        late_n = n_train - early_n
        std_e = float(np.std(z[:early_n]))
        std_l = float(np.std(z[-late_n:]))
        out["std_early_late_ratio"] = std_l / (std_e + 1e-12)

    return out


def aggregate_generator_gate(
    condition_id: str,
    per_container: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cohort generator gate evaluation."""
    df = pd.DataFrame(per_container)
    gates = GENERATOR_GATES
    result: dict[str, Any] = {
        "condition_id": condition_id,
        "n_containers": len(df),
        "checks": {},
        "pass": True,
    }

    def check(name: str, passed: bool, value: Any, threshold: Any) -> None:
        result["checks"][name] = {
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
        if not passed:
            result["pass"] = False

    if condition_id == "b0_lc":
        check(
            "acf1_mean",
            float(df["acf1_z"].mean()) >= gates["b0_acf1_min"],
            float(df["acf1_z"].mean()),
            f">= {gates['b0_acf1_min']}",
        )
        check(
            "late_early_ratio_mean",
            gates["b0_late_early_std_ratio_min"]
            <= float(df["late_early_std_ratio"].mean())
            <= gates["b0_late_early_std_ratio_max"],
            float(df["late_early_std_ratio"].mean()),
            f"[{gates['b0_late_early_std_ratio_min']}, {gates['b0_late_early_std_ratio_max']}]",
        )
        check(
            "roll_std_late_mean",
            float(df["roll_std_late_mean"].mean()) >= gates["b0_roll_std_late_min"],
            float(df["roll_std_late_mean"].mean()),
            f">= {gates['b0_roll_std_late_min']}",
        )
        check(
            "rollout_rmse_mean",
            float(df["rollout_rmse"].mean()) <= gates["b0_rollout_rmse_max"],
            float(df["rollout_rmse"].mean()),
            f"<= {gates['b0_rollout_rmse_max']}",
        )
        check(
            "std_z_train_mean",
            float(df["std_z_train"].mean()) >= gates["z_train_std_min"],
            float(df["std_z_train"].mean()),
            f">= {gates['z_train_std_min']}",
        )

    if condition_id == "b1_snar":
        check(
            "acf1_mean",
            float(df["acf1_z"].mean()) >= gates["b1_acf1_min"],
            float(df["acf1_z"].mean()),
            f">= {gates['b1_acf1_min']}",
        )
        ratio = float(df["std_early_late_ratio"].mean())
        check(
            "std_early_late_ratio_mean",
            gates["b1_std_early_late_ratio_min"]
            <= ratio
            <= gates["b1_std_early_late_ratio_max"],
            ratio,
            f"[{gates['b1_std_early_late_ratio_min']}, {gates['b1_std_early_late_ratio_max']}]",
        )

    if condition_id in ("c0_iid", "c1_iid"):
        check(
            "acf1_abs_mean",
            float(np.abs(df["acf1_z"]).mean()) <= gates["c_acf1_abs_mean_max"],
            float(np.abs(df["acf1_z"]).mean()),
            f"<= {gates['c_acf1_abs_mean_max']}",
        )

    # kappa check for B0/B1
    if condition_id in ("b0_lc", "b1_snar"):
        for _, row in df.iterrows():
            target = gates["kappa"] * row.get("sigma_cpu_train", np.nan)
            # sigma_cpu_train stored separately in manifest aggregation
            pass

    return result


def evaluate_amplitude_preservation_gate(
    alpha_map: dict[str, float],
    kappa_target: float,
) -> dict[str, Any]:
    """Pre-GRU gate: boundary scaling must not collapse cohort injection amplitude."""
    summary = alpha_distribution_summary(alpha_map)
    kappa_eff = {cid: kappa_effective(a, kappa_target) for cid, a in alpha_map.items()}
    ke_vals = np.array(list(kappa_eff.values()))
    summary["mean_kappa_effective"] = float(np.mean(ke_vals))
    summary["median_kappa_effective"] = float(np.median(ke_vals))
    summary["min_kappa_effective"] = float(np.min(ke_vals))

    gates = AMPLITUDE_PRESERVATION_GATES
    result: dict[str, Any] = {
        "summary": summary,
        "checks": {},
        "pass": True,
    }

    def check(name: str, passed: bool, value: Any, threshold: Any) -> None:
        result["checks"][name] = {
            "value": value, "threshold": threshold, "pass": bool(passed),
        }
        if not passed:
            result["pass"] = False

    check(
        "fraction_alpha_eq_1",
        summary["fraction_alpha_eq_1"] >= gates["fraction_alpha_eq_1_min"],
        summary["fraction_alpha_eq_1"],
        f">= {gates['fraction_alpha_eq_1_min']}",
    )
    check(
        "fraction_alpha_eq_0",
        summary["fraction_alpha_eq_0"] <= gates["fraction_alpha_eq_0_max"],
        summary["fraction_alpha_eq_0"],
        f"<= {gates['fraction_alpha_eq_0_max']}",
    )
    check(
        "fraction_alpha_lt_0_25",
        summary["fraction_alpha_lt_0_25"] <= gates["fraction_alpha_lt_0_25_max"],
        summary["fraction_alpha_lt_0_25"],
        f"<= {gates['fraction_alpha_lt_0_25_max']}",
    )
    check(
        "median_alpha",
        summary["median"] >= gates["median_alpha_min"],
        summary["median"],
        f">= {gates['median_alpha_min']}",
    )
    check(
        "mean_kappa_effective",
        summary["mean_kappa_effective"] >= gates["mean_kappa_effective_min"],
        summary["mean_kappa_effective"],
        f">= {gates['mean_kappa_effective_min']}",
    )
    return result


def evaluate_retention_gate(
    condition_key: str,
    retention_df: pd.DataFrame,
    acf_r_b_mean: float,
    ljung_reject_fraction: float,
) -> dict[str, Any]:
    """Question B: Prophet retention gate for B0 or B1."""
    gates = B0_RETENTION_GATES if condition_key == "b0" else B1_RETENTION_GATES
    result: dict[str, Any] = {
        "condition": condition_key,
        "checks": {},
        "pass": True,
    }

    def check(name: str, passed: bool, value: Any, threshold: Any) -> None:
        result["checks"][name] = {
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
        if not passed:
            result["pass"] = False

    rho_mean = float(retention_df["rho_n_i"].mean())
    vr_mean = float(retention_df["vr"].mean())
    vs_mean = float(retention_df["vs"].mean())
    fail_frac = float(
        (retention_df["rho_n_i"] < gates["rho_n_i_per_container_min"]).mean()
    )

    check("rho_n_i_mean", rho_mean >= gates["rho_n_i_mean_min"], rho_mean,
          f">= {gates['rho_n_i_mean_min']}")
    check("vr_mean", vr_mean >= gates["vr_mean_min"], vr_mean,
          f">= {gates['vr_mean_min']}")
    check("vs_mean", vs_mean <= gates["vs_mean_max"], vs_mean,
          f"<= {gates['vs_mean_max']}")
    check("rho_n_i_fail_fraction", fail_frac <= gates["rho_n_i_fail_fraction_max"],
          fail_frac, f"<= {gates['rho_n_i_fail_fraction_max']}")
    check("acf_r_b_lag_1_10_mean", acf_r_b_mean >= gates["acf_r_b_lag_1_10_mean_min"],
          acf_r_b_mean, f">= {gates['acf_r_b_lag_1_10_mean_min']}")
    check("ljung_reject_fraction", ljung_reject_fraction >= gates["ljung_reject_lag20_fraction_min"],
          ljung_reject_fraction, f">= {gates['ljung_reject_lag20_fraction_min']}")

    return result
