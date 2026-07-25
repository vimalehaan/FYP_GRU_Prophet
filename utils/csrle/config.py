"""Frozen CSRLE configuration constants (pre-specified, not GRU-tuned)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

CONDITION_IDS = ("control", "b0_lc", "b1_snar", "c0_iid", "c1_iid")

# Protocol Amendment v2 (2026-07-25): train-derived boundary-safety α_c.
PROTOCOL_VERSION = "v2_boundary_safe"
PRE_AMENDMENT_REFERENCE_RUN = (
    "experiments/synthetic_residual_learnability_2026-07-25_163013"
)

# Pre-specified amplitude-preservation gate (researcher-defined).
AMPLITUDE_PRESERVATION_GATES = {
    "fraction_alpha_eq_1_min": 0.85,
    "fraction_alpha_eq_0_max": 0.10,
    "fraction_alpha_lt_0_25_max": 0.10,
    "median_alpha_min": 0.95,
    "mean_kappa_effective_min": 0.08,
}

# Pre-specified experimental acceptance criteria (researcher-defined, not universal).
B0_RETENTION_GATES = {
    "rho_n_i_mean_min": 0.40,
    "vr_mean_min": 0.25,
    "vs_mean_max": 0.75,
    "rho_n_i_fail_fraction_max": 0.20,
    "rho_n_i_per_container_min": 0.20,
    "acf_r_b_lag_1_10_mean_min": 0.10,
    "ljung_reject_lag20_fraction_min": 0.50,
}

B1_RETENTION_GATES = {
    "rho_n_i_mean_min": 0.35,
    "vr_mean_min": 0.20,
    "vs_mean_max": 0.80,
    "rho_n_i_fail_fraction_max": 0.25,
    "rho_n_i_per_container_min": 0.20,
    "acf_r_b_lag_1_10_mean_min": 0.10,
    "ljung_reject_lag20_fraction_min": 0.50,
}

GENERATOR_GATES = {
    "kappa": 0.10,
    "kappa_tolerance_relative": 0.02,
    "prewarm_steps": 200,
    "z_train_std_min": 0.05,
    "b0_late_early_std_ratio_min": 0.5,
    "b0_late_early_std_ratio_max": 2.0,
    "b0_roll_std_late_min": 0.05,
    "b0_acf1_min": 0.30,
    "b0_rollout_rmse_max": 1e-6,
    "b1_acf1_min": 0.25,
    "b1_std_early_late_ratio_min": 0.70,
    "b1_std_early_late_ratio_max": 1.30,
    "clip_rate_container_fail_fraction": 0.05,
    "clip_rate_timestep_max": 0.01,
    "clip_rate_per_container_max": 0.05,
    "c_acf1_abs_mean_max": 0.05,
    "c_ljung_reject_lag20_min": 0.05,
    "c_ljung_reject_lag20_max": 0.20,
    "c_std_match_relative_max": 0.02,
}

CONTROL_REPRODUCTION_GATES = {
    "mae_abs_tolerance": 0.10,
    "rmse_abs_tolerance": 0.15,
    "rank_corr_min": 0.95,
    "mae_warn_tolerance": 0.20,
    "rank_corr_warn_min": 0.90,
    "frozen_mae_mean": 1.7458824024279418,
    "frozen_rmse_mean": 2.3878133967258304,
}


@dataclass(frozen=True)
class CSRLEConfig:
    """Authoritative CSRLE synthetic generator configuration."""

    global_seed: int = 42
    kappa: float = 0.10
    prewarm_steps: int = 200
    cpu_min: float = 0.0
    cpu_max: float = 100.0

    # B0-LC
    b0_rotation: tuple[tuple[float, float], tuple[float, float]] = (
        (0.92, -0.38),
        (0.38, 0.92),
    )
    b0_r_star: float = 0.70
    b0_init_low: float = -0.4
    b0_init_high: float = 0.4

    # B1-SNAR
    b1_phi1: float = 0.55
    b1_phi2: float = -0.15
    b1_psi: float = 0.08
    b1_sigma_eta: float = 0.15

    day1_horizon: int = 96

    def to_dict(self) -> dict:
        return asdict(self)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
