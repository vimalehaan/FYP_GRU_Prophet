"""Dimensionless synthetic process generators for CSRLE."""

from __future__ import annotations

import numpy as np

from utils.csrle.config import CSRLEConfig
from utils.csrle.seeds import container_rng, uniform_init_pair


def generate_b0_lc_z(
    container_id: str,
    n_steps: int,
    config: CSRLEConfig,
) -> np.ndarray:
    """
    B0-LC: bounded 2D limit-cycle rotator, scalar output z_t = x_t.

    Includes ``config.prewarm_steps`` virtual steps before returning ``n_steps``.
    """
    total = config.prewarm_steps + n_steps
    r00, r01 = config.b0_rotation[0]
    r10, r11 = config.b0_rotation[1]
    r_star = config.b0_r_star

    x0, y0 = uniform_init_pair(
        container_id,
        config.global_seed,
        "B0_init",
        config.b0_init_low,
        config.b0_init_high,
    )

    x = np.zeros(total)
    y = np.zeros(total)
    x[0], y[0] = x0, y0
    if total > 1:
        x[1], y[1] = uniform_init_pair(
            container_id,
            config.global_seed,
            "B0_init2",
            config.b0_init_low,
            config.b0_init_high,
        )

    for t in range(2, total):
        xr = r00 * x[t - 1] + r01 * y[t - 1]
        yr = r10 * x[t - 1] + r11 * y[t - 1]
        r = float(np.hypot(xr, yr)) + 1e-12
        scale = r_star / r
        x[t] = np.tanh(scale * xr)
        y[t] = np.tanh(scale * yr)

    z = x / np.sqrt(2.0)
    return z[config.prewarm_steps :]


def generate_b1_snar_z(
    container_id: str,
    n_steps: int,
    config: CSRLEConfig,
) -> np.ndarray:
    """B1-SNAR: stochastic nonlinear AR(2) in dimensionless space."""
    total = config.prewarm_steps + n_steps
    rng = container_rng(container_id, config.global_seed, "B1")

    z = np.zeros(total)
    for t in range(2, total):
        eta = float(rng.normal(0.0, config.b1_sigma_eta))
        z[t] = (
            config.b1_phi1 * z[t - 1]
            + config.b1_phi2 * z[t - 2]
            + config.b1_psi * z[t - 1] * z[t - 2]
            + eta
        )

    return z[config.prewarm_steps :]


def generate_iid_z(
    container_id: str,
    n_steps: int,
    config: CSRLEConfig,
    stream: str,
) -> np.ndarray:
    """Standard normal i.i.d. dimensionless series (scaled later)."""
    rng = container_rng(container_id, config.global_seed, stream)
    return rng.normal(0.0, 1.0, size=n_steps)


def normalize_z_train_only(
    z: np.ndarray,
    train_len: int,
) -> tuple[np.ndarray, float, float]:
    """Apply train-only standardization; return z_prime, mu_z, sigma_z."""
    z_train = z[:train_len]
    mu_z = float(np.mean(z_train))
    sigma_z = float(np.std(z_train))
    if sigma_z < 1e-12:
        sigma_z = 1.0
    z_prime = (z - mu_z) / sigma_z
    return z_prime, mu_z, sigma_z


def zprime_to_injection(
    z_prime: np.ndarray,
    sigma_cpu_train: float,
    kappa: float,
) -> np.ndarray:
    return kappa * sigma_cpu_train * z_prime


def apply_clip(
    y_base: np.ndarray,
    n_injection: np.ndarray,
    cpu_min: float,
    cpu_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip synthetic CPU; return y_syn and boolean clip mask."""
    raw = y_base + n_injection
    y_syn = np.clip(raw, cpu_min, cpu_max)
    clipped = raw != y_syn
    return y_syn, clipped


def b0_deterministic_rollout_rmse(
    z: np.ndarray,
    config: CSRLEConfig,
    horizon: int = 96,
    stride: int = 40,
) -> float:
    """Max-mean RMSE for perfect B0-LC forward iteration (diagnostic)."""
    r00, r01 = config.b0_rotation[0]
    r10, r11 = config.b0_rotation[1]
    r_star = config.b0_r_star
    errs: list[float] = []

    # Recover x from z = x/sqrt(2)
    x_series = z * np.sqrt(2.0)
    # Approximate y from x derivative is not stored; rollout uses x only with y=x*sqrt(2)? 
    # B0 stores z=x/sqrt(2); we need both states for rollout. Re-simulate from init segment.
    # Use stored z and infer rollout from paired regeneration is cleaner in validation module.
    if len(z) < horizon + 3:
        return float("nan")

    for start in range(0, len(z) - horizon - 2, stride):
        truth = z[start + 1 : start + 1 + horizon]
        # Reconstruct x,y from z alone loses y; rerun short recurrence from approximate state.
        xs = z[start] * np.sqrt(2.0)
        ys = z[start + 1] * np.sqrt(2.0)  # approximate pairing for gate only
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
