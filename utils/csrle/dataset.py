"""Load frozen read-only data and build CSRLE experiment datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.csrle.boundary import (
    apply_boundary_safe_injection,
    compute_alpha_train,
    kappa_effective,
)
from utils.csrle.config import CSRLEConfig
from utils.csrle.generators import (
    generate_b0_lc_z,
    generate_b1_snar_z,
    generate_iid_z,
    normalize_z_train_only,
    zprime_to_injection,
)
from utils.hybrid_evaluation import _evaluable_container_ids


def load_frozen_base_data(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, MinMaxScaler], np.ndarray]:
    data_dir = repo_root / "data"
    train_df = pd.read_parquet(data_dir / "train_df.parquet")
    val_df = pd.read_parquet(data_dir / "val_df.parquet")
    with (data_dir / "scalers.pkl").open("rb") as fh:
        scalers: dict[str, MinMaxScaler] = pickle.load(fh)
    selected = np.load(data_dir / "selected_containers.npy", allow_pickle=True)
    return train_df, val_df, scalers, selected


def evaluable_container_ids(
    selected: np.ndarray | list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
) -> tuple[list[str], list[tuple[str, str]]]:
    return _evaluable_container_ids(selected, train_df, val_df, scalers)


def inverse_cpu_real(
    cpu_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    return scaler.inverse_transform(cpu_scaled.reshape(-1, 1)).ravel()


def fit_transform_condition_scaler(
    y_train_real: np.ndarray,
) -> tuple[MinMaxScaler, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler.fit_transform(y_train_real.reshape(-1, 1)).ravel()
    return scaler, y_train_scaled


def transform_condition_scaler(
    scaler: MinMaxScaler,
    y_real: np.ndarray,
) -> np.ndarray:
    return scaler.transform(y_real.reshape(-1, 1)).ravel()


def build_container_timeline(
    container_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    frozen_scaler: MinMaxScaler,
) -> dict[str, Any]:
    """Chronological train+val timeline in real CPU % (read-only base)."""
    train_c = train_df[train_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    val_c = val_df[val_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    y_train = inverse_cpu_real(train_c["cpu_scaled"].values, frozen_scaler)
    y_val = inverse_cpu_real(val_c["cpu_scaled"].values, frozen_scaler)
    y_all = np.concatenate([y_train, y_val])
    timestamps = pd.concat(
        [train_c["time_stamp"], val_c["time_stamp"]],
        ignore_index=True,
    )
    return {
        "container_id": container_id,
        "train_len": len(y_train),
        "val_len": len(y_val),
        "y_base": y_all,
        "y_train": y_train,
        "y_val": y_val,
        "timestamps": timestamps,
        "train_timestamps": train_c["time_stamp"].values,
        "val_timestamps": val_c["time_stamp"].values,
        "sigma_cpu_train": float(np.std(y_train)),
    }


def _generate_n_target_b(
    condition_id: str,
    container_id: str,
    n_steps: int,
    train_len: int,
    sigma_cpu_train: float,
    config: CSRLEConfig,
) -> dict[str, Any]:
    """Target injection N_target before boundary scaling (B0/B1)."""
    if condition_id == "b0_lc":
        z = generate_b0_lc_z(container_id, n_steps, config)
    elif condition_id == "b1_snar":
        z = generate_b1_snar_z(container_id, n_steps, config)
    else:
        raise ValueError(f"Not a B condition: {condition_id}")

    z_prime, mu_z, sigma_z = normalize_z_train_only(z, train_len)
    n_target = zprime_to_injection(z_prime, sigma_cpu_train, config.kappa)
    return {
        "z": z,
        "z_prime": z_prime,
        "N_target": n_target,
        "mu_z": mu_z,
        "sigma_z": sigma_z,
    }


def _generate_n_target_c(
    condition_id: str,
    container_id: str,
    n_steps: int,
    train_len: int,
    sigma_eff_b_train: float,
    alpha_b: float,
    config: CSRLEConfig,
) -> dict[str, Any]:
    """
    C0/C1 target injection matched to effective B-scale with inherited α policy.

    N_target_C = (σ_eff_B / (α_B · std(z_train))) · z_iid

    so α_B · N_target_C has train std = σ_eff_B exactly when α_C = α_B.
    """
    stream = "C0" if condition_id == "c0_iid" else "C1"
    z = generate_iid_z(container_id, n_steps, config, stream)
    z_std_train = float(np.std(z[:train_len]))
    if z_std_train < 1e-12:
        z_std_train = 1.0
    if alpha_b <= 1e-12 or sigma_eff_b_train <= 1e-12:
        n_target = np.zeros(n_steps)
    else:
        scale = sigma_eff_b_train / (alpha_b * z_std_train)
        n_target = scale * z
    return {
        "z": z,
        "z_prime": z,
        "N_target": n_target,
        "mu_z": 0.0,
        "sigma_z": z_std_train,
    }


def _resolve_c_alpha(
    y_train: np.ndarray,
    n_target_c: np.ndarray,
    alpha_b: float,
) -> float:
    """α_C = min(α_B, α_strict(N_target_C)) — train-only, IID boundary check."""
    if alpha_b <= 1e-12:
        return 0.0
    alpha_feas = compute_alpha_train(y_train, n_target_c)
    return float(min(alpha_b, alpha_feas))


def build_b_condition_dataframes(
    condition_id: str,
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    frozen_scalers: dict[str, MinMaxScaler],
    config: CSRLEConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Build B0/B1 with train-derived α_c.

    Returns train_syn, val_syn, manifest, alpha_map, sigma_eff_train_map.
    """
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    alpha_map: dict[str, float] = {}
    sigma_eff_map: dict[str, float] = {}

    for cid in container_ids:
        timeline = build_container_timeline(
            cid, train_df, val_df, frozen_scalers[cid]
        )
        n_steps = len(timeline["y_base"])
        train_len = timeline["train_len"]

        inj = _generate_n_target_b(
            condition_id, cid, n_steps, train_len,
            timeline["sigma_cpu_train"], config,
        )
        n_target = inj["N_target"]
        alpha = compute_alpha_train(timeline["y_train"], n_target[:train_len])
        alpha_map[cid] = alpha

        n_eff, y_syn, clipped = apply_boundary_safe_injection(
            timeline["y_base"], n_target, alpha, config.cpu_min, config.cpu_max,
        )
        sigma_eff_map[cid] = float(np.std(n_eff[:train_len]))

        y_train_syn = y_syn[:train_len]
        y_val_syn = y_syn[train_len:]

        cond_scaler, cpu_scaled_train = fit_transform_condition_scaler(y_train_syn)
        cpu_scaled_val = transform_condition_scaler(cond_scaler, y_val_syn)

        train_rows.append(_make_frame(
            cid, timeline["train_timestamps"], y_train_syn, cpu_scaled_train, train_df,
        ))
        val_rows.append(_make_frame(
            cid, timeline["val_timestamps"], y_val_syn, cpu_scaled_val, val_df,
        ))

        for idx in range(n_steps):
            manifest_rows.append(_manifest_row(
                cid, timeline, idx, train_len, condition_id,
                inj, n_target[idx], n_eff[idx], alpha, y_syn[idx], clipped[idx],
                config.kappa,
            ))

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.DataFrame(manifest_rows),
        alpha_map,
        sigma_eff_map,
    )


def build_c_condition_dataframes(
    condition_id: str,
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    frozen_scalers: dict[str, MinMaxScaler],
    config: CSRLEConfig,
    alpha_b_map: dict[str, float],
    sigma_eff_b_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build C0/C1 matched to effective B injection scale."""
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    alpha_map: dict[str, float] = {}

    for cid in container_ids:
        timeline = build_container_timeline(
            cid, train_df, val_df, frozen_scalers[cid]
        )
        n_steps = len(timeline["y_base"])
        train_len = timeline["train_len"]
        alpha_b = alpha_b_map[cid]
        sigma_eff_b = sigma_eff_b_map[cid]

        inj = _generate_n_target_c(
            condition_id, cid, n_steps, train_len, sigma_eff_b, alpha_b, config,
        )
        n_target = inj["N_target"]
        alpha = _resolve_c_alpha(timeline["y_train"], n_target[:train_len], alpha_b)
        alpha_map[cid] = alpha

        n_eff, y_syn, clipped = apply_boundary_safe_injection(
            timeline["y_base"], n_target, alpha, config.cpu_min, config.cpu_max,
        )

        y_train_syn = y_syn[:train_len]
        y_val_syn = y_syn[train_len:]

        cond_scaler, cpu_scaled_train = fit_transform_condition_scaler(y_train_syn)
        cpu_scaled_val = transform_condition_scaler(cond_scaler, y_val_syn)

        train_rows.append(_make_frame(
            cid, timeline["train_timestamps"], y_train_syn, cpu_scaled_train, train_df,
        ))
        val_rows.append(_make_frame(
            cid, timeline["val_timestamps"], y_val_syn, cpu_scaled_val, val_df,
        ))

        for idx in range(n_steps):
            manifest_rows.append(_manifest_row(
                cid, timeline, idx, train_len, condition_id,
                inj, n_target[idx], n_eff[idx], alpha, y_syn[idx], clipped[idx],
                config.kappa,
                alpha_b_inherited=alpha_b,
            ))

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.DataFrame(manifest_rows),
        alpha_map,
    )


def build_control_dataframes(
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    frozen_scalers: dict[str, MinMaxScaler],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Condition A: real data, N=0."""
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for cid in container_ids:
        timeline = build_container_timeline(
            cid, train_df, val_df, frozen_scalers[cid]
        )
        train_len = timeline["train_len"]
        y_train = timeline["y_train"]
        y_val = timeline["y_val"]

        train_rows.append(_make_frame(
            cid, timeline["train_timestamps"], y_train,
            train_df[train_df["container_id"] == cid].sort_values("time_stamp")[
                "cpu_scaled"
            ].values,
            train_df,
        ))
        val_rows.append(_make_frame(
            cid, timeline["val_timestamps"], y_val,
            val_df[val_df["container_id"] == cid].sort_values("time_stamp")[
                "cpu_scaled"
            ].values,
            val_df,
        ))

        for idx, y in enumerate(timeline["y_base"]):
            manifest_rows.append({
                "container_id": cid,
                "time_stamp": timeline["timestamps"].iloc[idx],
                "condition_id": "control",
                "split": "train" if idx < train_len else "val",
                "y_base": float(y),
                "N": 0.0,
                "N_target": 0.0,
                "y_syn": float(y),
                "z": 0.0,
                "z_prime": 0.0,
                "alpha": 1.0,
                "kappa_effective": 0.0,
                "clipped": False,
            })

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.DataFrame(manifest_rows),
    )


def _make_frame(
    cid: str,
    timestamps: np.ndarray,
    y_real: np.ndarray,
    cpu_scaled: np.ndarray,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    has_mean = "cpu_mean" in source_df.columns
    part = source_df[source_df["container_id"] == cid].sort_values("time_stamp")
    return pd.DataFrame({
        "container_id": cid,
        "time_stamp": timestamps,
        "cpu_util_percent": y_real,
        "cpu_scaled": cpu_scaled,
        "cpu_mean": part["cpu_mean"].values[: len(y_real)] if has_mean else np.nan,
        "cpu_std": part["cpu_std"].values[: len(y_real)] if has_mean else np.nan,
    })


def _manifest_row(
    cid: str,
    timeline: dict[str, Any],
    idx: int,
    train_len: int,
    condition_id: str,
    inj: dict[str, Any],
    n_target: float,
    n_eff: float,
    alpha: float,
    y_syn: float,
    clipped: bool,
    kappa: float,
    alpha_b_inherited: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "container_id": cid,
        "time_stamp": timeline["timestamps"].iloc[idx],
        "condition_id": condition_id,
        "split": "train" if idx < train_len else "val",
        "y_base": float(timeline["y_base"][idx]),
        "N": float(y_syn - timeline["y_base"][idx]),
        "N_target": float(n_target),
        "N_effective_pre_clip": float(n_eff),
        "y_syn": float(y_syn),
        "z": float(inj["z"][idx]),
        "z_prime": float(inj["z_prime"][idx]),
        "alpha": float(alpha),
        "kappa_effective": float(kappa_effective(alpha, kappa)),
        "clipped": bool(clipped),
    }
    if alpha_b_inherited is not None:
        row["alpha_b_inherited"] = float(alpha_b_inherited)
    return row
