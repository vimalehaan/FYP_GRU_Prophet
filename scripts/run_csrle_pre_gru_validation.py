#!/usr/bin/env python3
"""CSRLE pre-GRU validation (Protocol v2: boundary-safe injection). No GRU training."""

from __future__ import annotations

from typing import Any

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.csrle.boundary import alpha_distribution_summary, kappa_effective  # noqa: E402
from utils.csrle.config import (  # noqa: E402
    AMPLITUDE_PRESERVATION_GATES,
    B0_RETENTION_GATES,
    B1_RETENTION_GATES,
    CONTROL_REPRODUCTION_GATES,
    CSRLEConfig,
    GENERATOR_GATES,
    PRE_AMENDMENT_REFERENCE_RUN,
    PROTOCOL_VERSION,
)
from utils.csrle.dataset import (  # noqa: E402
    build_b_condition_dataframes,
    build_c_condition_dataframes,
    build_control_dataframes,
    build_container_timeline,
    evaluable_container_ids,
    fit_transform_condition_scaler,
    load_frozen_base_data,
)
from utils.csrle.diagnostics import (  # noqa: E402
    avg_abs_acf,
    avg_abs_pacf,
    ljung_box_reject_fraction,
)
from utils.csrle.prophet_retention import (  # noqa: E402
    fit_prophet_forecast_val,
    run_prophet_retention_for_pair,
)
from utils.csrle.validation_gates import (  # noqa: E402
    evaluate_amplitude_preservation_gate,
    evaluate_generator_process,
    evaluate_retention_gate,
)
from utils.hybrid_config import DAY1_HORIZON  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _kappa_effective_check(
    manifest: pd.DataFrame,
    container_ids: list[str],
    kappa_target: float,
) -> dict:
    """Verify std(N_eff_train) ≈ κ_effective · std(y_base_train)."""
    rows = []
    for cid in container_ids:
        m = manifest[manifest["container_id"] == cid]
        train = m[m["split"] == "train"]
        y_base = train["y_base"].values
        n_train = train["N"].values
        alpha = float(train["alpha"].iloc[0])
        sigma_base = float(np.std(y_base))
        k_eff = kappa_effective(alpha, kappa_target)
        target = k_eff * sigma_base
        actual = float(np.std(n_train))
        rel_err = abs(actual - target) / (target + 1e-12)
        rows.append({
            "container_id": cid,
            "alpha": alpha,
            "kappa_effective": k_eff,
            "sigma_cpu_train": sigma_base,
            "target_std_n": target,
            "actual_std_n": actual,
            "relative_error": rel_err,
            "pass": rel_err <= GENERATOR_GATES["kappa_tolerance_relative"],
        })
    df = pd.DataFrame(rows)
    return {
        "per_container": rows,
        "pass": bool(df["pass"].all()),
        "mean_relative_error": float(df["relative_error"].mean()),
    }


def _clip_gate(manifest: pd.DataFrame, container_ids: list[str]) -> dict:
    rows = []
    for cid in container_ids:
        m = manifest[manifest["container_id"] == cid]
        rate = float(m["clipped"].mean())
        rows.append({"container_id": cid, "clip_rate": rate})
    df = pd.DataFrame(rows)
    fail_frac = float((df["clip_rate"] > GENERATOR_GATES["clip_rate_timestep_max"]).mean())
    return {
        "per_container": rows,
        "fail_fraction": fail_frac,
        "cohort_mean_clip_rate": float(df["clip_rate"].mean()),
        "worst_clip_rate": float(df["clip_rate"].max()),
        "worst_container_id": df.loc[df["clip_rate"].idxmax(), "container_id"],
        "pass": fail_frac <= GENERATOR_GATES["clip_rate_container_fail_fraction"],
    }


def _c_std_match_effective(
    manifest_b: pd.DataFrame,
    manifest_c: pd.DataFrame,
    alpha_b_map: dict[str, float],
    alpha_c_map: dict[str, float],
    container_ids: list[str],
) -> dict:
    """Match C effective std to B effective std (adjusted for α_C/α_B ratio)."""
    rows = []
    for cid in container_ids:
        nb = manifest_b[
            (manifest_b["container_id"] == cid) & (manifest_b["split"] == "train")
        ]["N"].std()
        nc = manifest_c[
            (manifest_c["container_id"] == cid) & (manifest_c["split"] == "train")
        ]["N"].std()
        alpha_b = alpha_b_map[cid]
        alpha_c = alpha_c_map[cid]
        if alpha_b <= 1e-12:
            expected_ratio = 1.0
        else:
            expected_ratio = alpha_c / alpha_b
        expected_nc = nb * expected_ratio
        rel = abs(nc - expected_nc) / (expected_nc + 1e-12)
        rows.append({
            "container_id": cid,
            "std_b_eff": float(nb),
            "std_c_eff": float(nc),
            "expected_std_c": float(expected_nc),
            "alpha_b": alpha_b,
            "alpha_c": alpha_c,
            "relative_error": float(rel),
            "pass": rel <= GENERATOR_GATES["c_std_match_relative_max"] or nb < 1e-12,
        })
    df = pd.DataFrame(rows)
    return {"per_container": rows, "pass": bool(df["pass"].all())}


def _prophet_control_metrics(
    control_train: pd.DataFrame,
    control_val: pd.DataFrame,
    frozen_scalers: dict,
    container_ids: list[str],
) -> pd.DataFrame:
    """Prophet-only MAE: Day-1 and full validation window."""
    from sklearn.metrics import mean_absolute_error

    rows = []
    for cid in container_ids:
        train_c = control_train[control_train["container_id"] == cid].sort_values(
            "time_stamp"
        )
        val_c = control_val[control_val["container_id"] == cid].sort_values(
            "time_stamp"
        )
        yhat_d1, y_d1 = fit_prophet_forecast_val(
            control_train, control_val, cid, frozen_scalers[cid]
        )
        # Full val forecast
        from prophet import Prophet

        prophet_train = pd.DataFrame({
            "ds": train_c["time_stamp"],
            "y": train_c["cpu_scaled"],
        })
        model = Prophet(daily_seasonality=True, weekly_seasonality=False)
        model.fit(prophet_train)
        future = pd.DataFrame({"ds": val_c["time_stamp"].values})
        forecast = model.predict(future)
        yhat_full = frozen_scalers[cid].inverse_transform(
            forecast["yhat"].values.reshape(-1, 1)
        ).ravel()
        y_full = frozen_scalers[cid].inverse_transform(
            val_c["cpu_scaled"].values.reshape(-1, 1)
        ).ravel()

        rows.append({
            "container_id": cid,
            "prophet_day1_mae": float(mean_absolute_error(y_d1, yhat_d1)),
            "prophet_full_val_mae": float(mean_absolute_error(y_full, yhat_full)),
            "n_val_steps": len(y_full),
        })
    return pd.DataFrame(rows)


def _run_generator_gates(
    condition_id: str,
    manifest: pd.DataFrame,
    container_ids: list[str],
    config: CSRLEConfig,
    alpha_map: dict[str, float],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    frozen_scalers: dict,
    matched_c_pair: tuple[pd.DataFrame, dict, dict] | None = None,
) -> dict[str, Any]:
    per = []
    for cid in container_ids:
        row = evaluate_generator_process(condition_id, manifest, cid, config)
        tl = build_container_timeline(cid, train_df, val_df, frozen_scalers[cid])
        row["sigma_cpu_train"] = tl["sigma_cpu_train"]
        row["alpha"] = alpha_map[cid]
        row["kappa_effective"] = kappa_effective(alpha_map[cid], config.kappa)
        z = manifest[manifest["container_id"] == cid].sort_values("time_stamp")["z"].values
        row["pacf1_z"] = float(np.corrcoef(z[1:], z[:-1])[0, 1]) if len(z) > 2 else float("nan")
        row["avg_abs_acf_z_1_10"] = avg_abs_acf(z, 1, 10)
        row["avg_abs_pacf_z_1_10"] = avg_abs_pacf(z, 1, 10)
        per.append(row)

    gate: dict[str, Any] = {
        "per_container": per,
        "kappa_effective": _kappa_effective_check(
            manifest, container_ids, config.kappa
        )
        if condition_id in ("b0_lc", "b1_snar")
        else None,
        "clip": _clip_gate(manifest, container_ids),
        "amplitude": evaluate_amplitude_preservation_gate(alpha_map, config.kappa)
        if condition_id in ("b0_lc", "b1_snar")
        else None,
    }

    passed = True
    df = pd.DataFrame(per)

    if condition_id == "b0_lc":
        passed = (
            float(df["acf1_z"].mean()) >= GENERATOR_GATES["b0_acf1_min"]
            and GENERATOR_GATES["b0_late_early_std_ratio_min"]
            <= float(df["late_early_std_ratio"].mean())
            <= GENERATOR_GATES["b0_late_early_std_ratio_max"]
            and float(df["roll_std_late_mean"].mean())
            >= GENERATOR_GATES["b0_roll_std_late_min"]
            and float(df["rollout_rmse"].mean())
            <= GENERATOR_GATES["b0_rollout_rmse_max"]
            and gate["kappa_effective"]["pass"]
            and gate["clip"]["pass"]
            and gate["amplitude"]["pass"]
        )
    elif condition_id == "b1_snar":
        passed = (
            float(df["acf1_z"].mean()) >= GENERATOR_GATES["b1_acf1_min"]
            and gate["kappa_effective"]["pass"]
            and gate["clip"]["pass"]
            and gate["amplitude"]["pass"]
        )
    elif condition_id in ("c0_iid", "c1_iid"):
        passed = (
            float(np.abs(df["acf1_z"]).mean())
            <= GENERATOR_GATES["c_acf1_abs_mean_max"]
            and gate["clip"]["pass"]
        )
        if matched_c_pair is not None:
            mb, ab, ac = matched_c_pair
            passed = passed and _c_std_match_effective(
                mb, manifest, ab, alpha_map, container_ids
            )["pass"]

    gate["pass"] = passed
    return gate


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config = CSRLEConfig()
    exp_dir = REPO_ROOT / "experiments" / f"synthetic_residual_learnability_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSRLE protocol: {PROTOCOL_VERSION}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Pre-amendment reference: {PRE_AMENDMENT_REFERENCE_RUN}")

    frozen_checksums = {
        "train_df.parquet": _sha256(REPO_ROOT / "data" / "train_df.parquet"),
        "val_df.parquet": _sha256(REPO_ROOT / "data" / "val_df.parquet"),
        "scalers.pkl": _sha256(REPO_ROOT / "data" / "scalers.pkl"),
        "selected_containers.npy": _sha256(
            REPO_ROOT / "data" / "selected_containers.npy"
        ),
    }
    _write_json(exp_dir / "verification" / "frozen_integrity_checksums.json", {
        "checksums": frozen_checksums,
        "protocol_version": PROTOCOL_VERSION,
        "pre_amendment_reference_run": PRE_AMENDMENT_REFERENCE_RUN,
        "recorded_at": timestamp,
    })

    train_df, val_df, frozen_scalers, selected = load_frozen_base_data(REPO_ROOT)
    container_ids, skipped = evaluable_container_ids(
        selected, train_df, val_df, frozen_scalers
    )
    print(f"Evaluable containers: {len(container_ids)}; skipped: {skipped}")

    control_train = train_df[train_df["container_id"].isin(container_ids)].copy()
    control_val = val_df[val_df["container_id"].isin(container_ids)].copy()

    manifests: dict[str, pd.DataFrame] = {}
    alpha_maps: dict[str, dict[str, float]] = {}

    # --- B0 ---
    print("Generating B0-LC (boundary-safe)...")
    b0_train, b0_val, b0_manifest, b0_alpha, b0_sigma_eff = build_b_condition_dataframes(
        "b0_lc", container_ids, train_df, val_df, frozen_scalers, config,
    )
    manifests["b0_lc"] = b0_manifest
    alpha_maps["b0_lc"] = b0_alpha
    _write_parquet(exp_dir / "data" / "b0_lc" / "train_syn.parquet", b0_train)
    _write_parquet(exp_dir / "data" / "b0_lc" / "val_syn.parquet", b0_val)

    # --- B1 ---
    print("Generating B1-SNAR (boundary-safe)...")
    b1_train, b1_val, b1_manifest, b1_alpha, b1_sigma_eff = build_b_condition_dataframes(
        "b1_snar", container_ids, train_df, val_df, frozen_scalers, config,
    )
    manifests["b1_snar"] = b1_manifest
    alpha_maps["b1_snar"] = b1_alpha
    _write_parquet(exp_dir / "data" / "b1_snar" / "train_syn.parquet", b1_train)
    _write_parquet(exp_dir / "data" / "b1_snar" / "val_syn.parquet", b1_val)

    # --- C0 / C1 ---
    print("Generating C0/C1 (matched effective B scale)...")
    c0_train, c0_val, c0_manifest, c0_alpha = build_c_condition_dataframes(
        "c0_iid", container_ids, train_df, val_df, frozen_scalers, config,
        b0_alpha, b0_sigma_eff,
    )
    c1_train, c1_val, c1_manifest, c1_alpha = build_c_condition_dataframes(
        "c1_iid", container_ids, train_df, val_df, frozen_scalers, config,
        b1_alpha, b1_sigma_eff,
    )
    manifests["c0_iid"] = c0_manifest
    manifests["c1_iid"] = c1_manifest
    alpha_maps["c0_iid"] = c0_alpha
    alpha_maps["c1_iid"] = c1_alpha
    _write_parquet(exp_dir / "data" / "c0_iid" / "train_syn.parquet", c0_train)
    _write_parquet(exp_dir / "data" / "c0_iid" / "val_syn.parquet", c0_val)
    _write_parquet(exp_dir / "data" / "c1_iid" / "train_syn.parquet", c1_train)
    _write_parquet(exp_dir / "data" / "c1_iid" / "val_syn.parquet", c1_val)

    # --- Control A ---
    print("Building control manifest...")
    ctrl_train, ctrl_val, control_manifest = build_control_dataframes(
        container_ids, train_df, val_df, frozen_scalers,
    )
    manifests["control"] = control_manifest
    _write_parquet(exp_dir / "data" / "control" / "train_syn.parquet", ctrl_train)
    _write_parquet(exp_dir / "data" / "control" / "val_syn.parquet", ctrl_val)

    all_manifest = pd.concat(manifests.values(), ignore_index=True)
    _write_parquet(
        exp_dir / "data" / "ground_truth" / "injection_manifest.parquet",
        all_manifest,
    )

    # Alpha / κ_effective artifacts
    alpha_summary = {
        cond: alpha_distribution_summary(am)
        for cond, am in alpha_maps.items()
    }
    for cond, am in alpha_maps.items():
        alpha_df = pd.DataFrame([
            {"container_id": cid, "alpha": a, "kappa_effective": kappa_effective(a, config.kappa)}
            for cid, a in am.items()
        ])
        out = exp_dir / "synthetic_validation" / f"alpha_map_{cond}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        alpha_df.to_csv(out, index=False)
    _write_json(exp_dir / "synthetic_validation" / "alpha_distribution_summary.json", alpha_summary)

    # --- Generator validation ---
    print("Running generator validation gates...")
    generator_results: dict[str, Any] = {}
    for cond in ("b0_lc", "b1_snar", "c0_iid", "c1_iid"):
        matched = None
        if cond == "c0_iid":
            matched = (manifests["b0_lc"], b0_alpha, c0_alpha)
        elif cond == "c1_iid":
            matched = (manifests["b1_snar"], b1_alpha, c1_alpha)
        generator_results[cond] = _run_generator_gates(
            cond, manifests[cond], container_ids, config,
            alpha_maps[cond], train_df, val_df, frozen_scalers, matched,
        )

    _write_json(
        exp_dir / "synthetic_validation" / "generator_gate_report.json",
        generator_results,
    )

    # --- Prophet retention ---
    print("Running Prophet retention analysis (P_A vs P_B)...")
    retention_results: dict[str, Any] = {}
    for cond_key, synth_train, synth_val, manifest in [
        ("b0", b0_train, b0_val, b0_manifest),
        ("b1", b1_train, b1_val, b1_manifest),
    ]:
        rows = []
        r_b_series: dict[str, np.ndarray] = {}
        for cid in container_ids:
            y_train_syn = synth_train[
                synth_train["container_id"] == cid
            ]["cpu_util_percent"].values
            cond_scaler, _ = fit_transform_condition_scaler(y_train_syn)
            val_manifest = manifest[
                (manifest["container_id"] == cid) & (manifest["split"] == "val")
            ]
            m = run_prophet_retention_for_pair(
                cid, control_train, control_val,
                synth_train, synth_val,
                frozen_scalers[cid], cond_scaler, val_manifest,
            )
            rows.append(m)
            yhat_b, y_b = fit_prophet_forecast_val(
                synth_train, synth_val, cid, cond_scaler
            )
            r_b_series[cid] = y_b - yhat_b

        ret_df = pd.DataFrame(rows)
        acf_mean = float(np.nanmean([
            avg_abs_acf(r_b_series[c], 1, 10) for c in container_ids
        ]))
        ljung_frac = ljung_box_reject_fraction(r_b_series, lag=20)
        gate = evaluate_retention_gate(cond_key, ret_df, acf_mean, ljung_frac)

        n_val_all = []
        for cid in container_ids:
            val_m = manifest[
                (manifest["container_id"] == cid) & (manifest["split"] == "val")
            ].sort_values("time_stamp")
            n_val_all.extend(val_m["N"].values[:DAY1_HORIZON])
        n_arr = np.array(n_val_all)
        period_steps = float("nan")
        if cond_key == "b0" and len(n_arr) > 1:
            fft = np.abs(np.fft.rfft(n_arr - np.mean(n_arr)))
            freqs = np.fft.rfftfreq(len(n_arr), d=1.0)
            peak = int(np.argmax(fft[1:]) + 1) if len(fft) > 1 else 0
            if peak and freqs[peak] > 0:
                period_steps = 1.0 / freqs[peak]

        retention_results[cond_key] = {
            "per_container": rows,
            "cohort_summary": {
                "rho_n_i_mean": float(ret_df["rho_n_i"].mean()),
                "rho_n_delta_p_mean": float(ret_df["rho_n_delta_p"].mean()),
                "vs_mean": float(ret_df["vs"].mean()),
                "vr_mean": float(ret_df["vr"].mean()),
                "identity_max_err": float(ret_df["identity_max_abs_err"].max()),
                "acf_r_b_1_10_mean": acf_mean,
                "ljung_reject_fraction_lag20": ljung_frac,
                "dominant_n_period_steps_pooled": period_steps,
                "dominant_n_period_hours_pooled": period_steps * 15 / 60
                if period_steps == period_steps else float("nan"),
            },
            "gate": gate,
        }
        ret_df.to_csv(
            exp_dir / "synthetic_validation" / f"prophet_retention_{cond_key}.csv",
            index=False,
        )

    _write_json(
        exp_dir / "synthetic_validation" / "prophet_retention_report.json",
        retention_results,
    )

    # --- Condition A Prophet reconciliation ---
    print("Prophet control metrics (Day-1 vs full val)...")
    prophet_control = _prophet_control_metrics(
        control_train, control_val, frozen_scalers, container_ids,
    )
    prophet_control.to_csv(
        exp_dir / "synthetic_validation" / "control_prophet_mae_reconciliation.csv",
        index=False,
    )
    mean_day1 = float(prophet_control["prophet_day1_mae"].mean())
    mean_full = float(prophet_control["prophet_full_val_mae"].mean())
    control_status = {
        "prophet_day1_mean_mae": mean_day1,
        "prophet_full_val_mean_mae": mean_full,
        "reference_residual_analysis_full_val_mae": 1.8474,
        "reference_residual_analysis_scope": "Full validation window (~150-154 steps)",
        "csrle_day1_scope": f"Day-1 only ({DAY1_HORIZON} steps)",
        "mae_discrepancy_explanation": (
            "1.733 (Day-1) vs 1.847 (full val) — different evaluation horizons, "
            "same 99 containers, same Prophet config, same real CPU space."
        ),
        "hybrid_reproduction_gate": "DEFERRED",
    }

    # --- Freeze decision ---
    b0_gen = generator_results["b0_lc"]["pass"]
    b1_gen = generator_results["b1_snar"]["pass"]
    c0_gen = generator_results["c0_iid"]["pass"]
    c1_gen = generator_results["c1_iid"]["pass"]
    b0_ret = retention_results["b0"]["gate"]["pass"]
    b1_ret = retention_results["b1"]["gate"]["pass"]

    permitted = ["control"]
    if b0_gen and b0_ret:
        permitted.append("b0_lc")
    if b1_gen and b1_ret:
        permitted.append("b1_snar")
    if c0_gen:
        permitted.append("c0_iid")
    if c1_gen:
        permitted.append("c1_iid")

    freeze_payload = {
        "frozen_at": timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "pre_amendment_reference_run": PRE_AMENDMENT_REFERENCE_RUN,
        "config": config.to_dict(),
        "generator_gates": GENERATOR_GATES,
        "amplitude_preservation_gates": AMPLITUDE_PRESERVATION_GATES,
        "b0_retention_gates": B0_RETENTION_GATES,
        "b1_retention_gates": B1_RETENTION_GATES,
        "generator_results_pass": {
            "b0_lc": b0_gen, "b1_snar": b1_gen,
            "c0_iid": c0_gen, "c1_iid": c1_gen,
        },
        "retention_results_pass": {"b0": b0_ret, "b1": b1_ret},
        "permitted_gru_conditions": permitted,
        "alpha_rule": (
            "α_c = min(1, min_t α_t^max) on train; "
            "N_eff = α_c · N_target; κ_eff = α_c · 0.10"
        ),
        "c_matching": (
            "N_target_C = (σ_eff_B / (α_B · std(z_train))) · z_iid; "
            "α_C = min(α_B, α_strict(N_target_C)); N_eff = α_C · N_target_C"
        ),
    }

    all_gen_pass = b0_gen and b1_gen and c0_gen and c1_gen
    if all_gen_pass and b0_ret and b1_ret:
        _write_json(exp_dir / "config" / "synthetic_config_frozen.json", freeze_payload)
        print(f"Frozen config: {exp_dir / 'config' / 'synthetic_config_frozen.json'}")
    else:
        print("Not all gates passed — synthetic_config_frozen.json NOT written.")

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "pre_amendment_reference_run": PRE_AMENDMENT_REFERENCE_RUN,
        "experiment_dir": str(exp_dir),
        "n_containers": len(container_ids),
        "generator_pass": freeze_payload["generator_results_pass"],
        "retention_pass": freeze_payload["retention_results_pass"],
        "alpha_summary": alpha_summary,
        "permitted_gru_conditions": permitted,
        "control_status": control_status,
        "gru_training_occurred": False,
    }
    _write_json(exp_dir / "synthetic_validation" / "pre_gru_summary.json", summary)

    print("\n=== PRE-GRU SUMMARY (v2 boundary-safe) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
