"""CSRLE Stage 2: per-condition Hybrid GRU training on frozen synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.csrle.control_reproduction import train_control_hybrid
from utils.hybrid_artifacts import save_hybrid_artifacts
from utils.hybrid_config import DEFAULT_INPUT_WINDOW


def load_condition_frames(exp_dir: Path, condition_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = exp_dir / "data" / condition_id
    return (
        pd.read_parquet(base / "train_syn.parquet"),
        pd.read_parquet(base / "val_syn.parquet"),
    )


def build_condition_scalers(train_df: pd.DataFrame) -> dict[str, MinMaxScaler]:
    """Reconstruct per-container MinMaxScalers from train cpu_util_percent."""
    scalers: dict[str, MinMaxScaler] = {}
    for cid, grp in train_df.groupby("container_id"):
        scaler = MinMaxScaler(feature_range=(0, 1))
        y = grp.sort_values("time_stamp")["cpu_util_percent"].values.reshape(-1, 1)
        scaler.fit(y)
        scalers[str(cid)] = scaler
    return scalers


def train_condition_gru(
    condition_id: str,
    train_df: pd.DataFrame,
    output_dir: Path,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train experiment-local GRU; save model, stats, metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model, res_mean, res_std, _, meta = train_control_hybrid(
        train_df,
        input_window=DEFAULT_INPUT_WINDOW,
        verbose=verbose,
    )

    (output_dir / "training").mkdir(parents=True, exist_ok=True)
    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "models" / "hybrid_gru.keras"
    stats_path = output_dir / "models" / "residual_stats.pkl"
    save_hybrid_artifacts(
        model, res_mean, res_std,
        model_path=model_path,
        residual_stats_path=stats_path,
        input_window=DEFAULT_INPUT_WINDOW,
    )

    hist = meta.pop("history")
    pd.DataFrame(hist).to_csv(output_dir / "training" / "training_history.csv", index=False)

    payload = {
        "condition_id": condition_id,
        **meta,
        "model_path": str(model_path),
        "stats_path": str(stats_path),
    }
    with (output_dir / "training" / "training_metadata.json").open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    with (output_dir / "config" / "condition_config.json").open("w") as fh:
        json.dump({"condition_id": condition_id, "input_window": DEFAULT_INPUT_WINDOW}, fh, indent=2)

    return {
        "model": model,
        "res_mean": res_mean,
        "res_std": res_std,
        "meta": payload,
        "scalers": build_condition_scalers(train_df),
    }
