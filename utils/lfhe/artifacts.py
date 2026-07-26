"""Load LFHE arm artifacts (handles custom DA-MSE loss for inference)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import tensorflow as tf

from utils.hybrid_config import DEFAULT_INPUT_WINDOW
from utils.lfhe.loss import da_mse_loss_registered, make_da_mse_loss


def _read_arm_config(arm_dir: Path) -> dict[str, Any]:
    cfg_path = arm_dir / "config" / "arm_config.json"
    if cfg_path.exists():
        with cfg_path.open() as fh:
            return json.load(fh)
    return {}


def load_lfhe_arm_artifacts(
    arm_dir: str | Path,
) -> tuple[Any, float, float, int]:
    """
    Load GRU weights + residual stats for one LFHE arm.

    Uses ``compile=False`` so inference does not require deserializing the
    training loss (standard MSE or custom DA-MSE).
    """
    arm_dir = Path(arm_dir)
    model_path = arm_dir / "models" / "hybrid_gru.keras"
    stats_path = arm_dir / "models" / "residual_stats.pkl"
    arm_cfg = _read_arm_config(arm_dir)

    custom_objects: dict[str, Any] = {"da_mse_loss": da_mse_loss_registered}
    if arm_cfg.get("loss_kind") == "da_mse":
        custom_objects["loss"] = make_da_mse_loss(
            lambda_disp=float(arm_cfg.get("lambda_disp", 1.0)),
            epsilon=float(arm_cfg.get("epsilon", 1e-6)),
        )

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
    )

    with stats_path.open("rb") as handle:
        residual_stats = pickle.load(handle)

    input_window = int(residual_stats.get("input_window", DEFAULT_INPUT_WINDOW))
    return (
        model,
        float(residual_stats["res_mean"]),
        float(residual_stats["res_std"]),
        input_window,
    )
