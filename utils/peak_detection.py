"""Peak threshold computation and sequence weight-matrix construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.peak_config import (
    EXPECTED_N_SEQUENCES,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    NON_PEAK_WEIGHT,
    PEAK_PERCENTILE,
    PEAK_WEIGHT,
)


def cpu_real_series(cpu_scaled: pd.Series, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform scaled CPU values to real utilization percent."""
    values = cpu_scaled.values.reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def compute_peak_thresholds(
    global_train: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    percentile: int = PEAK_PERCENTILE,
) -> dict[str, float]:
    """
    Compute per-container peak thresholds on the train period only.

    Parameters
    ----------
    global_train:
        Train-period dataframe with ``container_id``, ``time_stamp``,
        and ``cpu_scaled`` columns.
    scalers:
        Per-container MinMax scalers keyed by ``container_id``.
    percentile:
        Percentile for the upper-tail threshold (locked at 90 for this study).

    Returns
    -------
    Mapping from container ID to threshold in real CPU percent.
    """
    thresholds: dict[str, float] = {}

    for container_id, group in global_train.groupby("container_id"):
        group = group.sort_values("time_stamp")
        cpu_real = cpu_real_series(group["cpu_scaled"], scalers[container_id])
        thresholds[container_id] = float(np.percentile(cpu_real, percentile))

    return thresholds


def label_peak_timesteps(
    cpu_real: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a boolean mask where timesteps meet the peak rule."""
    return cpu_real >= threshold


def build_sequence_weight_matrix(
    global_train: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    thresholds: dict[str, float],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    peak_weight: float = PEAK_WEIGHT,
    non_peak_weight: float = NON_PEAK_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-timestep loss weights aligned with ``create_residual_sequences``.

    Window iteration mirrors ``utils.sequence_utils.create_residual_sequences``:
    for each container (``groupby`` order), chronologically sorted, generate
    one weight row per valid sliding window. Only forecast-horizon timesteps
    are weighted; peaks are labelled on real CPU percent.

    Parameters
    ----------
    global_train:
        Train-period dataframe used for sequence generation.
    scalers:
        Per-container MinMax scalers.
    thresholds:
        Train-fitted per-container thresholds in real CPU percent.
    input_window:
        Residual input context length.
    forecast_horizon:
        Residual target horizon length.
    peak_weight:
        Loss weight applied to peak horizon timesteps.
    non_peak_weight:
        Loss weight applied to non-peak horizon timesteps.

    Returns
    -------
    weight_matrix:
        Array of shape ``(n_sequences, forecast_horizon)``.
    sample_cids:
        Container ID for each sequence row, aligned with baseline sequences.
    """
    weights: list[np.ndarray] = []
    sample_cids: list[str] = []

    for container_id, group in global_train.groupby("container_id"):
        group = group.sort_values("time_stamp")
        cpu_real = cpu_real_series(group["cpu_scaled"], scalers[container_id])
        threshold = thresholds[container_id]
        max_idx = len(group) - input_window - forecast_horizon

        for start in range(max_idx):
            horizon_cpu = cpu_real[
                start + input_window : start + input_window + forecast_horizon
            ]
            peak_mask = label_peak_timesteps(horizon_cpu, threshold)
            row = np.where(peak_mask, peak_weight, non_peak_weight).astype(
                np.float32
            )
            weights.append(row)
            sample_cids.append(container_id)

    weight_matrix = np.stack(weights, axis=0)
    return weight_matrix, np.array(sample_cids)


def summarize_weight_matrix(
    weight_matrix: np.ndarray,
    peak_weight: float = PEAK_WEIGHT,
    non_peak_weight: float = NON_PEAK_WEIGHT,
) -> dict[str, float | int]:
    """
    Summarize peak vs non-peak weight coverage for sanity checks.

    Returns global fraction of weighted timesteps equal to ``peak_weight``.
    """
    total_timesteps = int(weight_matrix.size)
    peak_timesteps = int(np.sum(weight_matrix == peak_weight))
    non_peak_timesteps = int(np.sum(weight_matrix == non_peak_weight))

    return {
        "n_sequences": int(weight_matrix.shape[0]),
        "forecast_horizon": int(weight_matrix.shape[1]),
        "total_weighted_timesteps": total_timesteps,
        "peak_weighted_timesteps": peak_timesteps,
        "non_peak_weighted_timesteps": non_peak_timesteps,
        "peak_weight_fraction": (
            peak_timesteps / total_timesteps if total_timesteps else 0.0
        ),
        "peak_weight": peak_weight,
        "non_peak_weight": non_peak_weight,
    }


def assert_binary_weight_values(
    weight_matrix: np.ndarray,
    peak_weight: float = PEAK_WEIGHT,
    non_peak_weight: float = NON_PEAK_WEIGHT,
) -> None:
    """
    Raise ``AssertionError`` if any entry is not exactly peak or non-peak weight.
    """
    allowed = {float(peak_weight), float(non_peak_weight)}
    unique_values = {float(value) for value in np.unique(weight_matrix)}
    unexpected = sorted(unique_values - allowed)
    if unexpected:
        raise AssertionError(
            "Weight matrix contains unexpected values: "
            f"{unexpected}. Allowed: {sorted(allowed)}"
        )


def verify_binary_weight_values(
    weight_matrix: np.ndarray,
    peak_weight: float = PEAK_WEIGHT,
    non_peak_weight: float = NON_PEAK_WEIGHT,
) -> dict[str, bool | list[float] | int]:
    """
    Verify that ``weight_matrix`` contains only the two expected weight values.
    """
    allowed = [float(non_peak_weight), float(peak_weight)]
    unique_values = sorted(float(v) for v in np.unique(weight_matrix))
    unexpected = [v for v in unique_values if v not in allowed]
    peak_count = int(np.sum(weight_matrix == peak_weight))
    non_peak_count = int(np.sum(weight_matrix == non_peak_weight))
    total = int(weight_matrix.size)

    return {
        "only_expected_values": len(unexpected) == 0,
        "allowed_values": allowed,
        "unique_values_found": unique_values,
        "unexpected_values": unexpected,
        "peak_weight_count": peak_count,
        "non_peak_weight_count": non_peak_count,
        "total_entries": total,
        "counts_sum_to_total": (peak_count + non_peak_count) == total,
    }


def assert_expected_sequence_counts(
    n_sequences: int,
    weight_matrix: np.ndarray,
    expected_n: int = EXPECTED_N_SEQUENCES,
) -> None:
    """Raise ``AssertionError`` if sequence counts diverge from baseline."""
    if n_sequences != expected_n:
        raise AssertionError(
            f"Expected {expected_n} sequences, got {n_sequences}"
        )
    if weight_matrix.shape[0] != expected_n:
        raise AssertionError(
            f"Expected weight matrix rows {expected_n}, "
            f"got {weight_matrix.shape[0]}"
        )


def save_peak_thresholds(
    thresholds: dict[str, float],
    path: str | Path,
) -> None:
    """Persist per-container peak thresholds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(thresholds, handle)


def load_peak_thresholds(path: str | Path) -> dict[str, float]:
    """Load per-container peak thresholds."""
    with open(path, "rb") as handle:
        data: dict[str, float] = pickle.load(handle)
    return data


def align_with_residual_sequences(
    global_train: pd.DataFrame,
    train_residual_df: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
    thresholds: dict[str, float],
    input_window: int = INPUT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> dict[str, Any]:
    """
    Verify weight-matrix rows align with baseline ``create_residual_sequences``.

    Compares container-ID order row-by-row against the baseline builder.
    """
    from utils.sequence_utils import create_residual_sequences

    features = ["residual_scaled"]
    _, _, baseline_cids = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    _, peak_cids = build_sequence_weight_matrix(
        global_train=global_train,
        scalers=scalers,
        thresholds=thresholds,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    return {
        "n_baseline": int(len(baseline_cids)),
        "n_peak_weights": int(len(peak_cids)),
        "cid_arrays_equal": bool(np.array_equal(baseline_cids, peak_cids)),
    }
