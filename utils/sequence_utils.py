"""Sliding-window sequence builders for Hybrid and Global GRU pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_residual_sequences(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    input_window: int = 96,
    forecast_horizon: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for Hybrid GRU residual forecasting.

    Parameters
    ----------
    df:
        Dataframe with ``container_id``, ``time_stamp``, feature columns,
        and a target column.
    features:
        Input feature column names.
    target:
        Target column name.
    input_window:
        Number of past timesteps per input sequence.
    forecast_horizon:
        Number of future timesteps per target sequence.

    Returns
    -------
    X:
        Shape ``[n_sequences, input_window, n_features]``.
    y:
        Shape ``[n_sequences, forecast_horizon]``.
    sample_cids:
        Container ID for each sequence.
    """
    return _create_sliding_window_sequences(
        df=df,
        features=features,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )


def create_longterm_sequences(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    input_window: int = 96,
    forecast_horizon: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for Global GRU direct CPU forecasting.

    Uses the same chronological sliding-window logic as
    ``create_residual_sequences()``, but is the canonical entry point for
    the Global GRU baseline (multi-feature input, ``cpu_scaled`` target).

    Parameters
    ----------
    df:
        Dataframe with ``container_id``, ``time_stamp``, feature columns,
        and a target column. For baseline training, pass ``global_train``
        only — never concatenate the validation period.
    features:
        Input feature column names (baseline: ``cpu_scaled``, ``cpu_mean``,
        ``cpu_std``).
    target:
        Target column name (baseline: ``cpu_scaled``).
    input_window:
        Number of past timesteps per input sequence.
    forecast_horizon:
        Number of future timesteps per target sequence.

    Returns
    -------
    X:
        Shape ``[n_sequences, input_window, n_features]``.
    y:
        Shape ``[n_sequences, forecast_horizon]``.
    sample_cids:
        Container ID for each sequence.
    """
    return _create_sliding_window_sequences(
        df=df,
        features=features,
        target=target,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )


def _create_sliding_window_sequences(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    input_window: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared per-container chronological sliding-window builder."""
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    sample_cids: list[str] = []

    for cid, group in df.groupby("container_id"):
        group = group.sort_values("time_stamp")

        values = group[features].values
        target_values = group[target].values

        max_idx = len(group) - input_window - forecast_horizon

        for i in range(max_idx):
            X.append(values[i : i + input_window])
            y.append(
                target_values[
                    i + input_window : i + input_window + forecast_horizon
                ]
            )
            sample_cids.append(cid)

    return (
        np.array(X),
        np.array(y),
        np.array(sample_cids),
    )
