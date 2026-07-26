"""Frozen configuration constants for Temporal Memory Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW

PROTOCOL_VERSION = "tma_v1.0"
MAX_LAG = 672
DAILY_LAGS: tuple[int, ...] = (96, 192, 288, 384, 480, 576, 672)
CORRELATION_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3)
CANDIDATE_WINDOWS: tuple[int, ...] = (96, 192, 288, 384)
BOOTSTRAP_SEED = 12345
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 0.95
SAMPLING_MINUTES = 15
STEPS_PER_DAY = 96
FFT_FS = 1.0
FOURIER_HARMONICS = 10

BASELINE_INFERENCE_CACHE = (
    "experiments/peak_aware_2026-07-14_164518/evaluation/"
    "baseline_inference_cache.pkl"
)


@dataclass
class TMAConfig:
    """Runtime configuration for a TMA experiment run."""

    timestamp: str
    repo_root: Any
    max_lag: int = MAX_LAG
    daily_lags: tuple[int, ...] = DAILY_LAGS
    correlation_thresholds: tuple[float, ...] = CORRELATION_THRESHOLDS
    candidate_windows: tuple[int, ...] = CANDIDATE_WINDOWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_n: int = BOOTSTRAP_N
    bootstrap_ci: float = BOOTSTRAP_CI
    input_window_reference: int = DEFAULT_INPUT_WINDOW
    day1_horizon: int = DAY1_HORIZON
    protocol_version: str = PROTOCOL_VERSION
    experiment_name: str = "temporal_memory_analysis"
    extra: dict[str, Any] = field(default_factory=dict)
