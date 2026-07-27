"""Frozen configuration for RRE Stage 0 diagnostics."""

from __future__ import annotations

PROTOCOL_VERSION = "rre_diagnostics_v1.0"
EXPERIMENT_NAME = "residual_representation_diagnostics"

MAX_LAG = 96
FFT_FS = 1.0
STEPS_PER_DAY = 96
ENTROPY_BINS = 50
RANDOM_SEED = 42

REPRESENTATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "R0",
        "name": "Level residual (global z-score)",
        "description": "Frozen Hybrid baseline: global train mean/std on level residuals",
    },
    {
        "id": "R1",
        "name": "First-order difference (velocity)",
        "description": "Δr_t = r_t - r_{t-1}, global z-score on train Δ",
    },
    {
        "id": "R2",
        "name": "Per-container normalization",
        "description": "Level residual z-scored per container using train μ/σ only",
    },
    {
        "id": "R3",
        "name": "Robust scaling (median/MAD)",
        "description": "Level residual scaled by global train median and MAD",
    },
)

REPRESENTATION_IDS: tuple[str, ...] = tuple(r["id"] for r in REPRESENTATIONS)

BASELINE_REFERENCE = "experiments/baseline_reference_2026-07-14"
BASELINE_RES_MEAN = 1.5362158578354976e-05
BASELINE_RES_STD = 0.11123930781933586
