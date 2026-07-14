"""Shared Hybrid Prophet + GRU forecast configuration."""

from __future__ import annotations

# Production default until H1.2 ablation selects a final window.
DEFAULT_INPUT_WINDOW = 96

# Primary evaluation horizon (24 hours at 15-minute intervals).
DAY1_HORIZON = 96

# Total Prophet forecast length for supplementary Day 2 recursive analysis.
FORECAST_HORIZON = 192

# Candidate windows for the H1.2 ablation study.
ABLATION_INPUT_WINDOWS: tuple[int, ...] = (96, 288)
