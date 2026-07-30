# Monitoring (Layer 1)

**Modules:** `monitoring.py`, `rolling_metrics.py`

## Purpose

Simulate operational monitoring during a leakage-free walk-forward study. At each origin, the framework forecasts the next Day-1 window and records operational error metrics.

## Metrics

Per container, per origin:

- Day-1 Hybrid MAE / RMSE
- Day-1 Prophet MAE / RMSE (decomposition baseline for diagnostics)

Rolling aggregates (configurable window, default 3 origins):

- `rolling_hybrid_mae`, `rolling_hybrid_rmse`
- `rolling_prophet_mae`, `rolling_prophet_rmse`

## Walk-forward rules

- Origins start after `min_history_steps` (default 200).
- Stride defaults to 96 steps (one Day-1 block).
- Only history **strictly before** the origin is used for forecasting.
- Actuals for metric computation come from the **future** Day-1 window (already observed in simulation).

## Leakage assertions

`monitoring.assert_leakage_free_origin` verifies:

- Origin is within series bounds.
- Forecast horizon is available after origin.
- Candidate training cutoff ≤ trigger origin (enforced in retraining module).

## Outputs

`experiments/.../monitoring/origin_metrics.csv` — full per-origin metric table merged with frozen baselines.
