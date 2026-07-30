# Diagnostics (Layer 4)

**Module:** `diagnostics.py`

## Purpose

Diagnostics are a **primary decision component**, not visualization-only. The framework compares Prophet vs Hybrid error degradation to explain *why* drift may have occurred.

## Classification rules

| Class | Pattern | Interpretation |
|-------|---------|----------------|
| `GRU_STALE` | Hybrid ↑, Prophet stable | Residual/GRU path stale; offline retrain appropriate |
| `REGIME_CHANGE` | Hybrid ↑, Prophet ↑ | Workload regime shift; retrain with new history |
| `INVESTIGATE` | Prophet ↑, Hybrid stable | Check data/decomposition before GRU retrain |
| `ANOMALY` | Unusual compensation | Manual review |
| `STABLE` | No significant change | No action |

Each result includes **confidence**, **interpretation**, and **recommended_action**.

## Influence on decisions

`decision_engine.py` counts `GRU_STALE` and `REGIME_CHANGE` as retrain-supporting classes. `INVESTIGATE` and `ANOMALY` can suppress retraining when they dominate the cohort.

Cohort diagnostic confidence is a weighted aggregate used alongside the global trigger.

## Outputs

`diagnostics/diagnostics.csv` — full classification log per container per origin.
