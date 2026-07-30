# Architecture

## Layers

| Layer | Module | Role |
|-------|--------|------|
| L1 Monitoring | `monitoring.py`, `rolling_metrics.py` | Walk-forward Day-1 MAE/RMSE |
| L2 Drift | `drift_detection.py` | Per-container baseline thresholds |
| L3 Page-Hinkley | `page_hinkley.py` | Confirmatory drift (default) |
| L4 Diagnostics | `diagnostics.py` | Prophet vs Hybrid classification |
| L5 Decision | `decision_engine.py`, `trigger_policy.py` | Retrain / defer / investigate |
| L6 Retrain | `candidate_retraining.py` | Offline Hybrid v2, v3, … |
| L7 Evaluate | `candidate_evaluation.py` | Candidate vs incumbent on future origins |
| L8 Deploy rec | `candidate_evaluation.py` | Deploy / keep / investigate |
| Lifecycle | `lifecycle.py` | Central state machine |
| Versioning | `model_versioning.py` | Artifact metadata registry |

## Orchestration

`utils/adaptive_lifecycle/pipeline.py` and `scripts/run_adaptive_forecast_model_lifecycle.py`
