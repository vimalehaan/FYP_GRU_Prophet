# Adaptive Forecast Model Lifecycle Framework (AFMLF)

**Status:** Isolated post-forecasting research phase  
**Forecasting research:** Complete — Hybrid Prophet+GRU frozen as production model  

## What AFMLF is

AFMLF is an **operational lifecycle management layer** for a deployed Hybrid forecaster. It determines:

- **when** concept drift has occurred  
- **why** (Prophet vs Hybrid diagnostic reasoning)  
- **whether** offline retraining should be recommended  
- **whether** a candidate model should replace the incumbent  

AFMLF does **not** change Hybrid architecture, does **not** perform online learning, and does **not** automatically overwrite production artifacts.

## Why forecasting is frozen

Module 1 forecasting research is complete. Hybrid Prophet+GRU was validated in controlled experiments and selected for production. AFMLF evaluates **lifecycle governance** — not another accuracy benchmark.

## Why this is not online learning

Online learning would update GRU weights continuously at inference time. AFMLF triggers **offline batch retraining** of candidate versions (`hybrid_v2`, …) using historical data only, then evaluates on future windows. Production is never auto-replaced.

## Isolation

All outputs live under:

`experiments/adaptive_forecast_model_lifecycle_<timestamp>/`

Code lives under:

`utils/adaptive_lifecycle/`

Frozen experiments, production Hybrid, and forecast service are **read-only inputs**.

## Run

```bash
# Pilot (3 containers, 2 origins)
tf_metal_env/bin/python scripts/run_adaptive_forecast_model_lifecycle.py --pilot

# Full 99-container experiment (slow — Prophet per origin)
tf_metal_env/bin/python scripts/run_adaptive_forecast_model_lifecycle.py

# Monitoring/drift only (no candidate training)
tf_metal_env/bin/python scripts/run_adaptive_forecast_model_lifecycle.py --skip-candidate-retraining
```

## Documentation index

| Document | Topic |
|----------|--------|
| [research_motivation.md](research_motivation.md) | Why AFMLF exists |
| [architecture.md](architecture.md) | Layers and modules |
| [module_interactions.md](module_interactions.md) | Data flow and orchestration |
| [state_machine.md](state_machine.md) | Lifecycle states |
| [monitoring.md](monitoring.md) | Layer 1 walk-forward metrics |
| [drift_detection.md](drift_detection.md) | Layer 2 per-container thresholds |
| [page_hinkley.md](page_hinkley.md) | Layer 3 confirmatory test |
| [diagnostics.md](diagnostics.md) | Layer 4 Prophet vs Hybrid |
| [decision_engine.md](decision_engine.md) | Layer 5 trigger + decisions |
| [candidate_retraining.md](candidate_retraining.md) | Layer 6 offline candidates |
| [candidate_evaluation.md](candidate_evaluation.md) | Layer 7 future-window eval |
| [deployment_recommendation.md](deployment_recommendation.md) | Layer 8 human gate |
| [model_versioning.md](model_versioning.md) | hybrid_v1, v2, … registry |
| [retraining_methodology.md](retraining_methodology.md) | Frozen candidate dataset policy |
| [unseen_container_case_study.md](unseen_container_case_study.md) | Dual-scenario unseen deployment demo (stable + drifted) |
| [evaluation_methodology.md](evaluation_methodology.md) | Framework vs forecast metrics |
| [limitations.md](limitations.md) | Scope limits |
| [future_work.md](future_work.md) | Production integration |

## Notebook

```bash
tf_metal_env/bin/python scripts/build_afmlf_visualization_notebooks.py
```

| Notebook | Purpose |
|----------|---------|
| `notebooks/adaptive_forecast_model_lifecycle.ipynb` | Official 99-container results — **live plots** from CSV/JSON |
| `notebooks/afmlf_unseen_container_case_study.ipynb` | Two unseen containers: stable vs drifted deployment walkthrough |

Training disabled by default in the official notebook. Case study optionally trains a demo candidate under `notebooks/outputs/` only.
