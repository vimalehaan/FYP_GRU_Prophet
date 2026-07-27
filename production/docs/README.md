# Production Pipeline — Hybrid Prophet + GRU

Deployable production workspace for the final Hybrid CPU forecasting model. **Isolated from research experiments.**

## Quick start

```bash
# Train (when needed — does not run automatically)
tf_metal_env/bin/python production/scripts/train_production_hybrid.py

# Evaluate saved model (no retraining)
tf_metal_env/bin/python production/scripts/evaluate_production_model.py

# Single-container inference
tf_metal_env/bin/python production/scripts/predict_container.py c_1016 --split known

# Build demonstration notebook
tf_metal_env/bin/python production/scripts/build_production_notebook.py
jupyter notebook production/notebooks/production_hybrid_model.ipynb
```

## Layout

```
production/
├── hybrid/          # Model artifacts (frozen after training)
├── utils/           # Production Python modules
├── scripts/         # CLI entry points
├── notebooks/       # Demonstration notebook
└── docs/            # Production documentation
```

See [repository_structure.md](repository_structure.md) for the full tree.

## Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Hybrid Prophet + GRU production architecture |
| [training.md](training.md) | Training protocol and hyperparameters |
| [evaluation.md](evaluation.md) | Known vs unseen evaluation |
| [inference.md](inference.md) | Single-container inference |
| [deployment.md](deployment.md) | Artifact bundle and serving |
| [reproducibility.md](reproducibility.md) | Seeds, splits, and isolation |
| [repository_structure.md](repository_structure.md) | Full directory map |

## Research vs production

- **Research** (`experiments/`, `notebooks/`, `utils/`, `docs/`) — frozen hypothesis testing
- **Production** (`production/`) — final deployable model only

Production code **wraps** frozen research Hybrid modules (`utils/hybrid_training.py`, `utils/hybrid_inference.py`) without copying or modifying them.
