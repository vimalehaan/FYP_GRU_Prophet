# Production Pipeline

## End-to-end flow

```
data/df_resampled.parquet
        │
        ▼
┌───────────────────────────┐
│ Container split (90/10)   │  seed=42, immutable once saved
└───────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
 Train      Unseen
 containers  containers
   │
   ▼
 Per-container 80/20 temporal split
 MinMaxScaler fit on train 80% only
   │
   ▼
 Prophet residuals (train period)
 Global residual normalization
 Sequence generation
   │
   ▼
 Hybrid GRU training (frozen hyperparameters)
   │
   ▼
 Save artifacts → production/hybrid/
   │
   ▼
 Evaluate known + unseen containers
```

## Entry point

**Training:** `production/scripts/train_production_hybrid.py` only.

**Notebook:** `production/notebooks/production_hybrid_model.ipynb` loads artifacts and generates plots. Retraining is disabled by default (`RUN_TRAINING = False`).

## Module layout

```
production/utils/
  config.py       — paths and frozen hyperparameters
  split.py        — container-level 90/10 split
  preprocess.py   — temporal 80/20 + scaling
  training.py     — GRU training with history capture
  inference.py    — known and unseen container inference
  evaluation.py   — extended metrics and prediction export
  artifacts.py    — save/load production artifacts
```

Reuses frozen baseline functions from `utils/hybrid_training.py` and `utils/hybrid_inference.py` without modification.
