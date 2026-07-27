# Production Architecture

## Model

**Hybrid Prophet + GRU** — matches the frozen research baseline exactly.

1. **Prophet** — per-container trend + daily seasonality on scaled CPU
2. **GRU** — learns globally normalized Prophet residuals
3. **Forecast** — Prophet + denormalized GRU residual → real CPU %

## GRU architecture

```
Input (96, 1) residual_scaled
  → GRU(256) → Dropout(0.2)
  → GRU(128) → Dropout(0.2)
  → GRU(64)
  → Dense(128, relu)
  → Dense(96)
```

405,088 parameters | Adam | MSE loss | batch 64 | early stopping patience 10

## Module layering

```
production/utils/
  config.py      — paths, hyperparameters
  preprocess.py  — container + temporal splits
  training.py    — wraps utils.hybrid_training
  inference.py   — wraps utils.hybrid_inference
  evaluation.py  — metrics + batch eval
  artifacts.py   — save/load Keras model + pickles
```

Research modules are imported read-only from `utils/` at the repository root.

## Artifact storage

All trained outputs live under `production/hybrid/`:

| Directory | Contents |
|-----------|----------|
| `models/` | `model.keras` |
| `metadata/` | scalers, residual stats, container IDs |
| `config/` | production + training metadata JSON |
| `cache/` | preprocessed train/val/unseen parquet |
| `metrics/` | evaluation CSV + summaries |
| `predictions/` | per-container prediction pickles |
| `logs/` | training history CSV |
