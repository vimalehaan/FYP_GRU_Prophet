# Training

## Frozen hyperparameters

| Parameter | Value |
|-----------|-------|
| Input window | 96 |
| Output horizon | 96 |
| Optimizer | Adam |
| Loss | MSE |
| Metrics | MAE |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping | patience 10, monitor val_loss |
| Sequence val split | 80/20 (within train-period sequences) |
| Prophet | daily_seasonality=True, weekly_seasonality=False |
| Residual normalization | Global (train-period mean/std) |

## Architecture

```
GRU(256, return_sequences=True)
Dropout(0.2)
GRU(128, return_sequences=True)
Dropout(0.2)
GRU(64)
Dense(128, relu)
Dense(96)
```

Identical to `utils/hybrid_training.build_hybrid_gru_model`.

## Execution

```bash
tf_metal_env/bin/python production/scripts/train_production_hybrid.py
```

Optional flags:

- `--force-recreate-split` — recreate container split (initial setup only)
- `--verbose 0|1|2` — Keras verbosity

## Saved training outputs

- `model.keras` — best-epoch weights (early stopping)
- `logs/training_history.csv` — per-epoch loss and MAE
- `training_metadata.json` — best epoch, sequence counts, hyperparameters

## Random seeds

`RANDOM_SEED = 42` for NumPy, Python random, and TensorFlow.
