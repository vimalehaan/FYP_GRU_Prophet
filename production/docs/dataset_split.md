# Dataset Split

## Container-level holdout

| Parameter | Value |
|-----------|-------|
| Source | `data/df_resampled.parquet` |
| Total containers | ~484 |
| Train fraction | 90% (~436) |
| Unseen fraction | 10% (~48) |
| Random seed | 42 |
| Persistence | `train_container_ids.json`, `unseen_container_ids.json` |

The split is created once and **never regenerated** unless `--force-recreate-split` is explicitly passed (debugging only).

## Temporal split (unchanged from research)

Per container, chronologically:

- **80%** → train period (Prophet fitting, scaler fitting, GRU sequence generation)
- **20%** → validation/holdout (evaluation only)

Minimum 200 resampled rows per container. Degenerate containers (constant CPU in train period) are excluded from training.

## Leakage prevention

Unseen containers are excluded from:

- GRU weight updates
- Global residual mean/std computation during training
- Prophet models fit on training containers
- MinMax scaler fitting for training containers

## Unseen container inference protocol

At evaluation time, each unseen container:

1. Uses only its historical 80% for Prophet and scaler fitting
2. Computes residuals on the historical portion
3. Applies the **trained production GRU** with **global** `res_mean` / `res_std`
4. Forecasts the first 96 steps of the remaining 20% (Day-1 horizon)

This uses only information available at deployment time — no leakage.
