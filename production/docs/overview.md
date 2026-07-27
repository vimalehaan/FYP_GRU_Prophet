# Overview

## Objective

Train the final **production Hybrid model** using the frozen research baseline methodology on the complete high-quality dataset (~484 containers), while reserving 10% of containers as a completely unseen generalization holdout.

This is **not** a research experiment. Its purpose is to produce a deployable model and demonstrate inference on unseen containers.

## Model

Hybrid Prophet + GRU:

1. **Prophet** captures trend and daily seasonality per container.
2. **GRU** learns residual patterns from globally normalized Prophet residuals.
3. Final forecast = Prophet component + denormalized GRU residual.

Architecture matches the frozen Hybrid baseline exactly (96 input → 96 output, Adam, MSE, batch 64, early stopping patience 10).

## Data strategy

Two independent split layers:

1. **Container split (90/10):** Applied before any preprocessing. Unseen containers never participate in GRU training, Prophet fitting for training containers, or scaler fitting.
2. **Temporal split (80/20):** Applied per container chronologically — identical to frozen preprocessing.

## Outputs

- Trained Keras model and supporting artifacts in `production/hybrid/`
- Per-container evaluation on known and unseen containers
- Demonstration notebook with in-notebook visualizations
