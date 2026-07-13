---
name: experiments
description: Enforces reproducible ML experiment practices — random seeds, hyperparameter logging, saved models/predictions/metrics/plots, timestamped outputs, and fair model comparison with MAE/RMSE/MAPE. Use when running experiments, training models, evaluating forecasts, comparing Hybrid Prophet+GRU vs Global GRU, or generating research results.
---

# Experiments

Every experiment should be reproducible.

Always

set random seeds

log hyperparameters

save trained models

save predictions

save evaluation metrics

save plots

Never overwrite previous experiment results.

Create timestamped outputs whenever appropriate.

When comparing models

use identical datasets

identical train/validation splits

identical evaluation metrics

Preferred evaluation metrics

MAE

RMSE

MAPE

Whenever possible also generate

Actual vs Predicted plots

Residual plots

Prediction error distributions

Forecast horizon comparisons

Explain the strengths and weaknesses of every experiment.
