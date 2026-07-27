# Limitations

## Generalization

- Unseen container performance may degrade vs known containers due to workload heterogeneity.
- The 10% holdout (~48 containers) provides a point estimate, not a confidence interval.

## Forecast horizon

- Primary evaluation is **96 steps (24 h)**. Recursive Day-2 forecasting is not the production default.
- Validation periods longer than 96 steps are truncated for Day-1 metrics.

## Prophet assumptions

- Daily seasonality may not capture all workload patterns.
- Per-container Prophet refitting at inference adds latency.

## Data quality

- Containers with degenerate train CPU are excluded from training.
- Minimum 200 resampled rows required per container.

## Residual normalization

- Global train-period mean/std may not perfectly match unseen container residual distributions.

## Compute

- Full training on ~436 containers with per-container Prophet fitting is CPU/GPU intensive.
