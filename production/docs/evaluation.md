# Evaluation

## Primary horizon

**Day-1:** first 96 validation steps (24 hours at 15-minute resolution). Matches the frozen Hybrid baseline evaluation protocol.

## Metrics

Per container:

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error (real CPU %) |
| RMSE | Root mean squared error |
| MAPE | Mean absolute percentage error (ε=0.01 floor) |
| Pearson r | Linear correlation |
| R² | Coefficient of determination |

## Splits evaluated

1. **Known:** production training containers (temporal 20% holdout)
2. **Unseen:** completely held-out containers (local 80/20 split at inference)

## Outputs

- `evaluation_results.csv` — per-container metrics with `split` column
- `metrics/summary_by_split.csv` — aggregate statistics
- `predictions/known/*.pkl` — prediction arrays for notebook plotting
- `predictions/unseen/*.pkl` — unseen container predictions

## Comparison

The demonstration notebook compares known vs unseen distributions via boxplots, violin plots, and summary tables — all generated in-notebook.
