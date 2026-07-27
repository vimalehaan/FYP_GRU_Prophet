# Deployment

## Artifact bundle

Deploy these files together from `production/hybrid/`:

```
models/model.keras
metadata/scalers.pkl
metadata/residual_stats.pkl
config/production_config.json
metadata/prophet_metadata.pkl
```

## Serving workflow

For a **known** container with a saved scaler:

1. Load recent 80% historical CPU data
2. Scale with saved MinMaxScaler
3. Fit Prophet on scaled history
4. Build 96-step residual input
5. Run GRU inference
6. Denormalize residuals and combine with Prophet forecast

For a **new/unseen** container:

1. Collect ≥200 resampled observations
2. Split 80/20 temporally
3. Fit new MinMaxScaler on 80% only
4. Follow steps 3–6 above using global residual stats

## Dependencies

- TensorFlow / Keras
- Prophet
- scikit-learn
- pandas, numpy

## Monitoring

Track per-container Day-1 MAE and MAPE in production. Compare against holdout benchmarks in `evaluation_results.csv`.

## Retraining policy

Retrain only via `production/scripts/train_production_hybrid.py`. Do not modify the container split without explicit approval — it invalidates generalization comparisons.
