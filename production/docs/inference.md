# Inference

## CLI (recommended)

```bash
tf_metal_env/bin/python production/scripts/predict_container.py c_1016 --split known
tf_metal_env/bin/python production/scripts/predict_container.py c_1181 --split unseen
```

## Loading artifacts (Python)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from production.utils.artifacts import (
    load_production_model,
    load_production_scalers,
    load_residual_stats,
)

model = load_production_model()
scalers = load_production_scalers()
res_mean, res_std, input_window, forecast_horizon = load_residual_stats()
```

## Known containers

Use saved per-container scalers and preprocessed train/val frames from `production/hybrid/cache/`:

```python
from production.utils.inference import run_known_container_inference

result = run_known_container_inference(
    container_id, train_df, val_df, model, scalers, res_mean, res_std
)
```

## Unseen containers

Fit Prophet and scaler locally on historical 80% only:

```python
from production.utils.inference import run_unseen_container_inference

result = run_unseen_container_inference(
    container_id, unseen_raw, model, res_mean, res_std
)
```

## Inference steps (per container)

1. Fit Prophet on train-period `cpu_scaled`
2. Forecast validation timestamps
3. Build residual input window (last 96 scaled residuals)
4. GRU predicts 96-step residual horizon
5. Combine Prophet + residual, inverse-transform to real CPU %

Global `res_mean` and `res_std` from training are always used for residual denormalization — even for unseen containers.
