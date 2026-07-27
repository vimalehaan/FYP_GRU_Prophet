# Future Work

## Model improvements

- Context-aware features (memory, co-location) if production monitoring shows systematic errors
- Alternative loss functions (evaluated in LFHE research — MSE retained for production parity)

## Operational

- Model versioning and A/B deployment infrastructure
- Automated drift detection on incoming container telemetry
- Latency-optimized Prophet caching for known containers

## Evaluation

- Extended multi-day recursive forecasting for operational planning
- Confidence intervals via conformal prediction or ensemble methods
- Cross-cluster validation on non-Alibaba traces

## Infrastructure

- REST inference API wrapping `production/utils/inference.py`
- Container registry integration for automatic scaler/prophet lifecycle management
