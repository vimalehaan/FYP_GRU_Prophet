# Model artifacts (not in Git)

Binary files (`model.keras`, `*.pkl`) are excluded from the repository due to size. Copy the frozen production bundle locally before running the service:

```bash
# From repo root
mkdir -p forecast_service/artifacts/hybrid_v1
cp production/hybrid/models/model.keras forecast_service/artifacts/hybrid_v1/model.keras
cp production/hybrid/metadata/scalers.pkl forecast_service/artifacts/hybrid_v1/scalers.pkl
cp production/hybrid/metadata/residual_stats.pkl forecast_service/artifacts/hybrid_v1/residual_stats.pkl
cp production/hybrid/metadata/prophet_metadata.pkl forecast_service/artifacts/hybrid_v1/prophet_metadata.pkl
cp production/hybrid/config/production_config.json forecast_service/artifacts/hybrid_v1/production_config.json
cp production/hybrid/metadata/train_container_ids.json forecast_service/artifacts/hybrid_v1/train_container_ids.json
```

Or copy the entire bundle if already assembled under `forecast_service/artifacts/hybrid_v1/`.

See [docs/implementation/artifact_reference.md](../docs/implementation/artifact_reference.md).
