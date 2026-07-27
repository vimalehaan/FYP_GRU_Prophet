# Reproducibility

## Fixed seeds

| Component | Seed |
|-----------|------|
| Container split | 42 |
| GRU training | 42 |

## Immutable artifacts

Once created, these files must not change between evaluation runs:

- `train_container_ids.json`
- `unseen_container_ids.json`
- `container_split_metadata.json`

## Version pinning

Record in deployment logs:

- TensorFlow version
- Prophet version
- Training completion timestamp (`training_metadata.json`)

## Reproducing results

```bash
# Full pipeline from scratch (same split if JSON exists)
tf_metal_env/bin/python production/scripts/train_production_hybrid.py

# Refresh notebook only
tf_metal_env/bin/python production/scripts/build_production_notebook.py
jupyter notebook production/notebooks/production_hybrid_model.ipynb
```

## Isolation guarantee

Production code lives in `production/utils/` and writes only to `production/hybrid/`. No research experiment directories are modified.
