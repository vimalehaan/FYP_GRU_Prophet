# Candidate Retraining (Layer 6)

**Module:** `candidate_retraining.py`

See **[retraining_methodology.md](retraining_methodology.md)** for the full thesis-ready specification.

## Principles

1. **Offline only** — batch retrain; no online GRU weight updates.
2. **Never overwrite production** — candidates are `hybrid_v2`, `hybrid_v3`, … under the experiment directory.
3. **Reuse frozen training logic** — wraps `train_production_hybrid` without modifying Hybrid architecture.
4. **Original train + accumulated val** — full `train_df` per container plus validation-period rows strictly before the drift trigger.

## Scaler policy

Frozen `cpu_scaled` from preprocessing parquet — production scalers fit on original train only; **no refit** during candidate retraining.

## Leakage rules

- Accumulated val rows: unified index `< T` (trigger origin)
- Evaluation: origins strictly `> T` (+ warmup gap)

## Artifacts per version

```
experiments/.../versions/hybrid_vN/
  model.keras
  residual_stats.pkl
  metadata.json
```

## Configuration

Set `skip_candidate_retraining=True` (or `--skip-candidate-retraining`) to run monitoring/drift/decisions without training.

## Validation history

- Initial validation: [retraining_pipeline_validation.md](retraining_pipeline_validation.md)
- OLD vs NEW pilot: [retraining_pilot_comparison.md](retraining_pilot_comparison.md)
