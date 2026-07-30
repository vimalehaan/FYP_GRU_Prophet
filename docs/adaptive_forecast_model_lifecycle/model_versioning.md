# Model Versioning

**Module:** `model_versioning.py`

## Version lineage

| Version | Role |
|---------|------|
| `hybrid_v1` | Incumbent copied from frozen baseline reference |
| `hybrid_v2+` | Offline candidates triggered by AFMLF |

## Record schema (`ModelVersionRecord`)

Each version stores:

- `version_id`, `created_at`
- `configuration` — AFMLF config snapshot
- `training_dataset_summary` — row counts, trigger origin, training metadata
- `evaluation_summary` — incumbent vs candidate MAE comparison
- `deployment_recommendation` — pending → deploy / keep / investigate
- `artifact_dir`, `parent_version`, `notes`

## Registry

`versions/registry.json` lists all versions in chronological order. Individual `metadata.json` files mirror each record for artifact portability.

## Production boundary

Version directories live under the experiment folder. Promoting a candidate to `production/hybrid/` is an explicit human step outside AFMLF.
