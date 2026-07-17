# Global GRU Baseline Freeze Statement

**Baseline name:** Global GRU v1 (`global_gru_v1`)  
**Freeze date:** 2026-07-17  
**Status:** FROZEN — implementation and training complete; Phase 5 evaluation pending

---

## Experiment Identifier

**Source training run:** `global_gru_baseline_2026-07-17_074743`

This immutable reference folder captures the verified Phase 4 implementation and trained model. Cohort evaluation artifacts will be added under `evaluation/` after Phase 5 completes.

---

## Specification Authority

[Phase 0.5 — Global GRU Baseline Specification](../../docs/global-gru-baseline/phase-03-5-baseline-specification.md)

---

## Random Seed

**42** (Python, NumPy, TensorFlow)

---

## Architecture

```
GRU(128, return_sequences=True)
Dropout(0.2)
GRU(64)
Dense(64, ReLU)
Dense(96)
```

---

## Input Features

| Feature | Role |
|---------|------|
| `cpu_scaled` | Input feature and prediction target |
| `cpu_mean` | Static train-period container context |
| `cpu_std` | Static train-period container context |

---

## Forecast Horizon

- **Input window:** 96 timesteps (24 hours)
- **Forecast horizon:** 96 timesteps (Day 1 primary)

---

## Training Methodology

| Item | Value |
|------|-------|
| Sequence source | `global_train` only — never `concat(global_train, global_val)` |
| Loss | MSE |
| Optimizer | Adam |
| Training metric | MAE |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | `val_loss`, patience 3, `restore_best_weights=True` |
| Shuffle | `False` |
| Training val split | 80/20 index on pooled train sequences (early stopping only) |
| Epochs run | 9 |

---

## Production Artifact Locations

| Artifact | Path |
|----------|------|
| Model | `models/global_gru.keras` |
| Metadata | `models/global_gru_metadata.json` |
| Reference copy | `experiments/global_gru_reference_2026-07-17/models/` |

---

## Immutable Reference Location

```
experiments/global_gru_reference_2026-07-17/
├── config/           baseline_metadata.json
├── models/           global_gru.keras, global_gru_metadata.json
├── training/         training_metadata.json
├── evaluation/       (populated in Phase 5)
├── verification/     verification_notes.md
├── plots/            (populated in Phase 5)
└── FREEZE_STATEMENT.md
```

This folder is the **permanent Global GRU control reference**. Future experiments must not overwrite it.

---

## Frozen Components

The following must remain unchanged during Peak-Aware Global GRU and all subsequent research:

- `data/` preprocessing artifacts
- `models/global_gru.keras` and `models/global_gru_metadata.json` (production path)
- `experiments/global_gru_reference_2026-07-17/` (this reference)
- Phase 0.5 locked training configuration and model form
- `utils/global_*.py` modules as implemented in Phase 4
- `notebooks/global_gru_model.ipynb` baseline workflow (Sections 1–2)

Shared Hybrid baseline artifacts remain independently frozen under `experiments/baseline_reference_2026-07-14/`.

---

## Future Experiment Policy

All future experiments — including **Peak-Aware Global GRU** — must:

1. Compare against this frozen baseline as the control.
2. Implement changes only in **parallel modules** (e.g. `utils/global_training_peak_aware.py`).
3. Save outputs only under new `experiments/` directories.
4. **Never** retrain, overwrite, or modify this frozen baseline.

---

## Baseline Governance

> The Global GRU v1 baseline is now frozen. Any future modifications — including Peak-Aware Learning, architecture changes, preprocessing changes, feature engineering, or hyperparameter optimisation — must be implemented as separate experiments using this frozen baseline as the control. The baseline itself must remain unchanged to preserve reproducibility and ensure fair experimental comparison.

---

## Verification (Phase 4)

Pre-training and implementation verification completed 2026-07-17. See `verification/verification_notes.md`.

**Note:** Multi-container cohort metrics are **not** part of this freeze. Methodology comparison requires Phase 5 evaluation under the shared 99-container protocol.

---

## Next Phase

**Phase 5 — Evaluation & Verification** (populate `evaluation/` and `plots/` in this reference after cohort evaluation completes)
