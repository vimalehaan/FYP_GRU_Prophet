# Global GRU v1 — Phase 4 Verification Notes

**Date:** 2026-07-17  
**Reference:** `experiments/global_gru_reference_2026-07-17/`  
**Experiment identifier:** `global_gru_baseline_2026-07-17_074743`

---

## Pre-Training Gate

| Script | Result | Notes |
|--------|--------|-------|
| `scripts/verify_global_gru_data_split.py` | **PASS** | 41,650 sequences from `global_train` only; 0 val-period target leaks |

### Data-split audit summary

| Metric | Value |
|--------|-------|
| Sequences (`global_train` only) | 41,650 |
| Val-period leaks (correct path) | 0 |
| Val-period leaks (legacy `concat`) | 15,120 (contrast only) |
| Notebook `pd.concat([global_train, global_val])` | Absent |

---

## Hybrid Regression (Shared Utilities)

| Script | Result | Notes |
|--------|--------|-------|
| `scripts/verify_hybrid_train_eval_split.py` | **PASS** | `sequence_utils.py` extension preserves Hybrid train/load equivalence |

`utils/hybrid_*.py`, frozen Hybrid models, and `experiments/baseline_reference_2026-07-14/` were not modified during Global GRU Phase 4 implementation.

---

## Implementation Verification

| Check | Result |
|-------|--------|
| Phase 0.5 architecture (`Dropout(0.2)`) in saved model | **PASS** |
| Metadata includes `baseline_spec`, `environment`, `random_seed=42` | **PASS** |
| Production model loads and runs inference | **PASS** |
| Training sequences from `global_train` only | **PASS** |

---

## Deferred to Phase 5

The following verification scripts are **not** run at Phase 4 freeze:

- `verify_global_gru_inference.py`
- `verify_global_gru_evaluation.py`
- `verify_global_gru_train_eval_split.py`
- `verify_global_gru_methodology_parity.py`

Cohort evaluation and methodology comparison require Phase 5 multi-container evaluation under the shared 99-container protocol.

---

## Conclusion

Phase 4 implementation verification is complete. The Global GRU v1 baseline is frozen for training configuration and artifact persistence. Phase 5 will populate `evaluation/` and extend verification coverage.
