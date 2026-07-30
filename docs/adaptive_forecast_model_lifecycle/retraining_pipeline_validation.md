# Candidate Retraining Pipeline — Validation Report

**Date:** 2026-07-30 (initial validation)  
**Updated:** 2026-07-30 (methodology correction applied)  
**Scope:** `utils/adaptive_lifecycle/candidate_retraining.py`, `candidate_evaluation.py`  
**Pilot (NEW):** `experiments/adaptive_forecast_model_lifecycle_2026-07-30_070637`

---

## Status

| Phase | Result |
|-------|--------|
| Initial validation | **B** — prefix methodology confounded incumbent vs candidate |
| Corrective implementation | **Complete** — original train + accumulated val before trigger |
| Pilot re-run | **Passed** — see [retraining_pilot_comparison.md](retraining_pilot_comparison.md) |
| Full 99-container evaluation | **Approved to proceed** |

---

## Initial problem (July 2026 validation)

The first implementation built candidate training data as:

```
concat(train_df, val_df) → iloc[:cutoff]   # chronological prefix from t=0
```

At trigger **T = 200**, each container received **200 rows** while incumbent `hybrid_v1` was trained on **~614 rows** (full `train_df`). Candidate vs incumbent comparisons were confounded by **training dataset size**, not drift adaptation alone.

Additional issues:

- MinMax scalers were refit on the prefix (inconsistent with frozen production preprocessing)
- Prefix truncation discarded most of the original offline corpus at early triggers

---

## Revised methodology (current)

Full thesis-ready description: **[retraining_methodology.md](retraining_methodology.md)**

### Dataset construction

```
Retraining Dataset =
    FULL train_df[c]                                    # Original Training Window
  + val_df[c].iloc[:max(0, T − len(train_df[c]))]      # Accumulated Historical Window
```

| Window | Definition |
|--------|------------|
| **Original Training Window** | Complete per-container `train_df` partition (~80% temporal split) — always included |
| **Accumulated Historical Window** | Post-training `val_df` rows with unified index in `[n_train, T)` |
| **Retraining Dataset** | Union of the two; passed to `train_production_hybrid()` |

### Temporal diagram (revised)

```
Index:  |-------- Original Train (n_train) --------|-- Accumulated Val --| T | future
        [0 ........................... n_train-1]  [n_train .... T-1]   ^   eval only

T = 200, n_train = 614:  accumulated val = 0  →  614 rows (matches incumbent train volume)
T = 700, n_train = 614:  accumulated val = 86 →  700 rows (614 + 86)
```

### Scaler policy (revised)

**Reuse frozen `cpu_scaled` from `train_df` / `val_df` parquet** — no refit. Matches production preprocessing (scaler fit on original train only, applied to val).

### Temporal integrity assertions

**Training** (`candidate_retraining.py`):

- `assert_retraining_temporal_integrity()` — validates accumulated val count and `n_train + n_accumulated ≤ T` when val rows included

**Evaluation** (`candidate_evaluation.py`):

- `assert (eval_origins["origin_index"] > trigger_origin_index).all()`
- `assert (eval_origins["origin_index"] >= T + warmup × stride).all()`

---

## Why the previous approach was insufficient

1. **Unequal training volume** — prefix at T=200 used 32% of incumbent data  
2. **Wrong production metaphor** — real retraining retains original batch + new telemetry, not a short prefix from timeline start  
3. **Insufficient GRU sequences** — 24 sequences (OLD pilot) vs 1,265 (NEW pilot)  
4. **Scaler inconsistency** — refit on prefix vs frozen production scaling  

---

## Why the new approach is scientifically correct

1. **Preserves original offline corpus** — candidate and incumbent share the same baseline training set  
2. **Appends only legitimate new history** — val rows strictly before trigger T  
3. **Excludes future data** — accumulated val capped at `T − n_train`; eval on origins > T  
4. **Matches production retraining workflow** — original batch + accumulated observations → offline retrain → future evaluation  
5. **Cohort-global retraining retained** — global GRU trained on all cohort containers (unchanged design)  

---

## Pilot verification (NEW methodology)

Trigger T = 200, 3 containers:

| Container | Original train | Accumulated val | Total |
|-----------|----------------|-----------------|-------|
| c_10032 | 614 | 0 | 614 |
| c_10034 | 613 | 0 | 613 |
| c_10089 | 614 | 0 | 614 |

- Total rows: **1,841** (was 600 under OLD)  
- GRU sequences: **1,265** (was 24)  
- Eval at origin 296: incumbent MAE 1.117, candidate MAE 1.116, **keep_current_model**  

Comparison: [retraining_pilot_comparison.md](retraining_pilot_comparison.md)

---

## Final verdict (updated)

### **A — Implementation now matches the intended research design. Safe to run the full 99-container AFMLF evaluation.**

Only `candidate_retraining.py` dataset construction, related metadata in `pipeline.py`, and evaluation temporal assertions were revised. No other AFMLF layers were modified.

---

## Appendix — Key code references (revised)

**Dataset assembly** (`candidate_retraining.py`):

```python
train_part = train_df[train_df["container_id"] == cid].sort_values("time_stamp")
n_val_take = accumulated_val_row_count(trigger, len(train_part), len(val_part))
part = pd.concat([train_part, val_part.iloc[:n_val_take]], ignore_index=True)
# cpu_scaled from frozen parquet — no scaler refit
```

**Evaluation filter** (`candidate_evaluation.py`):

```python
eval_origins = origin_df[origin_df["origin_index"] >= T + warmup * stride]
assert (eval_origins["origin_index"] > T).all()
```

**Incumbent training (unchanged reference):**

```python
train_production_hybrid(train_df)  # production/scripts/train_production_hybrid.py
```
