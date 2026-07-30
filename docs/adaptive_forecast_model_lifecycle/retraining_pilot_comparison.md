# Retraining Pilot Comparison — OLD vs NEW Methodology

**Date:** 2026-07-30  
**Pilot config:** 3 containers, 2 origins per container, trigger at T = 200  
**OLD experiment:** `experiments/adaptive_forecast_model_lifecycle_2026-07-30_060448`  
**NEW experiment:** `experiments/adaptive_forecast_model_lifecycle_2026-07-30_070637`

Monitoring, drift, diagnostics, decisions, and evaluation protocol were identical between runs. Only candidate **dataset construction** changed.

---

## Methodology summary

| | OLD (prefix) | NEW (original + accumulated) |
|--|--------------|------------------------------|
| Dataset rule | `concat(train,val).iloc[:cutoff]` | `train_df` full + `val_df.iloc[:max(0,T−n_train)]` |
| Scaler | Refit MinMax on prefix | Frozen `cpu_scaled` from parquet |
| At T = 200 | 200 rows / container | ~614 rows / container (full original train) |

---

## Training data comparison

| Metric | OLD | NEW |
|--------|-----|-----|
| Rows per container (c_10032) | 200 | **614** |
| Rows per container (c_10034) | 200 | **613** |
| Rows per container (c_10089) | 200 | **614** |
| **Total training rows** | **600** | **1,841** |
| Accumulated val rows (T=200) | 0 (prefix within train) | 0 (T < n_train) |
| GRU sequences total | **24** | **1,265** |
| GRU train sequences | 19 | 1,012 |
| GRU val sequences | 5 | 253 |
| Training duration | ~16 s (full pilot)* | **13.25 s** (training only) |
| Training epochs | 22 | 11 |

\*OLD run did not record isolated training duration; ~16 s is total pilot wall time.

---

## Candidate evaluation (identical eval window)

Both runs evaluate at **origin 296** (T + warmup 96), 3 container forecasts.

| Metric | OLD | NEW |
|--------|-----|-----|
| Incumbent cohort MAE | 1.117 | 1.117 |
| Candidate cohort MAE | 1.127 | **1.116** |
| Delta MAE (candidate − incumbent) | +0.010 | **−0.0014** |
| Deployment recommendation | keep_current_model | keep_current_model |
| Recommendation quality flag | `True` (correct defer) | `None` (within margin) |

**Interpretation:**

- OLD candidate was **worse** than incumbent (+0.010 MAE) — likely confounded by training on 200 rows vs incumbent's 614.
- NEW candidate is **marginally better** (−0.0014 MAE) but below `candidate_improvement_margin` (0.01) → still **keep_current_model**.
- Recommendation outcome unchanged (no deploy), but NEW comparison is scientifically meaningful.

---

## Recommendation quality impact

| Scenario | OLD | NEW |
|----------|-----|-----|
| Recommend retraining at T=200 | Yes | Yes |
| Candidate vs incumbent | Worse → correct defer | Slightly better → inconclusive (within margin) |
| Prevented bad deployment | Yes | Yes (margin too small to deploy) |
| Primary framework metric | Correct defer | Correct non-deploy (no false promote) |

The NEW methodology removes the training-size confound. The pilot no longer penalizes the candidate for having 3× less data than the incumbent.

---

## Verdict

The NEW methodology aligns candidate training volume with incumbent training at T = 200 and produces sufficient GRU sequences for stable retraining. Pilot results support proceeding to the official 99-container AFMLF evaluation.

See [retraining_methodology.md](retraining_methodology.md) for full thesis-ready methodology.
