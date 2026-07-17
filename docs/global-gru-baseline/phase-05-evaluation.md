# Phase 5 — Evaluation & Verification

[← Overview](README.md) · [Phase 4 Implementation](phase-04-implementation.md)

---

## 6. Phase 5 — Evaluation & Verification

### 6.1 Objective

Execute **multi-container Day 1 evaluation** on the trained Global GRU v1 model, implement and run the verification script suite, and produce thesis-ready evaluation artifacts — without modifying the frozen Hybrid baseline or retraining the Global GRU model.

**Model authority:** `models/global_gru.keras` (from Phase 4)  
**Specification authority:** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)  
**Hybrid control:** `experiments/baseline_reference_2026-07-14/`

### 6.2 Locked Evaluation Inputs

| Item | Value |
|------|-------|
| Evaluation period | Temporal validation split — `data/val_df.parquet` |
| Train context | `data/train_df.parquet` (last 96 steps as GRU input) |
| Cohort | 100 selected containers → **99 evaluable** (`c_14674` skipped) |
| Primary horizon | Day 1 — **96 steps** (24 h) |
| Metrics | MAE, RMSE, MAPE (per container → aggregate mean/std) |
| Metric unit | **Real CPU %** after per-container inverse MinMax |
| MAPE epsilon | 0.01 percentage points (same as Hybrid) |
| Demo container | `c_11461` (same as Hybrid) |
| Inference | `utils/global_inference.py` |
| Evaluation orchestration | `utils/global_evaluation.py` (thin — delegates to shared metrics) |

### 6.3 Hybrid Control Metrics (Reference Only)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 1.745882 | 2.486804 |
| RMSE | 2.387813 | 3.219141 |
| MAPE | 111.940318 | 615.299100 |

**Source:** `experiments/baseline_reference_2026-07-14/evaluation/evaluation_summary.csv`

### 6.4 Phase 5 Constraints

| Constraint | Rule |
|------------|------|
| Frozen Hybrid | Do **not** modify or re-evaluate Hybrid baseline in this phase |
| No retraining | Load pre-trained Global GRU weights only |
| No specification changes | Do **not** change locked Phase 0.5 configuration |
| Shared protocol | Evaluation must match Hybrid Day 1 protocol (cohort, period, horizon, metrics) |
| Reuse metrics | Import `compute_day1_metrics`, `compute_day1_mape`, `_evaluable_container_ids` from `hybrid_evaluation.py` (read-only) — do not duplicate |
| Peak-Aware | **Out of scope** — no peak-subset metrics in Global GRU baseline |
| Outputs | Save under `experiments/global_gru_baseline_*/evaluation/` |
| Phase 6 scope | Baseline freeze and methodology comparison deferred to Phase 6 |

---

## 6.5 Evaluation Pipeline Design

### Per-Container Inference

```
For each container_id in selected_containers:
  1. Pre-filter via _evaluable_container_ids()  ← shared import
  2. run_global_inference(container_id, global_train, global_val, model, scalers)
  3. compute_day1_metrics(actual_day1_real, day1_pred_real)  ← shared import
  4. compute_day1_mape(actual_day1_real, day1_pred_real, epsilon=0.01)  ← shared import
  5. Append row to evaluation_df
```

### Aggregate Reporting

```python
evaluation_summary = summarize_evaluation_metrics(evaluation_df)
# Returns mean, std, min, max for day1_mae, day1_rmse, day1_mape
```

### Expected Cohort Results

| Outcome | Count |
|---------|-------|
| Selected containers | 100 |
| Evaluated | 99 |
| Skipped | 1 (`c_14674`) |
| Failures | 0 (target) |

---

## 6.6 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Implement thin `utils/global_evaluation.py` (orchestration only) | **Planned** |
| 2 | Run multi-container Day 1 evaluation | **Planned** |
| 3 | Implement verification scripts (5 scripts) | **Planned** |
| 4 | Run all verification scripts | **Planned** |
| 5 | Generate publication-quality plots | **Planned** |
| 6 | Save evaluation artifacts | **Planned** |
| 7 | Lock Phase 5 evaluation record | **Planned** |

---

### Task 1 — Implement `utils/global_evaluation.py`

**Status:** Planned

#### Design — Reuse, Do Not Duplicate

| Function | Approach |
|----------|----------|
| `compute_day1_metrics()` | **Import** from `utils/hybrid_evaluation.py` (read-only) |
| `compute_day1_mape()` | **Import** from `utils/hybrid_evaluation.py` (read-only) |
| `_evaluable_container_ids()` | **Import** from `utils/hybrid_evaluation.py` (read-only) |
| `summarize_evaluation_metrics()` | **Import** from `utils/hybrid_evaluation.py` (read-only) |
| `evaluate_selected_containers()` | **New** — Global GRU orchestration loop calling `run_global_inference()` |

This module contains **only** Global GRU–specific evaluation orchestration. Metric logic is shared, not copied.

---

### Task 2 — Run Multi-Container Evaluation

**Status:** Planned

#### Command / Notebook

```python
RUN_TRAINING = False
global_gru_model, metadata = load_global_artifacts(...)
evaluation_df, inference_results, failures, skipped = evaluate_selected_containers(...)
evaluation_summary = summarize_evaluation_metrics(evaluation_df)
```

#### Outputs

| File | Content |
|------|---------|
| `evaluation/evaluation_df.csv` | Per-container Day 1 metrics |
| `evaluation/evaluation_summary.csv` | Aggregate mean/std/min/max |
| `evaluation/inference_cache.pkl` | Optional full inference results dict |

---

### Task 3 — Verification Scripts

**Status:** Planned

| Script | Checks |
|--------|--------|
| `scripts/verify_global_gru_data_split.py` | Training sequences sourced from `global_train` only; zero val-period targets in train set |
| `scripts/verify_global_gru_inference.py` | Single-container (`c_11461`) inference matches manual MAE/RMSE |
| `scripts/verify_global_gru_evaluation.py` | 99 evaluated, 1 skipped; schema correct; MAPE present |
| `scripts/verify_global_gru_train_eval_split.py` | Loaded model predictions identical to in-memory model |
| `scripts/verify_global_gru_methodology_parity.py` | Same cohort, val period, metric formulas, MAPE ε as Hybrid (methodology comparison readiness) |

All scripts must pass before Phase 6 freeze.

---

### Task 4 — Run Verification Suite

**Status:** Planned

#### Expected Output

```
=== Global GRU Evaluation Verification ===
Selected containers : 100
Evaluable containers: 99
Evaluated containers: 99
Skipped containers  : 1 (c_14674)
Failures            : 0
All checks          : PASS
```

Save results to `experiments/global_gru_baseline_*/verification/verification_notes.md`.

---

### Task 5 — Publication-Quality Plots

**Status:** Planned

Per `.cursor/skills/visualization/SKILL.md`:

| Plot | Requirements |
|------|--------------|
| Actual vs Predicted (demo) | Title, axis labels, legend, train/val boundary, validation shading |
| Actual vs Predicted (samples) | 3–5 containers |
| Error distribution | Histogram of per-container MAE |
| Format | PNG + PDF saved to `plots/` |

**Demo container:** `c_11461` (identical to Hybrid for visual comparison across methodologies).

---

### Task 6 — Save Evaluation Artifacts

**Status:** Planned

```
experiments/global_gru_baseline_YYYY-MM-DD_HHMMSS/
├── evaluation/
│   ├── evaluation_df.csv
│   ├── evaluation_summary.csv
│   ├── evaluation_metadata.json
│   └── inference_cache.pkl
├── verification/
│   └── verification_notes.md
└── plots/
    ├── actual_vs_predicted/
    │   ├── c_11461.png
    │   └── c_11461.pdf
    └── error_distribution.png
```

---

## 6.7 Pre-Freeze Verification Checklist

```
Data integrity
[ ] Sequences built from global_train only
[ ] No val-period rows in training sequence targets
[ ] Same selected_containers.npy as Hybrid
[ ] Same scalers.pkl for inverse transform

Training integrity
[ ] RUN_TRAINING=False skips all fit() calls
[ ] Random seed 42 logged in metadata
[ ] Early-stop val split on train sequences only
[ ] shuffle=False during fit()
[ ] No hyperparameter tuning performed

Evaluation integrity
[ ] Per-container: last 96 train → first 96 val
[ ] Metrics in real CPU % (not scaled)
[ ] MAPE uses epsilon=0.01 (shared function)
[ ] 99 containers evaluated, c_14674 skipped
[ ] evaluation_df Day 1 only

Reproducibility
[ ] Full Phase 0.5 specification in metadata JSON
[ ] Timestamped experiment directory (no overwrite)
[ ] Plots saved PNG + PDF
[ ] All 5 verification scripts PASS

Methodology comparison readiness
[ ] Hybrid control unchanged in baseline_reference_2026-07-14/
[ ] Global GRU metrics in same table format as Hybrid
[ ] No Peak-Aware code paths active
[ ] Metric functions reused, not duplicated
```

---

## 6.8 Strengths and Weaknesses Template

Document after first complete evaluation run (required per experiments skill):

**Global GRU forecasting methodology — Strengths:**

- (To be completed after Phase 5 run)

**Global GRU forecasting methodology — Weaknesses:**

- (To be completed after Phase 5 run)

---

## 6.9 Exit Criteria (Before Phase 6)

- [ ] Thin `utils/global_evaluation.py` implemented (shared metrics imported, not duplicated)
- [ ] Multi-container evaluation complete (99 containers)
- [ ] `evaluation_df.csv` and `evaluation_summary.csv` saved
- [ ] All 5 verification scripts PASS
- [ ] Plots exported as PNG + PDF
- [ ] Strengths/weaknesses documented
- [ ] Frozen Hybrid files unmodified
- [ ] Phase 6 baseline freeze **not** started until all checks pass

---

## 6.10 Conclusion

Phase 5 is **planned**. It delivers multi-container Day 1 metrics and a verification suite under the locked Phase 0.5 specification, reusing shared metric utilities. Results become the input for Phase 6 baseline freeze and forecasting methodology comparison.

**Next phase (after Phase 5 complete):** [Phase 6 — Baseline Freeze & Methodology Comparison](phase-06-baseline-freeze-and-comparison.md)
