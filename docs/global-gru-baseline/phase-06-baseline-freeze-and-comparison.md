# Phase 6 — Baseline Freeze & Methodology Comparison

[← Overview](README.md) · [Phase 5 Evaluation](phase-05-evaluation.md)

---

## 7. Phase 6 — Baseline Freeze & Methodology Comparison

### 7.1 Objective

Freeze the verified Global GRU v1 model as an **immutable baseline reference**, then produce a **defensible forecasting methodology comparison** (Hybrid Prophet + GRU vs Global GRU) suitable for thesis reporting — without retraining either model or modifying frozen artifacts.

Phase 6 converts verified Global GRU measurements into a permanent control for future Peak-Aware Global GRU research and completes Research Objective 2 (*compare Hybrid Prophet + GRU against Global GRU under fair conditions*).

**Global GRU result authority:** `experiments/global_gru_baseline_*/evaluation/` (from Phase 5)  
**Hybrid control authority:** `experiments/baseline_reference_2026-07-14/`  
**Specification authority:** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)

### 7.2 Locked Inputs (From Phase 5 — Do Not Recompute)

| Item | Value |
|------|-------|
| Global GRU model | `global_gru_v1` — trained in Phase 4 |
| Cohort | 99 evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — 96 validation timesteps per container |
| Hybrid control | Frozen baseline Hybrid Prophet + GRU |
| Peak-Aware | **Out of scope** for this comparison |
| Intentional difference | **Forecasting methodology only** |

### 7.3 Comparison Philosophy

This phase reports a **forecasting methodology comparison** under an identical experimental protocol:

| Fixed (identical) | Differs by design |
|-------------------|-------------------|
| Preprocessing pipeline | Forecasting methodology |
| Temporal data split | |
| Evaluation procedure | |
| Metrics and reporting | |
| Container cohort | |

The research question is which **forecasting approach** performs better — not which neural network configuration is optimal.

### 7.4 Phase 6 Constraints

| Constraint | Rule |
|------------|------|
| No retraining | Do not retrain Global GRU or Hybrid |
| No methodology redesign | Do not change evaluation protocol |
| No baseline changes | Frozen references remain read-only after freeze |
| Findings vs explanations | Label claims explicitly (see §7.5) |
| Peak-Aware Global GRU | **Deferred** — separate study track after this freeze |
| Unseen container eval | **Deferred** — post-freeze follow-on experiment |

---

## 7.5 Writing Standard — Findings vs Possible Explanations

| Label | Definition | Example phrasing |
|-------|------------|------------------|
| **Experimental finding** | Directly supported by locked metrics | "Global GRU achieved mean Day 1 MAE of X on 99 containers under the shared protocol." |
| **Possible explanation** | Hypothesis not independently measured | "One possible explanation is that Hybrid's Prophet component captures daily seasonality that the Global GRU methodology must learn implicitly." |

**Do not** present methodology hypotheses as proven without supporting ablation evidence.

| Avoid (overstated) | Prefer (careful) |
|--------------------|------------------|
| "The smaller GRU causes worse performance." | "Under the locked Global GRU methodology, mean Day 1 MAE was X compared to Hybrid's Y." |
| "Global GRU is inferior." | "The Global GRU forecasting methodology did / did not outperform Hybrid on overall Day 1 accuracy under identical evaluation conditions." |

---

## 7.6 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Copy verified artifacts to frozen reference folder | **Planned** |
| 2 | Write `FREEZE_STATEMENT.md` and `baseline_metadata.json` | **Planned** |
| 3 | Build forecasting methodology comparison table | **Planned** |
| 4 | Paired per-container delta analysis | **Planned** |
| 5 | Generate cross-methodology comparison plots | **Planned** |
| 6 | Write strengths/weaknesses for both methodologies | **Planned** |
| 7 | Lock Phase 6 record and update project docs | **Planned** |

---

### Task 1 — Freeze Global GRU Baseline

**Status:** Planned

#### Reference Folder

```
experiments/global_gru_reference_YYYY-MM-DD/
├── FREEZE_STATEMENT.md
├── models/
│   ├── global_gru.keras
│   └── global_gru_metadata.json
├── evaluation/
│   ├── evaluation_df.csv
│   └── evaluation_summary.csv
├── config/
│   └── baseline_metadata.json
└── verification/
    └── verification_notes.md
```

#### Freeze Gate

All Phase 5 verification scripts must PASS before copying to reference folder.

#### Policy (Mirror Peak-Aware Baseline Freeze)

- Peak-Aware Global GRU experiments must save only under `experiments/peak_aware_global_*`
- Never overwrite `experiments/global_gru_reference_*/`
- Never overwrite `experiments/baseline_reference_2026-07-14/`

---

### Task 2 — Baseline Metadata Record

**Status:** Planned

`baseline_metadata.json` must include the complete [Phase 0.5 specification](phase-03-5-baseline-specification.md) plus:

| Section | Content |
|---------|---------|
| `baseline_version` | `global_gru_v1` |
| `frozen_date` | ISO date |
| Full locked specification | All Phase 0.5 fields |
| Evaluation | Aggregate metrics, evaluated/skipped containers |
| Verification | Script results (all PASS) |
| Hybrid control reference | Path + control metrics for side-by-side |
| Environment | Python, TensorFlow, platform |

---

### Task 3 — Forecasting Methodology Comparison Table

**Status:** Planned

#### Primary Results (96-step horizon, real CPU %, identical protocol)

| Forecasting methodology | MAE (mean ± std) | RMSE (mean ± std) | MAPE (mean ± std) | Containers |
|--------------------------|------------------|-------------------|-------------------|------------|
| Hybrid Prophet + GRU | 1.746 ± 2.487 | 2.388 ± 3.219 | 111.94 ± 615.30 | 99 |
| Global GRU | TBD | TBD | TBD | 99 |

#### Shared Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Alibaba Cluster Trace |
| Target | `cpu_util_percent` |
| Resampling | 15 min |
| Input window | 96 steps (24 h) |
| Forecast horizon | 96 steps (24 h) |
| Evaluation period | Preprocessing validation split |
| Container cohort | 100 selected / 99 evaluable |
| Random seed | 42 |
| Date | YYYY-MM-DD |

Save to:

```
experiments/hybrid_vs_global_YYYY-MM-DD/
├── comparison_table.csv
├── experiment_config.json
└── phase6_comparison.json
```

---

### Task 4 — Paired Per-Container Delta Analysis

**Status:** Planned

For each of 99 evaluable containers:

```
delta_mae  = global_gru_mae  - hybrid_mae
delta_rmse = global_gru_rmse - hybrid_rmse
delta_mape = global_gru_mape - hybrid_mape
```

#### Summary Statistics

| Statistic | Description |
|-----------|-------------|
| Mean delta | Average improvement/degradation |
| Improved count | Containers where Global GRU MAE < Hybrid MAE |
| Worsened count | Containers where Global GRU MAE > Hybrid MAE |
| Std of delta | Variability across containers |

Output: `experiments/hybrid_vs_global_*/per_container_delta.csv`

---

### Task 5 — Cross-Methodology Comparison Plots

**Status:** Planned

| Plot | Description |
|------|-------------|
| Side-by-side Actual vs Predicted | Same container (`c_11461`), same axes, both methodologies |
| MAE delta histogram | Distribution of per-container Hybrid − Global GRU MAE |
| Scatter: Hybrid MAE vs Global GRU MAE | Per-container paired comparison |
| Error distribution overlay | Both methodologies on same histogram |

All plots: PNG + PDF, publication quality per visualization skill.

---

### Task 6 — Strengths and Weaknesses (Both Methodologies)

**Status:** Planned

#### Hybrid Prophet + GRU Methodology

**Strengths:**

- Explicit trend and daily seasonality via Prophet
- Interpretable decomposition (Prophet vs GRU residual)
- Strong performance on containers with clear seasonal patterns

**Weaknesses:**

- Per-container Prophet refit at inference (computational cost)
- Two-stage pipeline complexity
- Global residual normalization may under-represent per-container scale

#### Global GRU Methodology

**Strengths:**

- Single model serves all containers (scalability, lower inference latency)
- No Prophet dependency at inference
- Learns shared temporal patterns across workloads

**Weaknesses:**

- No explicit seasonality decomposition
- Container context limited to static train stats (`cpu_mean`, `cpu_std`)
- May underperform where Prophet captures structure the Global GRU methodology cannot learn implicitly

---

## 7.7 Research Questions (Module 1)

| Question | Answered by |
|----------|-------------|
| How does the Global GRU forecasting methodology compare to Hybrid on overall Day 1 accuracy? | Task 3 comparison table |
| Which methodology performs better on which containers? | Task 4 delta analysis |
| What are the trade-offs for deployment scalability? | Task 6 strengths/weaknesses |

**Not answered in Phase 6 (deferred):**

- Unseen container generalization
- Peak-period accuracy (Peak-Aware tracks)
- Model-structure or hyperparameter ablations

---

## 7.8 Future Work (After Phase 6 — Not Part of Primary Workflow)

| Item | Note |
|------|------|
| **Peak-Aware Global GRU** | Requires frozen `global_gru_reference_*` as control; only loss function (or approved mechanism) may change |
| **Unseen container holdout** | Stratified split; new experiment directory |
| **Model-structure / feature ablations** | Separate experiments — not reinterpretations of Phase 6 |
| **Hyperparameter optimisation** | Separate experiment after baseline freeze |
| **Day 2 recursive forecast (Global GRU)** | Supplementary; not primary research horizon |

---

## 7.9 Exit Criteria (Phase 6 Complete)

- [ ] Global GRU reference copied to `experiments/global_gru_reference_*/`
- [ ] `FREEZE_STATEMENT.md` written with control metrics
- [ ] `baseline_metadata.json` complete with full Phase 0.5 provenance
- [ ] Forecasting methodology comparison table saved
- [ ] Per-container delta analysis saved
- [ ] Comparison plots exported (PNG + PDF)
- [ ] Strengths/weaknesses documented for both methodologies
- [ ] `docs/README.md` updated with Global GRU baseline status
- [ ] `docs/global-gru-baseline/README.md` index updated to Complete
- [ ] Frozen Hybrid reference unchanged
- [ ] Peak-Aware Global GRU **not** started

---

## 7.10 Handoff to Future Research

After Phase 6, the repository will contain two frozen forecasting methodology baselines:

| Forecasting methodology | Frozen reference |
|------------------------|------------------|
| Hybrid Prophet + GRU | `experiments/baseline_reference_2026-07-14/` |
| Global GRU | `experiments/global_gru_reference_YYYY-MM-DD/` |

Future Peak-Aware Global GRU research must:

1. Treat `global_gru_reference_*` as the permanent control
2. Implement only in parallel modules (e.g. `utils/global_training_peak_aware.py`)
3. Change **only** the training loss function (or another explicitly approved peak-aware mechanism)
4. Keep every other Phase 0.5 component identical
5. Never overwrite frozen reference folders

---

## 7.11 Conclusion

Phase 6 is **planned**. It completes the Global GRU baseline research track by freezing verified artifacts and producing a thesis-ready **forecasting methodology comparison** under a fair, shared evaluation protocol.

**Prerequisite:** Phases 4 and 5 complete; Phase 0.5 specification honoured; all verification scripts passing.

---

## Appendix A — Comparison Results Template

### Primary Results (to be filled after Phase 5)

| Forecasting methodology | MAE | RMSE | MAPE | N |
|------------------------|-----|------|------|---|
| Hybrid Prophet + GRU | 1.746 | 2.388 | 111.94 | 99 |
| Global GRU | — | — | — | 99 |
| **Delta (Global − Hybrid)** | — | — | — | — |

### Per-Container Delta Summary (to be filled)

| Statistic | MAE | RMSE |
|-----------|-----|------|
| Mean delta | — | — |
| Improved / Worsened | — / — | — / — |
| Std of delta | — | — |

---

## Appendix B — Related Documentation

| Document | Purpose |
|----------|---------|
| [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md) | Locked research contract |
| [Hybrid Prophet + GRU](../hybrid-prophet-gru.md) | Hybrid methodology specification |
| [Peak-Aware Hybrid](../peak-aware-hybrid/README.md) | Separate Hybrid enhancement study |
| [Recommended Evaluation Protocol](../recommended-evaluation-protocol.md) | Target methodology (implemented in Phases 4–5) |
| [Implementation Roadmap](../implementation-roadmap.md) | Project-wide action plan |
