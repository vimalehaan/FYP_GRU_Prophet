# Peak-Aware Global GRU — Research Documentation

**Project:** DracaSys Module 1 — Long-Term CPU Utilization Forecasting  
**Institution:** University of Moratuwa — Bachelor of Information Technology (Final Year Research Project)

**Thesis framing:** One Peak-Aware learning methodology applied to two forecasting architectures. This folder documents the **Global GRU application** of that shared methodology. It is **not** a second independent Peak-Aware research project.

---

## Methodology Tree

```
Peak-Aware Learning (thesis methodology)
│
├── Hybrid Prophet + GRU
│      ├── Control:  experiments/baseline_reference_2026-07-14/
│      └── Treatment: experiments/peak_aware_2026-07-14_164518/  [complete]
│
└── Global GRU
       ├── Control:  experiments/global_gru_reference_2026-07-17/  [frozen]
       └── Treatment: experiments/peak_aware_global_2026-07-17_164737/  [Stage 2 complete]
```

---

## Documentation Index

| Stage | Document | Status |
|-------|----------|--------|
| 1 | [Design Lock](01-design-lock.md) | **Complete** |
| 2 | [Implementation Plan](stage-02-implementation-plan.md) · [Implementation Record](02-implementation-record.md) | **Complete** |
| 3 | [Evaluation Plan](stage-03-evaluation-plan.md) · Evaluation Record | Plan ready · Record pending |
| 4 | Results & Discussion | Pending |

**Design lock artifact:** `experiments/peak_aware_global_design_2026-07-17/design_lock.json`  
**Primary training run:** `experiments/peak_aware_global_2026-07-17_164737/`  
**Stage 2 lock:** `experiments/peak_aware_global_2026-07-17_164737/phase2_implementation.json`

---

## Authority & Cross-References

| Resource | Role |
|----------|------|
| [Peak-Aware Hybrid](../peak-aware-hybrid/README.md) | **Methodology authority** — peak definition, λ, loss, evaluation protocol (Phases 1–6 complete) |
| [Global GRU Baseline](../global-gru-baseline/README.md) | Frozen control context — `global_gru_v1` specification |
| [Phase 0.5 Baseline Spec](../global-gru-baseline/phase-03-5-baseline-specification.md) | Future Peak-Aware single-variable contract |
| `experiments/global_gru_reference_2026-07-17/` | **Frozen Global control** for this study |
| `experiments/peak_aware_2026-07-14_164518/` | Locked Hybrid Peak-Aware treatment (methodology reference) |

---

## Study Stages

| Stage | Name | Status |
|-------|------|--------|
| 1 | Design Lock (including methodology inheritance) | **Complete** |
| 2 | Implementation + Fairness Verification | **Complete** |
| 3 | Evaluation + Verification | Plan ready — pending Task 1 |
| 4 | Results, Discussion & Architecture Sensitivity Synthesis | Pending |

---

## Primary vs Secondary Comparisons

| Comparison | Role |
|------------|------|
| Global GRU vs Peak-Aware Global GRU | **Primary** — answers Global-track research questions |
| Peak vs non-peak (within Global pair) | **Primary** — Tier 2 contribution metrics |
| Hybrid vs Peak-Aware Hybrid | **Already locked** — cite only |
| Hybrid vs Global (methodology) | **Already locked** — cite only |
| Architecture Sensitivity (Δ Hybrid vs Δ Global) | **Secondary synthesis** — Stage 4 §4.7 only |

---

## Fair Comparison Principle

The frozen Global GRU baseline (`experiments/global_gru_reference_2026-07-17/`) must not be modified. Peak-Aware Global work uses parallel modules and separate experiment outputs only. The only intended difference between control and treatment is the GRU training objective (timestep-weighted MSE with λ = 5).

**Mandatory sequence fairness rule (Stage 2 gate):** Peak-Aware Global must produce **identical** `X_all`, `y_all`, and sample ordering as the frozen baseline. Only `W_all` (the weight matrix) may differ. Fairness verification requires **10 checks PASS** (see [Stage 2 plan](stage-02-implementation-plan.md)).

---

## Frozen Files (Do Not Modify)

- `utils/global_config.py`
- `utils/global_training.py`
- `utils/global_inference.py`
- `utils/global_evaluation.py`
- `utils/global_artifacts.py`
- `utils/global_plotting.py`
- `utils/sequence_utils.py`
- `notebooks/global_gru_model.ipynb`
- `models/global_gru.keras`
- `models/global_gru_metadata.json`
- `experiments/global_gru_reference_2026-07-17/`

Shared peak utilities (`utils/peak_detection.py`) may be **extended** (e.g. `align_with_longterm_sequences()`) but existing Hybrid-proven functions must remain backward-compatible.

---

*Last updated: 2026-07-17 — Stage 2 complete; Stage 3 evaluation plan ready*
