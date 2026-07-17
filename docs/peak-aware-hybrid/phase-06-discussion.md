# Phase 6 — Discussion and Comparison

[← Overview](README.md) · [Phase 5 Evaluation](phase-05-evaluation.md) · [Phase 5 lock JSON](../experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json)

---

## 7. Phase 6 — Discussion and Comparison

### 7.1 Objective

Complete the Peak-Aware Hybrid research track by **interpreting the locked Phase 5 results** — not by introducing new research questions, redesigning methodology, or re-running the primary experiment.

Phase 6 converts verified measurements into a thesis-ready discussion: what was found, what can be concluded under the evaluated configuration, what may explain the outcome, and what limitations bound the claims.

**Result authority:** `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`  
**Evaluation artefacts:** `experiments/peak_aware_2026-07-14_164518/evaluation/`  
**Control reference:** `experiments/baseline_reference_2026-07-14/`

### 7.2 Locked Inputs (From Phase 5 — Do Not Recompute)

| Item | Value |
|------|-------|
| Cohort | 99 evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — 96 validation timesteps per container |
| Peak definition | P90 per-container, train-fitted — `peak/peak_thresholds.pkl` |
| Treatment | Timestep-weighted residual MSE, **λ = 5** |
| Control | Frozen baseline Hybrid Prophet + GRU |
| Experimental variable | GRU training objective only |
| Verification | Phase 4 fairness 8/8 PASS; Phase 5 evaluation 8/8 PASS |

**Locked pooled results (treatment − baseline):**

| Scope | MAE Δ | RMSE Δ | Interpretation |
|-------|-------|--------|----------------|
| Overall (Tier 1) | **−0.005** | **−0.003** | Negligible improvement |
| Peak subset (Tier 2) | **+0.024** | **+0.018** | No peak-subset improvement |
| Non-peak subset (Tier 3) | **−0.015** | **−0.040** | Slight non-peak improvement |

**Per-container:** mean ΔMAE −0.005; **45 improved / 54 worsened**; std(ΔMAE) ≈ 0.10.

### 7.3 Phase 6 Constraints

| Constraint | Rule |
|------------|------|
| No primary retraining | Do not modify or replace the locked λ = 5 experiment |
| No methodology redesign | Do not change peak definition, fairness contract, or eval protocol |
| No baseline changes | Frozen `utils/hybrid_*.py`, `models/`, `baseline_reference/` remain read-only |
| Findings vs explanations | Label claims explicitly (see §7.4) |
| Optional work | Error attribution and λ sensitivity are **not** required for Phase 6 completion |
| Scope boundary | Conclusions apply to **this configuration only** — not all peak-aware methods |

### 7.4 Writing Standard — Findings vs Possible Explanations

All Phase 6 prose must distinguish:

| Label | Definition | Example phrasing |
|-------|------------|------------------|
| **Experimental finding** | Directly supported by locked Phase 5 metrics | “Pooled peak-subset MAE increased by 0.024 under the evaluated configuration.” |
| **Possible explanation** | Interpretation or hypothesis not independently measured in Phase 5 | “One possible explanation is that peak-period error may originate partly outside the GRU residual path.” |

**Do not** present possible explanations as proven facts.

| Avoid (overstated) | Prefer (careful) |
|--------------------|------------------|
| “Prophet dominates peak error.” | “A possible explanation is that peak-period error may not be fully addressable by reweighting GRU residual learning alone.” |
| “Peak-aware learning does not work.” | “Under P90 + λ = 5 timestep-weighted residual training, peak-subset accuracy did not improve.” |
| “λ = 5 was too low.” | “λ = 5 was locked analytically; sensitivity to other λ values was not evaluated in the primary experiment.” |

### 7.5 Research Questions (Locked From Phase 3–5)

| Tier | Question |
|------|----------|
| **Tier 1** | Does peak-aware training change overall Day-1 accuracy vs baseline? |
| **Tier 2 (contribution)** | Does peak-aware training improve accuracy specifically on peak timesteps? |

Phase 6 must answer both using **only** locked Phase 5 results in §7.8 (Answer to the Research Question).

### 7.6 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Results synthesis | Pending |
| 2 | Answer to the research question | Pending |
| 3 | Mechanism discussion (possible explanations) | Pending |
| 4 | Stratified discussion | Pending |
| 5 | Limitations and threats to validity | Pending |
| 6 | Lock Phase 6 record | Pending |

**Not in primary workflow:** Error attribution (Prophet vs GRU) — see [Appendix A](#appendix-a--optional-exploratory-error-attribution).

---

### Task 1 — Results Synthesis

**Objective:** Present locked Phase 5 outcomes in structured, thesis-ready form.

**Inputs (read-only):**
- `phase5_evaluation.json`
- `evaluation/comparison_table.csv`
- `evaluation/per_container_delta.csv`
- `evaluation/peak_subset_summary.csv`
- `evaluation/plots/`

**Deliverables:**
- §7.7 Results Synthesis in this document
- Tables: overall, peak/non-peak, per-container delta summary

**Content rules:**
- Report absolute values and deltas (treatment − baseline)
- Classify effect sizes as negligible / mixed where appropriate
- **Findings only** — no causal claims in this section

**Exit criterion:** A reader can reproduce the numeric story without opening raw CSVs.

---

### Task 2 — Answer to the Research Question

**Objective:** State the research conclusion immediately following Results Synthesis (§7.8), using **only** locked Phase 5 evidence.

**Placement:** §7.8 follows §7.7 directly — this is the first interpretive subsection after numeric synthesis.

**Draft conclusion (to refine in implementation, scope-bound):**

> Under the evaluated configuration — per-container P90 peak definition, timestep-weighted GRU residual learning with λ = 5, and an otherwise identical Hybrid Prophet + GRU pipeline — the proposed Peak-Aware Hybrid model did **not** produce meaningful improvements in peak-subset Day-1 prediction accuracy. Pooled peak-subset MAE and RMSE were marginally **higher** than the frozen baseline (MAE +0.024; RMSE +0.018). Overall Day-1 performance remained **comparable** to the baseline, with negligible aggregate gains (MAE −0.005; RMSE −0.003) that are unlikely to be operationally significant. Per-container outcomes were mixed (45 containers improved, 54 worsened), indicating high heterogeneity relative to the small mean effect.
>
> These findings answer the contribution question for **this specific peak-aware training design** under the stated fairness protocol. They do **not** establish that peak-aware learning in general is ineffective, nor do they evaluate alternative peak definitions, weighting schemes, architectures, or inference-time mechanisms.

**Deliverables:**
- §7.8 Answer to the Research Question
- One paragraph + Tier 1 / Tier 2 bullet answers

**Exit criterion:** Conclusion is academically careful, scope-bound, and free of over-generalization.

---

### Task 3 — Mechanism Discussion (Possible Explanations)

**Objective:** Discuss **plausible reasons** for the locked findings without claiming unmeasured mechanisms as facts.

**Topics (all framed as possible explanations unless separately supported):**

1. **Train–select–evaluate objective alignment** — weighted training loss vs unweighted early stopping vs unweighted Day-1 evaluation
2. **Intervention scope** — only GRU residual learning was modified; Prophet fitting unchanged
3. **Peak label distribution** — train-time weighted peak fraction (~17%) vs evaluation peak fraction (~25.7%)
4. **Fixed λ = 5** — analytically locked in Phase 3; not empirically tuned on peak metrics
5. **Near-idle P90 artefact** — Phase 2 carry-forward may affect which timesteps receive high training weight

**Deliverables:**
- §7.9 Mechanism Discussion
- Each hypothesis explicitly tagged **Possible explanation**

**Exit criterion:** No mechanism stated as confirmed without Phase 5 (or appendix) evidence.

---

### Task 4 — Stratified Discussion

**Objective:** Describe heterogeneity in outcomes — avoid over-generalizing from pooled means alone.

**Inputs:**
- `evaluation/per_container_delta.csv`
- `evaluation/peak_subset_per_container_comparison.csv`
- `data/container_metadata.parquet` (workload tags from preprocessing)

**Questions (findings where supported by tables; otherwise possible patterns):**
- Which containers improved vs worsened?
- Are gains/losses associated with peak fraction or workload group?
- Does pooled near-tie conceal opposing subgroup effects?

**Deliverables:**
- §7.10 Stratified Discussion
- Optional: `discussion/stratified_summary.csv` under experiment dir

**Exit criterion:** Discussion acknowledges mixed per-container outcomes documented in Phase 5.

---

### Task 5 — Limitations and Threats to Validity

**Objective:** Thesis-grade limitations tied to Phases 1–5.

**Must include:**
1. Single primary configuration (P90, λ = 5) — conclusions not transferable to all peak-aware methods
2. λ not empirically swept in primary experiment (see [Appendix B](#appendix-b--future-work-and-supplementary-experiments))
3. Train/val peak-rate divergence under fixed thresholds
4. Near-idle P90 artefact (Phase 2)
5. MAPE instability near zero CPU
6. Single trace, 99-container cohort — external validity bounded
7. Negative peak-subset result is a **valid controlled finding**, not a failed experiment

**Deliverables:**
- §7.11 Limitations
- §7.12 Threats to validity (internal / external)

**Exit criterion:** Limitations clearly separate what was tested from what was not.

---

### Task 6 — Lock Phase 6 Record

**Objective:** Mark the Peak-Aware research track discussion complete.

**Deliverables:**
- `experiments/peak_aware_2026-07-14_164518/phase6_discussion.json`
- Final sections in this document (§7.7–§7.14)
- Update `docs/peak-aware-hybrid/README.md` — Phase 6 **Complete**
- `discussion/discussion_workspace.json` (optional tracker)
- Thesis handoff block (§7.14)

**Exit criterion:** Phases 1–6 documented end-to-end; frozen baseline untouched.

---

### 7.7 Results Synthesis *(pending — Task 1)*

*To be written during Task 1 implementation.*

---

### 7.8 Answer to the Research Question *(pending — Task 2)*

*To be written during Task 2 implementation. Use draft in Task 2 above as starting point.*

---

### 7.9 Mechanism Discussion *(pending — Task 3)*

*To be written during Task 3 implementation. Possible explanations only.*

---

### 7.10 Stratified Discussion *(pending — Task 4)*

*To be written during Task 4 implementation.*

---

### 7.11 Limitations *(pending — Task 5)*

*To be written during Task 5 implementation.*

---

### 7.12 Threats to Validity *(pending — Task 5)*

*To be written during Task 5 implementation.*

---

### 7.13 Thesis Narrative — Actual Contribution (Locked Focus)

Phase 6 discussion and thesis prose should emphasize **what was actually delivered**:

1. A **Peak-Aware Hybrid Prophet + GRU** model was **designed** (Phase 3) with one controlled experimental variable: timestep-weighted GRU residual training.
2. The design was **implemented** (Phase 4) in parallel modules without modifying the frozen baseline.
3. **Only the GRU training objective** was changed; Prophet, data, architecture, inference, and evaluation protocol remained identical.
4. The implementation **passed all fairness verification** (Phase 4: 8/8) and evaluation integrity checks (Phase 5: 8/8).
5. Under the evaluated configuration (P90, λ = 5), the method produced **negligible overall improvements** and **did not improve peak-subset accuracy**.
6. This is a **valid controlled research finding** — reported honestly with limitations and possible explanations, not as a universal verdict on peak-aware learning.

**Do not** frame Phase 6 as “fixing” the model or “proving” why it failed. Frame it as **completing the research** with an auditable negative/refinement result.

---

### 7.14 Thesis Handoff *(pending — Task 6)*

*Bullet list for Results + Discussion chapters — to be finalized in Task 6.*

---

## Appendix A — Optional Exploratory: Error Attribution

**Status:** Not required for Phase 6 completion.  
**Label in thesis:** *Exploratory / supplementary analysis* — not primary evidence.

### Why this is outside the primary workflow

The primary research question was answered by the **controlled paired comparison** in Phase 5 (overall + peak-subset metrics). Error attribution investigates a **different question**: where remaining error originates (e.g., Prophet vs GRU residual path). That is useful context but **not** part of the main contribution claim.

### If implemented later

| Item | Guidance |
|------|----------|
| Question | “At peak timesteps, how much error remains after Prophet alone vs after the full hybrid?” |
| Inputs | Inference caches or `notebooks/peak_aware_container_comparison.ipynb` |
| Output location | `experiments/.../discussion/supplementary/error_attribution/` |
| Writing rule | Results labelled **exploratory**; cannot upgrade §7.8 conclusion |
| Example phrasing | “Exploratory decomposition suggests…” — not “Prophet dominates peak error.” |

**Do not** block Phase 6 lock on completion of this appendix.

---

## Appendix B — Future Work and Supplementary Experiments

**Status:** Listed for completeness only. **Not** part of the primary Phase 6 workflow or locked conclusions.

| Item | Note |
|------|------|
| **λ sensitivity (λ ∈ {3, 5, 10})** | Would require **new training runs** in separate experiment directories. Does not modify the locked λ = 5 primary experiment or its conclusions. |
| **Peak-aligned early stopping** | Exploratory design variant; not the locked fairness contract. |
| **Global GRU comparison** | Requires fixing train/val leakage first (`docs/methodology-gap-analysis.md`). Out of peak-aware primary arc unless separately scoped. |
| **Alternative peak definitions** | New research question; not a reinterpretation of Phase 5. |

---

### 7.15 Exit Criteria (Phase 6 Complete)

- [ ] §7.7 Results synthesis written (findings only)
- [ ] §7.8 Research question answered (scope-bound, academic tone)
- [ ] §7.9 Mechanism discussion written (explanations labelled)
- [ ] §7.10 Stratified discussion written
- [ ] §7.11–§7.12 Limitations and threats to validity written
- [ ] §7.13–§7.14 Thesis narrative and handoff finalized
- [ ] `phase6_discussion.json` saved
- [ ] README updated — Phase 6 complete
- [ ] Frozen baseline and Phase 5 results unchanged
- [ ] Appendix A **not** required
- [ ] Appendix B items **not** presented as primary evidence

### 7.16 Estimated Effort

| Task | Effort |
|------|--------|
| 1 — Results synthesis | 1–2 hours |
| 2 — Research question answer | 1 hour |
| 3 — Mechanism discussion | 2–3 hours |
| 4 — Stratified discussion | 1–2 hours |
| 5 — Limitations | 1–2 hours |
| 6 — Lock record | ~1 hour |

**Total (primary workflow):** ~1–1.5 days of writing and light analysis on existing artefacts.

### 7.17 Handoff From Phase 5

Phase 5 delivered measurement only. Phase 6 delivers interpretation only.

**Start here:**
- `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`
- `evaluation/comparison_table.csv`, `per_container_delta.csv`, `peak_subset_summary.csv`
- `notebooks/peak_aware_container_comparison.ipynb` (manual inspection)

**Recommended task order:**

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

*Plan refined: 2026-07-14 — ready for Task 1 implementation*
