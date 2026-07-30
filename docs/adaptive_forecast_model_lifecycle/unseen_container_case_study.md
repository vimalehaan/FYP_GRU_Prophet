# Unseen Container Case Study

**Notebook:** `notebooks/afmlf_unseen_container_case_study.ipynb`  
**Status:** Operational demonstration — not a research experiment

---

## Purpose

The official AFMLF evaluation (`experiments/adaptive_forecast_model_lifecycle_2026-07-30_071206`) assessed lifecycle governance across **99 research containers** in aggregate. The unseen-container case study complements that evaluation by teaching AFMLF as **two distinct levels**:

1. **Individual container monitoring** — operational walkthrough on unseen workloads  
2. **Global model lifecycle management** — cohort-level retraining from the official experiment  

It answers: *“What does AFMLF do at the container level vs at the global model level?”*

---

## Why the notebook was restructured

An earlier version of this notebook showed a single drifting unseen container flowing directly into candidate retraining and deployment recommendation. That presentation was **conceptually inaccurate**:

- The deployed Hybrid model is a **global model** trained across many containers.
- Official AFMLF retraining is triggered only after **cohort-level monitoring** and the **global trigger policy** — not because one container drifts.

The restructured notebook explicitly separates:

| Section | Level | Content |
|---------|-------|---------|
| **Section 2** | Container-level | Two unseen scenarios (stable + drifting) — monitoring, drift, diagnostics, drift registry |
| **Section 3** | Cohort-level | Official 99-container evaluation — aggregate trigger, retrain, evaluation, recommendation |
| **Section 4** | Conceptual | Container-level vs global lifecycle + architecture diagram |

---

## What each section demonstrates

### Section 2 — Individual container monitoring

| Scenario | Container (auto-selected) | Outcome |
|----------|---------------------------|---------|
| **A — Stable** | e.g. `c_39702` | No drift; diagnostics STABLE; keep current model |
| **B — Drifting** | e.g. `c_29580` | Threshold breach; REGIME_CHANGE; **added to drift registry** |

**Critical point:** Scenario B **stops** after registering container-level degradation. It does **not** retrain the global Hybrid model. The notebook states:

> *This alone is NOT sufficient to retrain the global Hybrid model. The container is now part of the monitored population awaiting cohort-level lifecycle evaluation.*

### Section 3 — Cohort-level lifecycle (official experiment)

Uses frozen official artifacts only:

| Stage | Official result |
|-------|-----------------|
| Containers monitored | 99 |
| Drift suspected transitions | 65 |
| Global trigger satisfied | 1 (origin 200, cohort fraction) |
| Offline candidates trained | 1 (`hybrid_v2`) |
| Candidate evaluation | ΔMAE ≈ −0.003% |
| Deployment recommendation | **keep_current_model** |

Retraining decisions are based on the **monitored population**, not on an individual container.

---

## Why this is not a new experiment

| Aspect | Official evaluation | Case study |
|--------|---------------------|------------|
| Scope | 99-container cohort | Two unseen containers (monitoring) + official artifacts (lifecycle) |
| Outputs | Timestamped experiment directory | Notebook-only; reads official CSV/JSON |
| Methodology | Frozen AFMLF protocol | **Same frozen utilities** — no changes |
| Global retrain demo | Official experiment Section 3 | Not re-run; loaded from artifacts |
| Production model | Read-only `hybrid_v1` | Never modified |

- One unseen container demonstrates **operational monitoring** only.  
- It does **not** independently trigger retraining of the global Hybrid model.  
- Offline retraining is demonstrated separately using the **official cohort-level AFMLF evaluation**.

---

## How containers were selected (Section 2)

Automatic scan of 36 unseen containers using frozen monitoring/drift/diagnostic utilities:

| Scenario | Selection rule |
|----------|----------------|
| **A — Stable** | Zero breaches; all origins `STABLE`; highest threshold headroom |
| **B — Drifting** | Highest lifecycle score: consecutive breaches, meaningful diagnostics |

Scan cached at `notebooks/outputs/afmlf_case_study/unseen_scan.csv`.

---

## How it complements the official evaluation

| Official evaluation proves | Case study proves |
|---------------------------|-------------------|
| Scale (99 containers, 495 origin pairs) | Legibility of container-level monitoring |
| Cohort trigger and decision history | That one container ≠ global retrain |
| One offline retrain cycle | Full global lifecycle with official numbers |
| Deployment recommendation at scale | Two-level architecture understanding |

---

## Running the case study

```bash
# Regenerate notebook (optional)
tf_metal_env/bin/python scripts/build_afmlf_visualization_notebooks.py

# Open in Jupyter / VS Code — kernel: tf_metal_env
notebooks/afmlf_unseen_container_case_study.ipynb
```

Section 2 runs live monitoring (~1 min for two containers). Section 3 loads official experiment CSVs instantly — no retraining required.

---

## Related documentation

- [drift_detection.md](drift_detection.md) — per-container threshold formula  
- [decision_engine.md](decision_engine.md) — cohort-level retraining decisions  
- [AFMLF_FINAL_RESULTS.md](../../experiments/adaptive_forecast_model_lifecycle_2026-07-30_071206/reports/AFMLF_FINAL_RESULTS.md) — official thesis results  
- [README.md](README.md) — framework overview  
