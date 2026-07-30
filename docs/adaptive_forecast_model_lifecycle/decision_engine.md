# Decision Engine (Layer 5)

**Modules:** `decision_engine.py`, `trigger_policy.py`

## Separation of concerns

| Stage | Module | Question answered |
|-------|--------|-------------------|
| Detection | `drift_detection.py` | Did this container breach its baseline? |
| Diagnosis | `diagnostics.py` | Why might errors have changed? |
| Trigger | `trigger_policy.py` | Is the **cohort** degraded enough to act? |
| Decision | `decision_engine.py` | What should operators do? |

Execution (candidate retrain/eval) runs only after a positive decision.

## Valid decision outputs

Only three outputs are permitted:

1. **`recommend_retraining`** — offline candidate pipeline should run
2. **`do_not_retrain`** — continue monitoring incumbent model
3. **`needs_investigation`** — ambiguous signal; defer retrain pending review

## Global trigger policy

`trigger_policy.evaluate_global_trigger` fires when **either**:

- Fraction of containers with threshold breach ≥ `drifted_container_fraction` (default 20%), **or**
- Cohort median rolling MAE degradation ≥ `cohort_median_degradation` (default 15%)

Cooldown (`cooldown_origins`) prevents back-to-back retrain storms.

## Explainability

Every decision row includes `rationale`, diagnostic counts, and trigger reason — saved to `decisions/decisions.csv`.
