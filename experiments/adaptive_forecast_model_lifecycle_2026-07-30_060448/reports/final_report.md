# AFMLF Final Report

## Framework evaluation (primary)

- **confirmed_drift_events**: 0
- **false_alarm_events**: 1
- **detection_delay_origins**: []
- **recommendation_precision**: 1.0
- **deployment_success_rate**: 0.0
- **prevented_bad_deployments**: 1
- **diagnostic_distribution**: {'STABLE': 4, 'GRU_STALE': 2}
- **retraining_frequency**: 1
- **decision_latency_origins**: []
- **recommendation_quality**: [{'trigger_origin_index': 200, 'n_eval_origins': 3, 'incumbent_cohort_mae': 1.1171955687328308, 'candidate_cohort_mae': 1.1271508267866748, 'delta_mae': 0.009955258053844007, 'deployment_recommendation': 'keep_current_model', 'recommendation_correct': True, 'details': {'incumbent_cohort_mae': 1.1171955687328308, 'candidate_cohort_mae': 1.1271508267866748, 'delta_mae': 0.009955258053844007, 'n_eval_origins': 3}, 'candidate_version': 'hybrid_v2'}]
- **secondary_candidate_delta_mae**: [0.009955258053844007]

## Research answers

### Can concept drift be reliably detected?
The framework recorded 0 confirmed drift events with 1 deferred/no-retrain decisions for comparison.

### Are per-container baselines superior to global thresholds?
AFMLF uses per-container frozen Hybrid baselines because cohort MAE spans orders of magnitude; a single global threshold would mis-classify low- and high-error containers.

### Does the diagnostic layer distinguish GRU drift from workload drift?
Diagnostic distribution: {'STABLE': 4, 'GRU_STALE': 2}. GRU_STALE vs REGIME_CHANGE classes drive retraining confidence separately from INVESTIGATE.

### Is offline retraining appropriate?
Yes. The frozen global GRU and residual statistics require batch retraining; online weight updates were intentionally excluded.

### How does AFMLF extend the production forecasting pipeline?
AFMLF adds monitoring, drift detection, diagnostic reasoning, candidate versioning, and human-in-the-loop deployment recommendations without modifying Hybrid architecture.

