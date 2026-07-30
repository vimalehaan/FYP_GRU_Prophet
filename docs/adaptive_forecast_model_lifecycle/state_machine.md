# State Machine

Central AFMLF architecture — see `utils/adaptive_lifecycle/lifecycle.py`.

## States

`Stable → Warning → DriftSuspected → DriftConfirmed → DiagnosticAnalysis → Decision → CandidateRetraining → CandidateEvaluation → DeploymentRecommendation → Cooldown → Stable`

## Decision outputs (only three)

- `recommend_retraining`  
- `do_not_retrain`  
- `needs_investigation`  

## Deployment outputs

- `deploy_candidate`  
- `keep_current_model`  
- `needs_investigation`  

Production is **never** auto-overwritten.
