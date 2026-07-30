# Deployment Recommendation (Layer 8)

**Module:** `candidate_evaluation.py`

## Outputs

| Recommendation | Condition |
|----------------|-----------|
| `deploy_candidate` | Candidate MAE improves incumbent by more than `candidate_improvement_margin` |
| `keep_current_model` | Candidate not sufficiently better (or worse) |
| `needs_investigation` | No future evaluation origins available |

## Human-controlled deployment

AFMLF **never** replaces production artifacts automatically. Recommendations are written to:

- `recommendations/recommendations.csv`
- Version `metadata.json` (`deployment_recommendation` field)
- `reports/final_report.md`

Operators (or a future CI gate) decide whether to promote `hybrid_vN` to production.

## Why not automatic deployment?

1. Retraining can overfit recent history.
2. Diagnostic ambiguity may require manual review.
3. Production promotion is a governance decision separate from detection research.
4. Thesis contribution is **recommendation quality**, not unattended model swapping.
