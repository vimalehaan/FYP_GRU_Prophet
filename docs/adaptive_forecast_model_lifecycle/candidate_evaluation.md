# Candidate Evaluation (Layer 7)

**Module:** `candidate_evaluation.py`

## Purpose

Compare **candidate** vs **incumbent** on **future unseen origins only**.

## Evaluation window

After trigger at origin `T`:

- Skip `eval_warmup_origins × walk_forward_stride` steps (default 1 block).
- Evaluate on all remaining origins in the walk-forward simulation.

Both models forecast at the same origins using identical history cutoffs.

## Secondary metrics

- Incumbent cohort MAE
- Candidate cohort MAE
- Delta MAE (candidate − incumbent)

These are **secondary** to framework recommendation-quality metrics (see `evaluation_methodology.md`).

## Leakage rule

Evaluation origins satisfy `origin_index >= T + warmup`. Candidate training used only `origin_index < T` (with buffer). Assertions in `monitoring.py` enforce cutoff consistency.
