# Residual Pattern Analysis Documentation

> **This file has been superseded by the full notebook findings document:**
>
> **[`docs/residual-pattern-analysis-findings.md`](./residual-pattern-analysis-findings.md)**
>
> That document includes **all 14 sections** of `notebooks/residual_pattern_analysis.ipynb` (setup, Prophet diagnostics, ACF/PACF/Ljung–Box, distribution, frequency analysis, GRU quality, Conclusion C) **plus** the complete Section 13 evaluation audit.

---

## Quick reference — Section 13 audit verdict

**Verdict:** **A — Evaluation is correctly implemented; low correlation reflects genuine model behaviour**

| Question | Answer |
|----------|--------|
| Correct frozen GRU? | Yes — `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Same residual space for actual vs predicted? | Yes — raw scaled residual (`cpu_scaled − prophet`) |
| Timestamps aligned? | Yes — Day-1, 96 steps |
| Hybrid reconstruction matches official pipeline? | Yes — max diff = 0.0 |
| Why is Pearson r ≈ 0.02? | GRU outputs near-constant corrections (~30× lower variance than actual residuals) |

See the full audit (CHECK 1–8), cohort metrics, and cross-section synthesis in [`residual-pattern-analysis-findings.md`](./residual-pattern-analysis-findings.md).
