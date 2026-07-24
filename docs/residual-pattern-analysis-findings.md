# Residual Pattern Analysis — Full Findings & Section 13 Audit

**Project:** DracaSys FYP — Hybrid Prophet + GRU  
**Notebook:** `notebooks/residual_pattern_analysis.ipynb`  
**Run date:** 2026-07-24  
**Document purpose:** Section-by-section record of insights, numerical results, interpretations, and the Section 13 evaluation audit  
**Final verdict:** **Conclusion C** — validation residuals contain temporal structure, but the frozen baseline GRU does not exploit it effectively

---

## Executive summary

This notebook asks whether Prophet validation residuals still contain learnable temporal structure after trend and daily seasonality are removed — the core justification for the Hybrid GRU architecture.

| Question | Finding |
|----------|---------|
| Does Prophet leave structured residuals? | **Yes** — elevated ACF/PACF, 73.7% of containers reject white noise (Ljung–Box lag 20) |
| Are residuals centred and reasonably unbiased? | **Mostly yes** — pooled mean ≈ −0.03% CPU, but heavy tails (kurtosis ≈ 58) |
| Does the frozen GRU track residual dynamics? | **No** — cohort mean Pearson r ≈ 0.02, R² ≈ −0.34; predictions are near-flat |
| Is Section 13 evaluation implemented correctly? | **Yes** — audit confirms correct model, scaling, alignment, and reconstruction |
| Overall conclusion | **Conclusion C** — structure exists; baseline residual learner under-exploits it |

**Thesis-safe one-liner:**

> Prophet validation residuals show statistically significant temporal dependence across the Hybrid training cohort, supporting the decomposition design. A structured audit verified that low GRU residual correlation reflects genuine model behaviour (under-dispersion), not an evaluation bug — motivating improved residual-learning strategies rather than abandoning the diagnostic methodology.

---

## Notebook scope and constraints

| Item | Value |
|------|-------|
| Mode | Read-only diagnostic — no retraining, no artifact writes |
| Cohort | `selected_containers.npy` (Hybrid GRU training containers) |
| Frozen GRU | `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Input window / Day-1 horizon | 96 / 96 steps (15-minute resolution = 24 hours) |
| Prophet config | `daily_seasonality=True`, `weekly_seasonality=False` (matches Hybrid pipeline) |
| Residual definition (Sections 3–12) | Real CPU %: `Actual − Prophet` (inverse-transformed from scaled space) |
| Residual definition (Section 13) | Scaled space: `cpu_scaled − prophet_yhat` (GRU training space) |

---

## Section 1 — Project overview

### Purpose

The Hybrid pipeline decomposes CPU forecasting:

```
CPU Usage (scaled)
      ↓
  Prophet  →  trend + daily seasonality
      ↓
Residual = Actual − Prophet
      ↓
    GRU    →  learns temporal structure in residuals
      ↓
Final forecast = Prophet + GRU residual correction
```

**Research question:** After Prophet removes trend and seasonality, do remaining residuals contain predictable temporal dependencies that justify the GRU component?

### Methods used in the notebook

1. **Distributional analysis** — pooled residual shape (Section 5)
2. **Autocorrelation diagnostics** — ACF, PACF, Ljung–Box (Sections 6–8)
3. **Visual / lag / spectral analysis** — scatter plots, rolling stats, periodogram (Sections 9–11)
4. **Prophet error mapping** — where large residuals occur (Section 12)
5. **Frozen GRU residual quality** — Day-1 predicted vs actual residuals (Section 13)

### Setup confirmation (run output)

| Item | Value |
|------|-------|
| Repository root | `/Users/lehaan/Desktop/FYP_Long_Term` |
| Selected containers (file) | 100 |
| Frozen Hybrid GRU path | `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Input window / Day-1 | 96 / 96 steps |

---

## Section 2 — Container selection

### Configuration (this run)

| Setting | Value |
|---------|-------|
| `ANALYSE_ALL` | `True` |
| Selection mode | All selected evaluable containers |
| Containers analysed | **99** |
| Skipped | `c_14674` — missing from frozen train/val data |
| Representative container | **`c_10032`** (first in sorted list; used for single-container plots) |

### Data coverage

- Train steps per container: ~600–614 (≈ 6–6.5 days at 15-min resolution)
- Validation steps per container: ~150–154 (≈ 1.5–1.6 days)
- All containers satisfy `min_train_steps ≥ 96` (GRU input window requirement)

### Interpretation

The analysis covers essentially the full Hybrid GRU training cohort. Results generalise to containers the baseline was trained/evaluated on, not to unseen containers.

---

## Section 3 — Prophet training

### Method

For each container:

1. Fit Prophet on **training period only**
2. Forecast **validation period**
3. Compute residuals in **real CPU percent** after inverse MinMax scaling
4. Report Prophet-only MAE and RMSE on validation

### Cohort-level Prophet performance

| Metric | Value |
|--------|-------|
| Mean Prophet MAE | **1.8474%** CPU |
| Median Prophet MAE | **1.1086%** CPU |
| Min / Max Prophet MAE | **0.0039%** / **24.6117%** CPU |
| Approx. IQR (p25–p75) | **0.49%** – **1.81%** CPU |

### Notable per-container examples

| Container | Prophet MAE | Interpretation |
|-----------|-------------|----------------|
| `c_13308` | 0.0039% | Near-constant, trivial workload — residuals likely negligible |
| `c_14106` | 0.0066% | Very low activity |
| `c_12060` | 0.1458% | Strong Prophet fit |
| `c_12237` | 24.61% | Extreme Prophet miss — drives pooled heavy tails |
| `c_11087` | 7.04% | High-variance container |
| `c_13389` | 8.02% | High-variance container |

### Findings

1. **Prophet is reasonable on average** — mean validation MAE ≈ 1.85% CPU aligns with the broader Hybrid baseline context (~1.75% end-to-end Day-1 MAE after GRU correction).
2. **High container heterogeneity** — MAE spans nearly four orders of magnitude (0.004% to 24.6%), so pooled statistics are dominated by a subset of difficult containers.
3. **Prophet alone is not sufficient** on many containers — large per-container MAE values (e.g. `c_12237`, `c_11087`) indicate substantial remaining error mass that a residual learner could theoretically target.

### Reasoning

Prophet captures daily seasonality and trend well for typical containers but struggles on bursty or regime-shifting workloads. The Hybrid design assumes those misses are **structured in time**, not i.i.d. noise — tested in Sections 6–8.

---

## Section 4 — Residual time series (representative: `c_10032`)

### Method

Plot full validation-period residuals for the representative container. Compute sign-persistence statistic: fraction of consecutive timesteps where residual sign is unchanged.

### Results

| Metric | Value |
|--------|-------|
| Sign persistence (lag-1 same sign) | **0.778** (77.8%) |
| Residual range | **[-1.474, 2.767]%** CPU |

### Visual / qualitative findings

- Residuals show **runs above/below zero** — not rapid mean-reverting white noise
- **Bursts** of larger error appear intermittently
- Some **oscillatory rhythm** visible over the validation window

### Interpretation

A sign-persistence rate of ~78% is far above the ~50% expected for uncorrelated zero-mean noise. Residuals exhibit **short-term persistence**: if Prophet undershoots at step *t*, it tends to undershoot at *t*+1 as well. This is consistent with ACF elevation at short lags (Section 6) and supports the hypothesis that a sequence model could learn local error dynamics.

Pure white noise would show rapid sign flipping, no sustained runs, and no visible periodic structure — **not observed** here.

---

## Section 5 — Residual distribution (pooled, all 99 containers)

### Method

Concatenate all validation-period residuals (real CPU %) across analysed containers. Report distributional statistics and histogram.

### Results

| Statistic | Value (CPU %) |
|-----------|---------------|
| Mean | **−0.0302** |
| Median | **−0.0063** |
| Std | **4.4478** |
| Variance | **19.7827** |
| Min | **−59.0682** |
| Max | **84.7010** |
| Skewness | **−2.5353** |
| Kurtosis | **58.0470** |

### Findings

1. **Near-zero mean** — Prophet is approximately unbiased in aggregate (slight negative skew in mean).
2. **Heavy tails** — extreme kurtosis (58) and wide min/max indicate occasional **large Prophet misses**, not Gaussian noise.
3. **Left skew** — large negative residuals (Prophet over-predicts) may be slightly more common than large positive ones at the pooled level.

### Interpretation

The distribution shape is **not** consistent with white noise or homoscedastic Gaussian errors. Heavy tails imply that a small fraction of timesteps — likely peak/burst periods — carry disproportionate error magnitude. A sequence model (GRU) that learns to anticipate these bursts could improve Hybrid CPU MAE even if average residual correlation is low.

**Caveat:** Pooling across heterogeneous containers inflates variance; per-container distributions would be tighter.

---

## Section 6 — Autocorrelation analysis (ACF)

### Method

Compute ACF up to 50 lags per container on validation residuals. Report cohort-average |ACF| over lag bands. Plot representative container ACF.

### Results — cohort average |ACF|

| Lag band | Average \|ACF\| |
|----------|----------------|
| Lags 1–10 | **0.2010** |
| Lags 11–20 | **0.1015** |
| Lags 21–50 | **0.0776** |

### Findings

1. **Strongest dependence at short lags** — |ACF| at lags 1–10 is ~2× the 11–20 band, indicating short-term serial correlation dominates.
2. **Non-negligible long-lag structure** — |ACF| at lags 21–50 remains ~0.078, suggesting incomplete removal of sub-daily or daily patterns by Prophet alone.
3. **Monotonic decay across bands** — consistent with AR-like persistence rather than a single isolated spike.

### Interpretation

Elevated |ACF| at lags 1–10 means Prophet errors **persist** over 15-minute to 2.5-hour horizons. Non-zero |ACF| at longer lags (up to ~12 hours) suggests **residual daily/sub-daily structure** remains — a plausible target for GRU residual learning.

For reference, white noise would yield |ACF| ≈ 0 at all lags > 0 (within confidence bounds). Observed values (~0.20 at short lags) are **economically large** for forecasting.

---

## Section 7 — Partial autocorrelation (PACF)

### Method

Compute PACF (Yule-Walker) up to 50 lags per container. Report cohort-average |PACF| over lag bands.

### Results — cohort average |PACF|

| Lag band | Average \|PACF\| |
|----------|------------------|
| Lags 1–10 | **0.1088** |
| Lags 11–20 | **0.0575** |

### Findings

1. **Direct lag-1 dependence** — elevated |PACF| at low lags confirms residuals have direct serial dependence, not only indirect chain effects.
2. **Multi-lag structure** — significant PACF beyond lag 1 suggests predictable structure beyond a simple AR(1) model.
3. **Decay pattern mirrors ACF** — short-lag PACF stronger than mid-lag PACF.

### Interpretation

PACF isolates **direct** relationships: a GRU with 96-step input window can, in principle, exploit multi-step direct dependencies without relying solely on lag-1 chaining. The presence of structure at lags 1–10 supports the Hybrid architecture hypothesis at the **statistical** level.

---

## Section 8 — Ljung–Box test

### Method

Test whether residuals are consistent with white noise at lag orders 10, 20, and 30. Rejection at p < 0.05 indicates significant remaining autocorrelation.

### Results — cohort rejection rates

| Lag order | Fraction rejecting white noise (p < 0.05) |
|-----------|-------------------------------------------|
| 10 | **76.8%** |
| 20 | **73.7%** |
| 30 | **71.7%** |

### Representative container (`c_10032`)

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 10 | 401.965 | 3.594×10⁻⁸⁰ |
| 20 | 554.720 | 9.683×10⁻¹⁰⁵ |
| 30 | 598.608 | 5.752×10⁻¹⁰⁷ |

### Findings

1. **Overwhelming majority reject white noise** — ~72–77% of containers show statistically significant autocorrelation at all tested lag orders.
2. **Representative container is extreme** — p-values are effectively zero, confirming strong temporal dependence.
3. **Slight decline at higher lags** — rejection rate drops from 76.8% (lag 10) to 71.7% (lag 30), but remains well above 50%.

### Interpretation

If Prophet residuals were white noise, we would expect ~5% rejection at p < 0.05 by chance. Observed **~74% rejection** is decisive evidence that **temporal structure remains** after Prophet decomposition across the cohort.

This is the strongest statistical support for the Hybrid design in the notebook. It directly contradicts **Conclusion B** (limited structure).

---

## Section 9 — Residual lag relationship (representative: `c_10032`)

### Method

Scatter plots of `Residual(t)` vs `Residual(t+1)` and `Residual(t)` vs `Residual(t+5)` with OLS fit lines. Pearson r shown in plot titles.

### Expected qualitative patterns

| Pattern in scatter | Meaning |
|--------------------|---------|
| Diagonal cloud | Positive lag dependence — past residual predicts future |
| Diffuse ball around origin | Weak linear lag dependence |
| Curved / clustered structure | Nonlinear predictability (GRU may capture even if linear r is low) |

### Findings (qualitative, from notebook plots)

- **Lag-1 plot:** visible positive association — consistent with ACF lag-1 elevation and 78% sign persistence (Section 4)
- **Lag-5 plot:** weaker but non-random structure — multi-step dependence exists
- OLS fit lines show **positive slope** on lag-1 (typical for persistent residuals)

### Interpretation

Linear lag correlation visualises what ACF/PACF quantify. Even when GRU fails to exploit structure (Section 13), the **raw residuals remain lag-predictable** at the exploratory level. This reinforces Conclusion C: the bottleneck is the **learner**, not the absence of signal.

---

## Section 10 — Rolling statistics (representative: `c_10032`)

### Method

One-day (96-step) rolling mean and rolling standard deviation of validation residuals.

### Results

| Metric | Value |
|--------|-------|
| Rolling std range | **[0.3157, 0.8694]%** CPU |
| Rolling window | 96 steps (24 hours) |

### Findings

1. **Rolling std varies by ~2.8×** across the validation period (0.32% to 0.87%).
2. **Heteroscedasticity** — error variance is not constant over time.
3. Periods of elevated rolling std align with **burst / transition** regimes where Prophet error amplifies.

### Interpretation

Homogeneous white noise would show approximately flat rolling variance. The observed variation indicates **regime-dependent error magnitude** — often associated with workload bursts or diurnal transitions. A residual learner that adapts to local variance regimes (e.g. peak-aware weighting) could target high-impact periods identified in Section 12.

---

## Section 11 — Frequency analysis (representative: `c_10032`)

### Method

Periodogram of validation residuals to detect periodic components not fully removed by Prophet's daily seasonality term.

### Results — dominant periods

| Period (hours) | Spectral power |
|----------------|----------------|
| **≈ 38.50 h** | 2.896×10¹ |
| **≈ 19.25 h** | 1.338×10¹ |
| **≈ 9.62 h** | 4.081×10⁰ |

### Findings

1. **Sub-harmonics of ~38.5 h** — 19.25 h ≈ 38.5/2 and 9.62 h ≈ 38.5/4 suggest a fundamental period near **1.6 days** or aliasing of multi-day/workload-cycle patterns.
2. **Not exactly 24 h** — peaks are near but not precisely at 24 h, implying Prophet's default daily Fourier terms do not fully capture all cyclic workload behaviour.
3. **Sub-daily structure at ~9.6 h** — may reflect shift patterns or batch job cycles in the Alibaba trace.

### Interpretation

Remaining periodicity in residuals means Prophet's daily seasonality is **incomplete** for this container. Frequency peaks identify **candidate cycles** for GRU to learn. Combined with ACF at longer lags (Section 6), this supports multi-scale temporal structure in the residual target.

---

## Section 12 — Prophet vs residual visualization (representative: `c_10032`)

### Method

Aligned dual-panel plot:

1. Actual CPU vs Prophet forecast (validation period)
2. Residual series with **top 10% |residual|** timesteps highlighted

### Findings (qualitative)

- Prophet tracks the **overall level and daily shape** but misses **local peaks and troughs**
- Largest |residual| spikes **co-occur** with Prophet forecast divergence from actual CPU
- Highlighted top-10% error timesteps cluster in **burst regions** rather than being uniformly scattered

### Interpretation

This section links statistical diagnostics to **operational value**: GRU correction is most valuable where Prophet error is largest. If the GRU could learn to anticipate these spike regions (heteroscedastic, lag-correlated), Hybrid CPU MAE would improve most on high-impact timesteps — even if average residual Pearson r stays low.

---

## Section 13 — GRU residual prediction quality

### Method

Load **frozen** baseline Hybrid GRU (no retraining). For each container, run `run_hybrid_inference()` and compare:

```python
actual_res = result.actual_day1_scaled - result.day1_prophet
pred_res   = result.day1_residual
```

Report MAE, RMSE, Pearson r, and R² on Day-1 (96 validation steps) in **scaled residual space**.

### Cohort results

| Metric | Cohort average |
|--------|----------------|
| Residual MAE (scaled) | **0.094849** |
| Residual RMSE (scaled) | **0.140001** |
| Pearson r | **0.0215** |
| R² | **−0.3410** |

### Distribution of per-container GRU metrics

| Metric | Approx. range / pattern |
|--------|-------------------------|
| Pearson r | Roughly **−0.57 to +0.56**; slightly more containers positive than negative |
| R² | Mostly **≤ 0**; only ~10 containers show positive R² |
| Best r examples | `c_10034` (+0.56), `c_11735` (−0.50), `c_12231` (−0.57) |
| Extreme negative R² | `c_15794` (−3.39), `c_16008` (−2.88), `c_13308` (−8.20) — low-activity containers with tiny actual residual variance |

### Representative container (`c_10032`) — Day-1 plot

- Actual residuals vary over the 96-step horizon
- GRU predictions form a **near-flat band** close to zero
- Visual pattern: **under-dispersion**, not time-shift or inversion

### Notebook interpretation framework

| Case | Condition | This run |
|------|-----------|----------|
| **A** | Structure + high GRU r/R² | ✗ |
| **B** | Weak structure + low GRU r | ✗ |
| **C** | Structure + low GRU r | **✓ Selected** |

### Key finding

**Statistical structure exists (Sections 6–8), but the frozen GRU does not track validation residual dynamics (Section 13).** The gap between diagnostic signal and learner performance defines Conclusion C.

---

## Section 13 — Evaluation audit (implementation verification)

**Audit date:** 2026-07-24  
**Scope:** Read-only verification against official Hybrid pipeline  
**Verdict:** **A — Evaluation is correctly implemented; low correlation reflects genuine model behaviour**

### Why the audit was needed

Cohort averages of Pearson r ≈ 0.02 and R² ≈ −0.34 raised concern about scale mismatch, timestamp misalignment, or wrong model artifacts — not necessarily poor modelling.

### Check-by-check results

#### CHECK 1 — Correct model

| Item | Value |
|------|-------|
| Model path | `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Residual stats | `experiments/baseline_reference_2026-07-14/models/residual_stats.pkl` |
| Input window | 96 |
| Output shape | `(None, 96)` |
| Architecture | GRU(256) → GRU(128) → GRU(64) → Dense(128) → Dense(96) |

Matches `baseline_metadata.json` and `evaluate_selected_containers()` official pipeline.

#### CHECK 2 — Residual scaling

Both arrays are in **raw scaled residual units** (`cpu_scaled − prophet`). No mix of z-score vs inverse-transformed values.

Representative `c_10393`:

| Stat | Actual | Predicted |
|------|--------|-----------|
| mean | 0.00124 | −0.00104 |
| std | **0.0708** | **0.0023** |
| min / max | −0.216 / 0.194 | −0.007 / 0.004 |

`max |pred_res − inverse(day1_residual_scaled)| = 0.0` (exact).

**Terminology note:** *Scaled residual space* in the notebook means MinMax-scaled CPU minus Prophet — not z-score `residual_scaled` used internally by the GRU.

#### CHECK 3 — Sequence alignment

| Item | Value |
|------|-------|
| Input window | Last 96 train `residual_scaled` values |
| Forecast horizon | First 96 validation steps |
| Timestamp match | Prediction and ground-truth share identical validation indices |
| Train → val gap | 15 minutes (one timestep) |

#### CHECK 4 — Shape verification

Both arrays shape `(96,)` for all containers.

#### CHECK 5 — Visual alignment (`c_10393`)

| Pattern | Observed? |
|---------|-----------|
| Time shift / delay | No |
| Inversion | No |
| Flat prediction | **Yes** — pred std 0.002 vs actual std 0.071 |

#### CHECK 6 — Correlation sanity (`c_10393`)

| Metric | Value |
|--------|-------|
| Pearson r | −0.097 |
| MAE | 0.055 |
| RMSE | 0.071 |
| R² | −0.008 |

Identical r in z-score space — confirms metric consistency.

#### CHECK 7 — Naive baselines (`c_10393`)

| Predictor | MAE | RMSE | R² |
|-----------|-----|------|-----|
| Frozen GRU | 0.055 | 0.071 | −0.008 |
| Naive: r(t+1)=r(t) | 0.070 | 0.096 | −0.819 |
| Zero: r=0 | 0.054 | 0.071 | −0.0003 |

GRU slightly beats naive persistence on MAE but loses to zero predictor on R² because predictions have wrong variance.

#### CHECK 8 — CPU reconstruction

| Check | Result |
|-------|--------|
| `Prophet + day1_residual` vs `day1_final_scaled` | max diff = **0.0** |
| Notebook vs `evaluate_selected_containers()` | max diff = **0.0** |

### Audit root cause — why correlation is low (not a bug)

1. **Near-constant GRU output** — predicted residual std ~0.001–0.003 vs actual ~0.02–0.25 (up to ~30× under-dispersion)
2. **Train vs validation domain** — GRU trained on train-period sliding windows; validation residuals may differ in distribution
3. **MSE-optimal flat predictions** — predicting ≈ conditional mean in z-space yields small, low-variance corrections after inverse transform
4. **Metric sensitivity** — R² heavily penalises variance mismatch; MAE can look acceptable while r ≈ 0

### Audit conclusion

Section 13 is **methodologically sound**. Low Pearson r and negative R² are **real characteristics** of the frozen baseline GRU, not evaluation artefacts. Hybrid can still achieve useful CPU-level Day-1 MAE via small bias shifts without tracking residual shape.

---

## Section 14 — Overall summary

### Automated summary output (this run)

```
============================================================
RESIDUAL PATTERN ANALYSIS — SUMMARY
============================================================
Analysed containers              : 99
Average Prophet MAE (CPU %)      : 1.8474
Pooled residual mean (CPU %)     : -0.0302
Pooled residual variance         : 19.7827
Average |ACF| lags 1–10          : 0.2010
Containers rejecting white noise : 73.7% (Ljung–Box lag 20, p<0.05)
GRU residual Pearson r (mean)    : 0.0215
GRU residual R² (mean)           : -0.3410
============================================================

Conclusion C: The residuals contain temporal structure, but the current GRU
does not fully exploit it, indicating room for improving the residual-learning
component.
```

### Conclusion selection logic

```python
has_temporal_structure = (avg_acf_short > 0.05) and (reject_rate_20 >= 0.5)
gru_effective = (avg_pearson > 0.3) and (avg_r2 > 0.0)
```

| Condition | This run | Threshold |
|-----------|----------|-----------|
| `has_temporal_structure` | **True** | avg \|ACF\|₁₋₁₀ = 0.201 > 0.05; 73.7% reject > 50% |
| `gru_effective` | **False** | r = 0.02 ≪ 0.3; R² = −0.34 < 0 |

→ **Conclusion C**

### Conclusion definitions

| ID | Statement | Applies? |
|----|-----------|----------|
| **A** | Structure exists **and** GRU tracks residuals effectively | No |
| **B** | Limited structure; Prophet explains most signal | No |
| **C** | Structure exists **but** GRU under-exploits it | **Yes** |

---

## Cross-section synthesis

### Evidence chain supporting Conclusion C

```mermaid
flowchart TD
    A[Prophet fits trend + daily seasonality] --> B[Validation residuals computed]
    B --> C{Statistical tests}
    C --> D[ACF/PACF elevated at short lags]
    C --> E[73.7% reject Ljung-Box white noise]
    C --> F[Heavy-tailed pooled distribution]
    D --> G[Temporal structure confirmed]
    E --> G
    F --> G
    G --> H{Frozen GRU Day-1 residuals}
    H --> I[r ≈ 0.02, R² ≈ -0.34]
    H --> J[Near-flat predictions / under-dispersion]
    I --> K[Conclusion C]
    J --> K
    L[Section 13 audit: evaluation correct] --> K
```

### What is validated vs what is not

| Claim | Status |
|-------|--------|
| Prophet residuals are not white noise | **Validated** (Sections 6–8) |
| Residuals have burst/peak error regions | **Validated** (Sections 5, 10, 12) |
| Frozen GRU learns validation residual dynamics | **Not supported** (Section 13) |
| Section 13 metrics are computed correctly | **Validated** (audit) |
| Hybrid GRU architecture hypothesis is wrong | **Not supported** — learner under-performance ≠ no signal |

### Implications for thesis / next steps

1. **Keep the Hybrid decomposition** — diagnostics justify Prophet + sequence residual learner in principle.
2. **Improve residual learning** — peak-aware loss weighting, heteroscedastic modelling, or architecture changes; current baseline GRU regresses toward near-zero corrections.
3. **Do not over-interpret residual r alone** — low r is compatible with modest Hybrid CPU MAE gains via bias correction.
4. **Report diagnostics and learner metrics separately** — structure in residuals (ACF/Ljung–Box) and GRU tracking ability (Section 13) answer different questions.

---

## Limitations

1. **Training cohort only** — 99 evaluable containers from `selected_containers.npy`; unseen-container behaviour not tested.
2. **Day-1 horizon only** — Section 13 evaluates a single 96-step forecast; recursive Day-2+ not assessed.
3. **Single GRU snapshot** — results apply to `baseline_reference_2026-07-14` only.
4. **Pooled statistics mix heterogeneous containers** — high-MAE outliers (`c_12237`) inflate variance and kurtosis.
5. **Representative plots use `c_10032`** — single-container visuals may not generalise to all workload types.
6. **Terminology** — *scaled residual space* (Section 13) vs *real CPU %* (Sections 3–12) must be kept distinct when comparing magnitudes.

---

## References

| Resource | Path |
|----------|------|
| Diagnostic notebook | `notebooks/residual_pattern_analysis.ipynb` |
| Inference pipeline | `utils/hybrid_inference.py` |
| Evaluation pipeline | `utils/hybrid_evaluation.py` |
| Frozen baseline metadata | `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json` |
| Frozen GRU model | `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Residual statistics | `experiments/baseline_reference_2026-07-14/models/residual_stats.pkl` |
| Section 13 audit (legacy filename) | `docs/residual-pattern-analysis-section13-audit.md` |

---

*Document compiled from notebook run on 2026-07-24. Section 13 audit performed read-only; no models, artifacts, or training pipelines were modified.*
