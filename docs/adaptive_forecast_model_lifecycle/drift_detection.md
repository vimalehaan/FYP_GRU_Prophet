# Drift Detection (Layer 2)

**Module:** `drift_detection.py`

## Design

Drift is detected **per container** against that container's **own frozen Hybrid baseline** from completed forecasting research.

There is **no global MAE threshold**. A container with baseline MAE 0.5% and one with 5% each receive a threshold scaled to their own baseline — low-error and high-error containers are not judged by the same absolute cutoff.

---

## How the threshold is calculated (step by step)

AFMLF compares a **rolling operational error** against a **per-container threshold** derived from a **frozen baseline**.

### Step 1 — Frozen baseline (`baseline_hybrid_mae`)

For each container in the 99-container research cohort:

| Item | Detail |
|------|--------|
| **Source file** | `experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv` |
| **Column** | `day1_mae` (renamed to `baseline_hybrid_mae` in AFMLF) |
| **Meaning** | Day-1 Hybrid MAE recorded when the model was validated during completed forecasting research |
| **Mutability** | Fixed for the entire AFMLF run — never updated during walk-forward simulation |

This baseline represents “how well Hybrid forecast this container at deployment time.”

Implementation: `load_hybrid_baselines()` in `drift_detection.py`; merged into `origin_metrics.csv` during the experiment pipeline.

### Step 2 — Rolling monitoring signal (`rolling_hybrid_mae`)

At each walk-forward origin, Layer 1 monitoring:

1. Forecasts the next Day-1 window using only history before the origin.
2. Computes **Day-1 Hybrid MAE** (`hybrid_mae`) against ground truth.

Layer 1 rolling metrics then smooth errors over the last **N origins** per container (default `rolling_window_origins = 3`):

```
rolling_hybrid_mae = mean( hybrid_mae over last 3 walk-forward origins for this container )
```

Implementation: `append_rolling_metrics()` in `rolling_metrics.py`.

The drift detector compares **`rolling_hybrid_mae`** (current operational performance), not a single raw origin MAE, to reduce noise from one-off spikes.

### Step 3 — Threshold formula

Implemented in `drift_threshold()`:

```python
threshold = max(
    baseline_mae * (1.0 + relative_threshold),
    baseline_mae + absolute_threshold,
)
```

**Configurable parameters** (`AFMLFConfig`, saved in `config/afmlf_config.json`):

| Parameter | Default | Role |
|-----------|---------|------|
| `relative_threshold` | **0.25** | Allow up to 25% relative increase over baseline |
| `absolute_threshold` | **0.5** | Allow up to 0.5 percentage points absolute increase |

In plain language:

```
threshold = max( baseline × 1.25 ,  baseline + 0.5 )
```

Both bounds are computed; the ** larger** value becomes the threshold. This ensures:

- **High-baseline containers** — the relative term (× 1.25) often dominates.
- **Low-baseline containers** — the absolute term (+ 0.5 pp) prevents unrealistically tight cutoffs that would false-alarm on normal noise.

### Step 4 — Breach rule

```python
threshold_breach = rolling_hybrid_mae > threshold
```

Drift is flagged when rolling MAE is **strictly greater than** the threshold (equality is not a breach).

Implementation: `detect_container_drift()` returns `DriftDetectionResult` with `threshold_value` and `threshold_breach`.

---

## Worked examples

### Example A — Low-error container (official cohort)

Suppose `baseline_hybrid_mae = 0.35%`:

| Calculation | Value |
|-------------|-------|
| Relative bound | 0.35 × 1.25 = **0.438%** |
| Absolute bound | 0.35 + 0.5 = **0.85%** |
| **Threshold** | max(0.438, 0.85) = **0.85%** |

If `rolling_hybrid_mae = 0.90%` → **breach** (0.90 > 0.85).

### Example B — High-error container (case study `c_18764`)

For unseen container `c_18764`, the case study establishes deployment baseline from the **first monitoring origin** (see [unseen_container_case_study.md](unseen_container_case_study.md)). Using baseline ≈ **1.838%**:

| Calculation | Value |
|-------------|-------|
| Relative bound | 1.838 × 1.25 = **2.298%** |
| Absolute bound | 1.838 + 0.5 = **2.338%** |
| **Threshold** | max(2.298, 2.338) = **2.338%** |

At origin 392, `rolling_hybrid_mae ≈ 2.97%` → **breach** (2.97 > 2.338).

---

## What happens after a breach?

A threshold breach alone does **not** automatically recommend retraining. Subsequent frozen layers apply:

| Stage | Requirement |
|-------|-------------|
| **Persistence** | Two consecutive origin breaches → `drift_suspected` (see [state_machine.md](state_machine.md)) |
| **Page-Hinkley** | Confirmatory mode (default) — optional strengthening of confirmation |
| **Cohort trigger** | ≥ `drifted_container_fraction` (20%) of containers breaching **or** cohort median degradation ≥ 15% |
| **Diagnostics** | GRU_STALE / REGIME_CHANGE classes support the retraining decision |

Layer 2 answers: *“Has this container’s rolling error exceeded its personal baseline threshold?”*

---

## Unseen production containers

Containers **outside** the 99-cohort have no row in `baseline_reference` evaluation CSV.

For the operational case study only ([unseen_container_case_study.md](unseen_container_case_study.md)), deployment baseline is taken from the **first walk-forward origin’s Day-1 MAE** at go-live. The **same threshold formula** is then applied.

This is a demonstration convention for newly deployed workloads; the official 99-container evaluation uses frozen `baseline_reference` baselines exclusively.

---

## Drift vs diagnostic thresholds (do not confuse)

AFMLF uses **separate** thresholds for Layer 4 diagnostics (Prophet vs Hybrid classification):

| Config key | Default | Used for |
|------------|---------|----------|
| `relative_threshold` | 0.25 | **Layer 2 drift detection** |
| `absolute_threshold` | 0.5 | **Layer 2 drift detection** |
| `diagnostic_relative_threshold` | 0.15 | **Layer 4 diagnostics** |
| `diagnostic_absolute_threshold` | 0.3 | **Layer 4 diagnostics** |

Diagnostics classify *why* errors changed; drift detection decides *whether* rolling Hybrid error exceeded the deployment baseline by a configurable margin.

---

## Configuration and reproducibility

All parameters live in `AFMLFConfig` and are written to:

```
experiments/adaptive_forecast_model_lifecycle_<timestamp>/config/afmlf_config.json
```

Per-origin drift outputs are recorded in the experiment pipeline via `detect_container_drift()` at each walk-forward step. Threshold values can be reconstructed at any time from:

- `baseline_hybrid_mae` (from baseline CSV or deployment snapshot)
- `relative_threshold` and `absolute_threshold` (from config)

---

## Code reference

```python
# utils/adaptive_lifecycle/drift_detection.py

def drift_threshold(baseline_mae, relative_threshold, absolute_threshold):
    return max(
        baseline_mae * (1.0 + relative_threshold),
        baseline_mae + absolute_threshold,
    )

def detect_container_drift(..., rolling_hybrid_mae, baseline_hybrid_mae, config):
    threshold = drift_threshold(
        baseline_hybrid_mae,
        config.relative_threshold,
        config.absolute_threshold,
    )
    return DriftDetectionResult(..., threshold_breach=rolling_hybrid_mae > threshold)
```
