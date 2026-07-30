# Page-Hinkley (Layer 3)

**Module:** `page_hinkley.py`

## Role

Page-Hinkley (PH) is a sequential change-detection test on the per-container Hybrid MAE error stream. It is **confirmatory by default**, not the primary drift signal.

## Modes (`AFMLFConfig.page_hinkley_mode`)

| Mode | Behaviour |
|------|-----------|
| `off` | PH ignored; threshold persistence alone can confirm drift |
| `confirmatory` | **Default.** PH alarm strengthens confirmation at `drift_suspected` |
| `mandatory` | Drift confirmed only when PH fires |

## Parameters

- `page_hinkley_delta` — minimum detectable change (default 0.005)
- `page_hinkley_lambda` — alarm threshold on cumulative statistic (default 50)

## Outputs

`lifecycle/page_hinkley.csv` — per-container PH statistic and alarm flag at each origin.

Use ablation runs (`--page-hinkley-mode off|mandatory`) to study sensitivity without changing Hybrid architecture.
