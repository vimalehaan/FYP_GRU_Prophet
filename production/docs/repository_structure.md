# Repository Structure

## Overview

```
FYP_Long_Term/
├── README.md                 # Project overview (research + production)
├── data/                     # Shared research dataset (read-only for production)
├── experiments/              # RESEARCH — frozen experiment outputs
├── notebooks/                # RESEARCH — experiment notebooks
├── utils/                    # RESEARCH — shared ML utilities (Hybrid baseline frozen)
├── docs/                     # RESEARCH — experiment documentation
├── models/                   # RESEARCH — legacy model paths
│
└── production/               # PRODUCTION — isolated deployable pipeline
    ├── hybrid/               # Trained model artifacts
    │   ├── config/
    │   ├── models/
    │   ├── metadata/
    │   ├── cache/
    │   ├── metrics/
    │   ├── predictions/
    │   ├── logs/
    │   └── reports/
    ├── utils/                # Production Python package
    ├── scripts/              # CLI entry points
    ├── notebooks/            # Production demonstration notebook
    └── docs/                 # Production documentation
```

## Separation rules

| Rule | Detail |
|------|--------|
| Research frozen | `experiments/` must never be modified |
| Production isolated | All production code and artifacts under `production/` |
| No duplication | Production wraps `utils/hybrid_*` — does not copy research code |
| Shared data only | Production reads `data/df_resampled.parquet` — never writes to `data/` |

## Entry points

| Script | Purpose |
|--------|---------|
| `production/scripts/train_production_hybrid.py` | Full train + evaluate |
| `production/scripts/evaluate_production_model.py` | Evaluate saved model only |
| `production/scripts/predict_container.py` | Single-container inference |
| `production/scripts/build_production_notebook.py` | Regenerate demo notebook |

## Import convention

From production scripts/notebooks:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve().parent))  # repo root

from production.utils.artifacts import load_production_model
from utils.hybrid_inference import run_hybrid_inference  # frozen research
```
