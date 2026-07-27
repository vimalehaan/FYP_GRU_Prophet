# DracaSys — Module 1: Long-Term CPU Forecasting

Final Year Project: *Intelligent Deployment Helper for Containerized Applications*

This repository contains two **independent** pipelines:

1. **Research** — hypothesis testing, ablations, and methodology comparisons (frozen)
2. **Production** — final deployable Hybrid Prophet + GRU model

---

## Research Pipeline (read-only)

All experimental work lives outside `production/`. **Do not modify frozen experiment outputs.**

```
experiments/          # Timestamped experiment runs (HCERL, GGTCE, LFHE, CSRLE, …)
notebooks/            # Research analysis and demonstration notebooks
utils/                # Shared ML utilities (Hybrid baseline, Global GRU, sequence utils)
docs/                 # Research documentation and audit reports
data/                 # Preprocessed Alibaba cluster trace data
```

### Research documentation

Start at [`docs/README.md`](docs/README.md) for the full research documentation index.

### Key research areas (frozen)

| Area | Location |
|------|----------|
| Hybrid baseline | `experiments/hybrid_*`, `utils/hybrid_*.py` |
| Global GRU baseline | `experiments/global_gru_*` |
| Peak-aware, CSRLE, LFHE | `experiments/`, `docs/` |
| HCERL, GGTCE, TMA | `experiments/`, `docs/` |

---

## Production Pipeline

The **final selected Hybrid model** for deployment — completely isolated from research.

```
production/
├── hybrid/           # Trained artifacts (model, scalers, metrics, predictions)
├── utils/            # Production Python modules
├── scripts/          # train | evaluate | predict | build-notebook
├── notebooks/        # Thesis-ready demonstration notebook
└── docs/             # Production documentation
```

### Quick start

```bash
# Single-container inference (no retraining)
tf_metal_env/bin/python production/scripts/predict_container.py c_1016 --split known

# Open demonstration notebook
jupyter notebook production/notebooks/production_hybrid_model.ipynb
```

See [`production/docs/README.md`](production/docs/README.md) for full production documentation.

---

## Repository tree

```
FYP_Long_Term/
├── data/                     # Shared dataset
├── experiments/              # ← RESEARCH (frozen)
├── notebooks/                # ← RESEARCH notebooks
├── utils/                    # ← RESEARCH utilities (Hybrid baseline)
├── docs/                     # ← RESEARCH docs
│
└── production/               # ← PRODUCTION (isolated)
    ├── hybrid/               #     Model artifacts
    │   ├── config/
    │   ├── models/
    │   ├── metadata/
    │   ├── cache/
    │   ├── metrics/
    │   ├── predictions/
    │   └── logs/
    ├── utils/
    ├── scripts/
    ├── notebooks/
    └── docs/
```

---

## Environment

```bash
# TensorFlow + Prophet (Apple Silicon)
tf_metal_env/bin/python production/scripts/predict_container.py c_1016
```

---

## Agent skills

Project conventions live in `.cursor/skills/` (`dracasys/`, `experiments/`, `python-coding-standards/`, `visualization/`).
