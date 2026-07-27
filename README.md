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

## Forecast Service (REST API)

The **deployment integration layer** — exposes the frozen Hybrid model as a REST API for other project modules. Completely separate from research and from the `production/` training pipeline.

```
forecast_service/
├── app/                    # FastAPI application (inference only)
├── artifacts/hybrid_v1/    # Frozen model bundle (copied from production)
├── docs/                   # Integration & API documentation
├── examples/               # curl, Python, Java, JavaScript clients
└── tests/
```

### Quick start

```bash
cd forecast_service
pip install -r requirements.txt
python examples/generate_sample_request.py
python run.py
# → http://localhost:8000/docs
```

See [`forecast_service/README.md`](forecast_service/README.md) and [`forecast_service/docs/api/integration.md`](forecast_service/docs/api/integration.md).

---

## Repository tree

```
FYP_Long_Term/
├── data/                     # Shared dataset
├── experiments/              # ← RESEARCH (frozen)
├── notebooks/                # ← RESEARCH notebooks
├── utils/                    # ← RESEARCH utilities (Hybrid baseline)
├── docs/                     # ← RESEARCH docs
├── production/               # ← PRODUCTION training & CLI (isolated)
│   ├── hybrid/               #     Model artifacts
│   ├── utils/
│   ├── scripts/
│   └── docs/
└── forecast_service/         # ← DEPLOYMENT REST API (inference only)
    ├── app/
    ├── artifacts/hybrid_v1/
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
