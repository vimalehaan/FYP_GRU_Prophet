# Forecast Demo Frontend

React demonstration UI for the DracaSys Hybrid CPU Forecast API.

## Features

- Load preset examples (known + unseen containers with validation holdout)
- Upload custom request / validation JSON
- Call `POST /forecast` and plot history + predictions
- Optional overlay of held-out **actual** CPU for unseen containers
- Validation metrics: **MAE**, **RMSE**, **MAPE**, max error, bias

## Prerequisites

1. Forecast API running on port **8000** (see parent [README](../README.md))
2. Node.js 18+

## Run

Terminal 1 — API:

```bash
cd forecast_service
python run.py
```

Terminal 2 — demo UI:

```bash
cd forecast_service/demo
npm install
npm run dev
```

Open **http://localhost:5173**

The Vite dev server proxies `/forecast-api` → `http://localhost:8000` (CORS enabled on the API).

## Presets

| Preset | Validation |
|--------|------------|
| Known `c_1016` | No ground truth (forecast only) |
| Unseen `c_10312` | 96-step holdout in `public/examples/` |
| Unseen `c_12517` | 96-step holdout in `public/examples/` |

## Custom API URL

Set **API base** in the header to a full URL (e.g. `http://localhost:8000`) if not using the Vite proxy.

## Build for static hosting

```bash
npm run build
npm run preview
```

Serve `dist/` and point API base to your deployed forecast service.
