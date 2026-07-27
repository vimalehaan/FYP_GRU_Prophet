export interface ForecastRequest {
  container_id: string;
  historical_cpu: number[];
  timestamps: string[];
  sampling_interval_minutes?: number;
  prediction_horizon_steps?: number;
}

export interface ValidationGroundTruth {
  container_id: string;
  actual_cpu: number[];
  timestamps: string[];
  description?: string;
}

export interface ForecastMetadata {
  model_version: string;
  processing_time_ms: number;
  scaler_mode: "known" | "new";
  history_steps_used: number;
  horizon_steps: number;
  sampling_interval_minutes: number;
}

export interface ForecastData {
  timestamps: string[];
  predicted_cpu_percent: number[];
  prophet_component_percent: number[];
  gru_residual_component_percent: number[];
}

export interface ForecastResponse {
  status: "success";
  container_id: string;
  metadata: ForecastMetadata;
  forecast: ForecastData;
}

export interface ValidationMetrics {
  mae: number;
  rmse: number;
  mape: number;
  maxError: number;
  bias: number;
  n: number;
}

export interface ChartPoint {
  timestamp: string;
  label: string;
  actual?: number;
  predicted?: number;
  prophet?: number;
  residual?: number;
  kind: "history" | "forecast";
}

export interface Preset {
  id: string;
  label: string;
  requestPath: string;
  validationPath?: string;
}

export const PRESETS: Preset[] = [
  {
    id: "known",
    label: "Known container (c_1016) — 200 steps",
    requestPath: "/examples/sample_request.json",
  },
  {
    id: "unseen_10312",
    label: "Unseen container (c_10312) — 672 steps + validation",
    requestPath: "/examples/unseen_c_10312_forecast_request.json",
    validationPath: "/examples/unseen_c_10312_validation_ground_truth.json",
  },
  {
    id: "unseen_12517",
    label: "Unseen container (c_12517) — 672 steps + validation",
    requestPath: "/examples/unseen_c_12517_forecast_request.json",
    validationPath: "/examples/unseen_c_12517_validation_ground_truth.json",
  },
];
