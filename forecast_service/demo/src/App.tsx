import { useCallback, useEffect, useMemo, useState } from "react";
import {
  checkHealth,
  loadPresetRequest,
  loadValidation,
  resolveApiBase,
  runForecast,
} from "./api";
import { ForecastChart } from "./components/ForecastChart";
import { MetricsPanel } from "./components/MetricsPanel";
import { computeValidationMetrics } from "./metrics";
import type {
  ChartPoint,
  ForecastRequest,
  ForecastResponse,
  Preset,
  ValidationGroundTruth,
  ValidationMetrics,
} from "./types";
import { PRESETS } from "./types";

function buildChartData(
  request: ForecastRequest,
  response: ForecastResponse | null,
  validation: ValidationGroundTruth | null,
): ChartPoint[] {
  const points: ChartPoint[] = request.timestamps.map((ts, i) => ({
    timestamp: ts,
    label: new Date(ts).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }),
    actual: request.historical_cpu[i],
    kind: "history" as const,
  }));

  if (response) {
    response.forecast.timestamps.forEach((ts, i) => {
      points.push({
        timestamp: ts,
        label: new Date(ts).toLocaleString(undefined, {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
        predicted: response.forecast.predicted_cpu_percent[i],
        prophet: response.forecast.prophet_component_percent[i],
        residual: response.forecast.gru_residual_component_percent[i],
        actual: validation?.actual_cpu[i],
        kind: "forecast",
      });
    });
  }

  return points;
}

export default function App() {
  const [apiUrl, setApiUrl] = useState("/forecast-api");
  const [presetId, setPresetId] = useState(PRESETS[1].id);
  const [requestJson, setRequestJson] = useState("");
  const [validationJson, setValidationJson] = useState("");
  const [useValidation, setUseValidation] = useState(true);
  const [showComponents, setShowComponents] = useState(false);
  const [historyTail, setHistoryTail] = useState(96);
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ForecastResponse | null>(null);
  const [request, setRequest] = useState<ForecastRequest | null>(null);
  const [validation, setValidation] = useState<ValidationGroundTruth | null>(null);
  const [metrics, setMetrics] = useState<ValidationMetrics | null>(null);

  const loadPreset = useCallback(async (p: Preset, withValidation: boolean) => {
    setError(null);
    setResponse(null);
    setMetrics(null);
    try {
      const req = await loadPresetRequest(p.requestPath);
      setRequest(req);
      setRequestJson(JSON.stringify(req, null, 2));

      if (withValidation && p.validationPath) {
        const val = await loadValidation(p.validationPath);
        setValidation(val);
        setValidationJson(JSON.stringify(val, null, 2));
        setUseValidation(true);
      } else {
        setValidation(null);
        setValidationJson("");
        setUseValidation(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    const p = PRESETS.find((x) => x.id === presetId) ?? PRESETS[0];
    void loadPreset(p, p.validationPath != null);
  }, [presetId, loadPreset]);

  useEffect(() => {
    const base = resolveApiBase(apiUrl);
    void checkHealth(base).then(setHealthOk);
    const id = window.setInterval(() => {
      void checkHealth(base).then(setHealthOk);
    }, 15000);
    return () => window.clearInterval(id);
  }, [apiUrl]);

  const chartData = useMemo(
    () => (request ? buildChartData(request, response, useValidation ? validation : null) : []),
    [request, response, validation, useValidation],
  );

  async function handleRunForecast() {
    setError(null);
    setLoading(true);
    setResponse(null);
    setMetrics(null);

    try {
      let req: ForecastRequest;
      try {
        req = JSON.parse(requestJson) as ForecastRequest;
      } catch {
        throw new Error("Request JSON is invalid.");
      }
      setRequest(req);

      let val: ValidationGroundTruth | null = null;
      if (useValidation && validationJson.trim()) {
        try {
          val = JSON.parse(validationJson) as ValidationGroundTruth;
          setValidation(val);
        } catch {
          throw new Error("Validation JSON is invalid.");
        }
      } else {
        setValidation(null);
      }

      const base = resolveApiBase(apiUrl);
      const result = await runForecast(base, req);
      setResponse(result);

      if (val && val.actual_cpu.length > 0) {
        const m = computeValidationMetrics(
          val.actual_cpu,
          result.forecast.predicted_cpu_percent,
        );
        setMetrics(m);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleRequestFile(file: File) {
    file.text().then((text) => {
      setRequestJson(text);
      try {
        setRequest(JSON.parse(text) as ForecastRequest);
      } catch {
        /* ignore until run */
      }
    });
  }

  function handleValidationFile(file: File) {
    file.text().then((text) => {
      setValidationJson(text);
      setUseValidation(true);
      try {
        setValidation(JSON.parse(text) as ValidationGroundTruth);
      } catch {
        /* ignore until run */
      }
    });
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>DracaSys Hybrid Forecast Demo</h1>
          <p className="subtitle">24-hour CPU forecast · Prophet + GRU</p>
        </div>
        <div className="header-meta">
          <label className="inline-field">
            API base
            <input
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="/forecast-api"
            />
          </label>
          <span className={`health ${healthOk ? "ok" : "bad"}`}>
            {healthOk === null ? "Checking…" : healthOk ? "API ready" : "API unavailable"}
          </span>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <section className="panel">
            <h2>Input</h2>
            <label className="field">
              Example preset
              <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={useValidation}
                onChange={(e) => setUseValidation(e.target.checked)}
              />
              Compare with validation ground truth
            </label>

            <div className="file-row">
              <label className="file-btn">
                Upload request JSON
                <input type="file" accept=".json" onChange={(e) => e.target.files?.[0] && handleRequestFile(e.target.files[0])} />
              </label>
              {useValidation && (
                <label className="file-btn">
                  Upload validation JSON
                  <input type="file" accept=".json" onChange={(e) => e.target.files?.[0] && handleValidationFile(e.target.files[0])} />
                </label>
              )}
            </div>

            <details className="json-details">
              <summary>Request JSON ({request?.historical_cpu.length ?? "—"} steps)</summary>
              <textarea
                value={requestJson}
                onChange={(e) => setRequestJson(e.target.value)}
                rows={8}
                spellCheck={false}
              />
            </details>

            {useValidation && (
              <details className="json-details" open>
                <summary>Validation JSON ({validation?.actual_cpu.length ?? "—"} steps)</summary>
                <textarea
                  value={validationJson}
                  onChange={(e) => setValidationJson(e.target.value)}
                  rows={6}
                  spellCheck={false}
                />
              </details>
            )}

            <button
              className="primary"
              onClick={() => void handleRunForecast()}
              disabled={loading || !requestJson.trim()}
            >
              {loading ? "Running forecast…" : "Run forecast"}
            </button>

            {error && <p className="error">{error}</p>}
          </section>

          {response && (
            <section className="panel meta-panel">
              <h2>Result metadata</h2>
              <dl>
                <dt>Container</dt>
                <dd>{response.container_id}</dd>
                <dt>Scaler</dt>
                <dd>{response.metadata.scaler_mode}</dd>
                <dt>Horizon</dt>
                <dd>{response.metadata.horizon_steps} steps</dd>
                <dt>History used</dt>
                <dd>{response.metadata.history_steps_used} steps</dd>
                <dt>Processing</dt>
                <dd>{response.metadata.processing_time_ms.toFixed(0)} ms</dd>
                <dt>Model</dt>
                <dd>{response.metadata.model_version}</dd>
              </dl>
            </section>
          )}
        </aside>

        <main className="main">
          <section className="panel chart-panel">
            <div className="chart-toolbar">
              <h2>Forecast plot</h2>
              <div className="toolbar-controls">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showComponents}
                    onChange={(e) => setShowComponents(e.target.checked)}
                  />
                  Show Prophet / GRU components
                </label>
                <label className="inline-field compact">
                  History tail
                  <select
                    value={historyTail}
                    onChange={(e) => setHistoryTail(Number(e.target.value))}
                  >
                    <option value={48}>48 steps</option>
                    <option value={96}>96 steps</option>
                    <option value={192}>192 steps</option>
                    <option value={672}>All history</option>
                  </select>
                </label>
              </div>
            </div>

            {chartData.length > 0 && response ? (
              <ForecastChart
                data={chartData}
                showComponents={showComponents}
                historyTail={historyTail}
              />
            ) : (
              <div className="placeholder">
                Select a preset and run forecast to see the plot.
              </div>
            )}
          </section>

          <MetricsPanel metrics={metrics} enabled={useValidation} />
        </main>
      </div>
    </div>
  );
}
