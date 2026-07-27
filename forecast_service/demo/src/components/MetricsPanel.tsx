import type { ValidationMetrics } from "../types";
import { formatMetric } from "../metrics";

interface MetricsPanelProps {
  metrics: ValidationMetrics | null;
  enabled: boolean;
}

export function MetricsPanel({ metrics, enabled }: MetricsPanelProps) {
  if (!enabled) {
    return (
      <section className="panel metrics-panel muted">
        <h2>Validation metrics</h2>
        <p>Load validation ground truth to compare predicted vs actual CPU.</p>
      </section>
    );
  }

  if (!metrics || metrics.n === 0) {
    return (
      <section className="panel metrics-panel muted">
        <h2>Validation metrics</h2>
        <p>Run a forecast with validation data to see error metrics.</p>
      </section>
    );
  }

  const cards = [
    { label: "MAE", value: `${formatMetric(metrics.mae)}%`, hint: "Mean absolute error" },
    { label: "RMSE", value: `${formatMetric(metrics.rmse)}%`, hint: "Root mean squared error" },
    { label: "MAPE", value: `${formatMetric(metrics.mape)}%`, hint: "Mean absolute % error (non-zero actuals)" },
    { label: "Max |error|", value: `${formatMetric(metrics.maxError)}%`, hint: "Worst step" },
    { label: "Bias", value: `${formatMetric(metrics.bias)}%`, hint: "Mean signed error (+ over-predict)" },
    { label: "Steps", value: String(metrics.n), hint: "Compared forecast steps" },
  ];

  return (
    <section className="panel metrics-panel">
      <h2>Validation metrics</h2>
      <p className="subtitle">Predicted vs held-out actual CPU ({metrics.n} steps)</p>
      <div className="metric-grid">
        {cards.map((c) => (
          <div key={c.label} className="metric-card">
            <span className="metric-label">{c.label}</span>
            <span className="metric-value">{c.value}</span>
            <span className="metric-hint">{c.hint}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
