import type { SignalAnalysis, Suitability } from "../signalAnalysis";
import { formatMetric } from "../metrics";

interface SignalAnalysisPanelProps {
  analysis: SignalAnalysis | null;
}

function badgeClass(level: Suitability): string {
  if (level === "High") return "badge high";
  if (level === "Moderate") return "badge moderate";
  return "badge low";
}

function difficultyClass(level: Suitability): string {
  // Low difficulty is good (green); High difficulty is concerning (red)
  if (level === "Low") return "badge high";
  if (level === "Moderate") return "badge moderate";
  return "badge low";
}

export function SignalAnalysisPanel({ analysis }: SignalAnalysisPanelProps) {
  if (!analysis) {
    return (
      <section className="panel signal-panel muted">
        <h2>Signal predictability</h2>
        <p>
          Load a container request with at least 50 history steps to assess whether
          the CPU series looks forecastable (ACF, Ljung–Box, suitability).
        </p>
      </section>
    );
  }

  const cards = [
    {
      label: "Predictability",
      value: analysis.predictability,
      hint: "Overall structure in history",
      badge: badgeClass(analysis.predictability),
    },
    {
      label: "Score",
      value: `${analysis.score.toFixed(0)}/100`,
      hint: "Heuristic only (not a formal metric)",
      badge: "",
    },
    {
      label: "Lag-1 ACF",
      value: formatMetric(analysis.lag1Acf),
      hint: "Short-term dependence",
      badge: "",
    },
    {
      label: "Lag-96 ACF",
      value:
        analysis.lag96Acf == null ? "—" : formatMetric(analysis.lag96Acf),
      hint: "Daily cycle (15-min × 96)",
      badge: "",
    },
    {
      label: "Ljung–Box",
      value: analysis.rejectWhiteNoise ? "Reject WN" : "Accept H₀",
      hint: `p=${
        analysis.ljungBoxPValue < 0.001
          ? analysis.ljungBoxPValue.toExponential(2)
          : analysis.ljungBoxPValue.toFixed(3)
      }`,
      badge: analysis.rejectWhiteNoise ? "badge high" : "badge low",
    },
    {
      label: "CV",
      value: Number.isFinite(analysis.cv) ? formatMetric(analysis.cv) : "—",
      hint: `μ=${formatMetric(analysis.mean)}% · σ=${formatMetric(analysis.std)}%`,
      badge: "",
    },
    {
      label: "Prophet fit",
      value: analysis.prophetSuitability,
      hint: "Seasonality / daily cycle outlook",
      badge: badgeClass(analysis.prophetSuitability),
    },
    {
      label: "GRU fit",
      value: analysis.gruSuitability,
      hint: "Sequential residual outlook",
      badge: badgeClass(analysis.gruSuitability),
    },
    {
      label: "Hybrid fit",
      value: analysis.hybridSuitability,
      hint: "Combined architecture outlook",
      badge: badgeClass(analysis.hybridSuitability),
    },
    {
      label: "Difficulty",
      value: analysis.forecastDifficulty,
      hint: "Expected Day-1 forecasting difficulty",
      badge: difficultyClass(analysis.forecastDifficulty),
    },
  ];

  return (
    <section className="panel signal-panel">
      <h2>Signal predictability</h2>
      <p className="subtitle">
        History diagnostics for <strong>{analysis.containerId}</strong> ({analysis.n}{" "}
        steps) — updates when you change container / request. Research heuristic aligned
        with the container signal analysis notebook.
      </p>

      <div className="metric-grid signal-grid">
        {cards.map((c) => (
          <div key={c.label} className="metric-card">
            <span className="metric-label">{c.label}</span>
            <span className={`metric-value ${c.badge}`.trim()}>{c.value}</span>
            <span className="metric-hint">{c.hint}</span>
          </div>
        ))}
      </div>

      <div className={`insight insight-${analysis.predictability.toLowerCase()}`}>
        <strong>Insight</strong>
        <p>{analysis.insight}</p>
      </div>
    </section>
  );
}
