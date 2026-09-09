/**
 * Lightweight container CPU signal diagnostics for the demo UI.
 * Aligned with notebooks/container_signal_analysis.ipynb heuristics.
 * Not a substitute for full thesis statistical analysis.
 */

export type Suitability = "High" | "Moderate" | "Low";

export interface SignalAnalysis {
  containerId: string;
  n: number;
  mean: number;
  median: number;
  std: number;
  cv: number;
  lag1Acf: number;
  lag96Acf: number | null;
  ljungBoxStat: number;
  ljungBoxPValue: number;
  rejectWhiteNoise: boolean;
  predictability: Suitability;
  prophetSuitability: Suitability;
  gruSuitability: Suitability;
  hybridSuitability: Suitability;
  forecastDifficulty: Suitability;
  score: number;
  insight: string;
}

const DAY_LAG = 96;
const LJUNG_LAGS = 20;
const ALPHA = 0.05;
const MIN_STEPS = 50;

function mean(xs: number[]): number {
  return xs.reduce((s, v) => s + v, 0) / xs.length;
}

function median(xs: number[]): number {
  const a = [...xs].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function stdSample(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(v);
}

/** Autocorrelation at lag k (Pearson-style). */
export function acfAtLag(series: number[], lag: number): number {
  const n = series.length;
  if (lag <= 0 || lag >= n) return Number.NaN;
  const m = mean(series);
  let num = 0;
  let den = 0;
  for (let t = 0; t < n; t++) {
    den += (series[t] - m) ** 2;
  }
  if (den <= 1e-18) return 0;
  for (let t = 0; t < n - lag; t++) {
    num += (series[t] - m) * (series[t + lag] - m);
  }
  return num / den;
}

function normalCdf(z: number): number {
  // Abramowitz & Stegun approximation
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/** Approximate chi-squared survival function via Wilson–Hilferty. */
function chi2PValue(q: number, df: number): number {
  if (!Number.isFinite(q) || q <= 0) return 1;
  if (df <= 0) return Number.NaN;
  const h = 2 / (9 * df);
  const z = (Math.pow(q / df, 1 / 3) - (1 - h)) / Math.sqrt(h);
  return 1 - normalCdf(z);
}

/** Ljung–Box test for lags 1..h. */
export function ljungBox(series: number[], h = LJUNG_LAGS): { statistic: number; pValue: number } {
  const n = series.length;
  const effectiveH = Math.min(h, n - 2);
  if (effectiveH < 1) return { statistic: 0, pValue: 1 };

  let q = 0;
  for (let k = 1; k <= effectiveH; k++) {
    const r = acfAtLag(series, k);
    q += (r * r) / (n - k);
  }
  const statistic = n * (n + 2) * q;
  return { statistic, pValue: chi2PValue(statistic, effectiveH) };
}

function clip01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function level(high: boolean, medium: boolean): Suitability {
  if (high) return "High";
  if (medium) return "Moderate";
  return "Low";
}

function buildInsight(a: Omit<SignalAnalysis, "insight">): string {
  const cid = a.containerId;
  if (a.predictability === "High" && a.rejectWhiteNoise) {
    const daily =
      a.lag96Acf != null && Math.abs(a.lag96Acf) > 0.15
        ? " and clear daily seasonality"
        : "";
    return (
      `Container ${cid} shows meaningful temporal dependency${daily}. ` +
      `Ljung–Box rejects white noise (p=${a.ljungBoxPValue.toExponential(2)}), ` +
      `so the history contains exploitable structure. ` +
      (a.prophetSuitability !== "Low"
        ? "Prophet should capture seasonal patterns well; the GRU can use short-term residual dependence. "
        : "Daily seasonality is limited, so Prophet may mainly smooth the level; GRU residual learning may still help. ") +
      "Overall, this container looks comparatively forecastable with Hybrid Prophet + GRU."
    );
  }

  if (a.predictability === "Low" && !a.rejectWhiteNoise) {
    return (
      `Container ${cid} shows weak temporal dependency and behaviour close to white noise. ` +
      `Ljung–Box cannot reject white noise (p=${a.ljungBoxPValue.toFixed(3)}). ` +
      "Limited structure means forecast accuracy may stay constrained regardless of architecture."
    );
  }

  if (a.predictability === "Low") {
    return (
      `Container ${cid} has limited predictability (weak autocorrelation and/or near-constant variance). ` +
      "Hybrid forecasts may be difficult; larger errors can reflect the signal itself, not only the model."
    );
  }

  const lag96 =
    a.lag96Acf == null ? "n/a" : a.lag96Acf.toFixed(3);
  return (
    `Container ${cid} shows mixed temporal structure (predictability=${a.predictability.toLowerCase()}). ` +
    `Lag-1 ACF=${a.lag1Acf.toFixed(3)}, lag-96 ACF=${lag96}, ` +
    `Ljung–Box p=${a.ljungBoxPValue < 0.001 ? a.ljungBoxPValue.toExponential(2) : a.ljungBoxPValue.toFixed(3)}. ` +
    "Prophet and/or GRU may help partially; expect moderate, container-specific Day-1 accuracy."
  );
}

export function analyzeSignal(
  historicalCpu: number[],
  containerId: string,
): SignalAnalysis | null {
  const series = historicalCpu.filter((v) => Number.isFinite(v));
  if (series.length < MIN_STEPS) return null;

  const n = series.length;
  const m = mean(series);
  const med = median(series);
  const sd = stdSample(series);
  const cv = m !== 0 ? sd / Math.abs(m) : Number.NaN;
  const variance = sd * sd;

  const lag1 = acfAtLag(series, 1);
  const lag96 = n > DAY_LAG + 2 ? acfAtLag(series, DAY_LAG) : null;
  const lb = ljungBox(series, LJUNG_LAGS);
  const rejectWn = lb.pValue < ALPHA;

  // Significance band approximation
  const conf = 1.96 / Math.sqrt(n);

  let scorePoints = 0;
  if (Math.abs(lag1) >= 0.5) scorePoints += 2;
  else if (Math.abs(lag1) >= conf) scorePoints += 1;

  if (lag96 != null && Math.abs(lag96) >= 0.25) scorePoints += 2;
  else if (lag96 != null && Math.abs(lag96) > conf) scorePoints += 1;

  if (rejectWn) scorePoints += 2;

  if (variance < 1e-6 || (Number.isFinite(cv) && cv < 0.05)) {
    scorePoints = Math.max(0, scorePoints - 2);
  }

  const predictability: Suitability =
    scorePoints >= 5 ? "High" : scorePoints >= 3 ? "Moderate" : "Low";

  const prophetSuit = level(
    lag96 != null && Math.abs(lag96) >= 0.25,
    lag96 != null && Math.abs(lag96) > conf,
  );
  const gruSuit = level(Math.abs(lag1) >= 0.4 && rejectWn, Math.abs(lag1) > conf || rejectWn);

  let hybridSuit: Suitability;
  let difficulty: Suitability;
  if (predictability === "High" && (prophetSuit !== "Low" || gruSuit !== "Low")) {
    hybridSuit = "High";
    difficulty = "Low";
  } else if (predictability === "Low") {
    hybridSuit = "Low";
    difficulty = "High";
  } else {
    hybridSuit = "Moderate";
    difficulty = "Moderate";
  }

  const sLag1 = clip01(Math.abs(lag1));
  const sLag96 = lag96 == null ? 0 : clip01(Math.abs(lag96) / 0.5);
  const sLb = rejectWn ? 1 : 0;
  const sVar = variance < 1e-6 || (Number.isFinite(cv) && cv < 0.05) ? 0 : 1;
  const score = 100 * (0.3 * sLag1 + 0.3 * sLag96 + 0.3 * sLb + 0.1 * sVar);

  const base: Omit<SignalAnalysis, "insight"> = {
    containerId,
    n,
    mean: m,
    median: med,
    std: sd,
    cv,
    lag1Acf: lag1,
    lag96Acf: lag96,
    ljungBoxStat: lb.statistic,
    ljungBoxPValue: lb.pValue,
    rejectWhiteNoise: rejectWn,
    predictability,
    prophetSuitability: prophetSuit,
    gruSuitability: gruSuit,
    hybridSuitability: hybridSuit,
    forecastDifficulty: difficulty,
    score,
  };

  return { ...base, insight: buildInsight(base) };
}
