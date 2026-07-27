import type { ValidationMetrics } from "./types";

export function computeValidationMetrics(
  actual: number[],
  predicted: number[],
): ValidationMetrics {
  const n = Math.min(actual.length, predicted.length);
  if (n === 0) {
    return { mae: 0, rmse: 0, mape: 0, maxError: 0, bias: 0, n: 0 };
  }

  const a = actual.slice(0, n);
  const p = predicted.slice(0, n);
  const errors = a.map((v, i) => p[i] - v);
  const absErrors = errors.map((e) => Math.abs(e));

  const mae = absErrors.reduce((s, v) => s + v, 0) / n;
  const rmse = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / n);
  const bias = errors.reduce((s, e) => s + e, 0) / n;
  const maxError = Math.max(...absErrors);

  const nonZero = a.filter((v) => v !== 0);
  let mape = 0;
  if (nonZero.length > 0) {
    const mapeSum = a.reduce((s, v, i) => {
      if (v === 0) return s;
      return s + Math.abs((p[i] - v) / v);
    }, 0);
    mape = (mapeSum / nonZero.length) * 100;
  }

  return { mae, rmse, mape, maxError, bias, n };
}

export function formatMetric(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
