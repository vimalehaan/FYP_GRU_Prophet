import type { ForecastRequest, ForecastResponse, ValidationGroundTruth } from "./types";

const DEFAULT_API = "/forecast-api";

export async function checkHealth(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.model_loaded === true;
  } catch {
    return false;
  }
}

export async function runForecast(
  baseUrl: string,
  request: ForecastRequest,
): Promise<ForecastResponse> {
  const res = await fetch(`${baseUrl}/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  const data = await res.json();
  if (!res.ok) {
    const message = data.message ?? data.detail ?? res.statusText;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return data as ForecastResponse;
}

export async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json() as Promise<T>;
}

export async function loadPresetRequest(path: string): Promise<ForecastRequest> {
  return loadJson<ForecastRequest>(path);
}

export async function loadValidation(path: string): Promise<ValidationGroundTruth> {
  return loadJson<ValidationGroundTruth>(path);
}

export function resolveApiBase(customUrl: string): string {
  const trimmed = customUrl.trim();
  return trimmed || DEFAULT_API;
}
