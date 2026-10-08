// Single place where the frontend talks to the backend. No fallbacks to fake data:
// failures surface as errors that pages render with a Retry action.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(`/api${path}`, {
      method,
      signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Cannot reach the backend API. Is the server running?", 0);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(`Malformed response from ${path} (HTTP ${res.status})`, res.status);
  }
  if (!res.ok) throw new ApiError(data?.detail || `Request failed (HTTP ${res.status})`, res.status);
  return data;
}

const qs = (params = {}) => {
  const p = Object.entries(params).filter(([, v]) => v !== "" && v !== undefined && v !== null);
  return p.length ? "?" + new URLSearchParams(p).toString() : "";
};

export const healthApi = { get: (o) => request("/health", o) };
export const overviewApi = { get: (o) => request("/overview", o) };
export const liveCallsApi = {
  list: (params, o) => request(`/live-calls${qs(params)}`, o),
  scenarios: (o) => request("/live-calls/scenarios", o),
  get: (id, o) => request(`/live-calls/${encodeURIComponent(id)}`, o),
  step: (id, body, o) => request(`/live-calls/${encodeURIComponent(id)}/step`, { method: "POST", body, ...o }),
  predict: (id, o) => request(`/live-calls/${encodeURIComponent(id)}/predict`, { method: "POST", ...o }),
  generate: (id, body, o) => request(`/live-calls/${encodeURIComponent(id)}/generate`, { method: "POST", body, ...o }),
};
export const customersApi = {
  list: (params, o) => request(`/customers${qs(params)}`, o),
  get: (id, o) => request(`/customers/${encodeURIComponent(id)}`, o),
};
export const agentsApi = {
  list: (o) => request("/agents", o),
  get: (id, o) => request(`/agents/${encodeURIComponent(id)}`, o),
};
export const treatmentApi = {
  options: (params, o) => request(`/treatment/options${qs(params)}`, o),
  get: (ptpId, o) => request(`/treatment/${encodeURIComponent(ptpId)}`, o),
};
export const analyticsApi = { get: (params, o) => request(`/analytics${qs(params)}`, o) };
export const auditApi = {
  list: (params, o) => request(`/audit${qs(params)}`, o),
  get: (id, o) => request(`/audit/${encodeURIComponent(id)}`, o),
};
