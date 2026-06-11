const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

export type Material = "steel" | "aluminum" | "copper" | "titanium";

export interface Params {
  material: Material;
  length: number;     // [m]
  intensity: number;  // [W/m]
  x0: number;        // [m]
  velocity: number;  // [m/s]
  t_total: number;   // [s]
}

export interface QueryPoint {
  x_query: number;   // [m]
  t_query: number;   // [s]
}

export interface PredictResponse {
  temperature_K: number;
  temperature_C: number;
  delta_T: number;
  analytical_K: number;
  analytical_delta_T: number;
  relative_error: number;
  T_amb: number;
  device: string;
}

export interface HeatmapResponse {
  x_grid: number[];             // (nx,) positions [m]
  t_grid: number[];             // (nt,) times [s]
  delta_T_pinn: number[][];     // (nt, nx) PINN ΔT [K]
  delta_T_analytical: number[][]; // (nt, nx) reference ΔT [K]
  burner_positions: number[];   // (nt,) x_b(t) [m]
  T_amb: number;
}

export interface MaterialInfo {
  name: string;
  alpha: number;
  rho_c: number;
  thermal_conductivity_approx: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      detail = JSON.parse(text)?.detail ?? text;
    } catch {}
    throw new Error(`[${res.status}] ${detail}`);
  }
  return res.json();
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function predict(
  params: Params,
  query: QueryPoint
): Promise<PredictResponse> {
  return post("/predict", { ...params, ...query });
}

export async function heatmap(
  params: Params,
  nx = 60,
  nt = 60
): Promise<HeatmapResponse> {
  return post("/heatmap", { ...params, nx, nt });
}

export async function fetchMaterials(): Promise<MaterialInfo[]> {
  const res = await fetch(`${API_BASE}/materials`);
  if (!res.ok) throw new Error("Could not load materials");
  const data = await res.json();
  return data.materials;
}
