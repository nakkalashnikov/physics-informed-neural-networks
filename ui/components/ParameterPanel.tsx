"use client";

import { useMemo } from "react";
import type { Params, QueryPoint, Material } from "@/lib/api";

interface Props {
  params: Params;
  query: QueryPoint;
  onParamsChange: (p: Params) => void;
  onQueryChange: (q: QueryPoint) => void;
  onCompute: () => void;
  loading: boolean;
}

const MATERIALS: { id: Material; label: string; hint: string }[] = [
  { id: "steel",    label: "Steel",    hint: "α=1.2e-5" },
  { id: "aluminum", label: "Aluminum", hint: "α=9.7e-5" },
  { id: "copper",   label: "Copper",   hint: "α=1.2e-4" },
  { id: "titanium", label: "Titanium", hint: "α=2.9e-6" },
];

export default function ParameterPanel({
  params, query, onParamsChange, onQueryChange, onCompute, loading,
}: Props) {
  // Max velocity that keeps the burner inside the pipe
  const vMax = useMemo(
    () => Math.max((params.length - params.x0) / params.t_total, 0.005),
    [params.length, params.x0, params.t_total]
  );

  function set<K extends keyof Params>(key: K, val: Params[K]) {
    onParamsChange({ ...params, [key]: val });
  }

  function setQ<K extends keyof QueryPoint>(key: K, val: number) {
    onQueryChange({ ...query, [key]: val });
  }

  return (
    <aside className="w-[17rem] flex-shrink-0 bg-[#13151f] border-r border-white/[0.06] flex flex-col">
      {/* Brand */}
      <div className="px-5 pt-5 pb-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_6px_rgba(96,165,250,0.8)]" />
          <h2 className="text-sm font-semibold tracking-tight">PINN Parameters</h2>
        </div>
        <p className="text-[11px] text-slate-500 pl-4">1D pipe · insulated ends</p>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">

        {/* Material selector */}
        <Section label="Material">
          <div className="grid grid-cols-2 gap-1.5">
            {MATERIALS.map(({ id, label, hint }) => (
              <button
                key={id}
                onClick={() => set("material", id)}
                className={`flex flex-col items-start px-2.5 py-2 rounded-lg border text-left transition-all
                  ${params.material === id
                    ? "border-blue-500/70 bg-blue-500/10 text-blue-300"
                    : "border-white/[0.07] text-slate-400 hover:border-white/20 hover:text-slate-300"
                  }`}
              >
                <span className="text-xs font-medium">{label}</span>
                <span className="text-[10px] font-mono opacity-60">{hint}</span>
              </button>
            ))}
          </div>
        </Section>

        {/* Pipe */}
        <Section label="Pipe">
          <SliderRow
            label="Length" unit="m"
            value={params.length} min={0.5} max={3.0} step={0.1}
            onChange={(v) => set("length", v)}
          />
          <SliderRow
            label="Intensity" unit="W/m"
            value={params.intensity} min={500} max={10000} step={100}
            onChange={(v) => set("intensity", v)}
          />
        </Section>

        {/* Burner */}
        <Section label="Burner motion">
          <SliderRow
            label="Start x₀" unit="m"
            value={params.x0} min={0} max={parseFloat((params.length * 0.4).toFixed(1))} step={0.05}
            onChange={(v) => set("x0", v)}
          />
          <SliderRow
            label="Velocity" unit="m/s"
            value={Math.min(params.velocity, vMax)} min={0.005} max={parseFloat(vMax.toFixed(3))} step={0.005}
            onChange={(v) => set("velocity", v)}
          />
          <SliderRow
            label="Duration" unit="s"
            value={params.t_total} min={10} max={60} step={1}
            onChange={(v) => set("t_total", v)}
          />
          {/* Live burner endpoint preview */}
          <div className="text-[10px] text-slate-600 font-mono pt-0.5">
            end pos = {(params.x0 + params.velocity * params.t_total).toFixed(2)} m
            &nbsp;/&nbsp;{params.length} m
          </div>
        </Section>

        {/* Query point */}
        <Section label="Query point">
          <SliderRow
            label="Position x" unit="m"
            value={Math.min(query.x_query, params.length)} min={0} max={params.length} step={0.05}
            onChange={(v) => setQ("x_query", v)}
          />
          <SliderRow
            label="Time t" unit="s"
            value={Math.min(query.t_query, params.t_total)} min={0} max={params.t_total} step={0.5}
            onChange={(v) => setQ("t_query", v)}
          />
        </Section>

      </div>

      {/* Compute button */}
      <div className="px-5 pb-5 pt-3 border-t border-white/[0.06]">
        <button
          onClick={onCompute}
          disabled={loading}
          className={`w-full py-2.5 rounded-xl text-sm font-semibold transition-all
            ${loading
              ? "bg-blue-500/30 text-blue-300/50 cursor-not-allowed"
              : "bg-blue-500 hover:bg-blue-400 active:scale-95 text-white shadow-[0_0_16px_rgba(96,165,250,0.25)]"
            }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-3.5 h-3.5 rounded-full border-2 border-blue-300/40 border-t-blue-300 animate-spin" />
              Computing…
            </span>
          ) : (
            "Compute →"
          )}
        </button>
      </div>
    </aside>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
        {label}
      </p>
      {children}
    </div>
  );
}

interface SliderRowProps {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}

function SliderRow({ label, unit, value, min, max, step, onChange }: SliderRowProps) {
  const decimals = step < 1 ? (step < 0.01 ? 3 : 2) : 0;
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-blue-300 tabular-nums">
          {value.toFixed(decimals)} <span className="text-slate-600">{unit}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
