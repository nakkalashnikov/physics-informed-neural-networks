"use client";

import type { PredictResponse, Params, QueryPoint } from "@/lib/api";

interface Props {
  data: PredictResponse;
  params: Params & QueryPoint;
}

export default function QueryResult({ data, params }: Props) {
  const errPct  = (data.relative_error * 100).toFixed(2);
  const errColor =
    data.relative_error < 0.05 ? "text-emerald-400" :
    data.relative_error < 0.15 ? "text-yellow-400"  :
    "text-red-400";

  const dT_pinn = data.delta_T.toFixed(2);
  const dT_ref  = data.analytical_delta_T.toFixed(2);

  return (
    <div className="bg-[#1a1d27] border border-white/[0.06] rounded-xl p-5">
      {/* Top row */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <p className="text-sm font-semibold">Point query result</p>
          <p className="text-[11px] text-slate-500 mt-0.5 font-mono">
            x = {params.x_query} m &nbsp;·&nbsp; t = {params.t_query} s
            &nbsp;·&nbsp; {params.material}
            &nbsp;·&nbsp; i = {params.intensity} W/m
          </p>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-mono font-bold ${errColor}`}>
            {errPct}%
          </p>
          <p className="text-[9px] uppercase tracking-widest text-slate-600 mt-0.5">
            relative error
          </p>
        </div>
      </div>

      {/* Temperature cards */}
      <div className="grid grid-cols-3 gap-3">
        <TempCard
          label="PINN"
          tempK={data.temperature_K}
          tempC={data.temperature_C}
          deltaT={dT_pinn}
          accent="blue"
        />
        <TempCard
          label="Analytical"
          tempK={data.analytical_K}
          tempC={data.analytical_K - 273.15}
          deltaT={dT_ref}
          accent="emerald"
        />
        <DiffCard
          pinnK={data.temperature_K}
          refK={data.analytical_K}
          errColor={errColor}
        />
      </div>

      {/* Footer */}
      <div className="flex justify-between mt-4 pt-3 border-t border-white/[0.04] text-[11px] text-slate-600 font-mono">
        <span>T_amb = {data.T_amb.toFixed(1)} K  ({(data.T_amb - 273.15).toFixed(1)} °C)</span>
        <span>device: {data.device}</span>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

const ACCENT_MAP = {
  blue:    { bg: "bg-blue-500/10",    border: "border-blue-500/20",    text: "text-blue-300",    sub: "text-blue-400/60" },
  emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/20", text: "text-emerald-300", sub: "text-emerald-400/60" },
};

function TempCard({
  label, tempK, tempC, deltaT, accent,
}: {
  label: string; tempK: number; tempC: number; deltaT: string; accent: "blue" | "emerald";
}) {
  const c = ACCENT_MAP[accent];
  return (
    <div className={`${c.bg} border ${c.border} rounded-xl p-3.5`}>
      <p className={`text-[10px] uppercase tracking-widest ${c.sub} mb-2`}>{label}</p>
      <p className={`text-xl font-mono font-bold ${c.text} leading-none`}>
        {tempK.toFixed(1)}<span className="text-sm font-normal ml-1">K</span>
      </p>
      <p className="text-xs text-slate-500 mt-1">{tempC.toFixed(1)} °C</p>
      <p className={`text-[10px] font-mono mt-2 ${c.sub}`}>ΔT = {deltaT} K</p>
    </div>
  );
}

function DiffCard({
  pinnK, refK, errColor,
}: {
  pinnK: number; refK: number; errColor: string;
}) {
  const diff = (pinnK - refK).toFixed(2);
  const sign = pinnK >= refK ? "+" : "";
  return (
    <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-3.5">
      <p className="text-[10px] uppercase tracking-widest text-slate-600 mb-2">Δ (PINN − ref)</p>
      <p className={`text-xl font-mono font-bold ${errColor} leading-none`}>
        {sign}{diff}<span className="text-sm font-normal ml-1">K</span>
      </p>
      <p className="text-xs text-slate-500 mt-1">&nbsp;</p>
      <p className={`text-[10px] font-mono mt-2 ${errColor}`}>
        {pinnK >= refK ? "overestimate" : "underestimate"}
      </p>
    </div>
  );
}
