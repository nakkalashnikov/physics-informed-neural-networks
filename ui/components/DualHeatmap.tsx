"use client";

import dynamic from "next/dynamic";
import type { HeatmapResponse, QueryPoint } from "@/lib/api";

// Plotly has no SSR support — import client-side only
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  data: HeatmapResponse;
  query: QueryPoint;
}

const DARK_BG  = "#13151f";
const CARD_BG  = "#1a1d27";
const GRID_COL = "#1e2130";
const AXIS_COL = "#4a5568";

const LAYOUT_BASE: Partial<Plotly.Layout> = {
  paper_bgcolor: CARD_BG,
  plot_bgcolor:  CARD_BG,
  font:   { color: "#64748b", family: "ui-monospace, monospace", size: 11 },
  margin: { t: 36, l: 52, r: 16, b: 48 },
  xaxis: {
    title: { text: "x  [m]", standoff: 8 },
    color: AXIS_COL,
    gridcolor: GRID_COL,
    zerolinecolor: GRID_COL,
  },
  yaxis: {
    title: { text: "t  [s]", standoff: 8 },
    color: AXIS_COL,
    gridcolor: GRID_COL,
    zerolinecolor: GRID_COL,
  },
};

const PLOT_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

// Shared colourbar config
function colorbar(title: string): Partial<Plotly.ColorBar> {
  return {
    title: { text: title, font: { color: "#64748b", size: 10 } },
    tickfont: { color: "#64748b", size: 9 },
    thickness: 12,
    len: 0.9,
  };
}

export default function DualHeatmap({ data, query }: Props) {
  const { x_grid, t_grid, delta_T_pinn, delta_T_analytical, burner_positions } = data;

  // Global ΔT range so both heatmaps share the same colour scale
  const allVals = [...delta_T_pinn.flat(), ...delta_T_analytical.flat()];
  const zMin = Math.min(...allVals);
  const zMax = Math.max(...allVals);

  // Absolute error grid
  const errorZ = delta_T_pinn.map((row, i) =>
    row.map((v, j) => Math.abs(v - delta_T_analytical[i][j]))
  );
  const errMax = Math.max(...errorZ.flat());

  // Burner trajectory + query-point marker (reused in both main plots)
  const burnerTrace: Partial<Plotly.ScatterData> = {
    type:      "scatter",
    x:         burner_positions,
    y:         t_grid,
    mode:      "lines",
    line:      { color: "#f97316", width: 1.5, dash: "dot" },
    name:      "Burner",
    hovertemplate: "x_b = %{x:.2f} m<br>t = %{y:.1f} s<extra>Burner</extra>",
  };

  const queryTrace: Partial<Plotly.ScatterData> = {
    type:   "scatter",
    x:      [query.x_query],
    y:      [query.t_query],
    mode:   "markers",
    marker: { color: "#60a5fa", size: 9, symbol: "cross", line: { color: "#fff", width: 1.5 } },
    name:   "Query",
    hovertemplate: "x = %{x:.2f} m<br>t = %{y:.1f} s<extra>Query point</extra>",
  };

  function heatTrace(z: number[][], title: string, cs: Plotly.ColorScale): Partial<Plotly.HeatmapData> {
    return {
      type:      "heatmap",
      x:         x_grid,
      y:         t_grid,
      z,
      zmin:      zMin,
      zmax:      zMax,
      colorscale: cs,
      colorbar:  colorbar("ΔT  [K]"),
      hovertemplate: "x = %{x:.2f} m<br>t = %{y:.1f} s<br>ΔT = %{z:.2f} K<extra>" + title + "</extra>",
    };
  }

  const pinnPlot = {
    data:   [heatTrace(delta_T_pinn, "PINN", "Plasma"), burnerTrace, queryTrace],
    layout: { ...LAYOUT_BASE, title: { text: "PINN  —  predicted  ΔT(x, t)", font: { color: "#cbd5e1", size: 12 } } },
    config: PLOT_CONFIG,
  };

  const refPlot = {
    data:   [heatTrace(delta_T_analytical, "Analytical", "Plasma"), burnerTrace, queryTrace],
    layout: { ...LAYOUT_BASE, title: { text: "Analytical  —  exact  ΔT(x, t)", font: { color: "#cbd5e1", size: 12 } } },
    config: PLOT_CONFIG,
  };

  const errPlot = {
    data: [{
      type:      "heatmap" as const,
      x:         x_grid,
      y:         t_grid,
      z:         errorZ,
      zmin:      0,
      zmax:      errMax,
      colorscale: [[0, CARD_BG], [0.5, "#7c3aed"], [1, "#ef4444"]] as Plotly.ColorScale,
      colorbar:  colorbar("|err|  [K]"),
      hovertemplate: "x = %{x:.2f} m<br>t = %{y:.1f} s<br>|err| = %{z:.3f} K<extra>Error</extra>",
    }, burnerTrace, queryTrace],
    layout: { ...LAYOUT_BASE, title: { text: "|PINN − Analytical|", font: { color: "#cbd5e1", size: 12 } } },
    config: PLOT_CONFIG,
  };

  return (
    <div className="space-y-4">
      {/* Dual heatmap row */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <Plot {...pinnPlot} style={{ width: "100%", height: 320 }} />
        </Card>
        <Card>
          <Plot {...refPlot} style={{ width: "100%", height: 320 }} />
        </Card>
      </div>

      {/* Error heatmap */}
      <Card>
        <Plot {...errPlot} style={{ width: "100%", height: 220 }} />
      </Card>

      {/* Legend */}
      <div className="flex gap-6 px-1 text-[11px] text-slate-600">
        <LegendItem color="border-orange-500 border-dashed" label="Burner trajectory" />
        <LegendItem color="bg-blue-400 rounded-full" label="Query point" isCircle />
      </div>
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-[#1a1d27] border border-white/[0.06] rounded-xl overflow-hidden">
      {children}
    </div>
  );
}

function LegendItem({ color, label, isCircle = false }: { color: string; label: string; isCircle?: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <span className={isCircle ? `inline-block w-2.5 h-2.5 ${color}` : `inline-block w-7 border-t-2 ${color}`} />
      {label}
    </span>
  );
}
