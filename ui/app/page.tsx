"use client";

import { useState, useCallback } from "react";
import ParameterPanel from "@/components/ParameterPanel";
import DualHeatmap from "@/components/DualHeatmap";
import QueryResult from "@/components/QueryResult";
import {
  predict,
  heatmap,
  type Params,
  type QueryPoint,
  type PredictResponse,
  type HeatmapResponse,
} from "@/lib/api";

const DEFAULT_PARAMS: Params = {
  material: "steel",
  length: 2.0,
  intensity: 5000,
  x0: 0.2,
  velocity: 0.05,
  t_total: 30,
};

const DEFAULT_QUERY: QueryPoint = { x_query: 1.0, t_query: 15 };

export default function Home() {
  const [params, setParams]       = useState<Params>(DEFAULT_PARAMS);
  const [query, setQuery]         = useState<QueryPoint>(DEFAULT_QUERY);
  const [heatmapData, setHeatmap] = useState<HeatmapResponse | null>(null);
  const [predictData, setPredict] = useState<PredictResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const handleCompute = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [hData, pData] = await Promise.all([
        heatmap(params),
        predict(params, query),
      ]);
      setHeatmap(hData);
      setPredict(pData);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setLoading(false);
    }
  }, [params, query]);

  return (
    <div className="flex h-screen overflow-hidden">
      <ParameterPanel
        params={params}
        query={query}
        onParamsChange={setParams}
        onQueryChange={setQuery}
        onCompute={handleCompute}
        loading={loading}
      />

      <main className="flex-1 overflow-y-auto p-6 space-y-5">
        {/* Header */}
        <header className="flex items-baseline gap-4">
          <h1 className="text-xl font-semibold tracking-tight">
            Heat Equation — PINN vs Analytical
          </h1>
          <p className="text-xs font-mono text-slate-500">
            ∂T/∂t = α·∂²T/∂x² + (i/ρc)·δ(x − x_b(t))
          </p>
        </header>

        {/* Error */}
        {error && (
          <div className="bg-red-950/60 border border-red-700 text-red-300 rounded-xl px-4 py-3 text-sm">
            <span className="font-semibold">Error: </span>{error}
          </div>
        )}

        {/* Results */}
        {predictData && (
          <QueryResult data={predictData} params={{ ...params, ...query }} />
        )}

        {heatmapData ? (
          <DualHeatmap data={heatmapData} query={query} />
        ) : (
          <EmptyState loading={loading} />
        )}
      </main>
    </div>
  );
}

function EmptyState({ loading }: { loading: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] border border-dashed border-slate-700/60 rounded-2xl gap-4 text-slate-600">
      {loading ? (
        <>
          <Spinner />
          <p className="text-sm">Computing…</p>
        </>
      ) : (
        <>
          <p className="text-3xl">⟶</p>
          <p className="text-sm">Set parameters and click <strong className="text-slate-400">Compute</strong></p>
        </>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div className="w-8 h-8 rounded-full border-2 border-slate-700 border-t-blue-400 animate-spin" />
  );
}
