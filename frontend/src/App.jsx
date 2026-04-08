import React, { useState } from "react";
import {
  Upload, Activity, Zap, Info, Loader2, RefreshCcw,
  Database, Cloud, TableProperties, LayoutPanelLeft
} from "lucide-react";

const NUTRIENT_META = {
  calories:       { label: "Calories",         unit: "kcal",   valueColor: "text-amber-400",  dotColor: "bg-amber-400" },
  fat:            { label: "Total Fat",         unit: "g",      valueColor: "text-orange-400", dotColor: "bg-orange-400" },
  saturated_fat:  { label: "Saturated Fat",     unit: "g",      valueColor: "text-red-400",    dotColor: "bg-red-400" },
  carbohydrates:  { label: "Carbohydrates",     unit: "g",      valueColor: "text-blue-400",   dotColor: "bg-blue-400" },
  sugars:         { label: "of which Sugars",   unit: "g",      valueColor: "text-violet-400", dotColor: "bg-violet-400" },
  fibre:          { label: "Dietary Fibre",     unit: "g",      valueColor: "text-emerald-400",dotColor: "bg-emerald-400" },
  protein:        { label: "Protein",           unit: "g",      valueColor: "text-cyan-400",   dotColor: "bg-cyan-400" },
  sodium:         { label: "Sodium",            unit: "g",      valueColor: "text-slate-400",  dotColor: "bg-slate-400" },
};

function getFallbackMeta(key) {
  return {
    label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
    unit: "",
    valueColor: "text-slate-300",
    dotColor: "bg-slate-500",
  };
}

export default function App() {
  const [image, setImage]       = useState(null);
  const [loading, setLoading]   = useState(false);
  const [results, setResults]   = useState(null);
  const [activeTab, setActiveTab] = useState("per_100g");

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage({ file, preview: URL.createObjectURL(file) });
      setResults(null);
    }
  };

  const handleAnalyze = async () => {
    if (results) { setImage(null); setResults(null); return; }
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image.file);

    try {
      const response = await fetch("http://localhost:8000/analyze-label", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Pipeline Error");

      const data = await response.json();

      // ✅ Improvement 1: auto-select available tab
      if (data.per_100g) setActiveTab("per_100g");
      else if (data.per_serving) setActiveTab("per_serving");

      setResults(data);

    } catch (err) {
      alert("Pipeline Failure: Check Backend Terminal");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto">

        {/* HEADER */}
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-12 gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-cyan-500/20 p-2 rounded-lg border border-cyan-500/30">
              <Database className="h-8 w-8 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white">
                NutriScan <span className="text-cyan-400">Pipeline</span>
              </h1>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em]">
                Data Engineering Edition
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <StatusBadge icon={<Cloud className="h-3 w-3" />}    label="S3_LAKE" status="ACTIVE" color="text-cyan-500" />
            <StatusBadge icon={<Database className="h-3 w-3" />} label="DUCKDB"  status="SYNCED" color="text-purple-500" />
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* LEFT */}
          <div className="lg:col-span-4 space-y-6">
            <div className={`relative border-2 border-dashed rounded-3xl p-6 h-[400px] flex flex-col items-center justify-center transition-all
              ${image ? "border-cyan-500/40 bg-slate-900/30" : "border-slate-800 bg-slate-900/10"}`}>
              {!image ? (
                <>
                  <input
                    type="file"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    onChange={handleImageUpload}
                    accept="image/*"
                  />
                  <Upload className="h-8 w-8 text-slate-600 mb-2" />
                  <p className="text-slate-500 text-sm">Ingest Image</p>
                </>
              ) : (
                <img
                  src={image.preview}
                  alt="Preview"
                  className="max-h-full rounded-2xl shadow-2xl border border-slate-800"
                />
              )}
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading}
              className={`w-full py-4 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 shadow-xl
                ${results
                  ? "bg-slate-900 text-rose-400 border border-rose-900/30"
                  : "bg-cyan-500 text-slate-950"
                }`}
            >
              {loading
                ? <Loader2 className="animate-spin h-5 w-5" />
                : results
                  ? <RefreshCcw className="h-5 w-5" />
                  : <Zap className="h-5 w-5" />
              }
              {loading ? "PROCESSING..." : results ? "CLEAR_SESSION" : "RUN_EXTRACTION"}
            </button>
          </div>

          {/* RIGHT */}
          <div className="lg:col-span-8">
            {loading ? (
              <PipelineLoader />
            ) : results ? (
              <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8 space-y-6">

                {/* ✅ Improvement 2: disable tabs if missing */}
                <div className="flex p-1 bg-slate-950 border border-slate-800 rounded-xl w-full max-w-sm">
                  {["per_100g", "per_serving"].map((tab) => {
                    const hasData = !!results[tab];

                    return (
                      <button
                        key={tab}
                        onClick={() => hasData && setActiveTab(tab)}
                        disabled={!hasData}
                        className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all
                          ${!hasData
                            ? "text-slate-700 cursor-not-allowed"
                            : activeTab === tab
                              ? tab === "per_100g"
                                ? "bg-cyan-500 text-slate-950 shadow-lg"
                                : "bg-purple-500 text-slate-950 shadow-lg"
                              : "text-slate-500 hover:text-slate-300"
                          }`}
                      >
                        {tab === "per_100g" ? "PER 100G" : "PER SERVING"}
                      </button>
                    );
                  })}
                </div>

                {/* HEADER */}
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2">
                    <TableProperties className="h-4 w-4 text-slate-500" />
                    <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                      {activeTab === "per_100g" ? "Per 100g" : "Per Serving"} Schema
                    </h2>
                  </div>
                  {activeTab === "per_serving" && results.per_serving?.size && (
                    <span className="text-[10px] bg-purple-500/10 text-purple-400 px-3 py-1 rounded-full border border-purple-500/20 font-mono">
                      SIZE: {results.per_serving.size}
                    </span>
                  )}
                </div>

                <NutrientGrid data={results[activeTab]} />

                <details className="group border border-slate-800 rounded-xl bg-slate-950/30">
                  <summary className="p-3 text-[9px] font-mono text-slate-600 cursor-pointer list-none flex justify-between items-center uppercase">
                    <span>{">"} raw_data_payload</span>
                    <span className="group-open:rotate-180 transition-transform">▼</span>
                  </summary>
                  <div className="p-4 pt-0">
                    <pre className="text-[10px] text-cyan-800 font-mono overflow-x-auto">
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  </div>
                </details>
              </div>
            ) : (
              <div className="h-full border border-slate-900 bg-slate-900/5 rounded-3xl flex flex-col items-center justify-center p-12 text-center text-slate-700">
                <LayoutPanelLeft className="h-10 w-10 mb-4 opacity-10" />
                <p className="text-sm italic">Initialize Ingestion to View Analytics</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function NutrientGrid({ data }) {
  if (!data) {
    return (
      <div className="py-12 text-center border border-dashed border-slate-800 rounded-2xl">
        <span className="text-xs text-slate-600 italic">No data extracted</span>
      </div>
    );
  }

  const entries = Object.entries(data).filter(([key]) => key !== "size");

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {entries.map(([key, value]) => {
        const meta = NUTRIENT_META[key] ?? getFallbackMeta(key);

        return (
          <div key={key} className="bg-slate-950/50 border border-slate-800 p-5 rounded-2xl flex justify-between items-center">
            <div>
              <p className="text-[10px] font-bold text-slate-500 uppercase">{meta.label}</p>
            </div>
            <span className={`text-xl font-bold ${meta.valueColor}`}>{value}</span>
          </div>
        );
      })}
    </div>
  );
}

function StatusBadge({ icon, label, status, color }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border border-slate-800 rounded-lg bg-slate-900/50">
      <div className={color}>{icon}</div>
      <div className="flex flex-col">
        <span className="text-[8px] text-slate-600 font-bold">{label}</span>
        <span className={`text-[10px] font-mono ${color}`}>{status}</span>
      </div>
    </div>
  );
}

function PipelineLoader() {
  return (
    <div className="h-full flex flex-col items-center justify-center space-y-6">
      <Loader2 className="animate-spin text-cyan-400" />
      <p className="text-cyan-400 text-xs uppercase">Running Ingestion...</p>
    </div>
  );
}