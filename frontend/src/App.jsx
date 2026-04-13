import React, { useState, useRef } from "react";
import {
  Upload, Zap, Loader2, RefreshCcw,
  Database, Cloud, TableProperties, LayoutPanelLeft, Scale, X, ChevronLeft, ChevronRight
} from "lucide-react";

// Use environment variable or localhost for development
const API_URL = import.meta.env.DEV ? "http://localhost:8000" : (import.meta.env.VITE_API_URL || "https://nutritionaltracker.onrender.com");

console.log("🔧 API_URL:", API_URL); // This helps debug

const NUTRIENT_META = {
  calories:      { label: "Calories",       unit: "kcal", valueColor: "text-amber-400"   },
  fat:           { label: "Total Fat",       unit: "g",    valueColor: "text-orange-400"  },
  saturated_fat: { label: "Saturated Fat",   unit: "g",    valueColor: "text-red-400"     },
  carbohydrates: { label: "Carbohydrates",   unit: "g",    valueColor: "text-blue-400"    },
  sugars:        { label: "of which Sugars", unit: "g",    valueColor: "text-violet-400"  },
  fibre:         { label: "Dietary Fibre",   unit: "g",    valueColor: "text-emerald-400" },
  protein:       { label: "Protein",         unit: "g",    valueColor: "text-cyan-400"    },
  sodium:        { label: "Sodium",          unit: "g",    valueColor: "text-slate-400"   },
};

function getFallbackMeta(key) {
  return {
    label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
    unit: "",
    valueColor: "text-slate-300",
  };
}

function parseNumeric(value) {
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const match = value.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : null;
  }
  return null;
}

function extractServingGrams(size) {
  if (!size) return null;
  const gMatch = size.match(/(\d+(\.\d+)?)\s*g/i);
  if (gMatch) return parseFloat(gMatch[1]);
  const numMatch = size.match(/(\d+(\.\d+)?)/);
  return numMatch ? parseFloat(numMatch[1]) : null;
}

export default function App() {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [activeTab, setActiveTab] = useState("per_100g");
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    const newImages = files.map(file => ({ file, preview: URL.createObjectURL(file) }));
    setImages(prev => [...prev, ...newImages]);
    setResults(null);
    setActiveIndex(0);
    e.target.value = "";
  };

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
    if (activeIndex >= images.length - 1) {
      setActiveIndex(Math.max(0, images.length - 2));
    }
    setResults(null);
  };

  const handleClear = () => {
    images.forEach(img => URL.revokeObjectURL(img.preview));
    setImages([]);
    setResults(null);
    setActiveIndex(0);
    setActiveTab("per_100g");
  };

  const switchToIndex = (i, resultArr) => {
    setActiveIndex(i);
    const r = (resultArr ?? results)?.[i];
    if (r?.per_100g && Object.keys(r.per_100g).length > 0) {
      setActiveTab("per_100g");
    } else if (r?.per_serving && Object.keys(r.per_serving).length > 0) {
      setActiveTab("per_serving");
    }
  };

  const handleAnalyze = async () => {
    if (results) {
      handleClear();
      return;
    }
    if (!images.length) return;

    setLoading(true);
    const formData = new FormData();
    images.forEach(img => formData.append("files", img.file));

    try {
      console.log("📤 Sending request to:", `${API_URL}/analyze-labels`);
      const response = await fetch(`${API_URL}/analyze-labels`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pipeline Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      const arr = Array.isArray(data) ? data : [data];
      setResults(arr);
      switchToIndex(0, arr);
    } catch (err) {
      console.error("❌ Pipeline Failure:", err);
      alert(`Pipeline Failure: ${err.message}\n\nMake sure backend is running on http://localhost:8000`);
    } finally {
      setLoading(false);
    }
  };

  const currentResult = results?.[activeIndex] ?? null;
  const currentPreview = images[activeIndex]?.preview ?? null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto">
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
            <StatusBadge icon={<Cloud className="h-3 w-3" />} label="API" status={API_URL} color="text-cyan-500" />
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Panel */}
          <div className="lg:col-span-4 space-y-4">
            {images.length === 0 ? (
              <div
                className="relative border-2 border-dashed border-slate-800 bg-slate-900/10 rounded-3xl p-6 h-[400px] flex flex-col items-center justify-center cursor-pointer hover:border-slate-700 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple />
                <Upload className="h-8 w-8 text-slate-600 mb-2" />
                <p className="text-slate-500 text-sm">Click to ingest images</p>
                <p className="text-slate-700 text-[10px] font-mono mt-1">SUPPORTS MULTIPLE FILES</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="relative border border-slate-800 bg-slate-900/30 rounded-3xl p-4 h-[280px] flex items-center justify-center overflow-hidden">
                  <img
                    src={currentPreview}
                    alt={`Preview ${activeIndex + 1}`}
                    className="max-h-full max-w-full rounded-2xl object-contain border border-slate-800"
                  />
                  {images.length > 1 && (
                    <>
                      <button 
                        onClick={() => setActiveIndex(i => Math.max(0, i - 1))} 
                        disabled={activeIndex === 0}
                        className="absolute left-3 top-1/2 -translate-y-1/2 bg-slate-900/80 border border-slate-700 rounded-full p-1 disabled:opacity-20 hover:border-slate-500 transition-colors"
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>
                      <button 
                        onClick={() => setActiveIndex(i => Math.min(images.length - 1, i + 1))} 
                        disabled={activeIndex === images.length - 1}
                        className="absolute right-3 top-1/2 -translate-y-1/2 bg-slate-900/80 border border-slate-700 rounded-full p-1 disabled:opacity-20 hover:border-slate-500 transition-colors"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </>
                  )}
                </div>

                <div className="flex gap-2 flex-wrap">
                  {images.map((img, i) => (
                    <div 
                      key={i} 
                      onClick={() => setActiveIndex(i)}
                      className={`relative w-14 h-14 rounded-xl overflow-hidden border cursor-pointer flex-shrink-0 transition-all
                        ${i === activeIndex ? "border-cyan-500/70" : "border-slate-800 opacity-60 hover:opacity-100"}`}
                    >
                      <img src={img.preview} alt={`Thumb ${i + 1}`} className="w-full h-full object-cover" />
                      {!results && (
                        <button 
                          onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                          className="absolute top-0.5 right-0.5 bg-slate-900/90 rounded-full p-0.5 hover:text-rose-400 transition-colors"
                        >
                          <X className="h-2.5 w-2.5" />
                        </button>
                      )}
                      <div className="absolute bottom-0.5 left-0.5 bg-slate-900/90 rounded text-[8px] font-mono text-slate-400 px-1">
                        {i + 1}
                      </div>
                    </div>
                  ))}
                  {!results && (
                    <div 
                      onClick={() => fileInputRef.current?.click()}
                      className="w-14 h-14 rounded-xl border border-dashed border-slate-700 flex items-center justify-center cursor-pointer hover:border-slate-500 transition-colors flex-shrink-0"
                    >
                      <Upload className="h-4 w-4 text-slate-600" />
                    </div>
                  )}
                </div>

                <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple />
                <p className="text-[10px] font-mono text-slate-600 text-center">
                  {images.length} IMAGE{images.length !== 1 ? "S" : ""} QUEUED
                </p>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={loading || images.length === 0}
              className={`w-full py-4 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 shadow-xl
                ${results
                  ? "bg-slate-900 text-rose-400 border border-rose-900/30"
                  : images.length === 0
                    ? "bg-slate-900 text-slate-700 border border-slate-800 cursor-not-allowed"
                    : "bg-cyan-500 text-slate-950 hover:bg-cyan-400"
                }`}
            >
              {loading ? <Loader2 className="animate-spin h-5 w-5" /> : results ? <RefreshCcw className="h-5 w-5" /> : <Zap className="h-5 w-5" />}
              {loading
                ? `PROCESSING ${images.length} IMAGE${images.length !== 1 ? "S" : ""}...`
                : results ? "CLEAR_SESSION"
                : `RUN_EXTRACTION${images.length > 1 ? ` (${images.length})` : ""}`}
            </button>
          </div>

          {/* Right Panel */}
          <div className="lg:col-span-8">
            {loading ? (
              <div className="h-full flex flex-col items-center justify-center space-y-6">
                <Loader2 className="animate-spin text-cyan-400 h-8 w-8" />
                <p className="text-cyan-400 text-xs uppercase">
                  Running Batch Ingestion{images.length > 1 ? ` — ${images.length} Images` : ""}...
                </p>
              </div>
            ) : results ? (
              <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8 space-y-6">
                {results.length > 1 && (
                  <div className="space-y-2">
                    <p className="text-[10px] font-mono text-slate-600">SELECT RESULT — IN UPLOAD ORDER</p>
                    <div className="flex gap-2 flex-wrap">
                      {results.map((_, i) => (
                        <button
                          key={i}
                          onClick={() => switchToIndex(i, results)}
                          className={`flex items-center gap-2 px-2 py-1.5 rounded-xl border transition-all
                            ${activeIndex === i
                              ? "border-cyan-500/50 bg-cyan-500/10"
                              : "border-slate-800 hover:border-slate-600"}`}
                        >
                          <div className={`w-10 h-10 rounded-lg overflow-hidden border flex-shrink-0
                            ${activeIndex === i ? "border-cyan-500/50" : "border-slate-700"}`}>
                            <img src={images[i]?.preview} alt={`Label ${i + 1}`} className="w-full h-full object-cover" />
                          </div>
                          <div className="text-left">
                            <div className={`text-[10px] font-mono font-bold ${activeIndex === i ? "text-cyan-400" : "text-slate-500"}`}>
                              IMG_{String(i + 1).padStart(2, "0")}
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-3 bg-slate-950/50 border border-slate-800 rounded-2xl p-3">
                  <div className="w-16 h-16 rounded-xl overflow-hidden border border-slate-700 flex-shrink-0">
                    <img src={currentPreview} alt="Active label" className="w-full h-full object-cover" />
                  </div>
                  <div>
                    <div className="text-[10px] font-mono text-slate-600">ANALYZING LABEL</div>
                    <div className="text-sm font-bold text-slate-300 mt-0.5">
                      Image {activeIndex + 1} of {results.length}
                    </div>
                  </div>
                </div>

                <div className="flex p-1 bg-slate-950 border border-slate-800 rounded-xl w-full max-w-sm">
                  {["per_100g", "per_serving"].map((tab) => {
                    const hasData = currentResult?.[tab] && Object.keys(currentResult[tab]).length > 0;
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

                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2">
                    <TableProperties className="h-4 w-4 text-slate-500" />
                    <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                      {activeTab === "per_100g" ? "Per 100g" : "Per Serving"} Schema
                    </h2>
                  </div>
                  {activeTab === "per_serving" && currentResult?.per_serving?.size && (
                    <span className="text-[10px] bg-purple-500/10 text-purple-400 px-3 py-1 rounded-full border border-purple-500/20 font-mono">
                      SIZE: {currentResult.per_serving.size}
                    </span>
                  )}
                </div>

                <NutrientGrid
                  key={activeIndex}
                  data={currentResult?.[activeTab]}
                  activeTab={activeTab}
                  per100gData={currentResult?.per_100g}
                />

                <details className="group border border-slate-800 rounded-xl bg-slate-950/30">
                  <summary className="p-3 text-[9px] font-mono text-slate-600 cursor-pointer list-none flex justify-between items-center uppercase">
                    <span>{">"} raw_data_payload [{activeIndex + 1}/{results.length}]</span>
                    <span className="group-open:rotate-180 transition-transform">▼</span>
                  </summary>
                  <div className="p-4 pt-0">
                    <pre className="text-[10px] text-cyan-800 font-mono overflow-x-auto">
                      {JSON.stringify(currentResult, null, 2)}
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

function NutrientGrid({ data, activeTab, per100gData }) {
  const [customGrams, setCustomGrams] = useState("");

  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="py-12 text-center border border-dashed border-slate-800 rounded-2xl">
        <span className="text-xs text-slate-600 italic">No data extracted for this view</span>
      </div>
    );
  }

  const customG = parseFloat(customGrams);
  const isValid = !isNaN(customG) && customG > 0;

  let baseGrams = null;
  let scaleFrom = data;
  let baseLabel = null;
  let warnMsg = null;

  if (activeTab === "per_100g") {
    baseGrams = 100;
    baseLabel = "100g";
  } else {
    const servingG = extractServingGrams(data.size);
    if (servingG) {
      baseGrams = servingG;
      baseLabel = `${data.size} (${servingG}g)`;
    } else if (per100gData && Object.keys(per100gData).length > 0) {
      baseGrams = 100;
      scaleFrom = per100gData;
      baseLabel = "100g (fallback — no serving size on label)";
      warnMsg = "NO SERVING SIZE DETECTED — SCALING FROM PER_100G BASE";
    } else {
      warnMsg = "CANNOT SCALE — NO SIZE OR PER_100G DATA AVAILABLE";
    }
  }

  const scalingActive = isValid && baseGrams !== null;
  const factor = scalingActive ? customG / baseGrams : null;

  const getDisplay = (key, rawValue) => {
    if (!scalingActive) return { display: rawValue, adjusted: false };
    const sourceVal = scaleFrom[key] ?? rawValue;
    const base = parseNumeric(sourceVal);
    if (base === null) return { display: rawValue, adjusted: false };
    const scaled = parseFloat((base * factor).toFixed(2));
    return { display: scaled, adjusted: true, baseDisplay: rawValue };
  };

  const entries = Object.entries(data).filter(([key]) => key !== "size");

  return (
    <div className="space-y-5">
      <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Scale className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
            Custom Serving Calculator
          </span>
        </div>
        <div className="flex gap-3 items-center">
          <input
            type="number"
            min="1"
            step="any"
            placeholder="Enter your serving size, e.g. 58"
            value={customGrams}
            onChange={(e) => setCustomGrams(e.target.value)}
            className="flex-1 p-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200
                       placeholder-slate-600 focus:outline-none focus:border-cyan-500/60 transition-colors"
          />
          <span className="text-slate-400 text-sm font-mono font-bold">g</span>
          {customGrams && (
            <button onClick={() => setCustomGrams("")}
              className="text-[10px] text-slate-600 hover:text-rose-400 transition-colors font-mono px-2">
              CLEAR
            </button>
          )}
        </div>
        <div className="text-[10px] font-mono leading-relaxed">
          {warnMsg ? (
            <span className="text-amber-700">{warnMsg}</span>
          ) : (
            <span className="text-slate-600">
              BASE: {baseLabel}
              {scalingActive && (
                <span className="text-cyan-700 ml-2">
                  → {customG}g | FACTOR: ×{factor.toFixed(4)}
                </span>
              )}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {entries.map(([key, value]) => {
          const meta = NUTRIENT_META[key] ?? getFallbackMeta(key);
          const { display, adjusted, baseDisplay } = getDisplay(key, value);
          return (
            <div key={key}
              className={`bg-slate-950/50 border rounded-2xl p-5 flex justify-between items-center transition-all duration-200
                ${adjusted ? "border-cyan-500/30" : "border-slate-800"}`}>
              <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase">{meta.label}</p>
                {adjusted && (
                  <p className="text-[10px] font-mono text-slate-700 mt-0.5">base: {baseDisplay}</p>
                )}
              </div>
              <div className="text-right">
                <div className="flex items-baseline gap-1 justify-end">
                  <span className={`text-xl font-bold ${meta.valueColor}`}>{display}</span>
                  {meta.unit && <span className="text-[10px] text-slate-600 font-mono">{meta.unit}</span>}
                </div>
                {adjusted && (
                  <div className="text-[9px] text-cyan-700 font-mono mt-0.5 tracking-wider">ADJUSTED</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
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