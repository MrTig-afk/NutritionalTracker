import React, { useState, useRef, useEffect, useCallback } from "react";
import { apiFetch, runAnalysis } from "../lib/api";
import { applyPipelineToFile, createPreviewUrl } from "../lib/nutrition";
import { card, inputStyle, primaryBtn, mintBtn, ghostBtn, pillRow } from "../styles";
import { Icon, Spin } from "../components/Icon";
import ImageCropper from "../components/ImageCropper";
import SaveToFolderModal from "../components/SaveToFolderModal";
import NutrientGrid from "../components/NutrientGrid";

export default function ScanTab({ onAddToLog, onBusyChange }) {
  const [images, setImages] = useState([]);
  const [optimizedFiles, setOptimizedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [activeTab, setActiveTab] = useState("per_100g");
  const [cropperQueue, setCropperQueue] = useState([]);
  const [cropperFile, setCropperFile] = useState(null);
  const [saveModal, setSaveModal] = useState(null);
  const [logName, setLogName] = useState("");
  const [fileInputKey, setFileInputKey] = useState(0);
  const [usage, setUsage] = useState(null);
  const fileInputRef = useRef(null);
  const accumulatedOptimizedRef = useRef([]);
  const accumulatedImagesRef    = useRef([]);

  const fetchUsage = useCallback(async () => {
    try { setUsage(await apiFetch("/usage")); } catch (_) {}
  }, []);

  useEffect(() => { fetchUsage(); }, [fetchUsage]);
  useEffect(() => { if (!loading && results) fetchUsage(); }, [loading, results, fetchUsage]);
  useEffect(() => { onBusyChange?.(loading || !!saveModal || !!cropperFile); }, [loading, saveModal, cropperFile, onBusyChange]);

  const stateRef = useRef({ images, optimizedFiles, results, activeIndex, activeTab });
  useEffect(() => { stateRef.current = { images, optimizedFiles, results, activeIndex, activeTab }; }, [images, optimizedFiles, results, activeIndex, activeTab]);
  useEffect(() => {
    const handler = () => { if (document.visibilityState === "visible") { const s = stateRef.current; setImages(s.images); setOptimizedFiles(s.optimizedFiles); setResults(s.results); setActiveIndex(s.activeIndex); setActiveTab(s.activeTab); } };
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);

  const switchToIndex = useCallback((i, resultArr) => {
    setActiveIndex(i); setLogName("");
    const r = (resultArr ?? results)?.[i];
    if (r?.per_100g && Object.keys(r.per_100g).length > 0) setActiveTab("per_100g");
    else if (r?.per_serving && Object.keys(r.per_serving).length > 0) setActiveTab("per_serving");
  }, [results]);

  const handleImageUpload = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setCropperQueue(files); setCropperFile(files[0]);
  }, []);

  const handleCropConfirm = useCallback(async (cropData) => {
    const file = cropperFile; const remaining = cropperQueue.slice(1);
    setLoadingMsg("Processing...");
    const previewUrl = createPreviewUrl(file);
    const optimized  = await applyPipelineToFile(file, cropData);
    const imageEntry = { file, preview: previewUrl, cropData, persistentUrl: null };
    setImages(prev => [...prev, imageEntry]); setOptimizedFiles(prev => [...prev, optimized]);
    setResults(null); setError(null); setActiveIndex(0); setLoadingMsg("");
    accumulatedOptimizedRef.current = [...accumulatedOptimizedRef.current, optimized];
    accumulatedImagesRef.current    = [...accumulatedImagesRef.current, imageEntry];
    if (remaining.length > 0) { setCropperQueue(remaining); setCropperFile(remaining[0]); }
    else { setCropperQueue([]); setCropperFile(null); }
  }, [cropperFile, cropperQueue]);

  const handleCropCancel = useCallback(() => { accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = []; setCropperQueue([]); setCropperFile(null); }, []);

  const handleReCrop = useCallback((index) => {
    const img = images[index]; if (!img) return;
    setImages(prev => prev.filter((_, i) => i !== index)); setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setResults(null); setError(null); accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setCropperQueue([img.file]); setCropperFile(img.file);
  }, [images]);

  const removeImage = useCallback((index) => {
    setImages(prev => { const img = prev[index]; if (img?.preview && img.preview.startsWith("blob:") && !img.persistentUrl) URL.revokeObjectURL(img.preview); return prev.filter((_, i) => i !== index); });
    setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setActiveIndex(prev => (prev >= index && prev > 0 ? prev - 1 : prev)); setResults(null); setError(null);
  }, []);

  const handleClear = useCallback(() => {
    images.forEach(img => { if (img?.preview && img.preview.startsWith("blob:") && !img.persistentUrl) URL.revokeObjectURL(img.preview); });
    accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setImages([]); setOptimizedFiles([]); setResults(null); setError(null); setActiveIndex(0); setActiveTab("per_100g");
    setFileInputKey(k => k + 1);
  }, [images]);

  const handleAnalyze = useCallback(async () => {
    if (results) { handleClear(); return; } if (!images.length) return;
    const filesToSend = accumulatedOptimizedRef.current.length === images.length ? accumulatedOptimizedRef.current : optimizedFiles.length === images.length ? optimizedFiles : images.map(i => i.file);
    runAnalysis({ optimizedFiles: filesToSend, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex });
  }, [images, optimizedFiles, results, handleClear, switchToIndex]);

  const currentResult  = results?.[activeIndex] ?? null;
  const currentPreview = images[activeIndex]?.preview || images[activeIndex]?.persistentUrl || accumulatedImagesRef.current[activeIndex]?.preview || null;
  const allOptimized   = optimizedFiles.length === images.length && images.length > 0;

  return (
    <>
      {cropperFile && <ImageCropper file={cropperFile} onConfirm={handleCropConfirm} onCancel={handleCropCancel} />}
      {saveModal && <SaveToFolderModal result={saveModal.result} imageId={saveModal.imageId} onClose={() => setSaveModal(null)} onSaved={(name) => setLogName(name)} initialName={saveModal.name || ""} />}

      <div style={
        !images.length && !results
          ? { display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "stretch", minHeight: "62vh", width: "100%", maxWidth: 560, margin: "0 auto", gap: 16 }
          : { display: "grid", gap: 16, ...(!results && { maxWidth: 560, margin: "0 auto", width: "100%" }) }
      } className={results ? "ns-scan-grid" : ""}>
        {/* Upload / preview panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {images.length === 0 ? (
            <div style={{ border: "2px dashed var(--border)", borderRadius: 20, background: "var(--white)", padding: "48px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10, cursor: "pointer" }}
              onClick={() => fileInputRef.current?.click()}>
              <input key={fileInputKey} ref={fileInputRef} type="file" style={{ display: "none" }} onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
              <div style={{ width: 52, height: 52, borderRadius: 14, background: "var(--teal-lt)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon n="upload" size={22} style={{ color: "var(--teal)" }} />
              </div>
              <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)" }}>Upload nutrition label</div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>Crop → optimize → analyze</div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ position: "relative", border: "1px solid var(--border)", background: "var(--white)", borderRadius: 20, padding: 12, height: 280, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
                {currentPreview
                  ? <img src={currentPreview} alt="Preview" style={{ maxHeight: "100%", maxWidth: "100%", borderRadius: 12, objectFit: "contain" }} />
                  : <div style={{ color: "var(--muted)", fontSize: 13 }}>No preview</div>
                }
                {images.length > 1 && (
                  <>
                    <button onClick={() => setActiveIndex(i => Math.max(0, i - 1))} disabled={activeIndex === 0}
                      style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", width: 32, height: 32, borderRadius: "50%", background: "var(--white)", border: "1px solid var(--border)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", opacity: activeIndex === 0 ? 0.3 : 1 }}>
                      <Icon n="chevron_left" size={14} />
                    </button>
                    <button onClick={() => setActiveIndex(i => Math.min(images.length - 1, i + 1))} disabled={activeIndex === images.length - 1}
                      style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", width: 32, height: 32, borderRadius: "50%", background: "var(--white)", border: "1px solid var(--border)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", opacity: activeIndex === images.length - 1 ? 0.3 : 1 }}>
                      <Icon n="chevron_right" size={14} />
                    </button>
                  </>
                )}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {images.map((img, i) => {
                  const thumbSrc = img.preview || img.persistentUrl || accumulatedImagesRef.current[i]?.preview || null;
                  return (
                    <div key={i} onClick={() => setActiveIndex(i)} style={{ position: "relative", width: 56, height: 56, borderRadius: 12, overflow: "hidden", border: `2px solid ${i === activeIndex ? "var(--teal)" : "var(--border)"}`, cursor: "pointer", flexShrink: 0, opacity: i === activeIndex ? 1 : 0.6 }}>
                      {thumbSrc ? <img src={thumbSrc} alt={`Thumb ${i+1}`} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <div style={{ width: "100%", height: "100%", background: "var(--off)" }} />}
                      {!results && (
                        <>
                          <button onClick={e => { e.stopPropagation(); removeImage(i); }} style={{ position: "absolute", top: 2, right: 2, width: 16, height: 16, borderRadius: "50%", background: "white", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="close" size={9} /></button>
                          <button onClick={e => { e.stopPropagation(); handleReCrop(i); }} style={{ position: "absolute", bottom: 2, right: 2, width: 16, height: 16, borderRadius: "50%", background: "white", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="crop" size={9} /></button>
                        </>
                      )}
                      <div style={{ position: "absolute", bottom: 2, left: 4, fontSize: 9, fontWeight: 700, color: "white", textShadow: "0 1px 2px rgba(0,0,0,0.5)" }}>{i + 1}</div>
                    </div>
                  );
                })}
                {!results && (
                  <div onClick={() => fileInputRef.current?.click()} style={{ width: 56, height: 56, borderRadius: 12, border: "2px dashed var(--border)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", flexShrink: 0 }}>
                    <Icon n="upload" size={16} style={{ color: "var(--muted)" }} />
                  </div>
                )}
              </div>
              <input key={fileInputKey} ref={fileInputRef} type="file" style={{ display: "none" }} onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
              <p style={{ fontSize: 11, color: "var(--muted)", textAlign: "center" }}>
                {images.length} image{images.length !== 1 ? "s" : ""} queued
                {allOptimized && !loading && !results && <span style={{ color: "var(--teal)", marginLeft: 6 }}>· optimized ✓</span>}
                {loadingMsg && <span style={{ color: "var(--orange)", marginLeft: 6 }}>· {loadingMsg}</span>}
              </p>
            </div>
          )}

          <button onClick={handleAnalyze} disabled={loading || images.length === 0 || !!cropperFile}
            style={{ ...( results ? { ...ghostBtn, width: "100%", background: "var(--brown-lt)", borderColor: "var(--brown)", color: "var(--brown)" } : images.length === 0 || !!cropperFile ? { ...primaryBtn, opacity: 0.35, cursor: "not-allowed" } : primaryBtn ) }}>
            {loading ? <Spin size={16} color="white" /> : results ? <Icon n="refresh" size={16} /> : <Icon n="bolt" size={16} />}
            {loading ? (loadingMsg || `Processing ${images.length} image${images.length !== 1 ? "s" : ""}...`) : results ? "Clear session" : `Analyze${images.length > 1 ? ` (${images.length})` : ""}`}
          </button>

          {usage && (
            <p style={{ textAlign: "center", fontSize: 11, fontWeight: 600, margin: 0,
              color: usage.used >= usage.limit ? "var(--danger)" : "var(--muted)" }}>
              {usage.used} / {usage.limit} scans today
            </p>
          )}

          {error && !loading && images.length > 0 && !results && (
            <button onClick={handleAnalyze} style={{ ...ghostBtn, width: "100%", borderColor: "var(--orange)", color: "var(--orange)" }}>
              <Icon n="refresh" size={14} /> Retry (images preserved)
            </button>
          )}

          {currentResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <input value={logName} onChange={e => setLogName(e.target.value)} placeholder="Name this item before logging..." style={inputStyle} />
              <button onClick={() => onAddToLog({ name: logName.trim() || `Label ${activeIndex + 1}`, nutrition: currentResult })} style={mintBtn}>
                <Icon n="add" size={15} /> Log this item
              </button>
            </div>
          )}
        </div>

        {/* Results panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {error && !loading && (
            <div style={{ background: "#FDF0DC", border: "1px solid var(--orange)", borderRadius: 12, padding: "12px 16px", display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: "var(--orange)" }}>Error</span>
              <p style={{ flex: 1, fontSize: 13, color: "var(--brown)" }}>{error}</p>
              <button onClick={() => setError(null)} style={{ background: "none", border: "none", cursor: "pointer" }}><Icon n="close" size={14} style={{ color: "var(--muted)" }} /></button>
            </div>
          )}

          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, padding: "60px 0" }}>
              <Spin size={32} />
              <p style={{ fontSize: 13, color: "var(--teal)", fontWeight: 600 }}>{loadingMsg || "Running analysis..."}</p>
            </div>
          ) : results ? (
            <div style={{ ...card, padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
              {results.length > 1 && (
                <div>
                  <p style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Select result</p>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {results.map((_, i) => {
                      const thumbSrc = images[i]?.preview || images[i]?.persistentUrl || accumulatedImagesRef.current[i]?.preview || null;
                      return (
                        <button key={i} onClick={() => switchToIndex(i, results)}
                          style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 10, border: `1px solid ${activeIndex === i ? "var(--teal)" : "var(--border)"}`, background: activeIndex === i ? "var(--teal-lt)" : "var(--white)", cursor: "pointer" }}>
                          <div style={{ width: 36, height: 36, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
                            {thumbSrc ? <img src={thumbSrc} alt={`Label ${i+1}`} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <div style={{ width: "100%", height: "100%", background: "var(--off)" }} />}
                          </div>
                          <span style={{ fontSize: 11, fontWeight: 700, color: activeIndex === i ? "var(--teal)" : "var(--muted)" }}>IMG {i + 1}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              <div style={{ display: "flex", alignItems: "center", gap: 12, background: "var(--brown-lt)", borderRadius: 12, padding: "10px 14px" }}>
                <div style={{ width: 52, height: 52, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0, background: "var(--off)" }}>
                  {currentPreview ? <img src={currentPreview} alt="Label" style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon n="storage" size={16} style={{ color: "var(--muted)" }} /></div>}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px" }}>Analyzing label</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--brown)", marginTop: 2 }}>Image {activeIndex + 1} of {results.length}</div>
                  {images[activeIndex]?.cropData && <div style={{ fontSize: 10, color: "var(--teal)", marginTop: 2 }}>✂ Cropped & optimized</div>}
                </div>
                <button onClick={() => setSaveModal({ result: currentResult, imageId: currentResult?.image_id || "", name: logName.trim() })}
                  style={{ ...ghostBtn, flexShrink: 0 }}>
                  <Icon n="bookmark_add" size={13} /> Save
                </button>
              </div>

              <div style={pillRow}>
                {["per_100g", "per_serving"].map(tab => {
                  const hasData = currentResult?.[tab] && Object.keys(currentResult[tab]).length > 0;
                  const active = activeTab === tab;
                  return (
                    <button key={tab} onClick={() => hasData && setActiveTab(tab)} disabled={!hasData}
                      style={{ flex: 1, padding: "8px", borderRadius: 8, border: "none", fontSize: 12, fontWeight: 700, cursor: hasData ? "pointer" : "not-allowed", background: active ? "var(--teal)" : "transparent", color: !hasData ? "var(--border)" : active ? "white" : "var(--muted)" }}>
                      {tab === "per_100g" ? "Per 100g" : "Per serving"}
                    </button>
                  );
                })}
              </div>

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 12, borderBottom: "1px solid var(--off2)" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "var(--muted)", display: "flex", alignItems: "center", gap: 6 }}>
                  <Icon n="table_chart" size={13} style={{ color: "var(--muted)" }} /> {activeTab === "per_100g" ? "Per 100g" : "Per Serving"}
                </div>
                {activeTab === "per_serving" && currentResult?.per_serving?.size && (
                  <span style={{ fontSize: 11, background: "var(--purp-lt)", color: "var(--purple)", padding: "3px 10px", borderRadius: 20, fontWeight: 600 }}>
                    {currentResult.per_serving.size}
                  </span>
                )}
              </div>

              <NutrientGrid key={activeIndex} data={currentResult?.[activeTab]} activeTab={activeTab} per100gData={currentResult?.per_100g} />
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}
