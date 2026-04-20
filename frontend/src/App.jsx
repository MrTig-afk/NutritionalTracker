import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, Zap, Loader2, RefreshCcw, Database, Cloud, TableProperties,
  LayoutPanelLeft, Scale, X, ChevronLeft, ChevronRight, Crop, Check,
  FolderPlus, Folder, FolderOpen, BookmarkPlus, Target, Plus, Minus,
  Trash2, BarChart3, CalendarDays, ChevronDown, ChevronUp, Save, PenLine
} from "lucide-react";

// =============================================================================
// CONFIG
// =============================================================================
const API_URL = import.meta.env.DEV
  ? "http://localhost:8000"
  : (import.meta.env.VITE_API_URL || "https://nutritionaltracker.onrender.com");

console.log("🔧 API_URL:", API_URL);

const MAX_IMAGE_PX         = 1024; // FIX: increased for sharper API submission
const JPEG_QUALITY         = 0.85; // FIX: increased quality
const REQUEST_TIMEOUT_MS   = 25000;
const MAX_FRONTEND_RETRIES = 2;
const RETRY_DELAY_MS       = [1500, 3000];

// =============================================================================
// NUTRIENT META
// =============================================================================
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
  return { label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()), unit: "", valueColor: "text-slate-300" };
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

// =============================================================================
// kJ → kcal CONVERSION
// =============================================================================
function normalizeCalories(value) {
  const num = parseNumeric(value);
  if (num === null) return value;
  if (typeof value === "string" && /kj|kilojoule/i.test(value)) {
    return Math.round(num / 4.184);
  }
  return value;
}

function normalizeNutritionData(data) {
  if (!data || typeof data !== "object") return data;
  const result = { ...data };
  if (result.calories !== undefined) result.calories = normalizeCalories(result.calories);
  return result;
}

function normalizeResult(result) {
  if (!result) return result;
  return {
    ...result,
    per_serving: result.per_serving ? normalizeNutritionData(result.per_serving) : result.per_serving,
    per_100g:    result.per_100g    ? normalizeNutritionData(result.per_100g)    : result.per_100g,
  };
}

function resolveNutrition(nutrition) {
  if (!nutrition) return {};
  if (!nutrition.per_serving && !nutrition.per_100g) return nutrition;
  if (nutrition.per_serving && Object.keys(nutrition.per_serving).length > 0) return nutrition.per_serving;
  if (nutrition.per_100g    && Object.keys(nutrition.per_100g).length    > 0) return nutrition.per_100g;
  return nutrition;
}

// =============================================================================
// IMAGE PIPELINE: crop → resize → grayscale → compress (for API submission)
// =============================================================================
async function applyPipelineToFile(file, cropData) {
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const srcX      = cropData ? cropData.x      : 0;
      const srcY      = cropData ? cropData.y      : 0;
      const srcWidth  = cropData ? cropData.width  : img.naturalWidth;
      const srcHeight = cropData ? cropData.height : img.naturalHeight;
      const scale     = Math.min(1, MAX_IMAGE_PX / Math.max(srcWidth, srcHeight));
      const targetW   = Math.max(1, Math.round(srcWidth  * scale));
      const targetH   = Math.max(1, Math.round(srcHeight * scale));
      const canvas    = document.createElement("canvas");
      canvas.width = targetW; canvas.height = targetH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, srcX, srcY, srcWidth, srcHeight, 0, 0, targetW, targetH);
      const imageData = ctx.getImageData(0, 0, targetW, targetH);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        data[i] = data[i + 1] = data[i + 2] = gray;
      }
      ctx.putImageData(imageData, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) { resolve(file); return; }
        resolve(new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg", lastModified: Date.now() }));
      }, "image/jpeg", JPEG_QUALITY);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

// Preview: just create a blob URL from the original file.
// Skipping canvas entirely — iOS Safari reliably renders blob URLs from File objects.
// The full pipeline (grayscale + resize) still runs separately for the API submission.
function generatePreview(file) {
  return URL.createObjectURL(file);
}
// =============================================================================
// FETCH WITH TIMEOUT + RETRY
// =============================================================================
async function fetchWithRetry(url, options, maxRetries = MAX_FRONTEND_RETRIES) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (response.ok) return response;
      const status = response.status;
      const isRetryable = status !== 429 && (status === 500 || status === 503 || status === 504);
      const errorText = await response.text().catch(() => `HTTP ${status}`);
      if (!isRetryable || attempt === maxRetries) throw new Error(`Pipeline Error: ${status} — ${errorText}`);
      lastError = new Error(`Retryable error: ${status}`);
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000));
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        lastError = new Error("Request timed out. Please try again.");
        if (attempt < maxRetries) { await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000)); continue; }
      } else if (err.message.startsWith("Pipeline Error:")) {
        throw err;
      } else {
        lastError = err;
        if (attempt < maxRetries) { await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000)); continue; }
      }
    }
  }
  throw lastError || new Error("Request failed after retries");
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: `HTTP ${res.status}` }));
    throw new Error(err.message || `HTTP ${res.status}`);
  }
  return res.json();
}

// =============================================================================
// CORE ANALYSIS — standalone function, no stale closure issues
// =============================================================================
async function runAnalysis({ optimizedFiles, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex }) {
  if (!optimizedFiles.length) return;
  setLoading(true); setError(null);
  try {
    const formData = new FormData();
    if (optimizedFiles.length === 1) {
      formData.append("file", optimizedFiles[0]);
      setLoadingMsg("Analyzing label...");
      const response = await fetchWithRetry(`${API_URL}/analyze-label`, { method: "POST", body: formData });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => ({
        ...img,
        persistentUrl: arr[idx]?.processed_url || arr[idx]?.raw_url || null,
        preview: arr[idx]?.processed_url || arr[idx]?.raw_url || img.preview,
      })));
      switchToIndex(0, arr);
    } else {
      optimizedFiles.forEach(f => formData.append("files", f));
      setLoadingMsg(`Analyzing ${optimizedFiles.length} labels...`);
      const response = await fetchWithRetry(`${API_URL}/analyze-labels`, { method: "POST", body: formData });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => ({
        ...img,
        persistentUrl: arr[idx]?.processed_url || arr[idx]?.raw_url || null,
        preview: arr[idx]?.processed_url || arr[idx]?.raw_url || img.preview,
      })));
      switchToIndex(0, arr);
    }
  } catch (err) {
    console.error("❌ Pipeline Failure:", err);
    setError(err.message);
  } finally {
    setLoading(false); setLoadingMsg("");
  }
}

// =============================================================================
// IMAGE CROPPER COMPONENT
// =============================================================================
function ImageCropper({ file, onConfirm, onCancel }) {
  const canvasRef = useRef(null); const containerRef = useRef(null);
  const imgRef = useRef(null); const isDragging = useRef(false);
  const dragMode = useRef(null); const dragStart = useRef({ x: 0, y: 0 });
  const cropAtStart = useRef(null);
  const [cropRect, setCropRect] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [imgScale, setImgScale] = useState(1);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      const container = containerRef.current;
      const maxW = container ? container.clientWidth : 360;
      const maxH = Math.min(window.innerHeight * 0.55, 460);
      const s = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
      const cw = Math.round(img.naturalWidth * s); const ch = Math.round(img.naturalHeight * s);
      setCanvasSize({ w: cw, h: ch }); setImgScale(s);
      setCropRect({ x: Math.round(cw * 0.1), y: Math.round(ch * 0.1), w: Math.round(cw * 0.8), h: Math.round(ch * 0.8) });
      setImageLoaded(true);
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!imageLoaded || !cropRect || !canvasRef.current) return;
    const canvas = canvasRef.current; const ctx = canvas.getContext("2d");
    const { w: cw, h: ch } = canvasSize;
    ctx.clearRect(0, 0, cw, ch); ctx.drawImage(imgRef.current, 0, 0, cw, ch);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, cw, cropRect.y);
    ctx.fillRect(0, cropRect.y + cropRect.h, cw, ch - cropRect.y - cropRect.h);
    ctx.fillRect(0, cropRect.y, cropRect.x, cropRect.h);
    ctx.fillRect(cropRect.x + cropRect.w, cropRect.y, cw - cropRect.x - cropRect.w, cropRect.h);
    ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);
    ctx.strokeStyle = "rgba(34,211,238,0.22)"; ctx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      const gx = cropRect.x + (cropRect.w / 3) * i; const gy = cropRect.y + (cropRect.h / 3) * i;
      ctx.beginPath(); ctx.moveTo(gx, cropRect.y); ctx.lineTo(gx, cropRect.y + cropRect.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cropRect.x, gy); ctx.lineTo(cropRect.x + cropRect.w, gy); ctx.stroke();
    }
    ctx.fillStyle = "#22d3ee";
    [[cropRect.x, cropRect.y], [cropRect.x + cropRect.w, cropRect.y],
     [cropRect.x, cropRect.y + cropRect.h], [cropRect.x + cropRect.w, cropRect.y + cropRect.h]]
      .forEach(([cx, cy]) => ctx.fillRect(cx - 5, cy - 5, 10, 10));
  }, [imageLoaded, cropRect, canvasSize]);

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * (canvas.width / rect.width), y: (cy - rect.top) * (canvas.height / rect.height) };
  };
  const getZone = (pos, r) => {
    const hs = 18;
    const corners = { nw: [r.x, r.y], ne: [r.x + r.w, r.y], sw: [r.x, r.y + r.h], se: [r.x + r.w, r.y + r.h] };
    for (const [k, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(pos.x - cx) <= hs && Math.abs(pos.y - cy) <= hs) return k;
    }
    if (pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h) return "move";
    return null;
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const onDown = useCallback((e) => {
    if (!cropRect || !canvasRef.current) return; e.preventDefault();
    const pos = getPos(e, canvasRef.current); const zone = getZone(pos, cropRect);
    if (!zone) return;
    isDragging.current = true; dragMode.current = zone; dragStart.current = pos; cropAtStart.current = { ...cropRect };
  }, [cropRect]);
  const onMove = useCallback((e) => {
    if (!isDragging.current || !cropAtStart.current) return; e.preventDefault();
    const pos = getPos(e, canvasRef.current);
    const dx = pos.x - dragStart.current.x, dy = pos.y - dragStart.current.y;
    const { w: cw, h: ch } = canvasSize;
    const b = cropAtStart.current, MIN = 40, m = dragMode.current;
    setCropRect(() => {
      let { x, y, w, h } = b;
      if (m === "move") { x = clamp(b.x + dx, 0, cw - b.w); y = clamp(b.y + dy, 0, ch - b.h); }
      else if (m === "se") { w = clamp(b.w + dx, MIN, cw - b.x); h = clamp(b.h + dy, MIN, ch - b.y); }
      else if (m === "sw") { const nx = clamp(b.x + dx, 0, b.x + b.w - MIN); w = b.x + b.w - nx; x = nx; h = clamp(b.h + dy, MIN, ch - b.y); }
      else if (m === "ne") { w = clamp(b.w + dx, MIN, cw - b.x); const ny = clamp(b.y + dy, 0, b.y + b.h - MIN); h = b.y + b.h - ny; y = ny; }
      else if (m === "nw") { const nx = clamp(b.x + dx, 0, b.x + b.w - MIN); const ny = clamp(b.y + dy, 0, b.y + b.h - MIN); w = b.x + b.w - nx; x = nx; h = b.y + b.h - ny; y = ny; }
      return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
    });
  }, [canvasSize]);
  const onUp = useCallback((e) => { e.preventDefault(); isDragging.current = false; }, []);
  const handleConfirm = () => {
    if (!cropRect || !imgScale) { onConfirm(null); return; }
    onConfirm({ x: Math.round(cropRect.x / imgScale), y: Math.round(cropRect.y / imgScale), width: Math.round(cropRect.w / imgScale), height: Math.round(cropRect.h / imgScale) });
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/96 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-lg">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-xs font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
              <Crop className="h-4 w-4 text-cyan-400" /> Crop Nutrition Label
            </p>
            <p className="text-[10px] text-slate-600 font-mono mt-0.5">Drag corners to resize · Drag inside to move</p>
          </div>
          <button onClick={onCancel} className="text-slate-600 hover:text-rose-400 transition-colors"><X className="h-5 w-5" /></button>
        </div>
        <div ref={containerRef} className="w-full rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 touch-none select-none">
          {imageLoaded ? (
            <canvas ref={canvasRef} width={canvasSize.w} height={canvasSize.h}
              style={{ display: "block", width: "100%", touchAction: "none", cursor: "crosshair" }}
              onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
              onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp} />
          ) : (
            <div className="flex items-center justify-center h-64"><Loader2 className="animate-spin text-cyan-400 h-6 w-6" /></div>
          )}
        </div>
        {cropRect && <p className="text-[9px] font-mono text-slate-700 mt-2 text-center">REGION: {Math.round(cropRect.w / imgScale)}×{Math.round(cropRect.h / imgScale)}px</p>}
        <div className="flex gap-3 mt-4">
          <button onClick={onCancel} className="flex-1 py-3 rounded-xl border border-slate-800 text-slate-500 text-xs font-bold hover:border-slate-600 transition-colors">CANCEL</button>
          <button onClick={() => onConfirm(null)} className="flex-1 py-3 rounded-xl border border-slate-700 text-slate-400 text-xs font-bold hover:border-slate-500 transition-colors">FULL IMAGE</button>
          <button onClick={handleConfirm} className="flex-1 py-3 rounded-xl bg-cyan-500 text-slate-950 text-xs font-bold hover:bg-cyan-400 transition-colors flex items-center justify-center gap-2">
            <Check className="h-4 w-4" /> CONFIRM
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// SAVE TO FOLDER MODAL
// =============================================================================
function SaveToFolderModal({ result, imageId, onClose }) {
  const [folders, setFolders] = useState([]);
  const [newFolder, setNewFolder] = useState("");
  const [itemName, setItemName] = useState("");
  const [selectedFolder, setSelectedFolder] = useState("");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);

  useEffect(() => { apiFetch("/folders").then(setFolders).catch(() => {}); }, []);

  const createFolder = async () => {
    if (!newFolder.trim()) return;
    try {
      const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolder.trim() }) });
      setFolders(prev => [f, ...prev]); setSelectedFolder(f.folder_id); setNewFolder("");
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
  };

  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolder(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolder(newFolder.slice(0, pos) + newFolder.slice(pos).replace(/^\s*\S+/, "")); }
  };

  const save = async () => {
    if (!selectedFolder || !itemName.trim()) return;
    setSaving(true);
    try {
      const nutrition = result.per_serving ? { per_serving: result.per_serving, per_100g: result.per_100g } : result;
      await apiFetch(`/folders/${selectedFolder}/items`, { method: "POST", body: JSON.stringify({ name: itemName.trim(), image_id: imageId || "", nutrition }) });
      setStatus({ type: "ok", msg: "Saved to folder!" });
      setTimeout(onClose, 1200);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/90 flex items-center justify-center p-4">
      <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2"><BookmarkPlus className="h-4 w-4 text-cyan-400" /> Save to Folder</h3>
          <button onClick={onClose} className="text-slate-600 hover:text-rose-400"><X className="h-4 w-4" /></button>
        </div>
        <div>
          <label className="text-[10px] font-mono text-slate-600 uppercase mb-1 block">Item Name</label>
          <input value={itemName} onChange={e => setItemName(e.target.value)} placeholder="e.g. Greek Yogurt"
            className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/60" />
        </div>
        <div>
          <label className="text-[10px] font-mono text-slate-600 uppercase mb-1 block">Folder</label>
          <select value={selectedFolder} onChange={e => setSelectedFolder(e.target.value)}
            className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500/60">
            <option value="">— select folder —</option>
            {folders.map(f => <option key={f.folder_id} value={f.folder_id}>{f.name}</option>)}
          </select>
        </div>
        <div className="flex gap-2">
          <input value={newFolder} onChange={e => setNewFolder(e.target.value)} placeholder="New folder name..."
            onKeyDown={handleFolderKeyDown}
            className="flex-1 px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/60" />
          <button onClick={createFolder} className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-cyan-400 transition-colors">
            <FolderPlus className="h-4 w-4" />
          </button>
        </div>
        {status && <p className={`text-xs font-mono ${status.type === "ok" ? "text-emerald-500" : "text-rose-400"}`}>{status.msg}</p>}
        <button onClick={save} disabled={saving || !selectedFolder || !itemName.trim()}
          className="w-full py-3 rounded-xl bg-cyan-500 text-slate-950 font-bold text-sm hover:bg-cyan-400 transition-colors disabled:opacity-40 flex items-center justify-center gap-2">
          {saving ? <Loader2 className="animate-spin h-4 w-4" /> : <Save className="h-4 w-4" />}
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// ADD TO LOG MODAL
// FIX: Added "Manual" mode — user can type custom macro values directly
// =============================================================================
function AddToLogModal({ item, onClose, onAdded }) {
  const [mode, setMode]       = useState("serving");
  const [servings, setServings] = useState("1");
  const [grams, setGrams]     = useState("");
  const [saving, setSaving]   = useState(false);
  const [status, setStatus]   = useState(null);

  // Manual mode
  const [manualName,  setManualName]  = useState(item.name || "");
  const [manualCal,   setManualCal]   = useState("");
  const [manualProt,  setManualProt]  = useState("");
  const [manualCarb,  setManualCarb]  = useState("");
  const [manualFat,   setManualFat]   = useState("");
  const [manualFibre, setManualFibre] = useState("");

  const rawNutrition  = item.nutrition || {};
  const per100g       = rawNutrition.per_100g    && Object.keys(rawNutrition.per_100g).length > 0    ? rawNutrition.per_100g    : null;
  const perServing    = rawNutrition.per_serving && Object.keys(rawNutrition.per_serving).length > 0 ? rawNutrition.per_serving : null;
  const baseNutrition = perServing ?? per100g ?? resolveNutrition(rawNutrition);
  const servingGrams  = perServing ? extractServingGrams(perServing.size) : null;
  const servingsNum   = parseFloat(servings) || 1;
  const gramsNum      = parseFloat(grams)    || 0;

  let scaledNutrition = { ...baseNutrition };
  let scalingInfo     = null;

  if (mode === "serving") {
    scalingInfo = { factor: servingsNum, baseLabel: perServing ? `per serving${perServing.size ? ` (${perServing.size})` : ""}` : "per 100g" };
  } else if (mode === "grams") {
    if (per100g && gramsNum > 0) {
      scalingInfo = { factor: gramsNum / 100, baseLabel: "per 100g", targetLabel: `${gramsNum}g` }; scaledNutrition = per100g;
    } else if (perServing && servingGrams && gramsNum > 0) {
      scalingInfo = { factor: gramsNum / servingGrams, baseLabel: `per serving (${servingGrams}g)`, targetLabel: `${gramsNum}g` }; scaledNutrition = perServing;
    } else {
      scalingInfo = { factor: 1, baseLabel: "no size data", targetLabel: `${gramsNum}g`, warn: true };
    }
  }

  const factor = scalingInfo?.factor ?? 1;
  const getVal = (key) => (parseNumeric(scaledNutrition[key]) || 0) * factor;

  const cal  = mode === "manual" ? (parseFloat(manualCal)   || 0) : getVal("calories");
  const prot = mode === "manual" ? (parseFloat(manualProt)  || 0) : getVal("protein");
  const carb = mode === "manual" ? (parseFloat(manualCarb)  || 0) : getVal("carbohydrates");
  const fat  = mode === "manual" ? (parseFloat(manualFat)   || 0) : getVal("fat");
  const fib  = mode === "manual" ? (parseFloat(manualFibre) || 0) : getVal("fibre");

  const buildSubmitNutrition = () => {
    if (mode === "manual") {
      return { per_serving: { size: "1 serving", calories: Math.round(cal), protein: `${prot.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, fat: `${fat.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g` } };
    }
    if (mode === "serving") return rawNutrition;
    return {
      per_serving: {
        size: `${gramsNum}g`, calories: Math.round(cal),
        fat: `${fat.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, protein: `${prot.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g`,
        ...(scaledNutrition.saturated_fat ? { saturated_fat: `${(parseNumeric(scaledNutrition.saturated_fat) || 0) * factor}g` } : {}),
        ...(scaledNutrition.sugars        ? { sugars:        `${(parseNumeric(scaledNutrition.sugars)        || 0) * factor}g` } : {}),
        ...(scaledNutrition.sodium        ? { sodium:        `${(parseNumeric(scaledNutrition.sodium)        || 0) * factor}g` } : {}),
      },
    };
  };

  const canSave = mode === "serving" ? servingsNum > 0 : mode === "grams" ? gramsNum > 0 : manualCal !== "" || manualProt !== "" || manualCarb !== "" || manualFat !== "";

  const save = async () => {
    if (!canSave) return;
    setSaving(true);
    try {
      const submitServings = mode === "serving" ? servingsNum : 1;
      const submitName     = mode === "manual" && manualName.trim() ? manualName.trim() : item.name;
      await apiFetch("/log", { method: "POST", body: JSON.stringify({ name: submitName, servings: submitServings, nutrition: buildSubmitNutrition() }) });
      setStatus({ type: "ok", msg: "Added to today's log!" });
      onAdded();
      setTimeout(() => { onClose(); }, 900);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/90 flex items-center justify-center p-4">
      <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2"><CalendarDays className="h-4 w-4 text-emerald-400" /> Add to Today's Log</h3>
          <button onClick={onClose} className="text-slate-600 hover:text-rose-400"><X className="h-4 w-4" /></button>
        </div>

        <p className="text-sm text-slate-300 font-bold truncate">{item.name}</p>

        <div className="flex p-1 bg-slate-950 border border-slate-800 rounded-xl">
          {[["serving", "Per Serving"], ["grams", "By Weight"], ["manual", "Manual"]].map(([m, label]) => (
            <button key={m} onClick={() => setMode(m)}
              className={`flex-1 py-2 text-[10px] font-bold rounded-lg transition-all ${mode === m ? "bg-emerald-500 text-slate-950 shadow" : "text-slate-500 hover:text-slate-300"}`}>
              {label}
            </button>
          ))}
        </div>

        {mode === "serving" && (
          <div>
            <label className="text-[10px] font-mono text-slate-600 uppercase mb-1 block">Servings</label>
            <div className="flex items-center gap-3">
              <button onClick={() => setServings(s => String(Math.max(0.5, parseFloat(s) - 0.5)))}
                className="w-8 h-8 rounded-full border border-slate-700 flex items-center justify-center text-slate-400 hover:border-slate-500"><Minus className="h-3 w-3" /></button>
              <input type="number" min="0.5" step="0.5" value={servings} onChange={e => setServings(e.target.value)}
                className="w-20 text-center px-2 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500/60" />
              <button onClick={() => setServings(s => String(parseFloat(s) + 0.5))}
                className="w-8 h-8 rounded-full border border-slate-700 flex items-center justify-center text-slate-400 hover:border-slate-500"><Plus className="h-3 w-3" /></button>
            </div>
            {baseNutrition.size && <p className="text-[10px] font-mono text-slate-600 mt-1">Base: {baseNutrition.size}</p>}
          </div>
        )}

        {mode === "grams" && (
          <div>
            <label className="text-[10px] font-mono text-slate-600 uppercase mb-1 block">Amount (grams)</label>
            <div className="flex items-center gap-3">
              <input type="number" min="1" step="1" value={grams} onChange={e => setGrams(e.target.value)} placeholder="e.g. 150"
                className="flex-1 px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/60" />
              <span className="text-sm font-mono text-slate-400 font-bold">g</span>
            </div>
          </div>
        )}

        {mode === "manual" && (
          <div className="space-y-2">
            <div>
              <label className="text-[10px] font-mono text-slate-600 uppercase mb-1 block">Food Name</label>
              <input value={manualName} onChange={e => setManualName(e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500/60" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Calories (kcal)", val: manualCal,   set: setManualCal   },
                { label: "Protein (g)",     val: manualProt,  set: setManualProt  },
                { label: "Carbs (g)",       val: manualCarb,  set: setManualCarb  },
                { label: "Fat (g)",         val: manualFat,   set: setManualFat   },
                { label: "Fibre (g)",       val: manualFibre, set: setManualFibre },
              ].map(({ label, val, set }) => (
                <div key={label}>
                  <label className="text-[9px] font-mono text-slate-600 uppercase mb-1 block">{label}</label>
                  <input type="number" min="0" step="any" value={val} onChange={e => set(e.target.value)} placeholder="0"
                    className="w-full px-2 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500/60" />
                </div>
              ))}
            </div>
          </div>
        )}

        {scalingInfo && mode !== "manual" && (
          <div className="text-[10px] font-mono leading-relaxed">
            {scalingInfo.warn
              ? <span className="text-amber-600">⚠ No size data — using base values as-is</span>
              : <span className="text-slate-600">
                  BASE: {scalingInfo.baseLabel}
                  {mode === "grams" && gramsNum > 0 && scalingInfo.factor !== 1 && <span className="text-emerald-700 ml-2">→ {scalingInfo.targetLabel} | ×{scalingInfo.factor.toFixed(4)}</span>}
                  {mode === "serving" && servingsNum !== 1 && <span className="text-emerald-700 ml-2">× {servingsNum} servings</span>}
                </span>
            }
          </div>
        )}

        <div className="grid grid-cols-4 gap-2">
          {[
            { label: "Cal",   value: cal,  unit: "kcal", color: "text-amber-400"  },
            { label: "Prot",  value: prot, unit: "g",    color: "text-cyan-400"   },
            { label: "Carbs", value: carb, unit: "g",    color: "text-blue-400"   },
            { label: "Fat",   value: fat,  unit: "g",    color: "text-orange-400" },
          ].map(({ label, value, unit, color }) => (
            <div key={label} className="bg-slate-950/60 border border-slate-800 rounded-xl p-2 text-center">
              <p className="text-[9px] font-mono text-slate-600">{label}</p>
              <p className={`text-sm font-bold ${color}`}>{value.toFixed(1)}</p>
              <p className="text-[8px] font-mono text-slate-700">{unit}</p>
            </div>
          ))}
        </div>

        {status && <p className={`text-xs font-mono ${status.type === "ok" ? "text-emerald-500" : "text-rose-400"}`}>{status.msg}</p>}

        <button onClick={save} disabled={saving || !canSave}
          className="w-full py-3 rounded-xl bg-emerald-500 text-slate-950 font-bold text-sm hover:bg-emerald-400 transition-colors disabled:opacity-40 flex items-center justify-center gap-2">
          {saving ? <Loader2 className="animate-spin h-4 w-4" /> : <Plus className="h-4 w-4" />}
          {saving ? "Adding..." : "Add to Log"}
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// MACRO PROGRESS BAR
// =============================================================================
function MacroBar({ label, current, goal, color }) {
  const pct = goal > 0 ? Math.min(100, (current / goal) * 100) : 0;
  const over = current > goal;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] font-mono">
        <span className="text-slate-500">{label}</span>
        <span className={over ? "text-rose-400" : "text-slate-400"}>{current.toFixed(1)} / {goal}{over && " ⚠"}</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${over ? "bg-rose-500" : color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// =============================================================================
// TRACKER TAB
// =============================================================================
function TrackerTab({ refreshKey }) {
  const [goals, setGoals] = useState({ calories: 2000, protein: 150, carbs: 250, fat: 65 });
  const [logData, setLogData] = useState(null);
  const [editingGoals, setEditingGoals] = useState(false);
  const [goalDraft, setGoalDraft] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingGoals, setSavingGoals] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const today = new Date().toISOString().slice(0, 10);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [g, l] = await Promise.all([apiFetch("/goals"), apiFetch(`/log?log_date=${today}`)]);
      setGoals(g); setLogData(l);
    } catch (e) { console.error("Tracker load failed:", e); }
    finally { setLoading(false); }
  }, [today]);

  useEffect(() => { loadData(); }, [loadData, refreshKey]);

  const saveGoals = async () => {
    setSavingGoals(true);
    try { const updated = await apiFetch("/goals", { method: "POST", body: JSON.stringify(goalDraft) }); setGoals(updated); setEditingGoals(false); }
    catch (e) { console.error(e); }
    finally { setSavingGoals(false); }
  };

  const deleteEntry = async (logId) => {
    setDeletingId(logId);
    try { await apiFetch(`/log/${logId}`, { method: "DELETE" }); await loadData(); }
    catch (e) { console.error(e); }
    finally { setDeletingId(null); }
  };

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="animate-spin text-cyan-400 h-6 w-6" /></div>;

  const totals = logData?.totals || { calories: 0, protein: 0, carbs: 0, fat: 0 };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2"><Target className="h-4 w-4 text-cyan-400" /> Daily Goals</h3>
          <button onClick={() => { setGoalDraft({ ...goals }); setEditingGoals(v => !v); }}
            className="text-[10px] font-mono text-slate-600 hover:text-cyan-400 transition-colors">{editingGoals ? "CANCEL" : "EDIT"}</button>
        </div>
        {editingGoals ? (
          <div className="space-y-3">
            {["calories", "protein", "carbs", "fat", "fibre"].map(key => (
              <div key={key} className="flex items-center gap-3">
                <label className="text-[10px] font-mono text-slate-500 w-16 uppercase">{key}</label>
                <input type="number" min="0" value={goalDraft[key] || ""}
                  onChange={e => setGoalDraft(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))}
                  className="flex-1 px-3 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500/60" />
                <span className="text-[10px] text-slate-600 font-mono w-8">{key === "calories" ? "kcal" : "g"}</span>
              </div>
            ))}
            <button onClick={saveGoals} disabled={savingGoals}
              className="w-full py-2.5 rounded-xl bg-cyan-500 text-slate-950 font-bold text-xs hover:bg-cyan-400 transition-colors disabled:opacity-40">
              {savingGoals ? "Saving..." : "SAVE GOALS"}
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <MacroBar label="Calories" current={totals.calories}    goal={goals.calories} color="bg-amber-400"  />
            <MacroBar label="Protein"  current={totals.protein}     goal={goals.protein}  color="bg-cyan-400"   />
            <MacroBar label="Carbs"    current={totals.carbs}       goal={goals.carbs}    color="bg-blue-400"   />
            <MacroBar label="Fat"      current={totals.fat}         goal={goals.fat}      color="bg-orange-400" />
            {goals.fibre > 0 && <MacroBar label="Fibre" current={totals.fibre || 0} goal={goals.fibre} color="bg-emerald-400" />}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Calories", val: totals.calories, unit: "kcal", color: "text-amber-400"  },
          { label: "Protein",  val: totals.protein,  unit: "g",    color: "text-cyan-400"   },
          { label: "Carbs",    val: totals.carbs,    unit: "g",    color: "text-blue-400"   },
          { label: "Fat",      val: totals.fat,      unit: "g",    color: "text-orange-400" },
        ].map(({ label, val, unit, color }) => (
          <div key={label} className="bg-slate-900/40 border border-slate-800 rounded-2xl p-4 text-center">
            <p className="text-[9px] font-mono text-slate-600 uppercase">{label}</p>
            <p className={`text-xl font-bold ${color} mt-1`}>{(val || 0).toFixed(1)}</p>
            <p className="text-[9px] font-mono text-slate-700">{unit}</p>
          </div>
        ))}
      </div>

      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <CalendarDays className="h-4 w-4 text-emerald-400" /> Today's Log — {today}
          </h3>
          <span className="text-[10px] font-mono text-slate-600">{logData?.items?.length || 0} entries</span>
        </div>
        {!logData?.items?.length ? (
          <div className="p-8 text-center text-slate-700 text-sm italic">No entries yet. Add food from Library.</div>
        ) : (
          <div className="divide-y divide-slate-800/50">
            {logData.items.map(entry => (
              <div key={entry.log_id} className="flex items-center gap-3 p-4 hover:bg-slate-800/20 transition-colors">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-bold text-slate-300 truncate">{entry.name}</p>
                  <p className="text-[10px] font-mono text-slate-600 mt-0.5">
                    ×{entry.servings} serving{entry.servings !== 1 ? "s" : ""} ·
                    {entry.contribution.calories.toFixed(0)} kcal ·
                    P {entry.contribution.protein.toFixed(1)}g ·
                    C {entry.contribution.carbs.toFixed(1)}g ·
                    F {entry.contribution.fat.toFixed(1)}g
                  </p>
                </div>
                <button onClick={() => deleteEntry(entry.log_id)} disabled={deletingId === entry.log_id}
                  className="text-slate-700 hover:text-rose-400 transition-colors flex-shrink-0">
                  {deletingId === entry.log_id ? <Loader2 className="animate-spin h-4 w-4" /> : <Trash2 className="h-4 w-4" />}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// LIBRARY TAB
// FIX: always re-fetch folder items on open (no stale cache)
// =============================================================================
function LibraryTab({ onAddToLog }) {
  const [folders, setFolders] = useState([]);
  const [openFolder, setOpenFolder] = useState(null);
  const [folderData, setFolderData] = useState({});
  const [newFolderName, setNewFolderName] = useState("");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deletingItem, setDeletingItem] = useState(null);

  const loadFolders = useCallback(async () => {
    setLoading(true);
    setFolderData({}); // FIX: clear cache on every mount
    try { setFolders(await apiFetch("/folders")); }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadFolders(); }, [loadFolders]);

  const openFolderById = async (id) => {
    if (openFolder === id) { setOpenFolder(null); return; }
    setOpenFolder(id);
    // FIX: always re-fetch — don't use stale cache
    try {
      const data = await apiFetch(`/folders/${id}`);
      setFolderData(prev => ({ ...prev, [id]: data }));
    } catch (e) { console.error(e); }
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return;
    setCreating(true);
    try {
      const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolderName.trim() }) });
      setFolders(prev => [f, ...prev]); setNewFolderName("");
    } catch (e) { console.error(e); }
    finally { setCreating(false); }
  };

  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolderName(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolderName(newFolderName.slice(0, pos) + newFolderName.slice(pos).replace(/^\s*\S+/, "")); }
  };

  const deleteItem = async (folderId, itemId) => {
    setDeletingItem(itemId);
    try {
      await apiFetch(`/folders/${folderId}/items/${itemId}`, { method: "DELETE" });
      setFolderData(prev => ({ ...prev, [folderId]: { ...prev[folderId], items: prev[folderId].items.filter(i => i.item_id !== itemId) } }));
    } catch (e) { console.error(e); }
    finally { setDeletingItem(null); }
  };

  const deleteFolder = async (folderId) => {
    try {
      await apiFetch(`/folders/${folderId}`, { method: "DELETE" });
      setFolders(prev => prev.filter(f => f.folder_id !== folderId));
      if (openFolder === folderId) setOpenFolder(null);
    } catch (e) { console.error(e); }
  };

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="animate-spin text-cyan-400 h-6 w-6" /></div>;

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input value={newFolderName} onChange={e => setNewFolderName(e.target.value)}
          placeholder="New folder name..." onKeyDown={handleFolderKeyDown}
          className="flex-1 px-4 py-2.5 bg-slate-900 border border-slate-800 rounded-2xl text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/60" />
        <button onClick={createFolder} disabled={creating || !newFolderName.trim()}
          className="px-4 py-2.5 bg-cyan-500 text-slate-950 rounded-2xl font-bold text-xs hover:bg-cyan-400 transition-colors disabled:opacity-40 flex items-center gap-2">
          {creating ? <Loader2 className="animate-spin h-4 w-4" /> : <FolderPlus className="h-4 w-4" />} CREATE
        </button>
      </div>

      {folders.length === 0 ? (
        <div className="text-center py-12 text-slate-700 text-sm italic border border-dashed border-slate-800 rounded-2xl">No folders yet. Create one above.</div>
      ) : (
        <div className="space-y-2">
          {folders.map(folder => (
            <div key={folder.folder_id} className="border border-slate-800 rounded-2xl overflow-hidden">
              <div className="flex items-center gap-3 p-4 cursor-pointer hover:bg-slate-800/30 transition-colors"
                onClick={() => openFolderById(folder.folder_id)}>
                {openFolder === folder.folder_id ? <FolderOpen className="h-5 w-5 text-cyan-400 flex-shrink-0" /> : <Folder className="h-5 w-5 text-slate-500 flex-shrink-0" />}
                <span className="flex-1 text-sm font-bold text-slate-300">{folder.name}</span>
                <span className="text-[10px] font-mono text-slate-600">{folderData[folder.folder_id]?.items?.length ?? ""} items</span>
                <button onClick={e => { e.stopPropagation(); deleteFolder(folder.folder_id); }} className="text-slate-700 hover:text-rose-400 transition-colors ml-2">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
                {openFolder === folder.folder_id ? <ChevronUp className="h-4 w-4 text-slate-600" /> : <ChevronDown className="h-4 w-4 text-slate-600" />}
              </div>

              {openFolder === folder.folder_id && folderData[folder.folder_id] && (
                <div className="border-t border-slate-800 divide-y divide-slate-800/50">
                  {folderData[folder.folder_id].items.length === 0 ? (
                    <p className="p-4 text-center text-slate-700 text-xs italic">No items in this folder.</p>
                  ) : (
                    folderData[folder.folder_id].items.map(item => {
                      const nutrition = item.nutrition?.per_serving ?? item.nutrition ?? {};
                      const cal  = parseNumeric(nutrition.calories)      || 0;
                      const prot = parseNumeric(nutrition.protein)       || 0;
                      const carb = parseNumeric(nutrition.carbohydrates) || 0;
                      const fat  = parseNumeric(nutrition.fat)           || 0;
                      const imageUrl = item.nutrition?.processed_url || item.nutrition?.raw_url || item.processed_url || item.raw_url || null;
                      return (
                        <div key={item.item_id} className="flex items-center gap-3 p-3 pl-4 hover:bg-slate-800/20 transition-colors">
                          {imageUrl ? (
                            <div className="w-12 h-12 rounded-xl overflow-hidden border border-slate-700 flex-shrink-0 bg-slate-800">
                              <img src={imageUrl} alt={item.name} className="w-full h-full object-cover" onError={e => { e.target.style.display = "none"; }} />
                            </div>
                          ) : (
                            <div className="w-12 h-12 rounded-xl border border-slate-800 flex-shrink-0 bg-slate-900 flex items-center justify-center">
                              <Database className="h-4 w-4 text-slate-700" />
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-bold text-slate-300 truncate">{item.name}</p>
                            <p className="text-[10px] font-mono text-slate-600 mt-0.5">{cal}kcal · P {prot}g · C {carb}g · F {fat}g</p>
                          </div>
                          <button onClick={() => onAddToLog({ ...item })}
                            className="text-[10px] font-mono text-emerald-600 hover:text-emerald-400 transition-colors px-2 py-1 border border-emerald-900/40 rounded-lg flex items-center gap-1 flex-shrink-0">
                            <Plus className="h-3 w-3" /> LOG
                          </button>
                          <button onClick={() => deleteItem(folder.folder_id, item.item_id)} disabled={deletingItem === item.item_id}
                            className="text-slate-700 hover:text-rose-400 transition-colors flex-shrink-0">
                            {deletingItem === item.item_id ? <Loader2 className="animate-spin h-3.5 w-3.5" /> : <Trash2 className="h-3.5 w-3.5" />}
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// NUTRIENT GRID
// =============================================================================
function NutrientGrid({ data, activeTab, per100gData }) {
  const [customGrams, setCustomGrams] = useState("");
  if (!data || Object.keys(data).length === 0) {
    return <div className="py-12 text-center border border-dashed border-slate-800 rounded-2xl"><span className="text-xs text-slate-600 italic">No data extracted for this view</span></div>;
  }
  const customG = parseFloat(customGrams); const isValid = !isNaN(customG) && customG > 0;
  let baseGrams = null, scaleFrom = data, baseLabel = null, warnMsg = null;
  if (activeTab === "per_100g") { baseGrams = 100; baseLabel = "100g"; }
  else {
    const sg = extractServingGrams(data.size);
    if (sg) { baseGrams = sg; baseLabel = `${data.size} (${sg}g)`; }
    else if (per100gData && Object.keys(per100gData).length > 0) { baseGrams = 100; scaleFrom = per100gData; baseLabel = "100g (fallback)"; warnMsg = "NO SERVING SIZE DETECTED — SCALING FROM PER_100G BASE"; }
    else { warnMsg = "CANNOT SCALE — NO SIZE OR PER_100G DATA AVAILABLE"; }
  }
  const scalingActive = isValid && baseGrams !== null;
  const factor = scalingActive ? customG / baseGrams : null;
  const getDisplay = (key, rawValue) => {
    if (!scalingActive) return { display: rawValue, adjusted: false };
    const base = parseNumeric(scaleFrom[key] ?? rawValue);
    if (base === null) return { display: rawValue, adjusted: false };
    return { display: parseFloat((base * factor).toFixed(2)), adjusted: true, baseDisplay: rawValue };
  };
  const entries = Object.entries(data).filter(([k]) => k !== "size");
  return (
    <div className="space-y-5">
      <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-3">
        <div className="flex items-center gap-2"><Scale className="h-4 w-4 text-cyan-400" /><span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Custom Serving Calculator</span></div>
        <div className="flex gap-3 items-center">
          <input type="number" min="1" step="any" placeholder="Enter serving size, e.g. 58" value={customGrams}
            onChange={e => setCustomGrams(e.target.value)}
            className="flex-1 p-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/60 transition-colors" />
          <span className="text-slate-400 text-sm font-mono font-bold">g</span>
          {customGrams && <button onClick={() => setCustomGrams("")} className="text-[10px] text-slate-600 hover:text-rose-400 transition-colors font-mono px-2">CLEAR</button>}
        </div>
        <div className="text-[10px] font-mono leading-relaxed">
          {warnMsg ? <span className="text-amber-700">{warnMsg}</span>
            : <span className="text-slate-600">BASE: {baseLabel}{scalingActive && <span className="text-cyan-700 ml-2">→ {customG}g | FACTOR: ×{factor.toFixed(4)}</span>}</span>}
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {entries.map(([key, value]) => {
          const meta = NUTRIENT_META[key] ?? getFallbackMeta(key);
          const { display, adjusted, baseDisplay } = getDisplay(key, value);
          return (
            <div key={key} className={`bg-slate-950/50 border rounded-2xl p-5 flex justify-between items-center transition-all ${adjusted ? "border-cyan-500/30" : "border-slate-800"}`}>
              <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase">{meta.label}</p>
                {adjusted && <p className="text-[10px] font-mono text-slate-700 mt-0.5">base: {baseDisplay}</p>}
              </div>
              <div className="text-right">
                <div className="flex items-baseline gap-1 justify-end">
                  <span className={`text-xl font-bold ${meta.valueColor}`}>{display}</span>
                  {meta.unit && <span className="text-[10px] text-slate-600 font-mono">{meta.unit}</span>}
                </div>
                {adjusted && <div className="text-[9px] text-cyan-700 font-mono mt-0.5 tracking-wider">ADJUSTED</div>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// STATUS BADGE
// =============================================================================
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

// =============================================================================
// SCAN TAB
// FIX 1: accumulatedOptimizedRef + accumulatedImagesRef → auto-analysis
// FIX 2: generatePreview() → full-color data URL, no blur, persists
// =============================================================================
function ScanTab({ onAddToLog }) {
  const [images, setImages]             = useState([]);
  const [optimizedFiles, setOptimizedFiles] = useState([]);
  const [loading, setLoading]           = useState(false);
  const [loadingMsg, setLoadingMsg]     = useState("");
  const [results, setResults]           = useState(null);
  const [error, setError]               = useState(null);
  const [activeIndex, setActiveIndex]   = useState(0);
  const [activeTab, setActiveTab]       = useState("per_100g");
  const [cropperQueue, setCropperQueue] = useState([]);
  const [cropperFile, setCropperFile]   = useState(null);
  const [saveModal, setSaveModal]       = useState(null);
  const fileInputRef = useRef(null);

  const accumulatedOptimizedRef = useRef([]);
  const accumulatedImagesRef    = useRef([]);

  const stateRef = useRef({ images, optimizedFiles, results, activeIndex, activeTab });
  useEffect(() => { stateRef.current = { images, optimizedFiles, results, activeIndex, activeTab }; },
    [images, optimizedFiles, results, activeIndex, activeTab]);
  useEffect(() => {
    const handler = () => {
      if (document.visibilityState === "visible") {
        const s = stateRef.current;
        setImages(s.images); setOptimizedFiles(s.optimizedFiles); setResults(s.results);
        setActiveIndex(s.activeIndex); setActiveTab(s.activeTab);
      }
    };
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);

  const switchToIndex = useCallback((i, resultArr) => {
    setActiveIndex(i);
    const r = (resultArr ?? results)?.[i];
    if (r?.per_100g && Object.keys(r.per_100g).length > 0) setActiveTab("per_100g");
    else if (r?.per_serving && Object.keys(r.per_serving).length > 0) setActiveTab("per_serving");
  }, [results]);

  const handleImageUpload = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    e.target.value = "";
    accumulatedOptimizedRef.current = [];
    accumulatedImagesRef.current    = [];
    setCropperQueue(files);
    setCropperFile(files[0]);
  }, []);

  const handleCropConfirm = useCallback(async (cropData) => {
    const file      = cropperFile;
    const remaining = cropperQueue.slice(1);

    setLoadingMsg("Processing...");
    // Generate preview immediately (sync blob URL from original file — works on iOS)
    // Run API-optimized version in background
    const previewUrl = generatePreview(file);
    const optimized  = await applyPipelineToFile(file, cropData);

    const imageEntry = { file, preview: previewUrl, cropData, persistentUrl: null };

    // FIX: append to both state AND refs in the same order so indices always match
    setImages(prev => [...prev, imageEntry]);
    setOptimizedFiles(prev => [...prev, optimized]);
    setResults(null); setError(null); setActiveIndex(0);
    setLoadingMsg("");

    accumulatedOptimizedRef.current = [...accumulatedOptimizedRef.current, optimized];
    accumulatedImagesRef.current    = [...accumulatedImagesRef.current, imageEntry];

    if (remaining.length > 0) {
      setCropperQueue(remaining); setCropperFile(remaining[0]);
    } else {
      // FIX: just close the cropper — do NOT auto-trigger analysis
      // User presses the Analyze button manually when ready
      setCropperQueue([]); setCropperFile(null);
    }
  }, [cropperFile, cropperQueue]);

  const handleCropCancel = useCallback(() => {
    accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setCropperQueue([]); setCropperFile(null);
  }, []);

  const handleReCrop = useCallback((index) => {
    const img = images[index];
    if (!img) return;
    setImages(prev => prev.filter((_, i) => i !== index));
    setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setResults(null); setError(null);
    accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setCropperQueue([img.file]); setCropperFile(img.file);
  }, [images]);

  const removeImage = useCallback((index) => {
    setImages(prev => {
      const img = prev[index];
      // Only revoke local blob previews, not persistent S3 URLs
      if (img?.preview && img.preview.startsWith("blob:") && !img.persistentUrl) {
        URL.revokeObjectURL(img.preview);
      }
      return prev.filter((_, i) => i !== index);
    });
    setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setActiveIndex(prev => (prev >= index && prev > 0 ? prev - 1 : prev));
    setResults(null); setError(null);
  }, []);

  const handleClear = useCallback(() => {
    // Revoke all preview blob URLs to free memory
    images.forEach(img => {
      if (img?.preview && img.preview.startsWith("blob:")) URL.revokeObjectURL(img.preview);
    });
    accumulatedOptimizedRef.current = []; accumulatedImagesRef.current = [];
    setImages([]); setOptimizedFiles([]); setResults(null); setError(null); setActiveIndex(0); setActiveTab("per_100g");
  }, [images]);

  const handleAnalyze = useCallback(async () => {
    if (results) { handleClear(); return; }
    if (!images.length) return;
    // FIX: always use accumulatedOptimizedRef — these are guaranteed to be in the
    // same order as images state since both are appended together in handleCropConfirm.
    // Fallback to optimizedFiles state if ref is somehow empty (e.g. after page refresh).
    const filesToSend = accumulatedOptimizedRef.current.length === images.length
      ? accumulatedOptimizedRef.current
      : optimizedFiles.length === images.length
        ? optimizedFiles
        : images.map(i => i.file);
    runAnalysis({ optimizedFiles: filesToSend, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex });
  }, [images, optimizedFiles, results, handleClear, switchToIndex]);

  const currentResult  = results?.[activeIndex] ?? null;
  const currentPreview = images[activeIndex]?.persistentUrl || images[activeIndex]?.preview || null;
  const allOptimized   = optimizedFiles.length === images.length && images.length > 0;

  return (
    <>
      {cropperFile && <ImageCropper file={cropperFile} onConfirm={handleCropConfirm} onCancel={handleCropCancel} />}
      {saveModal && <SaveToFolderModal result={saveModal.result} imageId={saveModal.imageId} onClose={() => setSaveModal(null)} />}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4 space-y-4">
          {images.length === 0 ? (
            <div className="relative border-2 border-dashed border-slate-800 bg-slate-900/10 rounded-3xl p-6 h-[400px] flex flex-col items-center justify-center cursor-pointer hover:border-slate-700 transition-colors"
              onClick={() => fileInputRef.current?.click()}>
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
              <Upload className="h-8 w-8 text-slate-600 mb-2" />
              <p className="text-slate-500 text-sm">Click to upload — auto-analyzes after crop</p>
              <p className="text-slate-700 text-[10px] font-mono mt-1">CROP → OPTIMIZE → AUTO-ANALYZE</p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="relative border border-slate-800 bg-slate-900/30 rounded-3xl p-4 h-[280px] flex items-center justify-center overflow-hidden">
                {currentPreview
                  ? <img src={currentPreview} alt={`Preview ${activeIndex + 1}`} className="max-h-full max-w-full rounded-2xl object-contain border border-slate-800" />
                  : <div className="text-slate-700 text-xs font-mono">No preview</div>}
                {images.length > 1 && (
                  <>
                    <button onClick={() => setActiveIndex(i => Math.max(0, i - 1))} disabled={activeIndex === 0}
                      className="absolute left-3 top-1/2 -translate-y-1/2 bg-slate-900/80 border border-slate-700 rounded-full p-1 disabled:opacity-20 hover:border-slate-500 transition-colors">
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <button onClick={() => setActiveIndex(i => Math.min(images.length - 1, i + 1))} disabled={activeIndex === images.length - 1}
                      className="absolute right-3 top-1/2 -translate-y-1/2 bg-slate-900/80 border border-slate-700 rounded-full p-1 disabled:opacity-20 hover:border-slate-500 transition-colors">
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </>
                )}
              </div>
              <div className="flex gap-2 flex-wrap">
                {images.map((img, i) => {
                  const thumbSrc = img.persistentUrl || img.preview;
                  return (
                    <div key={i} onClick={() => setActiveIndex(i)}
                      className={`relative w-14 h-14 rounded-xl overflow-hidden border cursor-pointer flex-shrink-0 transition-all ${i === activeIndex ? "border-cyan-500/70" : "border-slate-800 opacity-60 hover:opacity-100"}`}>
                      {thumbSrc ? <img src={thumbSrc} alt={`Thumb ${i + 1}`} className="w-full h-full object-cover" /> : <div className="w-full h-full bg-slate-800 flex items-center justify-center"><Upload className="h-4 w-4 text-slate-600" /></div>}
                      {!results && (
                        <>
                          <button onClick={e => { e.stopPropagation(); removeImage(i); }} className="absolute top-0.5 right-0.5 bg-slate-900/90 rounded-full p-0.5 hover:text-rose-400 z-10"><X className="h-2.5 w-2.5" /></button>
                          <button onClick={e => { e.stopPropagation(); handleReCrop(i); }} className="absolute bottom-0.5 right-0.5 bg-slate-900/90 rounded-full p-0.5 hover:text-cyan-400 z-10"><Crop className="h-2.5 w-2.5" /></button>
                        </>
                      )}
                      <div className="absolute bottom-0.5 left-0.5 bg-slate-900/90 rounded text-[8px] font-mono text-slate-400 px-1">{i + 1}</div>
                      {img.cropData && <div className="absolute top-0.5 left-0.5 bg-cyan-500/80 rounded text-[7px] font-mono text-slate-950 px-1">✂</div>}
                    </div>
                  );
                })}
                {!results && (
                  <div onClick={() => fileInputRef.current?.click()}
                    className="w-14 h-14 rounded-xl border border-dashed border-slate-700 flex items-center justify-center cursor-pointer hover:border-slate-500 transition-colors flex-shrink-0">
                    <Upload className="h-4 w-4 text-slate-600" />
                  </div>
                )}
              </div>
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
              <p className="text-[10px] font-mono text-slate-600 text-center">
                {images.length} IMAGE{images.length !== 1 ? "S" : ""} QUEUED
                {allOptimized && !loading && !results && <span className="text-cyan-900 ml-2">· OPTIMIZED ✓</span>}
                {loadingMsg && <span className="text-amber-800 ml-2">· {loadingMsg.toUpperCase()}</span>}
              </p>
            </div>
          )}

          <button onClick={handleAnalyze} disabled={loading || images.length === 0 || !!cropperFile}
            className={`w-full py-4 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 shadow-xl
              ${results ? "bg-slate-900 text-rose-400 border border-rose-900/30"
                : images.length === 0 || !!cropperFile ? "bg-slate-900 text-slate-700 border border-slate-800 cursor-not-allowed"
                : "bg-cyan-500 text-slate-950 hover:bg-cyan-400"}`}>
            {loading ? <Loader2 className="animate-spin h-5 w-5" /> : results ? <RefreshCcw className="h-5 w-5" /> : <Zap className="h-5 w-5" />}
            {loading ? (loadingMsg || `PROCESSING ${images.length} IMAGE${images.length !== 1 ? "S" : ""}...`)
              : results ? "CLEAR_SESSION"
              : `RUN_EXTRACTION${images.length > 1 ? ` (${images.length})` : ""}`}
          </button>

          {error && !loading && images.length > 0 && !results && (
            <button onClick={handleAnalyze}
              className="w-full py-3 rounded-2xl font-bold text-sm bg-slate-900 text-amber-400 border border-amber-900/40 hover:border-amber-700/50 transition-all flex items-center justify-center gap-2">
              <RefreshCcw className="h-4 w-4" /> RETRY (IMAGES PRESERVED)
            </button>
          )}

          {currentResult && (
            <button onClick={() => onAddToLog({ name: `Label ${activeIndex + 1}`, nutrition: currentResult })}
              className="w-full py-3 rounded-2xl font-bold text-sm bg-emerald-900/30 text-emerald-400 border border-emerald-900/40 hover:border-emerald-700/50 transition-all flex items-center justify-center gap-2">
              <Plus className="h-4 w-4" /> LOG THIS ITEM
            </button>
          )}
        </div>

        <div className="lg:col-span-8">
          {error && !loading && (
            <div className="mb-4 bg-rose-950/40 border border-rose-800/50 rounded-2xl px-5 py-4 flex items-start gap-3">
              <span className="text-rose-400 text-sm font-mono font-bold mt-0.5">ERR</span>
              <div className="flex-1"><p className="text-rose-300 text-sm">{error}</p></div>
              <button onClick={() => setError(null)} className="text-rose-700 hover:text-rose-400"><X className="h-4 w-4" /></button>
            </div>
          )}

          {loading ? (
            <div className="h-full flex flex-col items-center justify-center space-y-6 py-20">
              <Loader2 className="animate-spin text-cyan-400 h-8 w-8" />
              <p className="text-cyan-400 text-xs uppercase">{loadingMsg || "Running Ingestion..."}</p>
              <p className="text-slate-700 text-[10px] font-mono">Auto-retry · Timeout: {REQUEST_TIMEOUT_MS / 1000}s</p>
            </div>
          ) : results ? (
            <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8 space-y-6">
              {results.length > 1 && (
                <div className="space-y-2">
                  <p className="text-[10px] font-mono text-slate-600">SELECT RESULT</p>
                  <div className="flex gap-2 flex-wrap">
                    {results.map((_, i) => {
                      const thumbSrc = images[i]?.persistentUrl || images[i]?.preview;
                      return (
                        <button key={i} onClick={() => switchToIndex(i, results)}
                          className={`flex items-center gap-2 px-2 py-1.5 rounded-xl border transition-all ${activeIndex === i ? "border-cyan-500/50 bg-cyan-500/10" : "border-slate-800 hover:border-slate-600"}`}>
                          <div className={`w-10 h-10 rounded-lg overflow-hidden border flex-shrink-0 ${activeIndex === i ? "border-cyan-500/50" : "border-slate-700"}`}>
                            {thumbSrc ? <img src={thumbSrc} alt={`Label ${i + 1}`} className="w-full h-full object-cover" /> : <div className="w-full h-full bg-slate-800" />}
                          </div>
                          <div className={`text-[10px] font-mono font-bold ${activeIndex === i ? "text-cyan-400" : "text-slate-500"}`}>IMG_{String(i + 1).padStart(2, "0")}</div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="flex items-center gap-3 bg-slate-950/50 border border-slate-800 rounded-2xl p-3">
                <div className="w-16 h-16 rounded-xl overflow-hidden border border-slate-700 flex-shrink-0 bg-slate-800">
                  {currentPreview ? <img src={currentPreview} alt="Active label" className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center"><Database className="h-4 w-4 text-slate-700" /></div>}
                </div>
                <div className="flex-1">
                  <div className="text-[10px] font-mono text-slate-600">ANALYZING LABEL</div>
                  <div className="text-sm font-bold text-slate-300 mt-0.5">Image {activeIndex + 1} of {results.length}</div>
                  {images[activeIndex]?.cropData && <div className="text-[9px] font-mono text-cyan-800 mt-0.5">✂ CROPPED · OPTIMIZED</div>}
                  {images[activeIndex]?.persistentUrl && <div className="text-[9px] font-mono text-emerald-900 mt-0.5">☁ PERSISTED</div>}
                </div>
                <button onClick={() => setSaveModal({ result: currentResult, imageId: currentResult?.image_id || "" })}
                  className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-colors text-xs font-bold flex-shrink-0">
                  <BookmarkPlus className="h-4 w-4" /> SAVE
                </button>
              </div>

              <div className="flex p-1 bg-slate-950 border border-slate-800 rounded-xl w-full max-w-sm">
                {["per_100g", "per_serving"].map(tab => {
                  const hasData = currentResult?.[tab] && Object.keys(currentResult[tab]).length > 0;
                  return (
                    <button key={tab} onClick={() => hasData && setActiveTab(tab)} disabled={!hasData}
                      className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${!hasData ? "text-slate-700 cursor-not-allowed" : activeTab === tab ? tab === "per_100g" ? "bg-cyan-500 text-slate-950 shadow-lg" : "bg-purple-500 text-slate-950 shadow-lg" : "text-slate-500 hover:text-slate-300"}`}>
                      {tab === "per_100g" ? "PER 100G" : "PER SERVING"}
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div className="flex items-center gap-2">
                  <TableProperties className="h-4 w-4 text-slate-500" />
                  <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">{activeTab === "per_100g" ? "Per 100g" : "Per Serving"} Schema</h2>
                </div>
                {activeTab === "per_serving" && currentResult?.per_serving?.size && (
                  <span className="text-[10px] bg-purple-500/10 text-purple-400 px-3 py-1 rounded-full border border-purple-500/20 font-mono">SIZE: {currentResult.per_serving.size}</span>
                )}
              </div>

              <NutrientGrid key={activeIndex} data={currentResult?.[activeTab]} activeTab={activeTab} per100gData={currentResult?.per_100g} />

              <details className="group border border-slate-800 rounded-xl bg-slate-950/30">
                <summary className="p-3 text-[9px] font-mono text-slate-600 cursor-pointer list-none flex justify-between items-center uppercase">
                  <span>{">"} raw_data_payload [{activeIndex + 1}/{results.length}]</span>
                  <span className="group-open:rotate-180 transition-transform">▼</span>
                </summary>
                <div className="p-4 pt-0"><pre className="text-[10px] text-cyan-800 font-mono overflow-x-auto">{JSON.stringify(currentResult, null, 2)}</pre></div>
              </details>
            </div>
          ) : (
            <div className="h-full border border-slate-900 bg-slate-900/5 rounded-3xl flex flex-col items-center justify-center p-12 text-center text-slate-700 min-h-[300px]">
              <LayoutPanelLeft className="h-10 w-10 mb-4 opacity-10" />
              <p className="text-sm italic">{images.length > 0 && !loading ? "Analysis running automatically..." : "Upload an image to begin"}</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// =============================================================================
// ROOT APP
// =============================================================================
export default function App() {
  const [activeMainTab, setActiveMainTab] = useState("scan");
  const [addToLogItem, setAddToLogItem]   = useState(null);
  const [logRefreshKey, setLogRefreshKey]         = useState(0);
  // FIX: increment every time user navigates to Library so it remounts and reloads
  const [libraryMountKey, setLibraryMountKey]     = useState(0);

  const handleAddToLog = useCallback((item) => { setAddToLogItem(item); }, []);
  const handleLogAdded = useCallback(() => { setLogRefreshKey(k => k + 1); }, []);

  const handleTabChange = useCallback((tabId) => {
    setActiveMainTab(tabId);
    // FIX: remount LibraryTab every time user clicks Library so folders/items reload fresh
    if (tabId === "library") setLibraryMountKey(k => k + 1);
  }, []);

  const TABS = [
    { id: "scan",    label: "Scan",    icon: <Zap      className="h-4 w-4" /> },
    { id: "library", label: "Library", icon: <Folder   className="h-4 w-4" /> },
    { id: "tracker", label: "Tracker", icon: <BarChart3 className="h-4 w-4" /> },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans">
      {addToLogItem && <AddToLogModal item={addToLogItem} onClose={() => setAddToLogItem(null)} onAdded={handleLogAdded} />}

      <div className="max-w-6xl mx-auto p-4 md:p-8">
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-cyan-500/20 p-2 rounded-lg border border-cyan-500/30"><Database className="h-8 w-8 text-cyan-400" /></div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white">NutriScan <span className="text-cyan-400">Pipeline</span></h1>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em]">Data Engineering Edition v3</p>
            </div>
          </div>
          <div className="flex gap-3"><StatusBadge icon={<Cloud className="h-3 w-3" />} label="API" status={API_URL} color="text-cyan-500" /></div>
        </header>

        <div className="flex gap-1 p-1 bg-slate-900/60 border border-slate-800 rounded-2xl mb-8 w-fit">
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => handleTabChange(tab.id)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${activeMainTab === tab.id ? "bg-slate-800 text-slate-100 shadow" : "text-slate-500 hover:text-slate-300"}`}>
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* ScanTab: display:none preserves upload/crop state across tab switches */}
        <div style={{ display: activeMainTab === "scan" ? "block" : "none" }}>
          <ScanTab onAddToLog={handleAddToLog} />
        </div>
        {/* LibraryTab: remounts on every visit via key so folders always reload fresh */}
        {activeMainTab === "library" && (
          <LibraryTab key={libraryMountKey} onAddToLog={handleAddToLog} />
        )}
        {/* TrackerTab: display:none + refreshKey for instant macro updates */}
        <div style={{ display: activeMainTab === "tracker" ? "block" : "none" }}>
          <TrackerTab refreshKey={logRefreshKey} />
        </div>
      </div>
    </div>
  );
}