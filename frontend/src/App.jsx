import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, Zap, Loader2, RefreshCcw,
  Database, Cloud, TableProperties, LayoutPanelLeft, Scale, X, ChevronLeft, ChevronRight,
  Crop, Check
} from "lucide-react";

const API_URL = import.meta.env.DEV
  ? "http://localhost:8000"
  : (import.meta.env.VITE_API_URL || "https://nutritionaltracker.onrender.com");

console.log("🔧 API_URL:", API_URL);

// ---------- CONSTANTS ----------
const MAX_IMAGE_PX = 640;
const JPEG_QUALITY = 0.5;
const REQUEST_TIMEOUT_MS = 25000;
const MAX_FRONTEND_RETRIES = 2;
const RETRY_DELAY_MS = [1500, 3000];

// ---------- NUTRIENT META ----------
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

// ---------- APPLY CROP + GRAYSCALE + OPTIMIZE ----------
// cropData: { x, y, width, height } in natural image pixels, or null (use full image)
async function applyPipelineToFile(file, cropData) {
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);

      // Step 1: determine source region (crop or full image)
      const srcX      = cropData ? cropData.x      : 0;
      const srcY      = cropData ? cropData.y      : 0;
      const srcWidth  = cropData ? cropData.width  : img.naturalWidth;
      const srcHeight = cropData ? cropData.height : img.naturalHeight;

      // Step 2: scale to MAX_IMAGE_PX
      const scale   = Math.min(1, MAX_IMAGE_PX / Math.max(srcWidth, srcHeight));
      const targetW = Math.max(1, Math.round(srcWidth  * scale));
      const targetH = Math.max(1, Math.round(srcHeight * scale));

      const canvas = document.createElement("canvas");
      canvas.width  = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext("2d");

      // Step 3: draw cropped + resized region
      ctx.drawImage(img, srcX, srcY, srcWidth, srcHeight, 0, 0, targetW, targetH);

      // Step 4: grayscale conversion
      const imageData = ctx.getImageData(0, 0, targetW, targetH);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        data[i] = data[i + 1] = data[i + 2] = gray;
      }
      ctx.putImageData(imageData, 0, 0);

      // Step 5: compress to JPEG
      canvas.toBlob(
        (blob) => {
          if (!blob) { resolve(file); return; }
          const optimized = new File(
            [blob],
            file.name.replace(/\.[^.]+$/, ".jpg"),
            { type: "image/jpeg", lastModified: Date.now() }
          );
          console.log(
            `📉 Pipeline: ${file.size} → ${optimized.size} bytes | ` +
            `crop(${Math.round(srcX)},${Math.round(srcY)},${Math.round(srcWidth)}×${Math.round(srcHeight)}) → ${targetW}×${targetH} → grayscale → JPEG`
          );
          resolve(optimized);
        },
        "image/jpeg",
        JPEG_QUALITY
      );
    };

    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

// ---------- FETCH WITH TIMEOUT + RETRY ----------
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
      // Never retry 429
      const isRetryable = status !== 429 && (status === 500 || status === 503 || status === 504);
      const errorText = await response.text().catch(() => `HTTP ${status}`);

      if (!isRetryable || attempt === maxRetries) {
        throw new Error(`Pipeline Error: ${status} — ${errorText}`);
      }

      lastError = new Error(`Retryable error: ${status}`);
      console.warn(`⚠️ Attempt ${attempt + 1} failed (${status}), retrying in ${RETRY_DELAY_MS[attempt] || 3000}ms...`);
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000));

    } catch (err) {
      clearTimeout(timer);

      if (err.name === "AbortError") {
        lastError = new Error("Request timed out. Please try again.");
        console.warn(`⏱️ Attempt ${attempt + 1} timed out`);
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000));
          continue;
        }
      } else if (err.message.startsWith("Pipeline Error:")) {
        throw err;
      } else {
        lastError = err;
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000));
          continue;
        }
      }
    }
  }

  throw lastError || new Error("Request failed after retries");
}

// =============================================================================
// CROPPER COMPONENT — native canvas, full touch + mouse support
// =============================================================================
// Emits cropData: { x, y, width, height } in natural image pixels
function ImageCropper({ file, onConfirm, onCancel }) {
  const canvasRef       = useRef(null);
  const containerRef    = useRef(null);
  const imgRef          = useRef(null);
  const isDragging      = useRef(false);
  const dragMode        = useRef(null); // "move" | "nw" | "ne" | "sw" | "se"
  const dragStart       = useRef({ x: 0, y: 0 });
  const cropAtDragStart = useRef(null);

  // cropRect in canvas-space pixels: { x, y, w, h }
  const [cropRect, setCropRect]     = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [imgScale, setImgScale]     = useState(1); // canvas px / natural px
  const [imageLoaded, setImageLoaded] = useState(false);

  // Load image and compute canvas dimensions + default crop
  useEffect(() => {
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      const container = containerRef.current;
      const maxW = container ? container.clientWidth : 360;
      const maxH = Math.min(window.innerHeight * 0.55, 460);
      const s   = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
      const cw  = Math.round(img.naturalWidth  * s);
      const ch  = Math.round(img.naturalHeight * s);
      setCanvasSize({ w: cw, h: ch });
      setImgScale(s);
      // Default crop: center 80%
      const pad = 0.1;
      setCropRect({
        x: Math.round(cw * pad),
        y: Math.round(ch * pad),
        w: Math.round(cw * 0.8),
        h: Math.round(ch * 0.8),
      });
      setImageLoaded(true);
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Redraw canvas whenever cropRect changes
  useEffect(() => {
    if (!imageLoaded || !cropRect || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const { w: cw, h: ch } = canvasSize;
    ctx.clearRect(0, 0, cw, ch);

    // Draw image
    ctx.drawImage(imgRef.current, 0, 0, cw, ch);

    // Darken outside crop
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, cw, cropRect.y);
    ctx.fillRect(0, cropRect.y + cropRect.h, cw, ch - cropRect.y - cropRect.h);
    ctx.fillRect(0, cropRect.y, cropRect.x, cropRect.h);
    ctx.fillRect(cropRect.x + cropRect.w, cropRect.y, cw - cropRect.x - cropRect.w, cropRect.h);

    // Crop border
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);

    // Rule-of-thirds grid lines
    ctx.strokeStyle = "rgba(34,211,238,0.22)";
    ctx.lineWidth   = 1;
    for (let i = 1; i <= 2; i++) {
      const gx = cropRect.x + (cropRect.w / 3) * i;
      const gy = cropRect.y + (cropRect.h / 3) * i;
      ctx.beginPath(); ctx.moveTo(gx, cropRect.y);       ctx.lineTo(gx, cropRect.y + cropRect.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cropRect.x, gy);       ctx.lineTo(cropRect.x + cropRect.w, gy); ctx.stroke();
    }

    // Corner handles
    const hs = 10;
    ctx.fillStyle = "#22d3ee";
    [
      [cropRect.x,              cropRect.y             ],
      [cropRect.x + cropRect.w, cropRect.y             ],
      [cropRect.x,              cropRect.y + cropRect.h],
      [cropRect.x + cropRect.w, cropRect.y + cropRect.h],
    ].forEach(([cx, cy]) => {
      ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
    });
  }, [imageLoaded, cropRect, canvasSize]);

  const getEventPos = (e, canvas) => {
    const rect    = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width  / rect.width),
      y: (clientY - rect.top)  * (canvas.height / rect.height),
    };
  };

  const getHitZone = (pos, rect) => {
    const hs = 18;
    const corners = {
      nw: [rect.x,           rect.y          ],
      ne: [rect.x + rect.w,  rect.y          ],
      sw: [rect.x,           rect.y + rect.h ],
      se: [rect.x + rect.w,  rect.y + rect.h ],
    };
    for (const [key, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(pos.x - cx) <= hs && Math.abs(pos.y - cy) <= hs) return key;
    }
    if (pos.x >= rect.x && pos.x <= rect.x + rect.w &&
        pos.y >= rect.y && pos.y <= rect.y + rect.h) return "move";
    return null;
  };

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const onPointerDown = useCallback((e) => {
    if (!cropRect || !canvasRef.current) return;
    e.preventDefault();
    const pos  = getEventPos(e, canvasRef.current);
    const zone = getHitZone(pos, cropRect);
    if (!zone) return;
    isDragging.current      = true;
    dragMode.current        = zone;
    dragStart.current       = pos;
    cropAtDragStart.current = { ...cropRect };
  }, [cropRect]);

  const onPointerMove = useCallback((e) => {
    if (!isDragging.current || !canvasRef.current || !cropAtDragStart.current) return;
    e.preventDefault();
    const pos = getEventPos(e, canvasRef.current);
    const dx  = pos.x - dragStart.current.x;
    const dy  = pos.y - dragStart.current.y;
    const { w: cw, h: ch } = canvasSize;
    const base = cropAtDragStart.current;
    const MIN  = 40;

    setCropRect(() => {
      let { x, y, w, h } = base;
      const mode = dragMode.current;
      if (mode === "move") {
        x = clamp(base.x + dx, 0, cw - base.w);
        y = clamp(base.y + dy, 0, ch - base.h);
      } else if (mode === "se") {
        w = clamp(base.w + dx, MIN, cw - base.x);
        h = clamp(base.h + dy, MIN, ch - base.y);
      } else if (mode === "sw") {
        const nx = clamp(base.x + dx, 0, base.x + base.w - MIN);
        w = base.x + base.w - nx; x = nx;
        h = clamp(base.h + dy, MIN, ch - base.y);
      } else if (mode === "ne") {
        w = clamp(base.w + dx, MIN, cw - base.x);
        const ny = clamp(base.y + dy, 0, base.y + base.h - MIN);
        h = base.y + base.h - ny; y = ny;
      } else if (mode === "nw") {
        const nx = clamp(base.x + dx, 0, base.x + base.w - MIN);
        const ny = clamp(base.y + dy, 0, base.y + base.h - MIN);
        w = base.x + base.w - nx; x = nx;
        h = base.y + base.h - ny; y = ny;
      }
      return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
    });
  }, [canvasSize]);

  const onPointerUp = useCallback((e) => {
    e.preventDefault();
    isDragging.current = false;
  }, []);

  const handleConfirm = () => {
    if (!cropRect || !imgScale) { onConfirm(null); return; }
    onConfirm({
      x:      Math.round(cropRect.x / imgScale),
      y:      Math.round(cropRect.y / imgScale),
      width:  Math.round(cropRect.w / imgScale),
      height: Math.round(cropRect.h / imgScale),
    });
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/96 flex flex-col items-center justify-center p-4 gap-4">
      <div className="w-full max-w-lg">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-xs font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
              <Crop className="h-4 w-4 text-cyan-400" /> Crop Nutrition Label
            </p>
            <p className="text-[10px] text-slate-600 font-mono mt-0.5">Drag corners to resize · Drag inside to move</p>
          </div>
          <button onClick={onCancel} className="text-slate-600 hover:text-rose-400 transition-colors p-1">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div
          ref={containerRef}
          className="w-full rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 touch-none select-none"
        >
          {imageLoaded ? (
            <canvas
              ref={canvasRef}
              width={canvasSize.w}
              height={canvasSize.h}
              style={{ display: "block", width: "100%", touchAction: "none", cursor: "crosshair" }}
              onMouseDown={onPointerDown}
              onMouseMove={onPointerMove}
              onMouseUp={onPointerUp}
              onMouseLeave={onPointerUp}
              onTouchStart={onPointerDown}
              onTouchMove={onPointerMove}
              onTouchEnd={onPointerUp}
            />
          ) : (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="animate-spin text-cyan-400 h-6 w-6" />
            </div>
          )}
        </div>

        {cropRect && imgScale > 0 && (
          <p className="text-[9px] font-mono text-slate-700 mt-2 text-center">
            CROP REGION: {Math.round(cropRect.w / imgScale)}×{Math.round(cropRect.h / imgScale)}px (natural image)
          </p>
        )}

        <div className="flex gap-3 mt-4">
          <button
            onClick={onCancel}
            className="flex-1 py-3 rounded-xl border border-slate-800 text-slate-500 text-xs font-bold hover:border-slate-600 transition-colors"
          >
            CANCEL
          </button>
          <button
            onClick={() => onConfirm(null)}
            className="flex-1 py-3 rounded-xl border border-slate-700 text-slate-400 text-xs font-bold hover:border-slate-500 transition-colors"
          >
            FULL IMAGE
          </button>
          <button
            onClick={handleConfirm}
            className="flex-1 py-3 rounded-xl bg-cyan-500 text-slate-950 text-xs font-bold hover:bg-cyan-400 transition-colors flex items-center justify-center gap-2"
          >
            <Check className="h-4 w-4" /> CONFIRM
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// APP
// =============================================================================
export default function App() {
  // images: { file, preview, cropData }[]
  const [images, setImages]                 = useState([]);
  const [optimizedFiles, setOptimizedFiles] = useState([]);
  const [loading, setLoading]               = useState(false);
  const [loadingMsg, setLoadingMsg]         = useState("");
  const [results, setResults]               = useState(null);
  const [error, setError]                   = useState(null);
  const [activeIndex, setActiveIndex]       = useState(0);
  const [activeTab, setActiveTab]           = useState("per_100g");
  const fileInputRef = useRef(null);

  // Cropper queue state
  const [cropperQueue, setCropperQueue] = useState([]); // pending File[]
  const [cropperFile, setCropperFile]   = useState(null);

  // Visibility state preservation (keeps state when iPhone locks screen)
  const stateRef = useRef({ images, optimizedFiles, results, activeIndex, activeTab });
  useEffect(() => {
    stateRef.current = { images, optimizedFiles, results, activeIndex, activeTab };
  }, [images, optimizedFiles, results, activeIndex, activeTab]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        const s = stateRef.current;
        setImages(s.images);
        setOptimizedFiles(s.optimizedFiles);
        setResults(s.results);
        setActiveIndex(s.activeIndex);
        setActiveTab(s.activeTab);
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  // File selection → push to crop queue
  const handleImageUpload = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    e.target.value = "";
    setCropperQueue(files);
    setCropperFile(files[0]);
  }, []);

  // Crop confirmed for current file → optimize → advance queue
  const handleCropConfirm = useCallback(async (cropData) => {
    const file      = cropperFile;
    const remaining = cropperQueue.slice(1);

    const preview  = URL.createObjectURL(file);
    const imgEntry = { file, preview, cropData };

    setImages(prev => [...prev, imgEntry]);
    setResults(null);
    setError(null);
    setActiveIndex(0);

    // Non-blocking optimization
    setLoadingMsg("Optimizing...");
    const optimized = await applyPipelineToFile(file, cropData);
    setOptimizedFiles(prev => [...prev, optimized]);
    setLoadingMsg("");

    if (remaining.length > 0) {
      setCropperQueue(remaining);
      setCropperFile(remaining[0]);
    } else {
      setCropperQueue([]);
      setCropperFile(null);
    }
  }, [cropperFile, cropperQueue]);

  const handleCropCancel = useCallback(() => {
    setCropperQueue([]);
    setCropperFile(null);
  }, []);

  // Re-crop an already-added image
  const handleReCrop = useCallback((index) => {
    const img = images[index];
    if (!img) return;
    setImages(prev => prev.filter((_, i) => i !== index));
    setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setResults(null);
    setError(null);
    setCropperQueue([img.file]);
    setCropperFile(img.file);
  }, [images]);

  const removeImage = useCallback((index) => {
    setImages(prev => {
      URL.revokeObjectURL(prev[index]?.preview);
      return prev.filter((_, i) => i !== index);
    });
    setOptimizedFiles(prev => prev.filter((_, i) => i !== index));
    setActiveIndex(prev => (prev >= index && prev > 0 ? prev - 1 : prev));
    setResults(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    images.forEach(img => URL.revokeObjectURL(img.preview));
    setImages([]);
    setOptimizedFiles([]);
    setResults(null);
    setError(null);
    setActiveIndex(0);
    setActiveTab("per_100g");
  }, [images]);

  const switchToIndex = useCallback((i, resultArr) => {
    setActiveIndex(i);
    const r = (resultArr ?? results)?.[i];
    if (r?.per_100g && Object.keys(r.per_100g).length > 0) {
      setActiveTab("per_100g");
    } else if (r?.per_serving && Object.keys(r.per_serving).length > 0) {
      setActiveTab("per_serving");
    }
  }, [results]);

  const handleAnalyze = useCallback(async () => {
    if (results) { handleClear(); return; }
    if (!images.length) return;

    setLoading(true);
    setError(null);

    try {
      const filesToSend = optimizedFiles.length === images.length
        ? optimizedFiles
        : images.map(i => i.file);

      const formData = new FormData();

      if (filesToSend.length === 1) {
        formData.append("file", filesToSend[0]);
        setLoadingMsg("Analyzing label...");
        console.log("📤 Routing to /analyze-label (single)");
        const response = await fetchWithRetry(`${API_URL}/analyze-label`, { method: "POST", body: formData });
        const data = await response.json();
        const arr = Array.isArray(data) ? data : [data];
        setResults(arr);
        switchToIndex(0, arr);
      } else {
        filesToSend.forEach(f => formData.append("files", f));
        setLoadingMsg(`Analyzing ${filesToSend.length} labels...`);
        console.log(`📤 Routing to /analyze-labels (batch: ${filesToSend.length})`);
        const response = await fetchWithRetry(`${API_URL}/analyze-labels`, { method: "POST", body: formData });
        const data = await response.json();
        const arr = Array.isArray(data) ? data : [data];
        setResults(arr);
        switchToIndex(0, arr);
      }
    } catch (err) {
      console.error("❌ Pipeline Failure:", err);
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  }, [images, optimizedFiles, results, handleClear, switchToIndex]);

  const currentResult  = results?.[activeIndex] ?? null;
  const currentPreview = images[activeIndex]?.preview ?? null;
  const allOptimized   = optimizedFiles.length === images.length && images.length > 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans">
      {/* Cropper overlay */}
      {cropperFile && (
        <ImageCropper
          file={cropperFile}
          onConfirm={handleCropConfirm}
          onCancel={handleCropCancel}
        />
      )}

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

        {error && (
          <div className="mb-6 bg-rose-950/40 border border-rose-800/50 rounded-2xl px-5 py-4 flex items-start gap-3">
            <span className="text-rose-400 text-sm font-mono font-bold mt-0.5">ERR</span>
            <div className="flex-1">
              <p className="text-rose-300 text-sm">{error}</p>
              <button onClick={() => setError(null)} className="text-[10px] font-mono text-rose-600 hover:text-rose-400 mt-1 transition-colors">
                DISMISS
              </button>
            </div>
            <button onClick={() => setError(null)} className="text-rose-700 hover:text-rose-400 transition-colors">
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Panel */}
          <div className="lg:col-span-4 space-y-4">
            {images.length === 0 ? (
              <div
                className="relative border-2 border-dashed border-slate-800 bg-slate-900/10 rounded-3xl p-6 h-[400px] flex flex-col items-center justify-center cursor-pointer hover:border-slate-700 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
                <Upload className="h-8 w-8 text-slate-600 mb-2" />
                <p className="text-slate-500 text-sm">Click to ingest images</p>
                <p className="text-slate-700 text-[10px] font-mono mt-1">CROP → OPTIMIZE → ANALYZE</p>
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
                        <>
                          <button
                            onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                            className="absolute top-0.5 right-0.5 bg-slate-900/90 rounded-full p-0.5 hover:text-rose-400 transition-colors z-10"
                          >
                            <X className="h-2.5 w-2.5" />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleReCrop(i); }}
                            className="absolute bottom-0.5 right-0.5 bg-slate-900/90 rounded-full p-0.5 hover:text-cyan-400 transition-colors z-10"
                            title="Re-crop"
                          >
                            <Crop className="h-2.5 w-2.5" />
                          </button>
                        </>
                      )}
                      <div className="absolute bottom-0.5 left-0.5 bg-slate-900/90 rounded text-[8px] font-mono text-slate-400 px-1">
                        {i + 1}
                      </div>
                      {img.cropData && (
                        <div className="absolute top-0.5 left-0.5 bg-cyan-500/80 rounded text-[7px] font-mono text-slate-950 px-1">
                          ✂
                        </div>
                      )}
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

                <input ref={fileInputRef} type="file" className="hidden" onChange={handleImageUpload} accept="image/*" multiple capture="environment" />

                <p className="text-[10px] font-mono text-slate-600 text-center">
                  {images.length} IMAGE{images.length !== 1 ? "S" : ""} QUEUED
                  {allOptimized && <span className="text-cyan-900 ml-2">· OPTIMIZED ✓</span>}
                  {loadingMsg && <span className="text-amber-800 ml-2">· {loadingMsg.toUpperCase()}</span>}
                </p>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={loading || images.length === 0 || !!cropperFile}
              className={`w-full py-4 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 shadow-xl
                ${results
                  ? "bg-slate-900 text-rose-400 border border-rose-900/30"
                  : images.length === 0 || !!cropperFile
                    ? "bg-slate-900 text-slate-700 border border-slate-800 cursor-not-allowed"
                    : "bg-cyan-500 text-slate-950 hover:bg-cyan-400"
                }`}
            >
              {loading
                ? <Loader2 className="animate-spin h-5 w-5" />
                : results
                  ? <RefreshCcw className="h-5 w-5" />
                  : <Zap className="h-5 w-5" />}
              {loading
                ? (loadingMsg || `PROCESSING ${images.length} IMAGE${images.length !== 1 ? "S" : ""}...`)
                : results
                  ? "CLEAR_SESSION"
                  : `RUN_EXTRACTION${images.length > 1 ? ` (${images.length})` : ""}`}
            </button>

            {error && !loading && images.length > 0 && !results && (
              <button
                onClick={handleAnalyze}
                className="w-full py-3 rounded-2xl font-bold text-sm bg-slate-900 text-amber-400 border border-amber-900/40 hover:border-amber-700/50 transition-all flex items-center justify-center gap-2"
              >
                <RefreshCcw className="h-4 w-4" />
                RETRY (IMAGES PRESERVED)
              </button>
            )}
          </div>

          {/* Right Panel */}
          <div className="lg:col-span-8">
            {loading ? (
              <div className="h-full flex flex-col items-center justify-center space-y-6">
                <Loader2 className="animate-spin text-cyan-400 h-8 w-8" />
                <p className="text-cyan-400 text-xs uppercase">
                  {loadingMsg || `Running Batch Ingestion${images.length > 1 ? ` — ${images.length} Images` : ""}...`}
                </p>
                <p className="text-slate-700 text-[10px] font-mono">Auto-retry enabled · Timeout: {REQUEST_TIMEOUT_MS / 1000}s</p>
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
                    {images[activeIndex]?.cropData && (
                      <div className="text-[9px] font-mono text-cyan-800 mt-0.5">✂ CROPPED · GRAYSCALE · {MAX_IMAGE_PX}px · JPEG {Math.round(JPEG_QUALITY * 100)}%</div>
                    )}
                    {!images[activeIndex]?.cropData && allOptimized && (
                      <div className="text-[9px] font-mono text-slate-700 mt-0.5">GRAYSCALE · {MAX_IMAGE_PX}px · JPEG {Math.round(JPEG_QUALITY * 100)}%</div>
                    )}
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

// ---------- NUTRIENT GRID ----------
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

// ---------- STATUS BADGE ----------
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
