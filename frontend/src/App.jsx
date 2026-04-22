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

const MAX_IMAGE_PX         = 1024;
const JPEG_QUALITY         = 0.85;
const REQUEST_TIMEOUT_MS   = 25000;
const MAX_FRONTEND_RETRIES = 2;
const RETRY_DELAY_MS       = [1500, 3000];

// =============================================================================
// SUPABASE CLIENT
// =============================================================================
const SUPABASE_URL      = "https://zdmsfftfqnajanpbvcgn.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpkbXNmZnRmcW5hamFucGJ2Y2duIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3MTI4NTQsImV4cCI6MjA5MjI4ODg1NH0.Hro4TSxUz9EAOfsxQ4Fg0RsvHO2yi7YhthmT4GJ3Uio";

function generateCodeVerifier() {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode(...array)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}
async function generateCodeChallenge(verifier) {
  const data = new TextEncoder().encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return btoa(String.fromCharCode(...new Uint8Array(digest))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

// Minimal Supabase auth helper — no SDK needed
const supabase = {
  async signInWithOtp(email) {
    const res = await fetch(`${SUPABASE_URL}/auth/v1/otp`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "apikey": SUPABASE_ANON_KEY },
      body: JSON.stringify({ email, create_user: true }),
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.message || "Failed to send magic link"); }
    return res.json();
  },
  getSession() {
    try {
      // Supabase stores session in localStorage under this key
      const raw = localStorage.getItem(`sb-zdmsfftfqnajanpbvcgn-auth-token`);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      // Check expiry
      if (parsed?.expires_at && parsed.expires_at * 1000 < Date.now()) {
        console.log("[auth] session expired, removing. expires_at:", parsed.expires_at, "now:", Math.floor(Date.now()/1000));
        localStorage.removeItem(`sb-zdmsfftfqnajanpbvcgn-auth-token`);
        return null;
      }
      return parsed;
    } catch { return null; }
  },
  signOut() {
    localStorage.removeItem(`sb-zdmsfftfqnajanpbvcgn-auth-token`);
    window.location.reload();
  },
  async signInWithGoogle() {
    const verifier   = generateCodeVerifier();
    const challenge  = await generateCodeChallenge(verifier);
    sessionStorage.setItem('supabase_pkce_verifier', verifier);
    const redirectTo = encodeURIComponent(window.location.origin);
    window.location.href = `${SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to=${redirectTo}&code_challenge=${challenge}&code_challenge_method=S256`;
  },
  // Handle callback from both magic link (hash) and Google OAuth (PKCE code in query params)
  async handleAuthCallback() {
    console.log("[auth] handleAuthCallback — hash:", window.location.hash, "search:", window.location.search);
    // Implicit flow — tokens in URL hash (magic link)
    const hash = window.location.hash;
    if (hash.includes("access_token")) {
      console.log("[auth] implicit flow detected");
      const params       = new URLSearchParams(hash.replace("#", ""));
      const accessToken  = params.get("access_token");
      const refreshToken = params.get("refresh_token");
      const expiresAt    = parseInt(params.get("expires_at") || "0");
      console.log("[auth] implicit — expiresAt:", expiresAt, "hasToken:", !!accessToken);
      if (!accessToken) return null;
      const session = { access_token: accessToken, refresh_token: refreshToken, expires_at: expiresAt };
      localStorage.setItem(`sb-zdmsfftfqnajanpbvcgn-auth-token`, JSON.stringify(session));
      window.history.replaceState({}, document.title, window.location.pathname);
      return session;
    }
    // PKCE flow — code in query params (Google OAuth)
    const searchParams = new URLSearchParams(window.location.search);
    const code  = searchParams.get("code");
    const error = searchParams.get("error");
    console.log("[auth] URL search:", window.location.search);
    console.log("[auth] code:", code, "error:", error);
    if (error) {
      console.error("[auth] OAuth error:", error, searchParams.get("error_description"));
      return null;
    }
    if (code) {
      const verifier = sessionStorage.getItem('supabase_pkce_verifier');
      console.log("[auth] verifier found:", !!verifier);
      sessionStorage.removeItem('supabase_pkce_verifier');
      if (verifier) {
        const res = await fetch(`${SUPABASE_URL}/auth/v1/token?grant_type=pkce`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'apikey': SUPABASE_ANON_KEY },
          body: JSON.stringify({ auth_code: code, code_verifier: verifier }),
        });
        const data = await res.json();
        console.log("[auth] token exchange status:", res.status, "data:", data);
        if (res.ok) {
          const session = {
            access_token:  data.access_token,
            refresh_token: data.refresh_token,
            expires_at:    Math.floor(Date.now() / 1000) + (data.expires_in || 3600),
          };
          localStorage.setItem(`sb-zdmsfftfqnajanpbvcgn-auth-token`, JSON.stringify(session));
          window.history.replaceState({}, document.title, window.location.pathname);
          return session;
        }
      }
    }
    return null;
  },
};

// =============================================================================
// PALETTE — injected as CSS vars, used everywhere via style={}
// =============================================================================
const PALETTE_CSS = `
  :root {
    --teal:      #006D77;
    --teal-lt:   #E0F2F3;
    --teal-md:   #004E56;
    --teal-dk:   #003940;
    --purple:    #8C86AA;
    --purp-lt:   #EEEDF6;
    --mint:      #AEF6C7;
    --mint-dk:   #1A6B3C;
    --brown:     #583E23;
    --brown-lt:  #F5EDE0;
    --brown-md:  #3D2B17;
    --orange:    #B66D0D;
    --orange-lt: #FDF0DC;
    --white:     #FDFCF9;
    --off:       #F2EEE7;
    --off2:      #EAE3D8;
    --border:    #D9D0C4;
    --border2:   #C4B9A8;
    --text:      #2C2017;
    --muted:     #7A6A55;
    --danger:    #C0392B;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Georgia', serif; background: var(--off); color: var(--text); }
  input, select, textarea, button { font-family: inherit; }
  input:focus, select:focus { outline: none; }
`;

// =============================================================================
// NUTRIENT META
// =============================================================================
const NUTRIENT_META = {
  calories:      { label: "Calories",       unit: "kcal", color: "var(--orange)"  },
  fat:           { label: "Total Fat",       unit: "g",    color: "var(--brown)"   },
  saturated_fat: { label: "Saturated Fat",   unit: "g",    color: "var(--danger)"  },
  carbohydrates: { label: "Carbohydrates",   unit: "g",    color: "var(--purple)"  },
  sugars:        { label: "of which Sugars", unit: "g",    color: "var(--purple)"  },
  fibre:         { label: "Dietary Fibre",   unit: "g",    color: "var(--mint-dk)" },
  protein:       { label: "Protein",         unit: "g",    color: "var(--teal)"    },
  sodium:        { label: "Sodium",          unit: "g",    color: "var(--muted)"   },
};

function getFallbackMeta(key) {
  return { label: key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()), unit: "", color: "var(--muted)" };
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

function normalizeCalories(value) {
  const num = parseNumeric(value);
  if (num === null) return value;
  if (typeof value === "string" && /kj|kilojoule/i.test(value)) return Math.round(num / 4.184);
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
// IMAGE PIPELINE
// =============================================================================
async function applyPipelineToFile(file, cropData) {
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const srcX = cropData ? cropData.x : 0; const srcY = cropData ? cropData.y : 0;
      const srcW = cropData ? cropData.width : img.naturalWidth;
      const srcH = cropData ? cropData.height : img.naturalHeight;
      const scale = Math.min(1, MAX_IMAGE_PX / Math.max(srcW, srcH));
      const tW = Math.max(1, Math.round(srcW * scale)); const tH = Math.max(1, Math.round(srcH * scale));
      const canvas = document.createElement("canvas"); canvas.width = tW; canvas.height = tH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, tW, tH);
      const id = ctx.getImageData(0, 0, tW, tH); const d = id.data;
      for (let i = 0; i < d.length; i += 4) { const g = 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]; d[i]=d[i+1]=d[i+2]=g; }
      ctx.putImageData(id, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) { resolve(file); return; }
        resolve(new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg", lastModified: Date.now() }));
      }, "image/jpeg", JPEG_QUALITY);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

function createPreviewUrl(file) { return URL.createObjectURL(file); }

// =============================================================================
// FETCH HELPERS
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
      } else if (err.message.startsWith("Pipeline Error:")) { throw err; }
      else { lastError = err; if (attempt < maxRetries) { await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000)); continue; } }
    }
  }
  throw lastError || new Error("Request failed after retries");
}

async function apiFetch(path, options = {}) {
  const session = supabase.getSession();
  const authHeader = session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {};
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeader, ...options.headers },
    ...options,
  });
  if (res.status === 401) {
    // Session expired — force logout
    supabase.signOut();
    throw new Error("Session expired. Please log in again.");
  }
  if (!res.ok) { const err = await res.json().catch(() => ({ message: `HTTP ${res.status}` })); throw new Error(err.message || `HTTP ${res.status}`); }
  return res.json();
}

// =============================================================================
// CORE ANALYSIS
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
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    } else {
      optimizedFiles.forEach(f => formData.append("files", f));
      setLoadingMsg(`Analyzing ${optimizedFiles.length} labels...`);
      const response = await fetchWithRetry(`${API_URL}/analyze-labels`, { method: "POST", body: formData });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    }
  } catch (err) { console.error("❌ Pipeline Failure:", err); setError(err.message); }
  finally { setLoading(false); setLoadingMsg(""); }
}

// =============================================================================
// SHARED STYLE HELPERS
// =============================================================================
const card = { background: "var(--white)", border: "1px solid var(--border)", borderRadius: 16, overflow: "hidden" };
const cardHeader = { background: "var(--brown-lt)", padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" };
const inputStyle = { width: "100%", padding: "9px 12px", background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 14, color: "var(--text)" };
const labelStyle = { fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", display: "block", marginBottom: 4 };
const primaryBtn = { width: "100%", padding: "13px", background: "var(--teal)", color: "white", border: "none", borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 };
const mintBtn = { width: "100%", padding: "13px", background: "var(--mint)", color: "var(--mint-dk)", border: "none", borderRadius: 12, fontSize: 14, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 };
const ghostBtn = { padding: "9px 14px", background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, fontWeight: 600, color: "var(--brown)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 };
const overlayBg = { position: "fixed", inset: 0, zIndex: 50, background: "rgba(44,32,23,0.75)", display: "flex", alignItems: "center", justifyContent: "center", padding: 16 };
const modalBox = { width: "100%", maxWidth: 400, background: "var(--white)", border: "1px solid var(--border2)", borderRadius: 20, padding: 24, display: "flex", flexDirection: "column", gap: 16 };
const modalHeader = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const modalTitle = { fontSize: 15, fontWeight: 700, color: "var(--text)", display: "flex", alignItems: "center", gap: 8 };
const pillRow = { display: "flex", padding: 4, background: "var(--off2)", border: "1px solid var(--border)", borderRadius: 10, gap: 4 };
const macroCells = (vals) => vals.map(({ label, value, unit, color }) => (
  <div key={label} style={{ flex: 1, background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 6px", textAlign: "center" }}>
    <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase" }}>{label}</div>
    <div style={{ fontSize: 16, fontWeight: 700, color, marginTop: 2 }}>{value.toFixed(1)}</div>
    <div style={{ fontSize: 10, color: "var(--muted)" }}>{unit}</div>
  </div>
));

// =============================================================================
// IMAGE CROPPER
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
    ctx.fillStyle = "rgba(44,32,23,0.55)";
    ctx.fillRect(0, 0, cw, cropRect.y); ctx.fillRect(0, cropRect.y + cropRect.h, cw, ch - cropRect.y - cropRect.h);
    ctx.fillRect(0, cropRect.y, cropRect.x, cropRect.h); ctx.fillRect(cropRect.x + cropRect.w, cropRect.y, cw - cropRect.x - cropRect.w, cropRect.h);
    ctx.strokeStyle = "#006D77"; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);
    ctx.strokeStyle = "rgba(0,109,119,0.3)"; ctx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      const gx = cropRect.x + (cropRect.w / 3) * i; const gy = cropRect.y + (cropRect.h / 3) * i;
      ctx.beginPath(); ctx.moveTo(gx, cropRect.y); ctx.lineTo(gx, cropRect.y + cropRect.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cropRect.x, gy); ctx.lineTo(cropRect.x + cropRect.w, gy); ctx.stroke();
    }
    ctx.fillStyle = "#006D77";
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
    for (const [k, [cx, cy]] of Object.entries(corners)) if (Math.abs(pos.x - cx) <= hs && Math.abs(pos.y - cy) <= hs) return k;
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
    const pos = getPos(e, canvasRef.current); const dx = pos.x - dragStart.current.x, dy = pos.y - dragStart.current.y;
    const { w: cw, h: ch } = canvasSize; const b = cropAtStart.current, MIN = 40, m = dragMode.current;
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
    <div style={overlayBg}>
      <div style={{ width: "100%", maxWidth: 520 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <p style={{ fontSize: 13, fontWeight: 700, color: "var(--white)", display: "flex", alignItems: "center", gap: 8 }}>
              <Crop size={14} color="var(--mint)" /> Crop Nutrition Label
            </p>
            <p style={{ fontSize: 11, color: "rgba(253,252,249,0.5)", marginTop: 2 }}>Drag corners to resize · Drag inside to move</p>
          </div>
          <button onClick={onCancel} style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(253,252,249,0.5)" }}><X size={18} /></button>
        </div>
        <div ref={containerRef} style={{ width: "100%", borderRadius: 14, overflow: "hidden", border: "1px solid var(--border2)", touchAction: "none", userSelect: "none" }}>
          {imageLoaded
            ? <canvas ref={canvasRef} width={canvasSize.w} height={canvasSize.h} style={{ display: "block", width: "100%", touchAction: "none", cursor: "crosshair" }}
                onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
                onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp} />
            : <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 240 }}><Loader2 size={24} color="var(--teal)" className="animate-spin" /></div>
          }
        </div>
        {cropRect && <p style={{ fontSize: 10, color: "rgba(253,252,249,0.4)", textAlign: "center", marginTop: 8 }}>REGION: {Math.round(cropRect.w / imgScale)}×{Math.round(cropRect.h / imgScale)}px</p>}
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          <button onClick={onCancel} style={{ flex: 1, padding: "11px", background: "rgba(253,252,249,0.08)", border: "1px solid rgba(253,252,249,0.2)", borderRadius: 10, fontSize: 13, fontWeight: 600, color: "rgba(253,252,249,0.7)", cursor: "pointer" }}>Cancel</button>
          <button onClick={() => onConfirm(null)} style={{ flex: 1, padding: "11px", background: "rgba(253,252,249,0.08)", border: "1px solid rgba(253,252,249,0.2)", borderRadius: 10, fontSize: 13, fontWeight: 600, color: "rgba(253,252,249,0.7)", cursor: "pointer" }}>Full image</button>
          <button onClick={handleConfirm} style={{ flex: 1, padding: "11px", background: "var(--teal)", border: "none", borderRadius: 10, fontSize: 13, fontWeight: 700, color: "white", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
            <Check size={14} /> Confirm
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
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><BookmarkPlus size={15} color="var(--teal)" /> Save to Folder</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><X size={16} /></button>
        </div>
        <div>
          <label style={labelStyle}>Item Name</label>
          <input value={itemName} onChange={e => setItemName(e.target.value)} placeholder="e.g. Greek Yogurt" style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>Folder</label>
          <select value={selectedFolder} onChange={e => setSelectedFolder(e.target.value)} style={inputStyle}>
            <option value="">— select folder —</option>
            {folders.map(f => <option key={f.folder_id} value={f.folder_id}>{f.name}</option>)}
          </select>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input value={newFolder} onChange={e => setNewFolder(e.target.value)} placeholder="New folder name..." onKeyDown={handleFolderKeyDown} style={{ ...inputStyle, flex: 1 }} />
          <button onClick={createFolder} style={{ ...ghostBtn, padding: "9px 12px" }}><FolderPlus size={14} /></button>
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || !selectedFolder || !itemName.trim()} style={{ ...primaryBtn, opacity: (saving || !selectedFolder || !itemName.trim()) ? 0.45 : 1 }}>
          {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// ADD TO LOG MODAL
// =============================================================================
function AddToLogModal({ item, onClose, onAdded }) {
  const [mode, setMode] = useState("serving");
  const [servings, setServings] = useState("1");
  const [grams, setGrams] = useState("");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);
  const [manualName, setManualName] = useState(item.name || "");
  const [manualCal, setManualCal] = useState("");
  const [manualProt, setManualProt] = useState("");
  const [manualCarb, setManualCarb] = useState("");
  const [manualFat, setManualFat] = useState("");
  const [manualFibre, setManualFibre] = useState("");

  const rawNutrition = item.nutrition || {};
  const per100g    = rawNutrition.per_100g    && Object.keys(rawNutrition.per_100g).length    > 0 ? rawNutrition.per_100g    : null;
  const perServing = rawNutrition.per_serving && Object.keys(rawNutrition.per_serving).length > 0 ? rawNutrition.per_serving : null;
  const baseNutrition = perServing ?? per100g ?? resolveNutrition(rawNutrition);
  const servingGrams = perServing ? extractServingGrams(perServing.size) : null;
  const servingsNum = parseFloat(servings) || 1; const gramsNum = parseFloat(grams) || 0;

  let scaledNutrition = { ...baseNutrition }; let scalingInfo = null;
  if (mode === "serving") {
    scalingInfo = { factor: servingsNum, baseLabel: perServing ? `per serving${perServing.size ? ` (${perServing.size})` : ""}` : "per 100g" };
  } else if (mode === "grams") {
    if (per100g && gramsNum > 0) { scalingInfo = { factor: gramsNum / 100, baseLabel: "per 100g", targetLabel: `${gramsNum}g` }; scaledNutrition = per100g; }
    else if (perServing && servingGrams && gramsNum > 0) { scalingInfo = { factor: gramsNum / servingGrams, baseLabel: `per serving (${servingGrams}g)`, targetLabel: `${gramsNum}g` }; scaledNutrition = perServing; }
    else { scalingInfo = { factor: 1, baseLabel: "no size data", targetLabel: `${gramsNum}g`, warn: true }; }
  }

  const factor = scalingInfo?.factor ?? 1;
  const getVal = (key) => (parseNumeric(scaledNutrition[key]) || 0) * factor;
  const cal  = mode === "manual" ? (parseFloat(manualCal)   || 0) : getVal("calories");
  const prot = mode === "manual" ? (parseFloat(manualProt)  || 0) : getVal("protein");
  const carb = mode === "manual" ? (parseFloat(manualCarb)  || 0) : getVal("carbohydrates");
  const fat  = mode === "manual" ? (parseFloat(manualFat)   || 0) : getVal("fat");
  const fib  = mode === "manual" ? (parseFloat(manualFibre) || 0) : getVal("fibre");

  const buildSubmitNutrition = () => {
    if (mode === "manual") return { per_serving: { size: "1 serving", calories: Math.round(cal), protein: `${prot.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, fat: `${fat.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g` } };
    if (mode === "serving") return rawNutrition;
    return { per_serving: { size: `${gramsNum}g`, calories: Math.round(cal), fat: `${fat.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, protein: `${prot.toFixed(1)}g`, fibre: `${fib.toFixed(1)}g`, ...(scaledNutrition.saturated_fat ? { saturated_fat: `${(parseNumeric(scaledNutrition.saturated_fat)||0)*factor}g` } : {}), ...(scaledNutrition.sugars ? { sugars: `${(parseNumeric(scaledNutrition.sugars)||0)*factor}g` } : {}), ...(scaledNutrition.sodium ? { sodium: `${(parseNumeric(scaledNutrition.sodium)||0)*factor}g` } : {}) } };
  };

  const canSave = mode === "serving" ? servingsNum > 0 : mode === "grams" ? gramsNum > 0 : manualCal !== "" || manualProt !== "" || manualCarb !== "" || manualFat !== "";

  const save = async () => {
    if (!canSave) return; setSaving(true);
    try {
      const submitServings = mode === "serving" ? servingsNum : 1;
      const submitName = manualName.trim() || item.name;
      await apiFetch("/log", { method: "POST", body: JSON.stringify({ name: submitName, servings: submitServings, nutrition: buildSubmitNutrition() }) });
      setStatus({ type: "ok", msg: "Added to today's log!" }); onAdded(); setTimeout(() => { onClose(); }, 900);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  const activePill = { flex: 1, padding: "7px", background: "var(--teal)", color: "white", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer" };
  const inactivePill = { flex: 1, padding: "7px", background: "transparent", color: "var(--muted)", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" };

  return (
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><CalendarDays size={15} color="var(--mint-dk)" /> Add to Today's Log</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><X size={16} /></button>
        </div>
        <input value={manualName} onChange={e => setManualName(e.target.value)} placeholder="Item name..." style={{ ...inputStyle, fontWeight: 700 }} />
        <div style={pillRow}>
          {[["serving", "Per Serving"], ["grams", "By Weight"], ["manual", "Manual"]].map(([m, label]) => (
            <button key={m} onClick={() => setMode(m)} style={mode === m ? activePill : inactivePill}>{label}</button>
          ))}
        </div>
        {mode === "serving" && (
          <div>
            <label style={labelStyle}>Servings</label>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button onClick={() => setServings(s => String(Math.max(0.5, parseFloat(s) - 0.5)))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Minus size={12} /></button>
              <input type="number" min="0.5" step="0.5" value={servings} onChange={e => setServings(e.target.value)} style={{ ...inputStyle, width: 80, textAlign: "center" }} />
              <button onClick={() => setServings(s => String(parseFloat(s) + 0.5))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Plus size={12} /></button>
            </div>
            {baseNutrition.size && <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Base: {baseNutrition.size}</p>}
          </div>
        )}
        {mode === "grams" && (
          <div>
            <label style={labelStyle}>Amount (grams)</label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input type="number" min="1" step="1" value={grams} onChange={e => setGrams(e.target.value)} placeholder="e.g. 150" style={{ ...inputStyle, flex: 1 }} />
              <span style={{ fontSize: 14, fontWeight: 700, color: "var(--muted)" }}>g</span>
            </div>
          </div>
        )}
        {mode === "manual" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[["Calories (kcal)", manualCal, setManualCal], ["Protein (g)", manualProt, setManualProt], ["Carbs (g)", manualCarb, setManualCarb], ["Fat (g)", manualFat, setManualFat], ["Fibre (g)", manualFibre, setManualFibre]].map(([label, val, set]) => (
                <div key={label}>
                  <label style={{ ...labelStyle, fontSize: 10 }}>{label}</label>
                  <input type="number" min="0" step="any" value={val} onChange={e => set(e.target.value)} placeholder="0" style={inputStyle} />
                </div>
              ))}
            </div>
          </div>
        )}
        {scalingInfo && mode !== "manual" && (
          <p style={{ fontSize: 11, color: scalingInfo.warn ? "var(--orange)" : "var(--muted)" }}>
            {scalingInfo.warn ? `⚠ No size data — using base values as-is`
              : `Base: ${scalingInfo.baseLabel}${scalingInfo.targetLabel ? ` → ${scalingInfo.targetLabel} ×${scalingInfo.factor.toFixed(3)}` : ""}${mode === "serving" && servingsNum !== 1 ? ` × ${servingsNum} servings` : ""}`}
          </p>
        )}
        <div style={{ display: "flex", gap: 8 }}>
          {macroCells([
            { label: "Cal",   value: cal,  unit: "kcal", color: "var(--orange)"  },
            { label: "Prot",  value: prot, unit: "g",    color: "var(--teal)"    },
            { label: "Carbs", value: carb, unit: "g",    color: "var(--purple)"  },
            { label: "Fat",   value: fat,  unit: "g",    color: "var(--brown)"   },
          ])}
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || !canSave} style={{ ...mintBtn, opacity: (saving || !canSave) ? 0.45 : 1 }}>
          {saving ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
          {saving ? "Adding..." : "Add to Log"}
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// EDIT LOG MODAL
// =============================================================================
function EditLogModal({ entry, onClose, onSaved }) {
  const [mode, setMode] = useState("serving");
  const [servings, setServings] = useState(String(entry.servings || 1));
  const [grams, setGrams] = useState("");
  const [name, setName] = useState(entry.name || "");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);

  const nutrition = entry.nutrition || {};
  const per100g    = nutrition.per_100g    && Object.keys(nutrition.per_100g).length    > 0 ? nutrition.per_100g    : null;
  const perServing = nutrition.per_serving && Object.keys(nutrition.per_serving).length > 0 ? nutrition.per_serving : null;
  const baseNutrition = perServing ?? per100g ?? nutrition;
  const servingGrams = perServing ? extractServingGrams(perServing.size) : null;
  const servingsNum = parseFloat(servings) || 1; const gramsNum = parseFloat(grams) || 0;

  let scaledNutrition = { ...baseNutrition }; let scalingInfo = null;
  if (mode === "serving") { scalingInfo = { factor: servingsNum, baseLabel: perServing?.size ? `per serving (${perServing.size})` : "per serving" }; }
  else {
    if (per100g && gramsNum > 0) { scalingInfo = { factor: gramsNum / 100, baseLabel: "per 100g", targetLabel: `${gramsNum}g` }; scaledNutrition = per100g; }
    else if (perServing && servingGrams && gramsNum > 0) { scalingInfo = { factor: gramsNum / servingGrams, baseLabel: `per serving (${servingGrams}g)`, targetLabel: `${gramsNum}g` }; scaledNutrition = perServing; }
    else { scalingInfo = { factor: 1, baseLabel: "no size data", warn: true }; }
  }

  const factor = scalingInfo?.factor ?? 1;
  const getVal = (key) => (parseNumeric(scaledNutrition[key]) || 0) * factor;
  const cal = getVal("calories"); const prot = getVal("protein"); const carb = getVal("carbohydrates"); const fat = getVal("fat");

  const buildNutrition = () => {
    if (mode === "serving") return entry.nutrition;
    return { per_serving: { size: `${gramsNum}g`, calories: Math.round(cal), fat: `${fat.toFixed(1)}g`, carbohydrates: `${carb.toFixed(1)}g`, protein: `${prot.toFixed(1)}g`, fibre: `${getVal("fibre").toFixed(1)}g` } };
  };

  const save = async () => {
    if (mode === "serving" && servingsNum <= 0) return; if (mode === "grams" && gramsNum <= 0) return;
    setSaving(true);
    try {
      const submitServings = mode === "serving" ? servingsNum : 1;
      await apiFetch(`/log/${entry.log_id}`, { method: "PUT", body: JSON.stringify({ name: name.trim() || entry.name, servings: submitServings, nutrition: buildNutrition() }) });
      setStatus({ type: "ok", msg: "Updated!" }); onSaved(); setTimeout(onClose, 700);
    } catch (e) { setStatus({ type: "error", msg: e.message }); }
    finally { setSaving(false); }
  };

  const activePill = { flex: 1, padding: "7px", background: "var(--teal)", color: "white", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer" };
  const inactivePill = { flex: 1, padding: "7px", background: "transparent", color: "var(--muted)", border: "none", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" };

  return (
    <div style={overlayBg}>
      <div style={modalBox}>
        <div style={modalHeader}>
          <div style={modalTitle}><PenLine size={15} color="var(--teal)" /> Edit Log Entry</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><X size={16} /></button>
        </div>
        <div>
          <label style={labelStyle}>Name</label>
          <input value={name} onChange={e => setName(e.target.value)} style={{ ...inputStyle, fontWeight: 700 }} />
        </div>
        <div style={pillRow}>
          {[["serving", "Per Serving"], ["grams", "By Weight"]].map(([m, label]) => (
            <button key={m} onClick={() => setMode(m)} style={mode === m ? activePill : inactivePill}>{label}</button>
          ))}
        </div>
        {mode === "serving" ? (
          <div>
            <label style={labelStyle}>Servings</label>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button onClick={() => setServings(s => String(Math.max(0.5, parseFloat(s) - 0.5)))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Minus size={12} /></button>
              <input type="number" min="0.5" step="0.5" value={servings} onChange={e => setServings(e.target.value)} style={{ ...inputStyle, width: 80, textAlign: "center" }} />
              <button onClick={() => setServings(s => String(parseFloat(s) + 0.5))} style={{ width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Plus size={12} /></button>
            </div>
            {baseNutrition?.size && <p style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Base: {baseNutrition.size}</p>}
          </div>
        ) : (
          <div>
            <label style={labelStyle}>Amount (grams)</label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input type="number" min="1" step="1" value={grams} onChange={e => setGrams(e.target.value)} placeholder="e.g. 150" style={{ ...inputStyle, flex: 1 }} />
              <span style={{ fontSize: 14, fontWeight: 700, color: "var(--muted)" }}>g</span>
            </div>
          </div>
        )}
        {scalingInfo && <p style={{ fontSize: 11, color: scalingInfo.warn ? "var(--orange)" : "var(--muted)" }}>{scalingInfo.warn ? "⚠ No size data" : `Base: ${scalingInfo.baseLabel}${scalingInfo.targetLabel ? ` → ${scalingInfo.targetLabel}` : ""}${mode === "serving" && servingsNum !== 1 ? ` × ${servingsNum}` : ""}`}</p>}
        <div style={{ display: "flex", gap: 8 }}>
          {macroCells([
            { label: "Cal",   value: cal,  unit: "kcal", color: "var(--orange)" },
            { label: "Prot",  value: prot, unit: "g",    color: "var(--teal)"   },
            { label: "Carbs", value: carb, unit: "g",    color: "var(--purple)" },
            { label: "Fat",   value: fat,  unit: "g",    color: "var(--brown)"  },
          ])}
        </div>
        {status && <p style={{ fontSize: 12, color: status.type === "ok" ? "var(--mint-dk)" : "var(--danger)" }}>{status.msg}</p>}
        <button onClick={save} disabled={saving || (mode === "serving" ? servingsNum <= 0 : gramsNum <= 0)} style={{ ...primaryBtn, opacity: saving ? 0.45 : 1 }}>
          {saving ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
          {saving ? "Saving..." : "Save Changes"}
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// MACRO BAR
// =============================================================================
function MacroBar({ label, current, goal, color }) {
  const pct = goal > 0 ? Math.min(100, (current / goal) * 100) : 0;
  const over = current > goal;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
        <span style={{ color: "var(--muted)", fontWeight: 600 }}>{label}</span>
        <span style={{ color: over ? "var(--danger)" : "var(--text)", fontWeight: 600 }}>{current.toFixed(1)} / {goal}{over && " ⚠"}</span>
      </div>
      <div style={{ height: 7, background: "var(--off2)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 4, background: over ? "var(--danger)" : color, width: `${pct}%`, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

// =============================================================================
// TRACKER TAB
// =============================================================================
function TrackerTab({ refreshKey, onEditEntry }) {
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
    try { const [g, l] = await Promise.all([apiFetch("/goals"), apiFetch(`/log?log_date=${today}`)]); setGoals(g); setLogData(l); }
    catch (e) { console.error("Tracker load failed:", e); }
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

  if (loading) return <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}><Loader2 size={24} color="var(--teal)" className="animate-spin" /></div>;

  const totals = logData?.totals || { calories: 0, protein: 0, carbs: 0, fat: 0 };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Goals card */}
      <div style={card}>
        <div style={cardHeader}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--brown)", display: "flex", alignItems: "center", gap: 8 }}>
            <Target size={14} color="var(--teal)" /> Daily Goals
          </div>
          <button onClick={() => { setGoalDraft({ ...goals }); setEditingGoals(v => !v); }}
            style={{ fontSize: 12, fontWeight: 600, color: "var(--teal)", background: "none", border: "none", cursor: "pointer" }}>
            {editingGoals ? "Cancel" : "Edit"}
          </button>
        </div>
        <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
          {editingGoals ? (
            <>
              {["calories", "protein", "carbs", "fat", "fibre"].map(key => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <label style={{ ...labelStyle, width: 60, marginBottom: 0 }}>{key}</label>
                  <input type="number" min="0" value={goalDraft[key] || ""} onChange={e => setGoalDraft(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))} style={{ ...inputStyle, flex: 1 }} />
                  <span style={{ fontSize: 11, color: "var(--muted)", width: 32 }}>{key === "calories" ? "kcal" : "g"}</span>
                </div>
              ))}
              <button onClick={saveGoals} disabled={savingGoals} style={{ ...primaryBtn, opacity: savingGoals ? 0.45 : 1 }}>
                {savingGoals ? "Saving..." : "Save Goals"}
              </button>
            </>
          ) : (
            <>
              <MacroBar label="Calories" current={totals.calories}    goal={goals.calories} color="var(--orange)" />
              <MacroBar label="Protein"  current={totals.protein}     goal={goals.protein}  color="var(--teal)"   />
              <MacroBar label="Carbs"    current={totals.carbs}       goal={goals.carbs}    color="var(--purple)" />
              <MacroBar label="Fat"      current={totals.fat}         goal={goals.fat}      color="var(--brown)"  />
              {goals.fibre > 0 && <MacroBar label="Fibre" current={totals.fibre || 0} goal={goals.fibre} color="var(--mint-dk)" />}
            </>
          )}
        </div>
      </div>

      {/* Summary grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {[
          { label: "Calories", val: totals.calories, unit: "kcal", color: "var(--orange)" },
          { label: "Protein",  val: totals.protein,  unit: "g",    color: "var(--teal)"   },
          { label: "Carbs",    val: totals.carbs,    unit: "g",    color: "var(--purple)"  },
          { label: "Fat",      val: totals.fat,      unit: "g",    color: "var(--brown)"  },
        ].map(({ label, val, unit, color }) => (
          <div key={label} style={{ background: "var(--white)", border: "1px solid var(--border)", borderRadius: 12, padding: "12px 8px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px" }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color, marginTop: 4 }}>{(val || 0).toFixed(0)}</div>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>{unit}</div>
          </div>
        ))}
      </div>

      {/* Log entries */}
      <div style={card}>
        <div style={{ ...cardHeader, background: "var(--off)" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--brown)", display: "flex", alignItems: "center", gap: 8 }}>
            <CalendarDays size={14} color="var(--teal)" /> Today's Log — {today}
          </div>
          <span style={{ fontSize: 11, color: "var(--muted)" }}>{logData?.items?.length || 0} entries</span>
        </div>
        {!logData?.items?.length ? (
          <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic" }}>No entries yet. Add food from Library.</div>
        ) : (
          logData.items.map((entry, idx) => (
            <div key={entry.log_id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: idx % 2 === 0 ? "var(--teal)" : "var(--orange)", flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{entry.name}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                  ×{entry.servings} serving{entry.servings !== 1 ? "s" : ""} · {entry.contribution.calories.toFixed(0)} kcal · P {entry.contribution.protein.toFixed(1)}g · C {entry.contribution.carbs.toFixed(1)}g · F {entry.contribution.fat.toFixed(1)}g
                </div>
              </div>
              <button onClick={() => onEditEntry && onEditEntry(entry)}
                style={{ width: 30, height: 30, borderRadius: 8, border: "1px solid var(--border)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <PenLine size={13} color="var(--teal)" />
              </button>
              <button onClick={() => deleteEntry(entry.log_id)} disabled={deletingId === entry.log_id}
                style={{ width: 30, height: 30, borderRadius: 8, border: "1px solid var(--border)", background: "var(--off)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {deletingId === entry.log_id ? <Loader2 size={13} color="var(--muted)" className="animate-spin" /> : <Trash2 size={13} color="var(--danger)" />}
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// =============================================================================
// LIBRARY TAB
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
    setLoading(true); setFolderData({});
    try { setFolders(await apiFetch("/folders")); }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadFolders(); }, [loadFolders]);

  const openFolderById = async (id) => {
    if (openFolder === id) { setOpenFolder(null); return; }
    setOpenFolder(id);
    try { const data = await apiFetch(`/folders/${id}`); setFolderData(prev => ({ ...prev, [id]: data })); }
    catch (e) { console.error(e); }
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return; setCreating(true);
    try { const f = await apiFetch("/folders", { method: "POST", body: JSON.stringify({ name: newFolderName.trim() }) }); setFolders(prev => [f, ...prev]); setNewFolderName(""); }
    catch (e) { console.error(e); }
    finally { setCreating(false); }
  };

  const handleFolderKeyDown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); createFolder(); return; }
    if (e.ctrlKey && e.key === "Backspace") { e.preventDefault(); setNewFolderName(prev => prev.replace(/\s*\S+\s*$/, "")); return; }
    if (e.ctrlKey && e.key === "Delete") { e.preventDefault(); const pos = e.target.selectionStart; setNewFolderName(newFolderName.slice(0, pos) + newFolderName.slice(pos).replace(/^\s*\S+/, "")); }
  };

  const deleteItem = async (folderId, itemId) => {
    setDeletingItem(itemId);
    try { await apiFetch(`/folders/${folderId}/items/${itemId}`, { method: "DELETE" }); setFolderData(prev => ({ ...prev, [folderId]: { ...prev[folderId], items: prev[folderId].items.filter(i => i.item_id !== itemId) } })); }
    catch (e) { console.error(e); }
    finally { setDeletingItem(null); }
  };

  const deleteFolder = async (folderId) => {
    try { await apiFetch(`/folders/${folderId}`, { method: "DELETE" }); setFolders(prev => prev.filter(f => f.folder_id !== folderId)); if (openFolder === folderId) setOpenFolder(null); }
    catch (e) { console.error(e); }
  };

  if (loading) return <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}><Loader2 size={24} color="var(--teal)" className="animate-spin" /></div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 8 }}>
        <input value={newFolderName} onChange={e => setNewFolderName(e.target.value)} placeholder="New folder name..." onKeyDown={handleFolderKeyDown}
          style={{ ...inputStyle, flex: 1 }} />
        <button onClick={createFolder} disabled={creating || !newFolderName.trim()}
          style={{ ...primaryBtn, width: "auto", padding: "9px 16px", fontSize: 13, opacity: (creating || !newFolderName.trim()) ? 0.45 : 1 }}>
          {creating ? <Loader2 size={13} className="animate-spin" /> : <FolderPlus size={13} />} Create
        </button>
      </div>

      {folders.length === 0 ? (
        <div style={{ padding: "48px 16px", textAlign: "center", color: "var(--muted)", fontSize: 14, fontStyle: "italic", border: "2px dashed var(--border)", borderRadius: 16 }}>No folders yet. Create one above.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {folders.map(folder => (
            <div key={folder.folder_id} style={card}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", cursor: "pointer" }} onClick={() => openFolderById(folder.folder_id)}>
                {openFolder === folder.folder_id
                  ? <FolderOpen size={18} color="var(--teal)" />
                  : <Folder size={18} color="var(--muted)" />}
                <span style={{ flex: 1, fontSize: 14, fontWeight: 600, color: "var(--text)" }}>{folder.name}</span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>{folderData[folder.folder_id]?.items?.length ?? ""} items</span>
                <button onClick={e => { e.stopPropagation(); deleteFolder(folder.folder_id); }} style={{ background: "none", border: "none", cursor: "pointer", marginLeft: 4 }}>
                  <Trash2 size={13} color="var(--muted)" />
                </button>
                {openFolder === folder.folder_id ? <ChevronUp size={14} color="var(--muted)" /> : <ChevronDown size={14} color="var(--muted)" />}
              </div>

              {openFolder === folder.folder_id && folderData[folder.folder_id] && (
                <div style={{ borderTop: "1px solid var(--off2)" }}>
                  {folderData[folder.folder_id].items.length === 0 ? (
                    <p style={{ padding: "16px", textAlign: "center", color: "var(--muted)", fontSize: 13, fontStyle: "italic" }}>No items in this folder.</p>
                  ) : (
                    folderData[folder.folder_id].items.map((item, idx) => {
                      const nutrition = item.nutrition?.per_serving ?? item.nutrition ?? {};
                      const cal  = parseNumeric(nutrition.calories)      || 0;
                      const prot = parseNumeric(nutrition.protein)       || 0;
                      const carb = parseNumeric(nutrition.carbohydrates) || 0;
                      const fat  = parseNumeric(nutrition.fat)           || 0;
                      const imageUrl = item.nutrition?.processed_url || item.nutrition?.raw_url || item.processed_url || item.raw_url || null;
                      return (
                        <div key={item.item_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px 10px 20px", borderTop: idx === 0 ? "none" : "1px solid var(--off2)" }}>
                          {imageUrl
                            ? <div style={{ width: 44, height: 44, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}><img src={imageUrl} alt={item.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} /></div>
                            : <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--teal-lt)", border: "1px solid var(--border)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}><Database size={14} color="var(--teal)" /></div>
                          }
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{cal}kcal · P {prot}g · C {carb}g · F {fat}g</div>
                          </div>
                          <button onClick={() => onAddToLog({ ...item })} style={{ padding: "5px 10px", background: "var(--mint)", border: "none", borderRadius: 8, fontSize: 11, fontWeight: 700, color: "var(--mint-dk)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
                            <Plus size={11} /> Log
                          </button>
                          <button onClick={() => deleteItem(folder.folder_id, item.item_id)} disabled={deletingItem === item.item_id} style={{ background: "none", border: "none", cursor: "pointer" }}>
                            {deletingItem === item.item_id ? <Loader2 size={13} color="var(--muted)" className="animate-spin" /> : <Trash2 size={13} color="var(--muted)" />}
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
    return <div style={{ padding: "48px 16px", textAlign: "center", border: "2px dashed var(--border)", borderRadius: 16, color: "var(--muted)", fontSize: 13, fontStyle: "italic" }}>No data extracted for this view</div>;
  }
  const customG = parseFloat(customGrams); const isValid = !isNaN(customG) && customG > 0;
  let baseGrams = null, scaleFrom = data, baseLabel = null, warnMsg = null;
  if (activeTab === "per_100g") { baseGrams = 100; baseLabel = "100g"; }
  else {
    const sg = extractServingGrams(data.size);
    if (sg) { baseGrams = sg; baseLabel = `${data.size} (${sg}g)`; }
    else if (per100gData && Object.keys(per100gData).length > 0) { baseGrams = 100; scaleFrom = per100gData; baseLabel = "100g (fallback)"; warnMsg = "No serving size — scaling from per 100g"; }
    else { warnMsg = "Cannot scale — no size or per 100g data"; }
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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ background: "var(--off)", border: "1px solid var(--border)", borderRadius: 12, padding: "14px 16px" }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--brown)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
          <Scale size={13} color="var(--teal)" /> Custom Serving Calculator
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <input type="number" min="1" step="any" placeholder="e.g. 58" value={customGrams} onChange={e => setCustomGrams(e.target.value)} style={{ ...inputStyle, flex: 1 }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "var(--muted)" }}>g</span>
          {customGrams && <button onClick={() => setCustomGrams("")} style={{ fontSize: 11, color: "var(--muted)", background: "none", border: "none", cursor: "pointer" }}>Clear</button>}
        </div>
        <p style={{ fontSize: 11, color: warnMsg ? "var(--orange)" : "var(--muted)", marginTop: 8 }}>
          {warnMsg || `Base: ${baseLabel}${scalingActive ? ` → ${customG}g · ×${factor.toFixed(4)}` : ""}`}
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, background: "var(--border)", borderRadius: 12, overflow: "hidden" }}>
        {entries.map(([key, value]) => {
          const meta = NUTRIENT_META[key] ?? getFallbackMeta(key);
          const { display, adjusted, baseDisplay } = getDisplay(key, value);
          return (
            <div key={key} style={{ background: adjusted ? "var(--teal-lt)" : "var(--white)", padding: "13px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px" }}>{meta.label}</div>
                {adjusted && <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>base: {baseDisplay}</div>}
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: meta.color }}>{display}</div>
                {meta.unit && <div style={{ fontSize: 10, color: "var(--muted)" }}>{meta.unit}{adjusted && <span style={{ color: "var(--teal)", marginLeft: 4 }}>adj.</span>}</div>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// SCAN TAB
// =============================================================================
function ScanTab({ onAddToLog }) {
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
  const fileInputRef = useRef(null);
  const accumulatedOptimizedRef = useRef([]);
  const accumulatedImagesRef    = useRef([]);

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
  // Use accumulatedImagesRef as fallback — immune to React batching delays after runAnalysis
  const currentPreview = images[activeIndex]?.persistentUrl || images[activeIndex]?.preview || accumulatedImagesRef.current[activeIndex]?.preview || null;
  const allOptimized   = optimizedFiles.length === images.length && images.length > 0;

  return (
    <>
      {cropperFile && <ImageCropper file={cropperFile} onConfirm={handleCropConfirm} onCancel={handleCropCancel} />}
      {saveModal && <SaveToFolderModal result={saveModal.result} imageId={saveModal.imageId} onClose={() => setSaveModal(null)} />}

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }} className="lg:grid-cols-scan">
        {/* Upload / preview panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {images.length === 0 ? (
            <div style={{ border: "2px dashed var(--border)", borderRadius: 20, background: "var(--white)", padding: "48px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10, cursor: "pointer" }}
              onClick={() => fileInputRef.current?.click()}>
              <input key={fileInputKey} ref={fileInputRef} type="file" style={{ display: "none" }} onChange={handleImageUpload} accept="image/*" multiple capture="environment" />
              <div style={{ width: 52, height: 52, borderRadius: 14, background: "var(--teal-lt)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Upload size={22} color="var(--teal)" />
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
                      <ChevronLeft size={14} />
                    </button>
                    <button onClick={() => setActiveIndex(i => Math.min(images.length - 1, i + 1))} disabled={activeIndex === images.length - 1}
                      style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", width: 32, height: 32, borderRadius: "50%", background: "var(--white)", border: "1px solid var(--border)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", opacity: activeIndex === images.length - 1 ? 0.3 : 1 }}>
                      <ChevronRight size={14} />
                    </button>
                  </>
                )}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {images.map((img, i) => {
                  const thumbSrc = img.persistentUrl || img.preview || accumulatedImagesRef.current[i]?.preview || null;
                  return (
                    <div key={i} onClick={() => setActiveIndex(i)} style={{ position: "relative", width: 56, height: 56, borderRadius: 12, overflow: "hidden", border: `2px solid ${i === activeIndex ? "var(--teal)" : "var(--border)"}`, cursor: "pointer", flexShrink: 0, opacity: i === activeIndex ? 1 : 0.6 }}>
                      {thumbSrc ? <img src={thumbSrc} alt={`Thumb ${i+1}`} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <div style={{ width: "100%", height: "100%", background: "var(--off)" }} />}
                      {!results && (
                        <>
                          <button onClick={e => { e.stopPropagation(); removeImage(i); }} style={{ position: "absolute", top: 2, right: 2, width: 16, height: 16, borderRadius: "50%", background: "white", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><X size={9} /></button>
                          <button onClick={e => { e.stopPropagation(); handleReCrop(i); }} style={{ position: "absolute", bottom: 2, right: 2, width: 16, height: 16, borderRadius: "50%", background: "white", border: "none", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Crop size={9} /></button>
                        </>
                      )}
                      <div style={{ position: "absolute", bottom: 2, left: 4, fontSize: 9, fontWeight: 700, color: "white", textShadow: "0 1px 2px rgba(0,0,0,0.5)" }}>{i + 1}</div>
                    </div>
                  );
                })}
                {!results && (
                  <div onClick={() => fileInputRef.current?.click()} style={{ width: 56, height: 56, borderRadius: 12, border: "2px dashed var(--border)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", flexShrink: 0 }}>
                    <Upload size={16} color="var(--muted)" />
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
            {loading ? <Loader2 size={16} className="animate-spin" /> : results ? <RefreshCcw size={16} /> : <Zap size={16} />}
            {loading ? (loadingMsg || `Processing ${images.length} image${images.length !== 1 ? "s" : ""}...`) : results ? "Clear session" : `Analyze${images.length > 1 ? ` (${images.length})` : ""}`}
          </button>

          {error && !loading && images.length > 0 && !results && (
            <button onClick={handleAnalyze} style={{ ...ghostBtn, width: "100%", borderColor: "var(--orange)", color: "var(--orange)" }}>
              <RefreshCcw size={14} /> Retry (images preserved)
            </button>
          )}

          {currentResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <input value={logName} onChange={e => setLogName(e.target.value)} placeholder="Name this item before logging..." style={inputStyle} />
              <button onClick={() => onAddToLog({ name: logName.trim() || `Label ${activeIndex + 1}`, nutrition: currentResult })} style={mintBtn}>
                <Plus size={15} /> Log this item
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
              <button onClick={() => setError(null)} style={{ background: "none", border: "none", cursor: "pointer" }}><X size={14} color="var(--muted)" /></button>
            </div>
          )}

          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, padding: "60px 0" }}>
              <Loader2 size={32} color="var(--teal)" className="animate-spin" />
              <p style={{ fontSize: 13, color: "var(--teal)", fontWeight: 600 }}>{loadingMsg || "Running analysis..."}</p>
              <p style={{ fontSize: 11, color: "var(--muted)" }}>Auto-retry · Timeout: {REQUEST_TIMEOUT_MS / 1000}s</p>
            </div>
          ) : results ? (
            <div style={{ ...card, padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
              {results.length > 1 && (
                <div>
                  <p style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Select result</p>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {results.map((_, i) => {
                      const thumbSrc = images[i]?.persistentUrl || images[i]?.preview || accumulatedImagesRef.current[i]?.preview || null;
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

              {/* Result header */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, background: "var(--brown-lt)", borderRadius: 12, padding: "10px 14px" }}>
                <div style={{ width: 52, height: 52, borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0, background: "var(--off)" }}>
                  {currentPreview ? <img src={currentPreview} alt="Label" style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}><Database size={16} color="var(--muted)" /></div>}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px" }}>Analyzing label</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--brown)", marginTop: 2 }}>Image {activeIndex + 1} of {results.length}</div>
                  {images[activeIndex]?.cropData && <div style={{ fontSize: 10, color: "var(--teal)", marginTop: 2 }}>✂ Cropped & optimized</div>}
                </div>
                <button onClick={() => setSaveModal({ result: currentResult, imageId: currentResult?.image_id || "" })}
                  style={{ ...ghostBtn, flexShrink: 0 }}>
                  <BookmarkPlus size={13} /> Save
                </button>
              </div>

              {/* Per 100g / Per serving toggle */}
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

              {/* Schema header */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 12, borderBottom: "1px solid var(--off2)" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "var(--muted)", display: "flex", alignItems: "center", gap: 6 }}>
                  <TableProperties size={13} color="var(--muted)" /> {activeTab === "per_100g" ? "Per 100g" : "Per Serving"}
                </div>
                {activeTab === "per_serving" && currentResult?.per_serving?.size && (
                  <span style={{ fontSize: 11, background: "var(--purp-lt)", color: "var(--purple)", padding: "3px 10px", borderRadius: 20, fontWeight: 600 }}>
                    {currentResult.per_serving.size}
                  </span>
                )}
              </div>

              <NutrientGrid key={activeIndex} data={currentResult?.[activeTab]} activeTab={activeTab} per100gData={currentResult?.per_100g} />

              <details style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
                <summary style={{ padding: "10px 14px", fontSize: 11, color: "var(--muted)", cursor: "pointer", listStyle: "none", display: "flex", justifyContent: "space-between" }}>
                  <span>Raw data payload [{activeIndex + 1}/{results.length}]</span>
                  <span>▼</span>
                </summary>
                <div style={{ padding: "12px 14px", background: "var(--off)" }}>
                  <pre style={{ fontSize: 10, color: "var(--brown)", fontFamily: "monospace", overflow: "auto" }}>{JSON.stringify(currentResult, null, 2)}</pre>
                </div>
              </details>
            </div>
          ) : (
            <div style={{ border: "2px dashed var(--border)", borderRadius: 20, background: "var(--white)", padding: "60px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10, minHeight: 280 }}>
              <LayoutPanelLeft size={36} color="var(--border)" />
              <p style={{ fontSize: 14, color: "var(--muted)", fontStyle: "italic" }}>
                {images.length > 0 && !loading ? "Upload more images or hit Analyze" : "Upload a label to begin"}
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// =============================================================================
// LOGIN SCREEN
// =============================================================================
function LoginScreen({ onLogin }) {
  const [email, setEmail]       = useState("");
  const [sending, setSending]   = useState(false);
  const [sent, setSent]         = useState(false);
  const [error, setError]       = useState(null);

  const handleSubmit = async () => {
    if (!email.trim()) return;
    setSending(true); setError(null);
    try {
      await supabase.signInWithOtp(email.trim());
      setSent(true);
    } catch (e) {
      setError(e.message);
    } finally { setSending(false); }
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--off)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div style={{ width: "100%", maxWidth: 400, display: "flex", flexDirection: "column", gap: 24 }}>
        {/* Logo */}
        <div style={{ textAlign: "center" }}>
          <div style={{ width: 56, height: 56, borderRadius: 16, background: "var(--teal)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <Database size={26} color="var(--mint)" />
          </div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.5px" }}>NutriScan</div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4, letterSpacing: "0.8px", textTransform: "uppercase" }}>Pipeline v4</div>
        </div>

        {/* Card */}
        <div style={{ background: "var(--white)", border: "1px solid var(--border)", borderRadius: 20, padding: 28, display: "flex", flexDirection: "column", gap: 16 }}>
          {!sent ? (
            <>
              <div>
                <div style={{ fontSize: 17, fontWeight: 700, color: "var(--text)" }}>Sign in</div>
                <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}>We'll send a magic link to your email. No password needed.</div>
              </div>
              <div>
                <label style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", display: "block", marginBottom: 6 }}>Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSubmit()}
                  placeholder="you@example.com"
                  style={{ width: "100%", padding: "10px 14px", background: "var(--off)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 15, color: "var(--text)", boxSizing: "border-box" }}
                  autoFocus
                />
              </div>
              {error && <p style={{ fontSize: 12, color: "var(--danger)", margin: 0 }}>{error}</p>}
              <button
                onClick={handleSubmit}
                disabled={sending || !email.trim()}
                style={{ width: "100%", padding: "13px", background: "var(--teal)", color: "white", border: "none", borderRadius: 12, fontSize: 15, fontWeight: 700, cursor: sending || !email.trim() ? "not-allowed" : "pointer", opacity: sending || !email.trim() ? 0.5 : 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                {sending ? <Loader2 size={16} className="animate-spin" /> : null}
                {sending ? "Sending..." : "Send Magic Link"}
              </button>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
                <span style={{ fontSize: 12, color: "var(--muted)" }}>or</span>
                <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
              </div>
              <button
                onClick={() => supabase.signInWithGoogle()}
                style={{ width: "100%", padding: "13px", background: "var(--white)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 15, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                  <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18z"/>
                  <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2.04a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17z"/>
                  <path fill="#FBBC05" d="M4.5 10.48A4.8 4.8 0 0 1 4.5 7.52V5.45H1.83a8 8 0 0 0 0 7.1l2.67-2.07z"/>
                  <path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.45L4.5 7.52A4.8 4.8 0 0 1 8.98 4.18z"/>
                </svg>
                Continue with Google
              </button>
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "8px 0" }}>
              <div style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--mint)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
                <Check size={22} color="var(--mint-dk)" />
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text)" }}>Check your email</div>
              <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 8, lineHeight: 1.6 }}>
                We sent a magic link to <strong>{email}</strong>.<br />
                Click the link to sign in.
              </div>
              <button onClick={() => { setSent(false); setEmail(""); }} style={{ marginTop: 16, fontSize: 12, color: "var(--teal)", background: "none", border: "none", cursor: "pointer", fontWeight: 600 }}>
                Use a different email
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// ROOT APP
// =============================================================================
export default function App() {
  const [session, setSession]             = useState(undefined); // undefined = loading
  const [activeMainTab, setActiveMainTab] = useState("scan");
  const [addToLogItem, setAddToLogItem]   = useState(null);
  const [logRefreshKey, setLogRefreshKey] = useState(0);
  const [libraryMountKey, setLibraryMountKey] = useState(0);
  const [editLogItem, setEditLogItem]     = useState(null);

  // Handle magic link callback and session restore on mount
  useEffect(() => {
    const init = async () => {
      // Check if returning from magic link
      const callbackSession = await supabase.handleAuthCallback();
      if (callbackSession) { setSession(callbackSession); return; }
      // Restore existing session
      const existing = supabase.getSession();
      setSession(existing);
    };
    init();
  }, []);

  const handleAddToLog  = useCallback((item) => { setAddToLogItem(item); }, []);
  const handleLogAdded  = useCallback(() => { setLogRefreshKey(k => k + 1); }, []);
  const handleEditEntry = useCallback((entry) => { setEditLogItem(entry); }, []);

  const handleTabChange = useCallback((tabId) => {
    setActiveMainTab(tabId);
    if (tabId === "library") setLibraryMountKey(k => k + 1);
  }, []);

  const TABS = [
    { id: "scan",    label: "Scan",    icon: <Zap size={13} />      },
    { id: "library", label: "Library", icon: <Folder size={13} />   },
    { id: "tracker", label: "Tracker", icon: <BarChart3 size={13} /> },
  ];

  // Loading state
  if (session === undefined) {
    return (
      <>
        <style>{PALETTE_CSS}</style>
        <div style={{ minHeight: "100vh", background: "var(--off)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Loader2 size={32} color="var(--teal)" className="animate-spin" />
        </div>
      </>
    );
  }

  // Not logged in
  if (!session) {
    return (
      <>
        <style>{PALETTE_CSS}</style>
        <LoginScreen />
      </>
    );
  }

  // Logged in — get user email for display
  const userEmail = (() => {
    try {
      const payload = JSON.parse(atob(session.access_token.split(".")[1]));
      return payload.email || payload.sub?.slice(0, 8) + "...";
    } catch { return "User"; }
  })();

  return (
    <>
      <style>{PALETTE_CSS}</style>
      <div style={{ minHeight: "100vh", background: "var(--off)", color: "var(--text)" }}>
        {addToLogItem && <AddToLogModal item={addToLogItem} onClose={() => setAddToLogItem(null)} onAdded={handleLogAdded} />}
        {editLogItem  && <EditLogModal  entry={editLogItem}  onClose={() => setEditLogItem(null)}  onSaved={() => { setLogRefreshKey(k => k + 1); }} />}

        {/* Header */}
        <div style={{ background: "var(--teal)", padding: "16px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: "rgba(174,246,199,0.2)", border: "1.5px solid rgba(174,246,199,0.35)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Database size={18} color="var(--mint)" />
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "white", letterSpacing: "-0.3px" }}>NutriScan</div>
              <div style={{ fontSize: 10, color: "rgba(174,246,199,0.7)", letterSpacing: "0.8px", textTransform: "uppercase" }}>Pipeline v4</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ fontSize: 11, color: "rgba(174,246,199,0.8)" }}>{userEmail}</div>
            <button onClick={() => supabase.signOut()} style={{ fontSize: 11, padding: "4px 12px", background: "rgba(174,246,199,0.12)", border: "1px solid rgba(174,246,199,0.25)", borderRadius: 20, color: "var(--mint)", cursor: "pointer" }}>
              Sign out
            </button>
          </div>
        </div>

        {/* Tab bar */}
        <div style={{ background: "var(--teal-md)", display: "flex", padding: "8px 16px", gap: 6 }}>
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => handleTabChange(tab.id)}
              style={{ padding: "7px 18px", borderRadius: 8, border: "none", fontSize: 13, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 6, background: activeMainTab === tab.id ? "var(--mint)" : "rgba(255,255,255,0.1)", color: activeMainTab === tab.id ? "var(--mint-dk)" : "rgba(255,255,255,0.65)", transition: "all 0.15s" }}>
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "20px 16px" }}>
          <div style={{ display: activeMainTab === "scan" ? "block" : "none" }}>
            <ScanTab onAddToLog={handleAddToLog} />
          </div>
          {activeMainTab === "library" && <LibraryTab key={libraryMountKey} onAddToLog={handleAddToLog} />}
          <div style={{ display: activeMainTab === "tracker" ? "block" : "none" }}>
            <TrackerTab refreshKey={logRefreshKey} onEditEntry={handleEditEntry} />
          </div>
        </div>
      </div>
    </>
  );
}