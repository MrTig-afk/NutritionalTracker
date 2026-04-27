import { createClient } from "@supabase/supabase-js";
import { normalizeResult } from "./nutrition";

export const API_URL = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? "http://localhost:8000" : "https://nutritionaltracker.onrender.com");

const REQUEST_TIMEOUT_MS   = 25000;
const MAX_FRONTEND_RETRIES = 2;
const RETRY_DELAY_MS       = [1500, 3000];

const SUPABASE_URL      = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  document.body.innerHTML = `<div style="font-family:monospace;padding:2rem;color:#c00">
    <h2>Missing environment variables</h2>
    <p>VITE_SUPABASE_URL: ${SUPABASE_URL ? "✓" : "✗ NOT SET"}</p>
    <p>VITE_SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY ? "✓" : "✗ NOT SET"}</p>
    <p>Check Vercel → Project Settings → Environment Variables.</p>
    <p>Names must start with <strong>VITE_</strong> and the deployment must be rebuilt after adding them.</p>
  </div>`;
  throw new Error("Missing Supabase environment variables");
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

export async function fetchWithRetry(url, options, maxRetries = MAX_FRONTEND_RETRIES) {
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
      if (!isRetryable || attempt === maxRetries) {
        if (status === 429) {
          const body = await response.json().catch(() => null);
          throw new Error(body?.message || "You've reached your daily scan limit. Come back tomorrow!");
        }
        const errorText = await response.text().catch(() => `HTTP ${status}`);
        throw new Error(`Pipeline Error: ${status} — ${errorText}`);
      }
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

export async function apiFetch(path, options = {}) {
  const { data: { session } } = await supabase.auth.getSession();
  const authHeader = session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {};
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeader, ...options.headers },
    ...options,
  });
  if (res.status === 401) {
    await supabase.auth.signOut();
    throw new Error("Session expired. Please log in again.");
  }
  if (!res.ok) { const err = await res.json().catch(() => ({ message: `HTTP ${res.status}` })); throw new Error(err.message || `HTTP ${res.status}`); }
  return res.json();
}

export async function runAnalysis({ optimizedFiles, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex }) {
  if (!optimizedFiles.length) return;
  setLoading(true); setError(null);
  try {
    const { data: { session } } = await supabase.auth.getSession();
    const authHeader = session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {};
    const d = new Date();
    const clientDate = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
    const dateHeader = { "X-Client-Date": clientDate };
    const formData = new FormData();
    if (optimizedFiles.length === 1) {
      formData.append("file", optimizedFiles[0]);
      setLoadingMsg("Analyzing label...");
      const response = await fetchWithRetry(`${API_URL}/analyze-label`, { method: "POST", body: formData, headers: { ...authHeader, ...dateHeader } });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    } else {
      optimizedFiles.forEach(f => formData.append("files", f));
      setLoadingMsg(`Analyzing ${optimizedFiles.length} labels...`);
      const response = await fetchWithRetry(`${API_URL}/analyze-labels`, { method: "POST", body: formData, headers: { ...authHeader, ...dateHeader } });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    }
  } catch (err) { console.error("❌ Pipeline Failure:", err); setError(err.message); }
  finally { setLoading(false); setLoadingMsg(""); }
}
