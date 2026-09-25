import { createClient } from "@supabase/supabase-js";
import { normalizeResult } from "./nutrition";
import { confirm } from "./confirm";

export const API_URL = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? "http://localhost:8000" : "https://nutritionaltracker.onrender.com");

// Must outlast the backend's worst case (GEMINI_TIMEOUT 45 s x 2 attempts + 3 s backoff
// + 45 s fallback = 138 s): aborting earlier made the UI report failure and retry while
// the server was still working, so one scan cost two or three Gemini calls.
const REQUEST_TIMEOUT_MS   = 140000;
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

// Every sign-out is this device only: a global one would also end the account's other sessions and Claude's
// connection (a deleted account keeps them paused, not ended, so Keep my account brings them back).
export const signOutHere = () => supabase.auth.signOut({ scope: "local" });

// A 401: the two account-deletion cases get one explanation, even when several requests fail at once (they all
// wait for the same dialog, so signing out cannot close it early), then this device signs out.
const WHY_401 = {
  account_deleted: "This account was deleted.",
  signed_out: "This account is being deleted, so its other devices were signed out. Sign in again to look, export or keep it.",
};
let notice = null;   // cleared by the next sign-in, so a late 401 after this sign-out asks nothing again
supabase.auth.onAuthStateChange((event) => { if (event === "SIGNED_IN") notice = null; });
async function signedOut(res) {
  const type = (await res.json().catch(() => ({})))?.detail?.error_type;
  const msg = WHY_401[type];
  if (msg) {
    notice ||= confirm(msg, { title: type === "account_deleted" ? "Account deleted" : "Signed out", okLabel: "OK", cancel: false });
    await notice;
  }
  await signOutHere();
  return msg || "Session expired. Please log in again.";
}

// Lane L: inside the 15 days nothing can change. The greyed controls are the visible half; this refuses the
// write itself (a keyboard press, a control that missed its mark) with the server's own words, before any request.
const DELETING = "This account is being deleted. You can still view and export everything, or keep your account.";
let viewOnly = false;
export const setViewOnly = (on) => { viewOnly = on; };
// The routes the server lets through inside the 15 days (allow_deleting); the server's 423 stays the authority.
const allowedWhileDeleting = (method, path) =>
  method === "GET" || method === "HEAD" || path === "/account/restore" || (method === "DELETE" && path === "/account")
  || path === "/push/unsubscribe" || (method === "DELETE" && path.startsWith("/settings/connected-apps/"));
const deletingNow = () => window.dispatchEvent(new Event("ns-deleting"));   // App re-reads GET /account/deletion

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
      if (status === 401) { const e = new Error(await signedOut(response)); e.noRetry = true; throw e; }
      if (!isRetryable || attempt === maxRetries) {
        // Show the server's own message (FastAPI wraps HTTPException in {detail}); never the raw body.
        const body = await response.json().catch(() => null);
        const msg  = body?.detail?.message || body?.message;
        if (body?.detail?.error_type === "account_scheduled_for_deletion") deletingNow();
        const err  = new Error(msg || (status === 429
          ? "You've reached your daily scan limit. Come back tomorrow!"
          : "Something went wrong. Please try again."));
        err.noRetry = true;  // a 4xx was the server's final answer: do not resend the file
        throw err;
      }
      lastError = new Error(`Retryable error: ${status}`);
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000));
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        lastError = new Error("Request timed out. Please try again.");
        if (attempt < maxRetries) { await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000)); continue; }
      } else if (err.noRetry) { throw err; }
      else { lastError = err; if (attempt < maxRetries) { await new Promise(r => setTimeout(r, RETRY_DELAY_MS[attempt] || 3000)); continue; } }
    }
  }
  throw lastError || new Error("Request failed after retries");
}

// Throws DELETING while view-only (and re-reads the state: kept on another device unlocks this one).
export function assertWritable() {
  if (!viewOnly) return;
  deletingNow();
  const e = new Error(DELETING);
  e.status = 423; e.errorType = "account_scheduled_for_deletion";
  throw e;
}

export async function apiFetch(path, options = {}) {
  if (!allowedWhileDeleting((options.method || "GET").toUpperCase(), path)) assertWritable();
  const { data: { session } } = await supabase.auth.getSession();
  const authHeader = session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {};
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeader, ...options.headers },
    ...options,
  });
  if (res.status === 401) throw new Error(await signedOut(res));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err.message || err.detail?.message || `HTTP ${res.status}`;  // FastAPI wraps HTTPException in {detail}
    const type = err.detail?.error_type;
    if (type === "account_scheduled_for_deletion") deletingNow();
    if (res.status === 423 && type === "account_locked" && !apiFetch._locked) {   // frozen server-side: tell once, then sign out
      apiFetch._locked = true;
      await confirm(msg, { title: "Account locked", okLabel: "OK", cancel: false });
      await signOutHere();
    }
    const e = new Error(msg);
    e.status = res.status;   // callers tell "not for this account" (403) from a failure
    e.errorType = type;      // e.g. account_scheduled_for_deletion, reauth_required
    throw e;
  }
  return res.json();
}

// G3: the file comes back as a download; fetch (not a link) because the request must carry the login.
export async function downloadExport(format) {
  const { data: { session } } = await supabase.auth.getSession();
  const res = await fetch(`${API_URL}/export?format=${format}`, {
    headers: session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {},
  });
  if (res.status === 401) throw new Error(await signedOut(res));
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((res.status === 429 || res.status === 503) && body.detail?.message
      ? body.detail.message : "Couldn't create the export. Try again.");
  }
  const blob = await res.blob();
  const name = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") || "")?.[1]
    || `nutriscan-export.${format === "csv" ? "zip" : "xlsx"}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function runAnalysis({ optimizedFiles, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex }) {
  if (!optimizedFiles.length) return;
  try { assertWritable(); } catch (e) { setError(e.message); return; }   // no photo leaves the device while view-only
  setLoading(true); setError(null);
  try {
    const { data: { session } } = await supabase.auth.getSession();
    const authHeader = session?.access_token ? { "Authorization": `Bearer ${session.access_token}` } : {};
    const d = new Date();
    const clientDate = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
    const scanId = crypto.randomUUID();
    const extraHeaders = { "X-Client-Date": clientDate, "X-Scan-ID": scanId };
    const formData = new FormData();
    if (optimizedFiles.length === 1) {
      formData.append("file", optimizedFiles[0]);
      setLoadingMsg("Analyzing label...");
      const response = await fetchWithRetry(`${API_URL}/analyze-label`, { method: "POST", body: formData, headers: { ...authHeader, ...extraHeaders } });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    } else {
      optimizedFiles.forEach(f => formData.append("files", f));
      setLoadingMsg(`Analyzing ${optimizedFiles.length} labels...`);
      const response = await fetchWithRetry(`${API_URL}/analyze-labels`, { method: "POST", body: formData, headers: { ...authHeader, ...extraHeaders } });
      const data = await response.json();
      const arr = (Array.isArray(data) ? data : [data]).map(normalizeResult);
      setResults(arr);
      setImages(prev => prev.map((img, idx) => { const url = arr[idx]?.processed_url || arr[idx]?.raw_url || null; return url ? { ...img, persistentUrl: url } : img; }));
      switchToIndex(0, arr);
    }
  } catch (err) { console.error("❌ Pipeline Failure:", err); setError(err.message); }
  finally { setLoading(false); setLoadingMsg(""); }
}
