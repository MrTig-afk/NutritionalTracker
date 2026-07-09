/**
 * apiDemo.js — drop-in replacement for lib/api.js in the PUBLIC DEMO build.
 *
 * When the build runs with VITE_DEMO=1, vite.config.js aliases "lib/api" to
 * this module, so every importer transparently talks to the synthetic dataset
 * in demoData.js instead of Supabase + the FastAPI backend. No network
 * requests are made anywhere in this file: auth is a canned session, scans
 * return fixture parses after a short fake latency, and mutations behave
 * coherently (log/edit/delete, goals, folders, templates) so the demo feels
 * real. Everything lives in page memory and resets on reload.
 *
 * MUST NOT import "./api" or "../lib/api" (the alias would map the import
 * back onto this file). Mirrors api.js's full export surface.
 */

import { normalizeResult, parseNumeric } from "../lib/nutrition";
import { SCAN_FIXTURES, FIXTURE_NAMES, buildInitialState, localISO, newId } from "./demoData";

export const API_URL = "";

const state = buildInitialState();

const delay = (ms = 180) => new Promise(r => setTimeout(r, ms));

// ── Fake Supabase auth: the demo is always signed in as the demo user ───────

const DEMO_USER = {
  id: "demo-user",
  email: "demo@nutriscan.app",
  user_metadata: { full_name: "Demo", avatar_url: null },
};
const DEMO_SESSION = { access_token: "demo", token_type: "bearer", user: DEMO_USER };

export const supabase = {
  auth: {
    async getSession() { return { data: { session: DEMO_SESSION }, error: null }; },
    onAuthStateChange(callback) {
      queueMicrotask(() => callback("SIGNED_IN", DEMO_SESSION));
      return { data: { subscription: { unsubscribe() {} } } };
    },
    // Sign-out is a no-op: there is no login screen to return to in the demo.
    async signOut() { return { error: null }; },
    async signInWithOtp() { return { data: {}, error: null }; },
    async signInWithOAuth() { return { data: {}, error: null }; },
    async verifyOtp() { return { data: { session: DEMO_SESSION }, error: null }; },
  },
};

// ── Nutrition math (mirrors the backend's contribution calculation) ─────────

function resolveBase(nutrition) {
  if (!nutrition) return {};
  if (nutrition.per_serving && Object.keys(nutrition.per_serving).length > 0) return nutrition.per_serving;
  if (nutrition.per_100g && Object.keys(nutrition.per_100g).length > 0) return nutrition.per_100g;
  return nutrition;
}

function contributionOf(nutrition, servings) {
  const base = resolveBase(nutrition);
  const num = k => (parseNumeric(base[k]) || 0) * servings;
  return {
    calories: Math.round(num("calories") * 10) / 10,
    protein:  Math.round(num("protein") * 10) / 10,
    carbs:    Math.round(num("carbohydrates") * 10) / 10,
    fat:      Math.round(num("fat") * 10) / 10,
    fibre:    Math.round(num("fibre") * 10) / 10,
  };
}

function dayLog(logDate) {
  const items = [];
  const totals = { calories: 0, protein: 0, carbs: 0, fat: 0, fibre: 0 };
  for (const e of state.diary) {
    if (e.date !== logDate) continue;
    const c = contributionOf(e.nutrition, e.servings);
    for (const k of Object.keys(totals)) totals[k] += c[k];
    items.push({
      log_id: e.log_id,
      name: e.name,
      servings: e.servings,
      nutrition: e.nutrition,
      meal_group: e.nutrition?._meal_group ?? null,
      meal_label: e.nutrition?._meal_label ?? null,
      contribution: c,
    });
  }
  return { date: logDate, items, totals: Object.fromEntries(Object.entries(totals).map(([k, v]) => [k, Math.round(v * 10) / 10])) };
}

// ── Scripted demo assistant (no AI behind it) ────────────────────────────────

function chatReply(message) {
  const today = dayLog(localISO(0));
  const t = today.totals;
  const g = state.goals;
  const left = k => Math.max(0, Math.round((g[k] || 0) - (t[k] || 0)));
  const week = [];
  for (let i = 6; i >= 0; i--) week.push(dayLog(localISO(-i)).totals);
  const avg = k => Math.round(week.reduce((a, d) => a + (d[k] || 0), 0) / week.length);
  const m = (message || "").toLowerCase();

  if (/protein/.test(m)) {
    return `You're at ${Math.round(t.protein)}g protein today, so ${left("protein")}g to go on your ${g.protein}g goal. Your 7-day average is ${avg("protein")}g. Something like Blue Creek Cottage Cheese (~12g) or a Peak Trail bar (~12g) would close the gap nicely. (Scripted demo reply — the real app answers with a live AI that knows your log.)`;
  }
  if (/calorie|kcal|left|remaining|budget/.test(m)) {
    return `Today you've logged ${Math.round(t.calories)} kcal of your ${g.calories} kcal goal, so about ${left("calories")} kcal left — comfortable room for dinner. This week you're averaging ${avg("calories")} kcal a day. (Scripted demo reply.)`;
  }
  if (/dinner|eat|meal|suggest|tonight/.test(m)) {
    return `With ${left("calories")} kcal and ${left("protein")}g protein left today, the Golden Wok Veggie Stir-fry (~452 kcal, 20g protein) fits well — or the Ember Grill Chicken Wrap if you want more protein. (Scripted demo reply.)`;
  }
  if (/week|trend|average|summary|how am i/.test(m)) {
    return `Over the last 7 days you've averaged ${avg("calories")} kcal, ${avg("protein")}g protein, ${avg("carbs")}g carbs and ${avg("fat")}g fat a day — steady, and protein is trending at ${Math.round((avg("protein") / g.protein) * 100)}% of goal. (Scripted demo reply.)`;
  }
  return `This is NutriScan's assistant tab. In the demo my replies are scripted from the synthetic diary — try asking about your protein, calories left, dinner ideas, or your week. In the real app this is a live AI that knows your actual log and goals.`;
}

// ── apiFetch: the whole backend, in memory ───────────────────────────────────

export async function apiFetch(path, options = {}) {
  await delay();
  const method = (options.method || "GET").toUpperCase();
  const [rawPath, rawQuery] = path.split("?");
  const q = new URLSearchParams(rawQuery || "");
  const body = options.body ? JSON.parse(options.body) : null;
  const seg = rawPath.split("/").filter(Boolean); // e.g. ["log","1234"]

  // /usage
  if (rawPath === "/usage") return { ...state.usage };

  // /goals
  if (rawPath === "/goals") {
    if (method === "POST") {
      for (const k of ["calories", "protein", "carbs", "fat", "fibre"]) {
        if (body[k] !== undefined) state.goals[k] = parseFloat(body[k]) || 0;
      }
    }
    return { ...state.goals };
  }

  // /log/calendar?year&month
  if (rawPath === "/log/calendar") {
    const y = q.get("year"), mo = String(q.get("month")).padStart(2, "0");
    const prefix = `${y}-${mo}-`;
    return { dates: [...new Set(state.diary.filter(e => e.date.startsWith(prefix)).map(e => e.date))] };
  }

  // /log/trends?range&client_date
  if (rawPath === "/log/trends") {
    const days = q.get("range") === "monthly" ? 30 : 7;
    const data = [];
    for (let i = days - 1; i >= 0; i--) {
      const date = localISO(-i);
      const { totals } = dayLog(date);
      data.push({ date, ...totals });
    }
    return { data };
  }

  // /log and /log/{id}
  if (seg[0] === "log") {
    if (seg.length === 1 && method === "GET") return dayLog(q.get("log_date") || localISO(0));
    if (seg.length === 1 && method === "POST") {
      state.diary.push({ log_id: newId(), date: body.log_date || localISO(0), name: body.name, servings: body.servings || 1, nutrition: body.nutrition || {} });
      return { ok: true };
    }
    const id = Number(seg[1]);
    const entry = state.diary.find(e => e.log_id === id);
    if (method === "PUT" && entry) {
      entry.name = body.name ?? entry.name;
      entry.servings = body.servings ?? entry.servings;
      if (body.nutrition) entry.nutrition = { ...body.nutrition, _meal_group: entry.nutrition?._meal_group, _meal_label: entry.nutrition?._meal_label };
      return { ok: true };
    }
    if (method === "DELETE") {
      state.diary = state.diary.filter(e => e.log_id !== id);
      return { ok: true };
    }
    return { ok: true };
  }

  // /folders...
  if (seg[0] === "folders") {
    if (seg.length === 1 && method === "GET") return state.folders.map(f => ({ folder_id: f.folder_id, name: f.name }));
    if (seg.length === 1 && method === "POST") {
      const f = { folder_id: `demo-folder-${newId()}`, name: body.name, items: [] };
      state.folders.unshift(f);
      return { folder_id: f.folder_id, name: f.name };
    }
    const folder = state.folders.find(f => f.folder_id === seg[1]);
    if (seg.length === 2 && method === "GET") return folder ? { ...folder } : { folder_id: seg[1], name: "", items: [] };
    if (seg.length === 2 && method === "DELETE") {
      state.folders = state.folders.filter(f => f.folder_id !== seg[1]);
      return { ok: true };
    }
    if (seg[2] === "items" && method === "POST" && folder) {
      const item = { item_id: `demo-fi-${newId()}`, name: body.name, nutrition: body.nutrition || {} };
      folder.items.push(item);
      return item;
    }
    if (seg[2] === "items" && method === "DELETE" && folder) {
      folder.items = folder.items.filter(i => i.item_id !== seg[3]);
      return { ok: true };
    }
    return { ok: true };
  }

  // /meal-templates...
  if (seg[0] === "meal-templates") {
    if (seg.length === 1 && method === "GET") return state.templates.map(t => ({ template_id: t.template_id, name: t.name, item_count: t.items.length }));
    if (seg.length === 1 && method === "POST") {
      const t = { template_id: `demo-tmpl-${newId()}`, name: body.name, items: [] };
      state.templates.unshift(t);
      return { template_id: t.template_id, name: t.name, item_count: 0 };
    }
    const tmpl = state.templates.find(t => t.template_id === seg[1]);
    if (seg.length === 2 && method === "GET") return tmpl ? { ...tmpl } : { template_id: seg[1], name: "", items: [] };
    if (seg.length === 2 && method === "DELETE") {
      state.templates = state.templates.filter(t => t.template_id !== seg[1]);
      return { ok: true };
    }
    if (seg[2] === "log" && method === "POST" && tmpl) {
      const gid = `demo-group-${newId()}`;
      const logDate = q.get("log_date") || localISO(0);
      for (const it of tmpl.items) {
        state.diary.push({
          log_id: newId(), date: logDate, name: it.name, servings: it.servings || 1,
          nutrition: { ...it.nutrition, _meal_group: gid, _meal_label: tmpl.name },
        });
      }
      return { logged: tmpl.items.length };
    }
    if (seg[2] === "items" && method === "POST" && tmpl) {
      const item = { item_id: `demo-ti-${newId()}`, name: body.name, servings: body.servings || 1, nutrition: body.nutrition || {} };
      tmpl.items.push(item);
      return item;
    }
    if (seg[2] === "items" && tmpl) {
      const item = tmpl.items.find(i => i.item_id === seg[3]);
      if (method === "PUT" && item) { item.servings = body.servings || item.servings; return { ok: true }; }
      if (method === "DELETE") { tmpl.items = tmpl.items.filter(i => i.item_id !== seg[3]); return { ok: true }; }
    }
    return { ok: true };
  }

  // /chat
  if (rawPath === "/chat") {
    await delay(700); // thinking…
    return { reply: chatReply(body?.message) };
  }

  // /settings/notifications
  if (rawPath === "/settings/notifications") {
    if (method === "PUT" && body?.prefs) state.prefs = { ...state.prefs, ...body.prefs };
    return { prefs: { ...state.prefs } };
  }

  // /account (delete): pretend, but stay signed in — it's a shared demo.
  if (rawPath === "/account") return { ok: true };

  // /push/* (the demo build reports push as unsupported; these are inert)
  if (seg[0] === "push") return { public_key: "", ok: true };

  return { ok: true };
}

// fetchWithRetry only backs the real scan pipeline; nothing should reach it in
// the demo (runAnalysis below never calls it).
export async function fetchWithRetry() {
  throw new Error("Network access is disabled in the demo build.");
}

// ── The scan flow: canned parses after a believable latency ─────────────────

const FIXTURE_KEYS = Object.keys(SCAN_FIXTURES);
let fixtureCursor = 0;

function parseForFile(file) {
  const base = (file?.name || "").toLowerCase().replace(/\.[^.]+$/, "");
  const key = FIXTURE_KEYS.find(k => base.includes(k)) ?? FIXTURE_KEYS[fixtureCursor++ % FIXTURE_KEYS.length];
  return { ...SCAN_FIXTURES[key], _demo_name: FIXTURE_NAMES[key] };
}

export async function runAnalysis({ optimizedFiles, setLoading, setLoadingMsg, setError, setResults, setImages, switchToIndex }) {
  if (!optimizedFiles.length) return;
  setLoading(true); setError(null);
  setLoadingMsg(optimizedFiles.length === 1 ? "Analyzing label..." : `Analyzing ${optimizedFiles.length} labels...`);
  try {
    await delay(1600); // believable "AI is reading the label" latency
    const arr = optimizedFiles.map(f => normalizeResult(parseForFile(f)));
    state.usage.used = Math.min(state.usage.limit, state.usage.used + optimizedFiles.length);
    setResults(arr);
    setImages(prev => prev); // previews stay as local blobs; nothing is uploaded
    switchToIndex(0, arr);
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false); setLoadingMsg("");
  }
}
