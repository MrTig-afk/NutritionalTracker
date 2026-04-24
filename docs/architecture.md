# NutriScan — Architecture

## What this app is

A nutrition label scanner. Users photograph food packaging, the app sends it to Gemini via a FastAPI backend, extracts macros, and lets users log meals, track daily goals, and view trends. Auth is Supabase magic-link + Google OAuth.

---

## Backend (`backend/main.py`)

Single-file FastAPI app. All routes in one file by design — the backend is simple enough that splitting would add overhead without benefit.

**Key decisions:**
- Image optimization (resize + grayscale) happens on the *frontend* before upload, not the backend. Reduces bandwidth and Gemini input tokens.
- Rate limiting is per-user per-day, tracked in Postgres via `usage_log` table. Limit enforced in `check_and_track()`.
- All DB access goes through a connection pool (`get_pool`). No ORM.
- Chat endpoint (`/chat`) uses Groq (not Gemini) — cheaper for conversational use. It fetches today's log summary and injects it as system context.

**Routes:**
| Route | Purpose |
|-------|---------|
| `POST /analyze-label` | Single image → Gemini extraction |
| `POST /analyze-labels` | Batch (multi-image) → Gemini |
| `GET/POST /goals` | Read/write macro targets |
| `GET /log` | Daily log for a date |
| `POST /log` | Add entry |
| `PUT /log/:id` | Edit entry |
| `DELETE /log/:id` | Remove entry |
| `GET /log/calendar` | Dates with entries (for DatePicker dots) |
| `GET /log/trends` | Aggregated weekly/monthly data |
| `GET/POST /folders` | Library folder management |
| `GET/POST /folders/:id/items` | Items within a folder |
| `POST /chat` | AI nutrition assistant |

---

## Frontend (`frontend/src/`)

React + Vite, no router. Single-page app with tab navigation managed in `App.jsx`.

### Module map

```
lib/api.js          — Supabase client, apiFetch (auth-aware), fetchWithRetry, runAnalysis
lib/nutrition.js    — Pure functions: parse/normalize nutrition data, date helpers, image pipeline (resize+grayscale canvas)
styles.jsx          — CSS vars (PALETTE_CSS injected via <style>), shared inline style objects, macroCells JSX helper
```

```
components/
  Icon.jsx              — Material Symbols wrapper (Icon) + CSS spinner (Spin)
  MacroBar.jsx          — Progress bar for a single macro (label, current, goal, color)
  DatePicker.jsx        — Calendar dropdown; fetches tracked dates from /log/calendar to show dots
  NutrientGrid.jsx      — Nutrient table with custom serving calculator (scales values by gram input)
  ImageCropper.jsx      — Canvas-based crop UI; touch+mouse drag, corner handles
  ChatAssistant.jsx     — Floating chat panel; calls /chat, session-only (no persistence)
  LoginScreen.jsx       — Magic link + Google OAuth login
  SaveToFolderModal.jsx — Save a scan result to a library folder
  AddToLogModal.jsx     — Log a library item (serving / by-weight / manual modes)
  EditLogModal.jsx      — Edit an existing log entry
```

```
tabs/
  ScanTab.jsx      — Image upload, crop queue, analysis trigger, results display
  LibraryTab.jsx   — Folder list, item search, accordion expand
  TrackerTab.jsx   — Daily log view, macro goal bars, date navigation
  TrendsTab.jsx    — Weekly/monthly bar charts per macro (SVG, no chart library)
```

```
App.jsx           — Auth gate, tab shell (top nav + bottom nav), modal orchestration
```

### State flow

`App.jsx` owns:
- `session` — auth state from Supabase
- `addToLogItem` — drives `AddToLogModal` (set by ScanTab and LibraryTab via `onAddToLog` prop)
- `editLogItem` — drives `EditLogModal` (set by TrackerTab via `onEditEntry` prop)
- `logRefreshKey` — incremented when a log entry is added/edited; passed to TrackerTab to trigger reload
- `libraryMountKey` — remounts LibraryTab on tab switch to force a fresh fetch

Each tab is self-contained for its own data fetching. Tabs don't talk to each other directly — they communicate upward via callback props.

### Image pipeline

1. User picks file → `ImageCropper` shows canvas crop UI
2. On confirm → `applyPipelineToFile` (in `lib/nutrition.js`): crop region → resize to max 1024px → grayscale → JPEG 0.85 quality
3. Optimized `File` object passed to `runAnalysis` (in `lib/api.js`)
4. `runAnalysis` POSTs to `/analyze-label` or `/analyze-labels`, maps response through `normalizeResult`

### Nutrition normalization

`normalizeResult` and friends (in `lib/nutrition.js`) handle:
- kJ → kcal conversion
- Numeric string extraction (`"12.5g"` → `12.5`)
- Graceful fallback when per_serving or per_100g is absent

`resolveNutrition` picks the best available view (per_serving preferred over per_100g).

---

## Infrastructure

- **Frontend**: Vercel (auto-deploy from `main`)
- **Backend**: Render (free tier, sleeps after inactivity — UptimeRobot pings `/health` to keep it warm)
- **Database**: Supabase Postgres (row-level security enabled)
- **Storage**: Supabase Storage (label images uploaded by backend after Gemini call)
- **AI**: Gemini 1.5 Flash (label extraction), Groq (chat assistant)

---

## Non-obvious decisions

- **`ScanTab` stays mounted** (hidden via `display:none`) even when another tab is active, so in-progress scans aren't lost on tab switch. LibraryTab is remounted on each visit (fresh fetch). TrackerTab uses `refreshKey` prop instead.
- **No chart library** — `MacroTrendChart` is raw SVG. Avoided recharts/chart.js to keep the bundle small.
- **Frontend grayscale** — Gemini reads nutrition labels better in grayscale. This is done client-side to save backend processing time and API costs.
- **Retry logic** — `fetchWithRetry` handles 500/503/504 with exponential backoff (1.5s, 3s). 429 (rate limit) is surfaced immediately to the user, not retried.
