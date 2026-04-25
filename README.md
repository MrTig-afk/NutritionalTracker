# NutriScan

A full-stack nutrition tracking PWA that uses AI to extract macros from food label photos. Point your camera at any nutrition label — NutriScan parses it, lets you log it, and tracks your daily intake. Installable on iOS and Android, works offline after first load.

**Live:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app)

---

## How to Use

**1. Sign in** — use your email (magic link) or Google OAuth. No password needed.

**2. Scan** — go to the Scan tab and upload a photo of any nutrition label. The AI extracts all macro data instantly. You can upload up to 10 images at once. Each account gets 10 free scans per day.

**3. Log** — from scan results or your Library, tap Log to add a food to today's tracker. Choose how you want to log it — by serving, by weight, or enter values manually.

**4. Track** — the Tracker tab shows today's macro totals (calories, protein, carbs, fat, fibre) vs your personal goals. Use the calendar to view any past day.

**5. Library** — save scanned foods to named folders for quick re-logging without scanning again.

**6. Trends** — view a daily breakdown table of the last 7 days with per-macro columns and average summary cards.

**7. AI Assistant** — open the chat panel to ask nutrition questions. It knows your today's full food log, remaining macros, and 7-day averages — multi-turn conversation, context-aware answers.

**8. Install as PWA** — on iOS: Share → Add to Home Screen. On Android: browser install prompt. Auto-updates silently and shows an "Update" pill when a new version is ready.

---

## Stack

| Layer | Tech | Notes |
|---|---|---|
| Frontend | ![React](https://img.shields.io/badge/React_19-20232A?style=flat&logo=react&logoColor=61DAFB) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white) | React 19, Vite, CSS custom properties |
| PWA | ![PWA](https://img.shields.io/badge/PWA-5A0FC8?style=flat&logo=pwa&logoColor=white) | vite-plugin-pwa, Workbox, service worker auto-update |
| UI | ![Material Symbols](https://img.shields.io/badge/Material_Symbols-4285F4?style=flat&logo=google&logoColor=white) | Material Symbols Outlined, Manrope font |
| Backend | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | FastAPI (Python), Uvicorn |
| AI — Vision | ![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat&logo=google&logoColor=white) | Gemini 2.5 Flash (primary), 2.0 Flash (fallback) |
| AI — Chat | ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white) | Groq (openai/gpt-oss-120b), multi-turn, context-aware |
| Database | ![Neon](https://img.shields.io/badge/Neon_PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) | Neon (PostgreSQL), psycopg2 |
| Auth | ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) | OTP email + Google OAuth, JWT verified server-side (PyJWT RS256/HS256) |
| Storage | ![AWS S3](https://img.shields.io/badge/AWS_S3-FF9900?style=flat&logo=amazons3&logoColor=white) | Raw + processed image versions |
| Deploy | ![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white) ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white) | Vercel (frontend), Render (backend) |

---

## Architecture

```
User (browser / phone)
        │
        ▼
  React + Vite  ──────────────────────────────►  Vercel CDN
        │  Authorization: Bearer <jwt>
        ▼
  FastAPI on Render
   ├── verify JWT (PyJWT RS256/HS256)
   ├── enforce daily scan limit (api_usage table)
   ├── upsert user record (users table)
   ├── optimize image (Pillow)
   ├── upload raw + processed to S3
   ├── call Gemini API → parse JSON
   ├── call Groq API → chat with log context
   └── persist record to Neon
        │
        ▼
  Neon (PostgreSQL)
  tables: image_records · folders · folder_items
          daily_log · user_goals · users · api_usage
```

---

## Project Structure

```
NutritionDE/
├── backend/
│   ├── main.py           # FastAPI app — all routes + business logic
│   └── init_db.py
├── frontend/
│   └── src/
│       ├── App.jsx              # Root: auth gate, tab shell, modal orchestration
│       ├── styles.jsx           # CSS vars + shared style objects
│       ├── lib/
│       │   ├── api.js           # Supabase client, apiFetch, retry, runAnalysis
│       │   └── nutrition.js     # Parse/normalize utils, date helpers, image pipeline
│       ├── components/          # Shared UI (Icon, MacroBar, DatePicker, modals, chat, etc.)
│       └── tabs/
│           ├── ScanTab.jsx
│           ├── LibraryTab.jsx
│           ├── TrackerTab.jsx
│           └── TrendsTab.jsx
├── docs/
│   └── architecture.md   # Design decisions, state flow, route reference
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/usage` | Today's scan count for the authenticated user |
| POST | `/analyze-label` | Analyze a single nutrition label image |
| POST | `/analyze-labels` | Batch-analyze up to 10 images |
| GET / POST | `/goals` | Get or set daily macro goals |
| GET / POST | `/log` | Get daily log or add an entry |
| PUT / DELETE | `/log/{id}` | Update or delete a log entry |
| GET | `/log/calendar` | Dates with entries for a given month (calendar dots) |
| GET | `/log/trends` | Aggregated daily macro data for trends view |
| POST | `/chat` | AI nutrition assistant (multi-turn, knows full log + 7-day trends) |
| GET / POST | `/folders` | List or create folders |
| GET / DELETE | `/folders/{id}` | Get folder contents or delete folder |
| POST | `/folders/{id}/items` | Add item to folder |
| DELETE | `/folders/{id}/items/{item_id}` | Remove item from folder |

All endpoints (except `/health`) require `Authorization: Bearer <supabase_jwt>`.

---

## Author

Built by [@MrTig-afk](https://github.com/MrTig-afk)
