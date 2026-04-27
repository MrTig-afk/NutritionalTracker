# NutriScan

A full-stack nutrition tracking PWA that uses AI to extract macros from food label photos. Point your camera at any nutrition label — NutriScan parses it, lets you log it, and tracks your daily intake against your personal goals. Installable on iOS and Android, works offline after first load.

**Live:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app)

---

## How to Use

**1. Sign in** — email magic link or Google OAuth. No password needed.

**2. Scan** — go to the Scan tab and upload a photo of any nutrition label. The AI extracts all macro data instantly. Up to 10 images at once. Each account gets 10 free scans per day.

**3. Log** — from scan results or your Library, tap Log to add a food to today's tracker.

**4. Track** — the Tracker tab shows today's macro totals (calories, protein, carbs, fat, fibre) vs your personal goals with progress bars. Use the calendar to view any past day.

**5. Library** — save scanned foods to named folders for quick re-logging without scanning again. Build **Meal Templates** to log a full set of foods (e.g. "Breakfast") to the tracker in one tap.

**6. Trends** — daily breakdown of the last 7 days with per-macro columns and average summary cards.

**7. AI Assistant** — chat panel that knows your today's food log, remaining macros, and 7-day averages. Multi-turn, context-aware answers.

**8. Push Notifications** — enable from the Tracker tab to get notified when you hit your daily calorie goal or reach your scan limit.

**9. Install as PWA** — iOS: Share → Add to Home Screen. Android: browser install prompt. When a new version is available a changelog modal shows what's new — you choose when to apply it.

---

## Stack

| Layer | Tech | Notes |
|---|---|---|
| Frontend | ![React](https://img.shields.io/badge/React_19-20232A?style=flat&logo=react&logoColor=61DAFB) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white) | React 19, Vite, CSS custom properties |
| PWA | ![PWA](https://img.shields.io/badge/PWA-5A0FC8?style=flat&logo=pwa&logoColor=white) | vite-plugin-pwa, Workbox injectManifest, user-controlled updates |
| UI | ![Material Symbols](https://img.shields.io/badge/Material_Symbols-4285F4?style=flat&logo=google&logoColor=white) | Material Symbols Outlined, Manrope font |
| Backend | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | FastAPI, Uvicorn, psycopg2 |
| AI — Vision | ![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat&logo=google&logoColor=white) | Gemini 2.5 Flash (primary), 2.0 Flash (fallback) |
| AI — Chat | ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white) | Groq (openai/gpt-oss-120b), multi-turn, context-aware |
| Database | ![Neon](https://img.shields.io/badge/Neon_PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) | Neon serverless PostgreSQL, psycopg2 connection pool |
| Auth | ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) | OTP email + Google OAuth, JWT verified server-side (PyJWT RS256/HS256) |
| Storage | ![AWS S3](https://img.shields.io/badge/AWS_S3-FF9900?style=flat&logo=amazons3&logoColor=white) | Raw + processed image versions |
| Push | ![pywebpush](https://img.shields.io/badge/pywebpush-3776AB?style=flat&logo=python&logoColor=white) | VAPID keys, Web Push API, browser subscriptions stored in DB |
| Alerts | ![ntfy](https://img.shields.io/badge/ntfy.sh-009485?style=flat&logo=ntfy&logoColor=white) | Developer push alerts on API quota hits and startup failures |
| Deploy | ![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white) ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white) | Vercel (frontend), Render (backend) |

---

## Architecture

```
NutriScan/
├── backend/
│   ├── main.py              FastAPI app — all routes, JWT auth, DB pool, AI calls
│   └── requirements.txt
│
└── frontend/
    ├── public/
    │   ├── favicon.svg      Adaptive icon (light/dark via prefers-color-scheme)
    │   └── icon-{192,512}.png
    ├── src/
    │   ├── lib/
    │   │   ├── api.js       Supabase client · apiFetch (auth-aware) · retry · runAnalysis
    │   │   └── nutrition.js Parse/normalize macros · image pipeline (resize → grayscale)
    │   ├── components/
    │   │   ├── ChatAssistant.jsx     Floating chat panel — Groq, multi-turn, context-aware
    │   │   ├── AddToLogModal.jsx     Log a food (serving / by-weight / manual modes)
    │   │   ├── EditLogModal.jsx      Edit an existing log entry
    │   │   ├── SaveToFolderModal.jsx Save a scan result to a library folder
    │   │   ├── ImageCropper.jsx      Canvas crop UI — touch + mouse, corner handles
    │   │   ├── NutrientGrid.jsx      Nutrient table with custom serving calculator
    │   │   ├── DatePicker.jsx        Calendar dropdown with tracked-day dots
    │   │   ├── MacroBar.jsx          Single macro progress bar
    │   │   ├── LoginScreen.jsx       Magic link + Google OAuth
    │   │   └── Icon.jsx              Material Symbols wrapper + CSS spinner
    │   ├── tabs/
    │   │   ├── ScanTab.jsx      Upload · crop queue · Gemini analysis · results
    │   │   ├── LibraryTab.jsx   Folders · items · meal templates (log a set in one tap)
    │   │   ├── TrackerTab.jsx   Daily log · macro goal bars · push notification toggle
    │   │   └── TrendsTab.jsx    7-day macro breakdown — raw SVG, no chart library
    │   ├── App.jsx          Auth gate · tab shell · changelog/update modal
    │   ├── main.jsx         Service worker registration + update detection
    │   ├── sw.js            Workbox precache · push notification handler · SKIP_WAITING
    │   └── styles.jsx       CSS custom properties · shared style tokens
    └── index.html
```

**Database (Neon PostgreSQL — 10 tables)**

```
users · api_usage · image_records · folders · folder_items
daily_log · user_goals · meal_templates · meal_template_items · push_subscriptions
```

**External services**

| Service | Role |
|---|---|
| Supabase | Auth (magic link + Google OAuth), JWKS endpoint |
| AWS S3 | Raw + processed label images |
| Gemini 2.5 Flash | Nutrition label extraction (2.0 Flash fallback) |
| Groq | AI chat assistant (openai/gpt-oss-120b) |
| pywebpush / VAPID | User push notifications — goal hit, scan limit |
| ntfy.sh | Developer alerts — API quota hits, startup failures |

---

## Author

Built by [@MrTig-afk](https://github.com/MrTig-afk)
