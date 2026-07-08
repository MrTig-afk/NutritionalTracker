# NutriScan

**Live:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app)

## About

NutriScan turns a photo of any nutrition label into tracked macros in seconds. Snap the label, and AI vision reads the calories, protein, carbs, fat and fibre for you. No barcode database, no manual typing, no guesswork with foods that aren't in an app's catalog.

From there it works like a full nutrition tracker: log foods against personal daily goals, save favourites into folders, build meal templates that log an entire breakfast in one tap, review weekly trends, and chat with an AI assistant that already knows your log, your goals and your last seven days. Meal reminders and goal notifications arrive as real push notifications at times you choose.

It ships as an installable PWA: add it to your home screen on iOS or Android and it looks, launches and notifies like a native app, with light and dark mode and user-controlled updates. Under the hood it is a React frontend and a FastAPI backend on serverless Postgres, with row-level security isolating every user's data at the database layer.

---

## How to Use

**1. Sign in.** Google OAuth (with account picker) or an emailed 6-digit code. No password needed.

**2. Scan.** Go to the Scan tab and upload a photo of any nutrition label. The AI extracts all macro data instantly. Up to 10 images at once, and each account gets 10 free scans per day.

**3. Log.** From scan results or your Library, tap Log to add a food to today's tracker.

**4. Track.** The Tracker tab shows today's macro totals (calories, protein, carbs, fat, fibre) against your personal goals with progress bars. Use the calendar to view any past day.

**5. Library.** Save scanned foods to named folders for quick re-logging without scanning again. Build **Meal Templates** by multi-selecting foods, then log a full set (e.g. "Breakfast") to the tracker in one tap.

**6. Trends.** Daily breakdown of the last 7 days with per-macro columns and average summary cards.

**7. AI Assistant.** Its own tab: a chat that knows today's food log, remaining macros, and 7-day averages. Multi-turn, context-aware, with server-side safety guardrails.

**8. Notifications.** Enable them in Settings. Goal-reached and scan-limit alerts are automatic; meal reminders and a Sunday weekly summary are opt-in, each at a time you pick (like 9:00 AM or 11:00 AM).

**9. Settings.** Notification controls, contact links, dark/light mode from the top bar, and full account deletion.

**10. Install as PWA.** iOS: Share, then Add to Home Screen. Android: browser install prompt. When a new version is available, a banner shows exactly what's new and you choose when to apply it.

---

## Stack

| Layer | Tech | Notes |
|---|---|---|
| Frontend | ![React](https://img.shields.io/badge/React_19-20232A?style=flat&logo=react&logoColor=61DAFB) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white) | React 19, Vite, CSS custom properties, light/dark themes |
| PWA | ![PWA](https://img.shields.io/badge/PWA-5A0FC8?style=flat&logo=pwa&logoColor=white) | vite-plugin-pwa, Workbox injectManifest, version-synced changelog, user-controlled updates |
| UI | ![Material Symbols](https://img.shields.io/badge/Material_Symbols-4285F4?style=flat&logo=google&logoColor=white) | Material Symbols Outlined, Manrope font |
| Backend | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | FastAPI, Uvicorn, psycopg2 |
| AI Vision | ![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat&logo=google&logoColor=white) | Gemini 2.5 Flash (primary), 2.0 Flash (fallback) |
| AI Chat | ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white) | Groq, multi-turn, context-aware, layered safety guardrails |
| Database | ![Neon](https://img.shields.io/badge/Neon_PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) | Neon serverless PostgreSQL, connection pool, row-level security per user |
| Auth | ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) | Google OAuth + email OTP codes, ES256 JWTs verified server-side via JWKS |
| Push | ![pywebpush](https://img.shields.io/badge/pywebpush-3776AB?style=flat&logo=python&logoColor=white) | VAPID keys, Web Push API, per-user notification preferences and reminder times |
| Deploy | ![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white) ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white) | Vercel (frontend), Render (backend) |

---

## Architecture

```
NutriScan/
├── backend/
│   ├── main.py              FastAPI app: all routes, JWT auth, DB pool, AI calls,
│   │                        reminder scheduler, notification prefs, RLS binding
│   ├── rls_policies.sql     Row-level security policies (per-user data isolation)
│   └── requirements.txt
│
└── frontend/
    ├── public/
    │   ├── favicon.svg      Adaptive icon (light/dark via prefers-color-scheme)
    │   ├── changelog.v2.json  Versioned release notes shown by the update banner
    │   └── icon-{192,512}.png
    ├── src/
    │   ├── lib/
    │   │   ├── api.js       Supabase client, apiFetch (auth-aware), retry, runAnalysis
    │   │   ├── push.js      Web-push subscribe/unsubscribe helpers
    │   │   └── nutrition.js Parse/normalize macros, image pipeline (resize, grayscale)
    │   ├── components/
    │   │   ├── ChatAssistant.jsx     AI chat screen: Groq, multi-turn, context-aware
    │   │   ├── AddToLogModal.jsx     Log a food (serving / by-weight / manual modes)
    │   │   ├── EditLogModal.jsx      Edit an existing log entry
    │   │   ├── SaveToFolderModal.jsx Save a scan result to a library folder
    │   │   ├── ImageCropper.jsx      Canvas crop UI, touch + mouse, corner handles
    │   │   ├── NutrientGrid.jsx      Nutrient table with custom serving calculator
    │   │   ├── DatePicker.jsx        Calendar dropdown with tracked-day dots
    │   │   ├── MacroBar.jsx          Single macro progress bar
    │   │   ├── LoginScreen.jsx       Google OAuth + email OTP code entry
    │   │   └── Icon.jsx              Material Symbols wrapper + CSS spinner
    │   ├── tabs/
    │   │   ├── ScanTab.jsx      Upload, crop queue, Gemini analysis, results
    │   │   ├── LibraryTab.jsx   Folders, items, meal templates with multi-select
    │   │   ├── TrackerTab.jsx   Daily log, macro goal bars, date rollover handling
    │   │   ├── TrendsTab.jsx    7-day macro breakdown, raw SVG, no chart library
    │   │   └── SettingsTab.jsx  Notification prefs + times, contact, delete account
    │   ├── App.jsx          Auth gate, tab shell, theme toggle, update banner
    │   ├── main.jsx         Service worker registration + update detection
    │   ├── sw.js            Workbox precache, push handler, navigation fallback
    │   ├── version.js       Build's changelog version (drives the update notes)
    │   └── styles.jsx       CSS custom properties, light + dark palettes
    └── index.html
```

**Database (Neon PostgreSQL, 11 tables, all under row-level security)**

```
users · api_usage · image_records · folders · folder_items · daily_log
user_goals · meal_templates · meal_template_items · push_subscriptions · notification_prefs
```

---

## Author

Built by [Kaushik N](https://github.com/MrTig-afk) · [LinkedIn](https://www.linkedin.com/in/kaushikn2002/)
