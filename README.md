# NutriScan

A full-stack nutrition tracking web app that uses AI to extract macros from food label photos. Point your camera at any nutrition label — NutriScan parses it, lets you log it, and tracks your daily intake.

**Live:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app)

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS v4 |
| UI | Material Symbols Outlined, Manrope, CSS custom properties |
| Backend | FastAPI (Python), Uvicorn |
| AI | Google Gemini 2.5 Flash (primary), Gemini 2.0 Flash (fallback) |
| Database | Neon (PostgreSQL) |
| Auth | Supabase Auth — OTP email + Google OAuth, JWT verified server-side (PyJWT RS256/HS256) |
| Database | Neon (PostgreSQL), accessed via psycopg2 |
| Storage | AWS S3 — raw + processed image versions |
| Deploy | Vercel (frontend), Render (backend) |

---

## Features

**Scan**
- Upload one or multiple nutrition label photos; AI extracts all macro data
- Client-side crop, grayscale, and resize pipeline before upload
- Batch mode — up to 10 images in a single API call
- kJ → kcal auto-conversion; per-100g and per-serving data
- Custom serving size calculator on results
- 10 AI scans per user per day with a live usage counter

**Library**
- Save scanned items to named folders, persistent across sessions
- Raw and processed images stored in S3

**Tracker**
- Log items from scan results or library with serving counts
- Three logging modes: per serving, by weight (scales from per 100g), manual entry
- Edit logged entries in place
- Daily macro totals with progress bars vs personal goals
- Goals: calories, protein, carbs, fat, fibre

**Auth & Security**
- Supabase magic link (OTP email) and Google OAuth sign-in
- JWT verified server-side on every request; session expiry handled gracefully
- Per-user rate limiting enforced at the API layer

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
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # entire UI — auth, scan, library, tracker
│   │   ├── main.jsx
│   │   └── index.css     # Tailwind v4 + design tokens
│   └── index.html
├── backend/
│   ├── main.py           # FastAPI app — all routes + business logic
│   └── init_db.py
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
| GET / POST | `/folders` | List or create folders |
| GET / DELETE | `/folders/{id}` | Get folder contents or delete folder |
| POST | `/folders/{id}/items` | Add item to folder |
| DELETE | `/folders/{id}/items/{item_id}` | Remove item from folder |

All endpoints (except `/health`) require `Authorization: Bearer <supabase_jwt>`.

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
# set: DATABASE_URL, GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_JWT_SECRET
uvicorn main:app --reload

# Frontend
cd frontend
npm install
# create .env with VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
# set VITE_API_URL to point at production API to skip running backend locally
npm run dev
```

---

## Environment Variables

**Backend**
```
DATABASE_URL            Neon PostgreSQL connection string
GOOGLE_API_KEY          Gemini API key
SUPABASE_URL            Supabase project URL
SUPABASE_JWT_SECRET     JWT secret (HS256 token verification)
AWS_ACCESS_KEY_ID       S3 credentials (optional)
AWS_SECRET_ACCESS_KEY
S3_BUCKET_NAME
AWS_REGION
```

**Frontend**
```
VITE_SUPABASE_URL
VITE_SUPABASE_ANON_KEY
VITE_API_URL            Optional — override API base URL
```

---

## Author

Built by [@MrTig-afk](https://github.com/MrTig-afk)
