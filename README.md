# NutriScan Pipeline v3

A full-stack nutrition label scanner that uses AI to extract, log, and track macros from food packaging.

Point your phone at a label → crop it → AI reads it → log it.

🔗 **Live App:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app/)  

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React (Vite), custom CSS palette |
| Backend | FastAPI (Python) |
| AI | Google Gemini 2.5 Flash |
| Database | PostgreSQL (Render) |
| Storage | AWS S3 |
| Frontend Deploy | Vercel |
| Backend Deploy | Render |

---

## Features

### Scan
- Upload one or multiple nutrition label images
- Crop to the label region before sending
- Images are grayscale-optimised before API submission
- Batch analysis — multiple labels sent in a single Gemini API call
- Extracts per 100g and per serving data
- kJ → kcal auto-conversion
- Custom serving calculator on results

### Library
- Save scanned labels to named folders
- Persistent across sessions (PostgreSQL)
- Images stored on S3 with raw and processed versions

### Tracker
- Log food items from Library or directly from scan results
- Three logging modes: Per Serving, By Weight (scales from per 100g), Manual entry
- Edit logged entries in place without deleting and re-adding
- Daily macro totals with progress bars vs custom goals
- Goals support: calories, protein, carbs, fat, fibre

---

## Project Structure

```
NutritionDE/
├── frontend/
│   ├── src/
│   │   └── App.jsx          # Full React app
│   ├── index.html
│   └── vite.config.js
├── backend/
│   ├── main.py              # FastAPI app
│   └── requirements.txt
└── README.md
```

---

## Environment Variables

### Backend (Render)
```
GOOGLE_API_KEY=        # Gemini API key
DATABASE_URL=          # PostgreSQL internal URL (Render)
AWS_ACCESS_KEY_ID=     # S3 access key
AWS_SECRET_ACCESS_KEY= # S3 secret key
AWS_REGION=            # e.g. ap-southeast-2
S3_BUCKET_NAME=        # your bucket name
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze-label` | Analyze single label image |
| POST | `/analyze-labels` | Batch analyze up to 10 images |
| GET/POST | `/goals` | Get or set daily macro goals |
| POST | `/log` | Add a food log entry |
| GET | `/log` | Get daily log entries |
| PUT | `/log/{id}` | Edit an existing log entry |
| DELETE | `/log/{id}` | Delete a log entry |
| GET/POST | `/folders` | List or create folders |
| GET | `/folders/{id}` | Get folder with items |
| POST | `/folders/{id}/items` | Add item to folder |
| DELETE | `/folders/{id}/items/{item_id}` | Remove item from folder |
| DELETE | `/folders/{id}` | Delete folder |
| GET | `/health` | Health check |

---

## Known Limitations

- **Gemini free tier:** 5 RPM, 20 RPD — use batch endpoint to conserve quota
- **Image preview on iOS:** Blob URL lifecycle issues on iOS WebKit — under investigation
- **No authentication yet:** All data stored under a default user ID

---

## Roadmap

- [ ] Google / Email authentication via Supabase
- [ ] UI redesign with warm colour palette
- [ ] Image preview fix for iOS Safari
- [ ] Historical log view (past dates)
- [ ] Nutrition trends / charts over time
- [ ] Barcode scanning support

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## Author

Built by [@MrTig-afk](https://github.com/MrTig-afk)