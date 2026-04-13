# 🥗 NutriScan Pipeline
### Cloud-Native Nutritional Data Extraction & Analytical Warehousing

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=flat-square&logo=vercel)](https://nutritional-tracker-delta.vercel.app/)
[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?style=flat-square&logo=render)](https://nutritionaltracker.onrender.com/docs)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## 🔍 Overview

NutriScan is an end-to-end **Data Engineering pipeline** that transforms physical nutrition label images into structured, queryable analytical data — in a single request.

Upload one or multiple nutrition label images. The system uploads the raw assets to **AWS S3**, passes them to **Google Gemini 2.5 Flash** for multimodal inference in a single batch API call, stores the structured output in a **DuckDB analytical warehouse**, and renders the results in a clean, interactive UI — complete with a custom serving size macro calculator.

**Live:** [nutritional-tracker-delta.vercel.app](https://nutritional-tracker-delta.vercel.app/)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                       │
│              React + Tailwind CSS  (Vercel)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /analyze-labels
                           │  (multipart/form-data, 1–10 images)
┌──────────────────────────▼──────────────────────────────────┐
│                      FASTAPI BACKEND                        │
│                    Python  (Render)                         │
│                                                             │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│   │  AWS S3     │    │  Gemini 2.5  │    │   DuckDB     │   │
│   │  Data Lake  │    │  Flash (LLM) │    │  Warehouse   │   │
│   │  Raw Images │    │  Batch OCR   │    │  JSON Store  │   │
│   └─────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

The pipeline follows a decoupled **ELT (Extract, Load, Transform)** pattern:

1. **Extract** — Gemini 2.5 Flash performs multimodal inference on all images in a single API call, returning structured JSON
2. **Load** — Raw images are archived to AWS S3; structured JSON is loaded into DuckDB
3. **Transform** — The frontend applies schema-on-read mapping and real-time macro scaling

---

## ✨ Features

- **Batch image processing** — upload up to 10 nutrition label images and process them all in a single Gemini API request
- **Per-image preview** — results are displayed in upload order with thumbnail navigation so you always know which label you're looking at
- **Custom serving size calculator** — enter any gram value and all macros auto-scale in real time with the original base value shown alongside
- **Dual schema views** — toggle between Per 100g and Per Serving data for each label
- **Data Lake archival** — every raw image is stored in AWS S3 as the source of truth
- **Analytical warehousing** — structured extraction results are persisted in DuckDB using a Schema-on-Read JSON column, ready for SQL analysis
- **Schema resilience** — frontend transformation layer handles inconsistent OCR output keys dynamically

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 19, Tailwind CSS, Lucide Icons | UI & real-time macro calculations |
| **Backend** | FastAPI, Python | REST API & pipeline orchestration |
| **LLM Engine** | Google Gemini 2.5 Flash | Multimodal image-to-JSON extraction |
| **Data Lake** | AWS S3 (ap-southeast-2) | Raw image archival & source of truth |
| **Warehouse** | DuckDB | OLAP-optimised analytical storage |
| **Frontend Host** | Vercel | CI/CD on push, global CDN |
| **Backend Host** | Render | Containerised Python service |

---

## 📐 How the Batch Pipeline Works

Traditional approaches call the LLM once per image. NutriScan sends **all images in a single API call**:

```python
# All images packed into one Gemini request
contents = [batch_prompt]
for image in images:
    contents.append(types.Part.from_bytes(data=image.bytes, mime_type=image.content_type))

response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents   # 1 prompt + N images → 1 response with N results
)
```

Gemini returns a JSON array with one nutrition object per image, in submission order. This reduces API round trips from N to 1, lowering latency and cost at scale.

---

## 📊 Data Model

Each extraction is stored in DuckDB with this schema:

```sql
CREATE TABLE extractions (
    id          VARCHAR PRIMARY KEY,
    created_at  TIMESTAMP,
    s3_url      TEXT,
    raw_json    JSON        -- Schema-on-Read: full nutrition payload
);
```

The `raw_json` column stores the complete Gemini output, enabling flexible SQL querying without rigid schema constraints:

```sql
-- Example: query all extractions for protein content
SELECT
    id,
    created_at,
    raw_json->'$.per_100g.protein' AS protein_per_100g
FROM extractions
ORDER BY created_at DESC;
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- AWS account with S3 bucket
- Google AI API key (Gemini access)

### 1. Clone the repo

```bash
git clone https://github.com/MrTig-afk/NutritionalTracker.git
cd NutritionalTracker
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
GOOGLE_API_KEY=your_google_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-southeast-2
S3_BUCKET_NAME=your_bucket_name
DATABASE_NAME=nutriscan.duckdb
```

```bash
python main.py
# Backend live at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 3. Frontend setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
VITE_API_URL=http://localhost:8000
```

```bash
npm run dev
# Frontend live at http://localhost:5173
```

---

## 🌐 Deployment

| Service | Platform | Config |
|---------|----------|--------|
| Frontend | Vercel | Root Directory: `frontend`, Build: `npm run build`, Output: `dist` |
| Backend | Render | Root Directory: `backend`, Start: `uvicorn main:app --host 0.0.0.0 --port 10000` |

Environment variables are set directly in each platform's dashboard. `VITE_API_URL` on Vercel should point to your Render service URL.

---

## 📡 API Reference

### `POST /analyze-labels`
Batch endpoint — processes 1–10 images in a single request.

**Request:** `multipart/form-data` with one or more `files` fields

**Response:** JSON array of nutrition objects, one per image in submission order

```json
[
  {
    "per_100g": {
      "calories": 389,
      "fat": "7.1g",
      "protein": "8.9g",
      "carbohydrates": "72g",
      "sugars": "4.2g",
      "fibre": "3.1g",
      "sodium": "0.6g"
    },
    "per_serving": {
      "size": "45g",
      "calories": 175,
      "fat": "3.2g",
      "protein": "4g"
    }
  }
]
```

### `POST /analyze-label`
Legacy single-image endpoint — kept for backward compatibility.

### `GET /health`
Returns service status and client configuration state.

---

## ⚡ Technical Highlights

**Batch LLM inference** — packing multiple images into a single Gemini API call reduces latency from O(n) sequential requests to O(1), with cost savings proportional to the batch size.

**Storage decoupling** — separating raw image storage (S3) from structured metadata (DuckDB) follows the Lakehouse pattern: the data lake preserves raw fidelity while the warehouse optimises for query performance.

**Schema resilience** — Gemini's OCR output is non-deterministic in key naming (e.g. `"per 100g"` vs `"per 100ml"`). The frontend transformation layer dynamically maps extracted keys to a normalised UI schema, making the pipeline tolerant of model variance.

**Real-time macro scaling** — all scaling happens client-side with zero additional API calls. The calculator handles three cases: per-100g base, per-serving base (size extracted from label string), and a graceful fallback to per-100g data when no serving size is present.

---

## 📁 Project Structure

```
NutritionalTracker/
├── backend/
│   ├── main.py          # FastAPI app — single + batch endpoints
│   ├── init_db.py       # DuckDB warehouse initialisation
│   ├── requirements.txt
│   └── .env             # (not committed)
└── frontend/
    ├── src/
    │   ├── App.jsx      # Main React app + NutrientGrid + macro calculator
    │   └── index.css    # Tailwind + custom theme variables
    ├── .env.local        # (not committed) VITE_API_URL for local dev
    └── package.json
```

---

## 🔮 Roadmap


- [ ] Daily intake tracking & goal setting
- [ ] Export extractions to CSV / Parquet from DuckDB
- [ ] User authentication & personal extraction history

---

## 👤 Author

**MrTig** — Data Engineering & Full Stack  
[GitHub](https://github.com/MrTig-afk)

---

*Built with FastAPI, React, Google Gemini, AWS S3, and DuckDB.*
