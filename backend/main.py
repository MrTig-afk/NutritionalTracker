import os
import json
import uuid
import boto3
import duckdb
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
DB_NAME = os.getenv("DATABASE_NAME", "nutriscan.duckdb")

print("📦 USING DB:", os.path.abspath(DB_NAME))

# ---------- CLIENTS ----------
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- FASTAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PROMPT ----------
PROMPT = """Analyze this nutrition label image carefully.

Extract all nutrient values for BOTH the "per 100g" and "per serving" columns if present.

Return ONLY a valid JSON object with this exact structure (no markdown, no backticks):
{
  "per_serving": {
    "size": "<serving size as a string>",
    "calories": <integer>,
    "fat": "<string>",
    "saturated_fat": "<string>",
    "carbohydrates": "<string>",
    "sugars": "<string>",
    "fibre": "<string>",
    "protein": "<string>",
    "sodium": "<string>"
  },
  "per_100g": {
    "calories": <integer>,
    "fat": "<string>",
    "saturated_fat": "<string>",
    "carbohydrates": "<string>",
    "sugars": "<string>",
    "fibre": "<string>",
    "protein": "<string>",
    "sodium": "<string>"
  }
}

Rules:
- Omit missing columns or nutrients
- No hallucination
- Calories = integer
- Others = strings with units
- No markdown
"""

# ---------- DB INIT ----------
def init_db():
    con = duckdb.connect(DB_NAME)
    con.execute('''
        CREATE TABLE IF NOT EXISTS extractions (
            id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            s3_url TEXT,
            raw_json JSON
        )
    ''')
    con.close()

@app.on_event("startup")
def startup():
    init_db()


# ---------- ENDPOINT ----------
@app.post("/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    try:
        # new DB connection per request (important)
        db = duckdb.connect(DB_NAME)

        file_id = str(uuid.uuid4())
        s3_key = f"raw_labels/{file_id}.jpg"
        image_bytes = await file.read()

        # ---------- S3 UPLOAD ----------
        s3_client.put_object(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Key=s3_key,
            Body=image_bytes,
            ContentType=file.content_type
        )

        s3_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.amazonaws.com/{s3_key}"

        # ---------- GEMINI ----------
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )

        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()

        # validate JSON
        parsed = json.loads(raw_json_str)

        # ---------- INSERT ----------
        db.execute(
            "INSERT INTO extractions VALUES (?, ?, ?, ?)",
            [file_id, datetime.now(), s3_url, parsed]
        )

        db.close()

        return parsed

    except Exception as e:
        print(f"❌ Pipeline Failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)