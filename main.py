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
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------- CONFIG ----------
DB_NAME = os.getenv("DATABASE_NAME", "nutriscan.duckdb")

logger.info(f"📦 USING DB: {os.path.abspath(DB_NAME)}")

# ---------- CLIENTS ----------
# Only initialize S3 if credentials exist
s3_client = None
if all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), os.getenv("S3_BUCKET_NAME")]):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-southeast-2")
    )
    logger.info("✅ S3 client initialized")
else:
    logger.warning("⚠️ S3 credentials missing - running in demo mode")

# Initialize Gemini
gemini_client = None
if os.getenv("GOOGLE_API_KEY"):
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("✅ Gemini client initialized")
else:
    logger.warning("⚠️ Gemini API key missing")

# ---------- FASTAPI ----------
app = FastAPI(title="NutriScan API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PROMPTS ----------
PROMPT_SINGLE = """Analyze this nutrition label image carefully.

Extract all nutrient values for BOTH the "per 100g" and "per serving" columns if present.

Return ONLY a valid JSON object with this exact structure (no markdown, no backticks):
{
  "per_serving": {
    "size": "<serving size as a string>",
    "calories": 0,
    "fat": "<string>",
    "saturated_fat": "<string>",
    "carbohydrates": "<string>",
    "sugars": "<string>",
    "fibre": "<string>",
    "protein": "<string>",
    "sodium": "<string>"
  },
  "per_100g": {
    "calories": 0,
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

def build_batch_prompt(n: int) -> str:
    return f"""CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON array with exactly {n} objects.
NO markdown formatting, NO backticks, NO explanatory text before or after.

For EACH of the {n} nutrition label images provided (in order):

Extract nutrient values for BOTH the "per 100g" and "per serving" columns if present.

Each object in the array must follow this structure:
{{
  "per_serving": {{
    "size": "<serving size string or null>",
    "calories": 0,
    "fat": "<string or null>",
    "saturated_fat": "<string or null>",
    "carbohydrates": "<string or null>",
    "sugars": "<string or null>",
    "fibre": "<string or null>",
    "protein": "<string or null>",
    "sodium": "<string or null>"
  }},
  "per_100g": {{
    "calories": 0,
    "fat": "<string or null>",
    "saturated_fat": "<string or null>",
    "carbohydrates": "<string or null>",
    "sugars": "<string or null>",
    "fibre": "<string or null>",
    "protein": "<string or null>",
    "sodium": "<string or null>"
  }}
}}

Rules:
- Use null for missing values
- Calories as integer, others as strings with units
- No hallucination
- Output format example: [{{...}}, {{...}}]

RESPOND WITH JSON ONLY:"""

# ---------- DB INIT ----------
def init_db():
    try:
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
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")

@app.on_event("startup")
def startup():
    init_db()
    # Print all routes
    logger.info("\n📋 Registered Routes:")
    for route in app.routes:
        logger.info(f"   {route.methods if hasattr(route, 'methods') else 'GET'} {route.path}")

# ---------- HEALTH CHECK ----------
@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "s3_configured": s3_client is not None,
        "gemini_configured": gemini_client is not None
    }

# ---------- ORIGINAL SINGLE-IMAGE ENDPOINT ----------
@app.post("/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    """Process a single nutrition label image"""
    try:
        if not gemini_client:
            raise HTTPException(status_code=503, detail="Gemini API not configured")
        
        db = duckdb.connect(DB_NAME)
        file_id = str(uuid.uuid4())
        image_bytes = await file.read()
        
        # Upload to S3 if configured
        s3_url = None
        if s3_client:
            s3_key = f"raw_labels/{file_id}.jpg"
            s3_client.put_object(
                Bucket=os.getenv("S3_BUCKET_NAME"),
                Key=s3_key,
                Body=image_bytes,
                ContentType=file.content_type
            )
            s3_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.amazonaws.com/{s3_key}"
            logger.info(f"📤 Uploaded to S3: {s3_key}")
        
        # Process with Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                PROMPT_SINGLE,
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )
        
        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_json_str)
        
        # Save to database
        db.execute(
            "INSERT INTO extractions VALUES (?, ?, ?, ?)",
            [file_id, datetime.now(), s3_url or "", parsed]
        )
        db.close()
        
        return parsed
        
    except Exception as e:
        logger.error(f"❌ Pipeline Failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- BATCH ENDPOINT ----------
@app.post("/analyze-labels")
async def analyze_labels(files: List[UploadFile] = File(...)):
    """
    Accepts 1-10 images in a single request.
    Returns a JSON array of nutrition objects.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    MAX_IMAGES = 10
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum allowed: {MAX_IMAGES}"
        )
    
    logger.info(f"📥 Received {len(files)} file(s) for batch processing")
    
    try:
        db = duckdb.connect(DB_NAME)
        
        # Read all files
        images_data = []
        for idx, f in enumerate(files):
            raw = await f.read()
            images_data.append({
                "file_id": str(uuid.uuid4()),
                "bytes": raw,
                "content_type": f.content_type,
                "filename": f.filename,
                "index": idx
            })
            logger.info(f"   📄 File {idx + 1}: {f.filename} ({len(raw)} bytes)")
        
        # Upload to S3 if configured
        s3_urls = []
        if s3_client:
            for img in images_data:
                s3_key = f"raw_labels/{img['file_id']}.jpg"
                s3_client.put_object(
                    Bucket=os.getenv("S3_BUCKET_NAME"),
                    Key=s3_key,
                    Body=img["bytes"],
                    ContentType=img["content_type"]
                )
                s3_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.amazonaws.com/{s3_key}"
                s3_urls.append(s3_url)
                logger.info(f"📤 S3 upload: {s3_key}")
        else:
            s3_urls = [""] * len(images_data)
        
        # Build Gemini request
        n = len(images_data)
        contents = [build_batch_prompt(n)]
        
        for img in images_data:
            contents.append(
                types.Part.from_bytes(data=img["bytes"], mime_type=img["content_type"])
            )
        
        logger.info(f"🤖 Sending {n} image(s) to Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        
        # Parse response
        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_json_str)
        
        # Ensure we have an array
        if isinstance(parsed, dict):
            parsed = [parsed]
        
        # Ensure correct length
        if len(parsed) != n:
            logger.warning(f"⚠️ Expected {n} results, got {len(parsed)}")
            while len(parsed) < n:
                parsed.append({"error": "Missing data", "per_100g": {}, "per_serving": {}})
            parsed = parsed[:n]
        
        # Save to database
        for i, (img, s3_url, result) in enumerate(zip(images_data, s3_urls, parsed)):
            db.execute(
                "INSERT INTO extractions VALUES (?, ?, ?, ?)",
                [img["file_id"], datetime.now(), s3_url, result]
            )
        
        db.close()
        logger.info(f"✅ Batch complete: {n} image(s) processed")
        
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse failure: {e}")
        logger.error(f"   Raw response: {response.text[:500]}")
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"❌ Batch Pipeline Failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)