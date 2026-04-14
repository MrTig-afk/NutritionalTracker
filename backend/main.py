import os
import io
import json
import uuid
import asyncio
import boto3
import duckdb
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------- CONFIG ----------
DB_NAME = os.getenv("DATABASE_NAME", "nutriscan.duckdb")
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-1.5-flash"
MAX_IMAGE_PX = 1024
JPEG_QUALITY = 60
GEMINI_TIMEOUT = 45  # seconds
MAX_RETRIES = 2      # reduced from 3

logger.info(f"📦 USING DB: {os.path.abspath(DB_NAME)}")

# ---------- CLIENTS ----------
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

gemini_client = None
if os.getenv("GOOGLE_API_KEY"):
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("✅ Gemini client initialized")
else:
    logger.warning("⚠️ Gemini API key missing")

# ---------- FASTAPI ----------
app = FastAPI(title="NutriScan API", version="2.1")

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


# ---------- IMAGE OPTIMIZATION ----------
def optimize_image(image_bytes: bytes, content_type: str = "image/jpeg") -> bytes:
    """Resize to max 1024px and compress JPEG to quality 60. Kept as defensive backend layer."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        if w > MAX_IMAGE_PX or h > MAX_IMAGE_PX:
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.LANCZOS)
            logger.info(f"   🔲 Resized from {w}x{h} → {img.size}")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        optimized = buf.getvalue()
        logger.info(f"   📉 Optimized: {len(image_bytes)} → {len(optimized)} bytes")
        return optimized
    except Exception as e:
        logger.warning(f"   ⚠️ Image optimization failed ({e}), using original")
        return image_bytes


# ---------- GEMINI RETRY + FALLBACK ----------
async def call_gemini_with_retry(contents: list, label: str = "") -> str:
    """
    Call Gemini with exponential backoff retry on 503 errors.
    Does NOT retry on 429 (quota exceeded) — raises immediately.
    Falls back to FALLBACK_MODEL after primary exhausts retries.
    Enforces GEMINI_TIMEOUT per attempt.
    """
    delays = [1, 2]  # matches MAX_RETRIES=2

    async def _attempt(model: str, attempt: int) -> str:
        logger.info(f"   🤖 [{label}] {model} attempt {attempt + 1}/{MAX_RETRIES}")
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(
                    model=model,
                    contents=contents
                )
            ),
            timeout=GEMINI_TIMEOUT
        )
        return response.text

    # Try primary model
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await _attempt(PRIMARY_MODEL, attempt)
        except asyncio.TimeoutError:
            last_error = Exception(f"Timeout after {GEMINI_TIMEOUT}s")
            logger.warning(f"   ⏱️ [{label}] Primary model timeout on attempt {attempt + 1}")
        except Exception as e:
            last_error = e
            err_str = str(e)

            # Do NOT retry on 429 — raise immediately
            is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_429:
                logger.warning(f"   🚫 [{label}] Quota exceeded (429), not retrying")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error_type": "quota_exceeded",
                        "retryable": False,
                        "message": "API quota exceeded. Try again later."
                    }
                )

            is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "unavailable" in err_str.lower()
            logger.warning(f"   ⚠️ [{label}] Primary model error on attempt {attempt + 1}: {err_str[:120]}")
            if not is_503:
                break  # Non-retriable error, go straight to fallback

        if attempt < MAX_RETRIES - 1:
            wait = delays[attempt]
            logger.info(f"   ⏳ [{label}] Retrying in {wait}s...")
            await asyncio.sleep(wait)

    # Try fallback model (single attempt)
    logger.warning(f"   🔄 [{label}] Falling back to {FALLBACK_MODEL}")
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(
                    model=FALLBACK_MODEL,
                    contents=contents
                )
            ),
            timeout=GEMINI_TIMEOUT
        )
        logger.info(f"   ✅ [{label}] Fallback model succeeded")
        return response.text
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error_type": "timeout",
                "retryable": True,
                "message": f"Both models timed out after {GEMINI_TIMEOUT}s. Please try again."
            }
        )
    except Exception as e:
        err_str = str(e)
        # Also guard 429 on fallback
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise HTTPException(
                status_code=429,
                detail={
                    "error_type": "quota_exceeded",
                    "retryable": False,
                    "message": "API quota exceeded. Try again later."
                }
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error_type": "api_unavailable",
                "retryable": True,
                "message": f"Gemini API unavailable on both models: {str(e)}"
            }
        )


# ---------- ASYNC S3 UPLOAD ----------
async def upload_to_s3_async(image_bytes: bytes, file_id: str, content_type: str) -> str:
    """Upload to S3 in background thread without blocking inference."""
    if not s3_client:
        return ""
    try:
        s3_key = f"raw_labels/{file_id}.jpg"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=os.getenv("S3_BUCKET_NAME"),
                Key=s3_key,
                Body=image_bytes,
                ContentType=content_type
            )
        )
        s3_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.amazonaws.com/{s3_key}"
        logger.info(f"   📤 S3 upload: {s3_key}")
        return s3_url
    except Exception as e:
        logger.warning(f"   ⚠️ S3 upload failed (non-fatal): {e}")
        return ""


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


# ---------- SINGLE-IMAGE ENDPOINT ----------
@app.post("/analyze-label")
async def analyze_label(file: UploadFile = File(...)):
    """Process a single nutrition label image"""
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration",
            "retryable": False,
            "message": "Gemini API not configured"
        })

    file_id = str(uuid.uuid4())
    image_bytes = await file.read()
    logger.info(f"📥 Single image: {file.filename} ({len(image_bytes)} bytes)")

    # Backend optimization layer (defensive — frontend already optimized)
    optimized_bytes = optimize_image(image_bytes, file.content_type or "image/jpeg")

    contents = [
        PROMPT_SINGLE,
        types.Part.from_bytes(data=optimized_bytes, mime_type="image/jpeg")
    ]

    # Run inference and S3 upload concurrently
    gemini_task = asyncio.create_task(call_gemini_with_retry(contents, label="single"))
    s3_task     = asyncio.create_task(upload_to_s3_async(image_bytes, file_id, file.content_type or "image/jpeg"))

    try:
        raw_text = await gemini_task
    except HTTPException:
        s3_task.cancel()
        raise

    s3_url = await s3_task

    try:
        raw_json_str = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error",
            "retryable": True,
            "message": f"Gemini returned invalid JSON: {e}"
        })

    try:
        db = duckdb.connect(DB_NAME)
        db.execute(
            "INSERT INTO extractions VALUES (?, ?, ?, ?)",
            [file_id, datetime.now(), s3_url, parsed]
        )
        db.close()
    except Exception as e:
        logger.error(f"❌ DB save failed (non-fatal): {e}")

    logger.info(f"✅ Single image processed: {file_id}")
    return parsed


# ---------- BATCH ENDPOINT ----------
@app.post("/analyze-labels")
async def analyze_labels(files: List[UploadFile] = File(...)):
    """
    Accepts 1-10 images in a single request.
    Smart routing: single image reuses single-image logic internally.
    Returns a JSON array of nutrition objects.
    """
    if not files:
        raise HTTPException(status_code=400, detail={
            "error_type": "validation",
            "retryable": False,
            "message": "No files provided"
        })

    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration",
            "retryable": False,
            "message": "Gemini API not configured"
        })

    MAX_IMAGES = 10
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail={
            "error_type": "validation",
            "retryable": False,
            "message": f"Too many images. Maximum allowed: {MAX_IMAGES}"
        })

    logger.info(f"📥 Received {len(files)} file(s) for batch processing")

    # Read all files
    images_data = []
    for idx, f in enumerate(files):
        raw = await f.read()
        images_data.append({
            "file_id": str(uuid.uuid4()),
            "bytes": raw,
            "content_type": f.content_type or "image/jpeg",
            "filename": f.filename,
            "index": idx
        })
        logger.info(f"   📄 File {idx + 1}: {f.filename} ({len(raw)} bytes)")

    # Smart routing: single image → single-image logic
    if len(images_data) == 1:
        logger.info("   🔀 Smart routing: 1 image → single-image pipeline")
        img = images_data[0]
        optimized = optimize_image(img["bytes"], img["content_type"])
        contents = [
            PROMPT_SINGLE,
            types.Part.from_bytes(data=optimized, mime_type="image/jpeg")
        ]
        gemini_task = asyncio.create_task(call_gemini_with_retry(contents, label="routed-single"))
        s3_task     = asyncio.create_task(upload_to_s3_async(img["bytes"], img["file_id"], img["content_type"]))

        raw_text = await gemini_task
        s3_url   = await s3_task

        try:
            raw_json_str = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw_json_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail={
                "error_type": "parse_error",
                "retryable": True,
                "message": f"Gemini returned invalid JSON: {e}"
            })

        if isinstance(parsed, dict):
            parsed = [parsed]

        try:
            db = duckdb.connect(DB_NAME)
            db.execute(
                "INSERT INTO extractions VALUES (?, ?, ?, ?)",
                [img["file_id"], datetime.now(), s3_url, parsed[0]]
            )
            db.close()
        except Exception as e:
            logger.error(f"❌ DB save failed (non-fatal): {e}")

        logger.info("✅ Smart-routed single image processed")
        return parsed

    # Multi-image batch path — optimize all images (defensive backend layer)
    for img in images_data:
        img["optimized_bytes"] = optimize_image(img["bytes"], img["content_type"])

    n = len(images_data)
    contents = [build_batch_prompt(n)]
    for img in images_data:
        contents.append(
            types.Part.from_bytes(data=img["optimized_bytes"], mime_type="image/jpeg")
        )

    logger.info(f"🤖 Sending {n} optimized image(s) to Gemini...")

    try:
        raw_text = await call_gemini_with_retry(contents, label=f"batch-{n}")
    except HTTPException:
        raise

    try:
        raw_json_str = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse failure: {e}")
        logger.error(f"   Raw response: {raw_text[:500]}")
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error",
            "retryable": True,
            "message": f"Gemini returned invalid JSON: {e}"
        })

    if isinstance(parsed, dict):
        parsed = [parsed]

    if len(parsed) != n:
        logger.warning(f"⚠️ Expected {n} results, got {len(parsed)}")
        while len(parsed) < n:
            parsed.append({"error": "Missing data", "per_100g": {}, "per_serving": {}})
        parsed = parsed[:n]

    # S3 uploads after inference (concurrent)
    s3_tasks = [
        upload_to_s3_async(img["bytes"], img["file_id"], img["content_type"])
        for img in images_data
    ]
    s3_urls = await asyncio.gather(*s3_tasks)

    try:
        db = duckdb.connect(DB_NAME)
        for img, s3_url, result in zip(images_data, s3_urls, parsed):
            db.execute(
                "INSERT INTO extractions VALUES (?, ?, ?, ?)",
                [img["file_id"], datetime.now(), s3_url, result]
            )
        db.close()
    except Exception as e:
        logger.error(f"❌ DB save failed (non-fatal): {e}")

    logger.info(f"✅ Batch complete: {n} image(s) processed")
    return parsed


# ---------- EXCEPTION HANDLER ----------
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "internal_error",
            "retryable": True,
            "message": str(exc)
        }
    )


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
