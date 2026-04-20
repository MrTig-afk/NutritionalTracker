import os
import io
import json
import uuid
import asyncio
import boto3
import duckdb
from datetime import datetime, date
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Optional
import logging
from PIL import Image
from pydantic import BaseModel
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------- CONFIG ----------
DB_NAME        = os.getenv("DATABASE_NAME", "nutriscan.duckdb")
PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-1.5-flash"
MAX_IMAGE_PX   = 1024
JPEG_QUALITY   = 60
GEMINI_TIMEOUT = 45
MAX_RETRIES    = 2

logger.info(f"📦 USING DB: {os.path.abspath(DB_NAME)}")

# ---------- CLIENTS ----------
s3_client = None
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
if all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), S3_BUCKET]):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
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
app = FastAPI(title="NutriScan API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MOCK AUTH ----------
DEFAULT_USER_ID = "user_default"

def get_user_id(x_user_id: Optional[str] = None) -> str:
    return x_user_id if x_user_id else DEFAULT_USER_ID

# ---------- PYDANTIC MODELS ----------
class FolderCreate(BaseModel):
    name: str

class FolderItemCreate(BaseModel):
    name: str
    image_id: str
    nutrition: dict

class MacroGoal(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float
    fibre: Optional[float] = 0.0  # FIX: added fibre goal support

class LogEntry(BaseModel):
    name: str
    servings: float
    nutrition: dict

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
- Calories = integer (ALWAYS in kcal — if label shows kJ, convert: kcal = kJ / 4.184)
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
- Calories as integer in kcal (if label shows kJ, convert: kcal = kJ / 4.184)
- Others as strings with units
- No hallucination
- Output format example: [{{...}}, {{...}}]

RESPOND WITH JSON ONLY:"""


# =============================================================================
# KJ → KCAL CONVERSION HELPERS
# Applied as a safety net after Gemini parsing in case the model missed the
# instruction. Detects implausibly large calorie values (>900 kcal/100g is
# unusual for food) and converts if they appear to be kJ values.
# =============================================================================

KJ_PER_KCAL = 4.184
# Heuristic: if calories > 900 for a 100g serving it's almost certainly kJ
KJ_HEURISTIC_THRESHOLD = 900

def _convert_kj_if_needed(calories_raw) -> float:
    """Convert kJ to kcal if the value is suspiciously large."""
    val = _parse_num(calories_raw)
    if val > KJ_HEURISTIC_THRESHOLD:
        converted = round(val / KJ_PER_KCAL)
        logger.info(f"   🔄 kJ→kcal conversion: {val} → {converted}")
        return converted
    return val

def normalize_nutrition_section(section: dict) -> dict:
    """Normalize a per_serving or per_100g dict: convert kJ calories."""
    if not section or not isinstance(section, dict):
        return section
    result = dict(section)
    if "calories" in result and result["calories"] is not None:
        result["calories"] = int(_convert_kj_if_needed(result["calories"]))
    return result

def normalize_extracted_data(data: dict) -> dict:
    """Apply kJ→kcal normalization to a full extraction result."""
    if not data or not isinstance(data, dict):
        return data
    result = dict(data)
    if "per_serving" in result and result["per_serving"]:
        result["per_serving"] = normalize_nutrition_section(result["per_serving"])
    if "per_100g" in result and result["per_100g"]:
        result["per_100g"] = normalize_nutrition_section(result["per_100g"])
    return result


# =============================================================================
# STORAGE LAYER
# =============================================================================

def optimize_image(image_bytes: bytes, content_type: str = "image/jpeg") -> bytes:
    """Defensive backend optimization. Frontend already optimizes; this is a safety net."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        if w > MAX_IMAGE_PX or h > MAX_IMAGE_PX:
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.LANCZOS)
            logger.info(f"   🔲 Resized {w}x{h} → {img.size}")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        optimized = buf.getvalue()
        logger.info(f"   📉 Optimized: {len(image_bytes)} → {len(optimized)} bytes")
        return optimized
    except Exception as e:
        logger.warning(f"   ⚠️ Optimization failed ({e}), using original")
        return image_bytes


async def upload_raw_and_processed(
    raw_bytes: bytes,
    processed_bytes: bytes,
    user_id: str,
    image_id: str,
    content_type: str,
) -> tuple:
    """
    Upload both raw and processed images to structured S3 paths:
        users/{user_id}/raw/{image_id}.jpg
        users/{user_id}/processed/{image_id}.jpg
    Returns (raw_url, processed_url).
    """
    if not s3_client:
        return ("", "")

    raw_key       = f"users/{user_id}/raw/{image_id}.jpg"
    processed_key = f"users/{user_id}/processed/{image_id}.jpg"

    async def _put(key: str, body: bytes) -> str:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=body,
                    ContentType="image/jpeg",
                ),
            )
            url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
            logger.info(f"   📤 S3: {key}")
            return url
        except Exception as e:
            logger.warning(f"   ⚠️ S3 upload failed for {key}: {e}")
            return ""

    raw_url, processed_url = await asyncio.gather(
        _put(raw_key, raw_bytes),
        _put(processed_key, processed_bytes),
    )
    return raw_url, processed_url


# =============================================================================
# EXTRACTION LAYER
# =============================================================================

async def call_gemini_with_retry(contents: list, label: str = "") -> str:
    delays = [1, 2]

    async def _attempt(model: str, attempt: int) -> str:
        logger.info(f"   🤖 [{label}] {model} attempt {attempt + 1}/{MAX_RETRIES}")
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(model=model, contents=contents),
            ),
            timeout=GEMINI_TIMEOUT,
        )
        return response.text

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await _attempt(PRIMARY_MODEL, attempt)
        except asyncio.TimeoutError:
            last_error = Exception(f"Timeout after {GEMINI_TIMEOUT}s")
            logger.warning(f"   ⏱️ [{label}] Primary timeout attempt {attempt + 1}")
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise HTTPException(status_code=429, detail={
                    "error_type": "quota_exceeded", "retryable": False,
                    "message": "API quota exceeded. Try again later.",
                })
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "unavailable" in err_str.lower()
            logger.warning(f"   ⚠️ [{label}] Primary error attempt {attempt + 1}: {err_str[:120]}")
            if not is_503:
                break
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(delays[attempt])

    logger.warning(f"   🔄 [{label}] Falling back to {FALLBACK_MODEL}")
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(model=FALLBACK_MODEL, contents=contents),
            ),
            timeout=GEMINI_TIMEOUT,
        )
        logger.info(f"   ✅ [{label}] Fallback succeeded")
        return response.text
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "error_type": "timeout", "retryable": True,
            "message": f"Both models timed out after {GEMINI_TIMEOUT}s.",
        })
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise HTTPException(status_code=429, detail={
                "error_type": "quota_exceeded", "retryable": False,
                "message": "API quota exceeded. Try again later.",
            })
        raise HTTPException(status_code=503, detail={
            "error_type": "api_unavailable", "retryable": True,
            "message": f"Gemini unavailable: {str(e)}",
        })


def parse_gemini_json(raw_text: str):
    raw = raw_text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# =============================================================================
# DATABASE LAYER
# =============================================================================

def get_db():
    return duckdb.connect(DB_NAME)


def init_db():
    try:
        con = get_db()

        # Legacy table — kept for backward compat
        con.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                s3_url TEXT,
                raw_json JSON
            )
        """)

        # Extended image records with dual S3 URLs
        con.execute("""
            CREATE TABLE IF NOT EXISTS image_records (
                image_id      VARCHAR PRIMARY KEY,
                user_id       VARCHAR NOT NULL,
                created_at    TIMESTAMP,
                raw_url       TEXT,
                processed_url TEXT,
                extracted_data JSON
            )
        """)

        # Folders
        con.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                folder_id  VARCHAR PRIMARY KEY,
                user_id    VARCHAR NOT NULL,
                name       VARCHAR NOT NULL,
                created_at TIMESTAMP
            )
        """)

        # Folder items
        con.execute("""
            CREATE TABLE IF NOT EXISTS folder_items (
                item_id    VARCHAR PRIMARY KEY,
                folder_id  VARCHAR NOT NULL,
                user_id    VARCHAR NOT NULL,
                image_id   VARCHAR,
                name       VARCHAR NOT NULL,
                nutrition  JSON,
                created_at TIMESTAMP
            )
        """)

        # User macro goals — FIX: added fibre column with migration support
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_goals (
                user_id    VARCHAR PRIMARY KEY,
                calories   DOUBLE,
                protein    DOUBLE,
                carbs      DOUBLE,
                fat        DOUBLE,
                fibre      DOUBLE DEFAULT 0,
                updated_at TIMESTAMP
            )
        """)

        # Migration: add fibre column if it doesn't exist yet (backward compat)
        try:
            con.execute("ALTER TABLE user_goals ADD COLUMN fibre DOUBLE DEFAULT 0")
            logger.info("✅ Migrated user_goals: added fibre column")
        except Exception:
            pass  # Column already exists

        # Daily log — one row per entry
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_log (
                log_id     VARCHAR PRIMARY KEY,
                user_id    VARCHAR NOT NULL,
                date       VARCHAR NOT NULL,
                name       VARCHAR NOT NULL,
                servings   DOUBLE,
                nutrition  JSON,
                created_at TIMESTAMP
            )
        """)

        con.close()
        logger.info("✅ Database initialized (all tables)")
    except Exception as e:
        logger.error(f"❌ DB init failed: {e}")


@app.on_event("startup")
def startup():
    init_db()
    logger.info("\n📋 Registered Routes:")
    for route in app.routes:
        logger.info(f"   {getattr(route, 'methods', {'GET'})} {route.path}")


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "s3_configured": s3_client is not None,
        "gemini_configured": gemini_client is not None,
        "version": "3.1",
    }


# =============================================================================
# EXTRACTION ENDPOINTS
# =============================================================================

@app.post("/analyze-label")
async def analyze_label(
    file: UploadFile = File(...),
    x_user_id: Optional[str] = Header(default=None),
):
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration", "retryable": False,
            "message": "Gemini API not configured",
        })

    user_id   = get_user_id(x_user_id)
    image_id  = str(uuid.uuid4())
    raw_bytes = await file.read()
    logger.info(f"📥 Single image: {file.filename} ({len(raw_bytes)} bytes) user={user_id}")

    processed_bytes = optimize_image(raw_bytes, file.content_type or "image/jpeg")

    contents  = [PROMPT_SINGLE, types.Part.from_bytes(data=processed_bytes, mime_type="image/jpeg")]
    gemini_task = asyncio.create_task(call_gemini_with_retry(contents, label="single"))
    s3_task     = asyncio.create_task(
        upload_raw_and_processed(raw_bytes, processed_bytes, user_id, image_id, file.content_type or "image/jpeg")
    )

    try:
        raw_text = await gemini_task
    except HTTPException:
        s3_task.cancel()
        raise

    raw_url, processed_url = await s3_task

    try:
        parsed = parse_gemini_json(raw_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error", "retryable": True,
            "message": f"Gemini returned invalid JSON: {e}",
        })

    # FIX: Apply kJ→kcal normalization as a safety net
    parsed = normalize_extracted_data(parsed)

    try:
        db = get_db()
        db.execute("INSERT INTO extractions VALUES (?, ?, ?, ?)",
                   [image_id, datetime.now(), processed_url, json.dumps(parsed)])
        db.execute("INSERT INTO image_records VALUES (?, ?, ?, ?, ?, ?)",
                   [image_id, user_id, datetime.now(), raw_url, processed_url, json.dumps(parsed)])
        db.close()
    except Exception as e:
        logger.error(f"❌ DB save failed (non-fatal): {e}")

    logger.info(f"✅ Single processed: {image_id}")
    return {**parsed, "image_id": image_id, "raw_url": raw_url, "processed_url": processed_url}


@app.post("/analyze-labels")
async def analyze_labels(
    files: List[UploadFile] = File(...),
    x_user_id: Optional[str] = Header(default=None),
):
    if not files:
        raise HTTPException(status_code=400, detail={
            "error_type": "validation", "retryable": False, "message": "No files provided",
        })
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration", "retryable": False, "message": "Gemini API not configured",
        })

    MAX_IMAGES = 10
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail={
            "error_type": "validation", "retryable": False,
            "message": f"Too many images. Max: {MAX_IMAGES}",
        })

    user_id = get_user_id(x_user_id)
    logger.info(f"📥 Batch {len(files)} file(s) user={user_id}")

    images_data = []
    for idx, f in enumerate(files):
        raw = await f.read()
        images_data.append({
            "image_id":    str(uuid.uuid4()),
            "user_id":     user_id,
            "bytes":       raw,
            "processed":   optimize_image(raw, f.content_type or "image/jpeg"),
            "content_type": f.content_type or "image/jpeg",
            "filename":    f.filename,
        })
        logger.info(f"   📄 File {idx + 1}: {f.filename} ({len(raw)} bytes)")

    # Smart routing: 1 image → single prompt
    if len(images_data) == 1:
        img      = images_data[0]
        contents = [PROMPT_SINGLE, types.Part.from_bytes(data=img["processed"], mime_type="image/jpeg")]
        g_task   = asyncio.create_task(call_gemini_with_retry(contents, label="routed-single"))
        s_task   = asyncio.create_task(
            upload_raw_and_processed(img["bytes"], img["processed"], user_id, img["image_id"], img["content_type"])
        )
        raw_text = await g_task
        raw_url, processed_url = await s_task

        try:
            parsed = parse_gemini_json(raw_text)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail={
                "error_type": "parse_error", "retryable": True,
                "message": f"Gemini returned invalid JSON: {e}",
            })

        if isinstance(parsed, dict):
            parsed = [parsed]

        # FIX: Apply kJ→kcal normalization
        parsed = [normalize_extracted_data(p) for p in parsed]

        try:
            db = get_db()
            db.execute("INSERT INTO extractions VALUES (?, ?, ?, ?)",
                       [img["image_id"], datetime.now(), processed_url, json.dumps(parsed[0])])
            db.execute("INSERT INTO image_records VALUES (?, ?, ?, ?, ?, ?)",
                       [img["image_id"], user_id, datetime.now(), raw_url, processed_url, json.dumps(parsed[0])])
            db.close()
        except Exception as e:
            logger.error(f"❌ DB save failed (non-fatal): {e}")

        return [{**parsed[0], "image_id": img["image_id"], "raw_url": raw_url, "processed_url": processed_url}]

    # Multi-image batch
    n = len(images_data)
    contents = [build_batch_prompt(n)]
    for img in images_data:
        contents.append(types.Part.from_bytes(data=img["processed"], mime_type="image/jpeg"))

    logger.info(f"🤖 Sending {n} images to Gemini...")

    try:
        raw_text = await call_gemini_with_retry(contents, label=f"batch-{n}")
    except HTTPException:
        raise

    try:
        parsed = parse_gemini_json(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse failure. Raw: {raw_text[:500]}")
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error", "retryable": True,
            "message": f"Gemini returned invalid JSON: {e}",
        })

    if isinstance(parsed, dict):
        parsed = [parsed]
    while len(parsed) < n:
        parsed.append({"error": "Missing data", "per_100g": {}, "per_serving": {}})
    parsed = parsed[:n]

    # FIX: Apply kJ→kcal normalization to all results
    parsed = [normalize_extracted_data(p) for p in parsed]

    s3_tasks   = [
        upload_raw_and_processed(img["bytes"], img["processed"], user_id, img["image_id"], img["content_type"])
        for img in images_data
    ]
    s3_results = await asyncio.gather(*s3_tasks)

    try:
        db = get_db()
        for img, (raw_url, processed_url), result in zip(images_data, s3_results, parsed):
            db.execute("INSERT INTO extractions VALUES (?, ?, ?, ?)",
                       [img["image_id"], datetime.now(), processed_url, json.dumps(result)])
            db.execute("INSERT INTO image_records VALUES (?, ?, ?, ?, ?, ?)",
                       [img["image_id"], user_id, datetime.now(), raw_url, processed_url, json.dumps(result)])
        db.close()
    except Exception as e:
        logger.error(f"❌ DB save failed (non-fatal): {e}")

    logger.info(f"✅ Batch complete: {n} images")
    return [
        {**result, "image_id": img["image_id"], "raw_url": raw_url, "processed_url": processed_url}
        for img, (raw_url, processed_url), result in zip(images_data, s3_results, parsed)
    ]


# =============================================================================
# FOLDER ENDPOINTS
# =============================================================================

@app.post("/folders")
async def create_folder(
    body: FolderCreate,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id   = get_user_id(x_user_id)
    folder_id = str(uuid.uuid4())
    try:
        db = get_db()
        db.execute("INSERT INTO folders VALUES (?, ?, ?, ?)",
                   [folder_id, user_id, body.name, datetime.now()])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"folder_id": folder_id, "user_id": user_id, "name": body.name}


@app.get("/folders")
async def list_folders(x_user_id: Optional[str] = Header(default=None)):
    user_id = get_user_id(x_user_id)
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT folder_id, name, created_at FROM folders WHERE user_id = ? ORDER BY created_at DESC",
            [user_id],
        ).fetchall()
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return [{"folder_id": r[0], "name": r[1], "created_at": str(r[2])} for r in rows]


@app.post("/folders/{folder_id}/items")
async def add_folder_item(
    folder_id: str,
    body: FolderItemCreate,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    item_id = str(uuid.uuid4())
    try:
        db  = get_db()
        row = db.execute(
            "SELECT folder_id FROM folders WHERE folder_id = ? AND user_id = ?",
            [folder_id, user_id],
        ).fetchone()
        if not row:
            db.close()
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Folder not found"})
        db.execute("INSERT INTO folder_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                   [item_id, folder_id, user_id, body.image_id, body.name, json.dumps(body.nutrition), datetime.now()])
        db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"item_id": item_id, "folder_id": folder_id, "name": body.name}


@app.get("/folders/{folder_id}")
async def get_folder(
    folder_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    try:
        db     = get_db()
        folder = db.execute(
            "SELECT folder_id, name, created_at FROM folders WHERE folder_id = ? AND user_id = ?",
            [folder_id, user_id],
        ).fetchone()
        if not folder:
            db.close()
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Folder not found"})
        items = db.execute(
            "SELECT item_id, image_id, name, nutrition, created_at FROM folder_items WHERE folder_id = ? ORDER BY created_at DESC",
            [folder_id],
        ).fetchall()
        db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    return {
        "folder_id":  folder[0],
        "name":       folder[1],
        "created_at": str(folder[2]),
        "items": [
            {
                "item_id":    r[0],
                "image_id":   r[1],
                "name":       r[2],
                "nutrition":  json.loads(r[3]) if isinstance(r[3], str) else r[3],
                "created_at": str(r[4]),
            }
            for r in items
        ],
    }


@app.delete("/folders/{folder_id}/items/{item_id}")
async def delete_folder_item(
    folder_id: str,
    item_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    try:
        db = get_db()
        db.execute("DELETE FROM folder_items WHERE item_id = ? AND folder_id = ? AND user_id = ?",
                   [item_id, folder_id, user_id])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


@app.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    try:
        db = get_db()
        db.execute("DELETE FROM folder_items WHERE folder_id = ? AND user_id = ?", [folder_id, user_id])
        db.execute("DELETE FROM folders WHERE folder_id = ? AND user_id = ?", [folder_id, user_id])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


# =============================================================================
# MACRO GOALS ENDPOINTS — FIX: added fibre support
# =============================================================================

@app.post("/goals")
async def set_goals(
    body: MacroGoal,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    fibre   = body.fibre or 0.0
    try:
        db       = get_db()
        existing = db.execute("SELECT user_id FROM user_goals WHERE user_id = ?", [user_id]).fetchone()
        if existing:
            db.execute(
                "UPDATE user_goals SET calories=?, protein=?, carbs=?, fat=?, fibre=?, updated_at=? WHERE user_id=?",
                [body.calories, body.protein, body.carbs, body.fat, fibre, datetime.now(), user_id],
            )
        else:
            db.execute("INSERT INTO user_goals VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [user_id, body.calories, body.protein, body.carbs, body.fat, fibre, datetime.now()])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {
        "user_id": user_id, "calories": body.calories, "protein": body.protein,
        "carbs": body.carbs, "fat": body.fat, "fibre": fibre,
    }


@app.get("/goals")
async def get_goals(x_user_id: Optional[str] = Header(default=None)):
    user_id = get_user_id(x_user_id)
    try:
        db  = get_db()
        row = db.execute(
            "SELECT calories, protein, carbs, fat, fibre FROM user_goals WHERE user_id = ?", [user_id]
        ).fetchone()
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    if not row:
        return {"calories": 2000.0, "protein": 150.0, "carbs": 250.0, "fat": 65.0, "fibre": 30.0}
    # FIX: handle old rows without fibre column gracefully
    fibre = row[4] if len(row) > 4 and row[4] is not None else 0.0
    return {"calories": row[0], "protein": row[1], "carbs": row[2], "fat": row[3], "fibre": fibre}


# =============================================================================
# DAILY LOG ENDPOINTS
# =============================================================================

def _parse_num(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d.]+", str(v))
    return float(m.group()) if m else 0.0


@app.post("/log")
async def add_log_entry(
    body: LogEntry,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    log_id  = str(uuid.uuid4())
    today   = date.today().isoformat()
    try:
        db = get_db()
        db.execute("INSERT INTO daily_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                   [log_id, user_id, today, body.name, body.servings, json.dumps(body.nutrition), datetime.now()])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"log_id": log_id, "date": today, "name": body.name, "servings": body.servings}


@app.get("/log")
async def get_daily_log(
    log_date: Optional[str] = None,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id     = get_user_id(x_user_id)
    target_date = log_date or date.today().isoformat()
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT log_id, name, servings, nutrition FROM daily_log WHERE user_id = ? AND date = ? ORDER BY created_at",
            [user_id, target_date],
        ).fetchall()
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    items  = []
    # FIX: include fibre in totals
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fibre": 0.0}

    for row in rows:
        log_id, name, servings, nutrition_raw = row
        nutrition = json.loads(nutrition_raw) if isinstance(nutrition_raw, str) else nutrition_raw

        # FIX: Robust per_serving resolution — prefer per_serving, fallback to per_100g
        if nutrition.get("per_serving") and len(nutrition["per_serving"]) > 0:
            per_serving = nutrition["per_serving"]
        elif nutrition.get("per_100g") and len(nutrition["per_100g"]) > 0:
            per_serving = nutrition["per_100g"]
            logger.info(f"   ℹ️ [{name}] No per_serving — falling back to per_100g for log calculation")
        else:
            # Flat nutrition dict (pre-scaled grams mode submission)
            per_serving = nutrition

        # FIX: Apply kJ→kcal conversion on stored data too (handles legacy entries)
        cal_raw = per_serving.get("calories", 0)
        cal_val = _parse_num(cal_raw)
        if cal_val > KJ_HEURISTIC_THRESHOLD:
            cal_val = round(cal_val / KJ_PER_KCAL, 1)

        cal  = cal_val                                              * servings
        prot = _parse_num(per_serving.get("protein",       0))     * servings
        carb = _parse_num(per_serving.get("carbohydrates", 0))     * servings
        fat  = _parse_num(per_serving.get("fat",           0))     * servings
        fib  = _parse_num(per_serving.get("fibre",         0))     * servings

        totals["calories"] += cal
        totals["protein"]  += prot
        totals["carbs"]    += carb
        totals["fat"]      += fat
        totals["fibre"]    += fib

        items.append({
            "log_id":   log_id,
            "name":     name,
            "servings": servings,
            # Return full nutrition object so frontend can scale by weight in edit modal
            "nutrition": nutrition,
            "contribution": {
                "calories": round(cal,  1),
                "protein":  round(prot, 1),
                "carbs":    round(carb, 1),
                "fat":      round(fat,  1),
                "fibre":    round(fib,  1),
            },
        })

    return {
        "date":   target_date,
        "items":  items,
        "totals": {k: round(v, 1) for k, v in totals.items()},
    }


@app.put("/log/{log_id}")
async def update_log_entry(
    log_id: str,
    body: LogEntry,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    try:
        db  = get_db()
        row = db.execute(
            "SELECT log_id FROM daily_log WHERE log_id = ? AND user_id = ?",
            [log_id, user_id],
        ).fetchone()
        if not row:
            db.close()
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Log entry not found"})
        db.execute(
            "UPDATE daily_log SET name=?, servings=?, nutrition=? WHERE log_id=? AND user_id=?",
            [body.name, body.servings, json.dumps(body.nutrition), log_id, user_id],
        )
        db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"log_id": log_id, "name": body.name, "servings": body.servings}


@app.delete("/log/{log_id}")
async def delete_log_entry(
    log_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = get_user_id(x_user_id)
    try:
        db = get_db()
        db.execute("DELETE FROM daily_log WHERE log_id = ? AND user_id = ?", [log_id, user_id])
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


# =============================================================================
# EXCEPTION HANDLER
# =============================================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"❌ Unhandled: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error_type": "internal_error", "retryable": True, "message": str(exc)},
    )


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)