import os
import io
import json
import uuid
import asyncio
import boto3
import psycopg2
import psycopg2.extras
import jwt as pyjwt
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
DATABASE_URL   = os.getenv("DATABASE_URL")  # Set to Supabase connection string
PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-1.5-flash"
MAX_IMAGE_PX   = 1024
JPEG_QUALITY   = 60
GEMINI_TIMEOUT = 45
MAX_RETRIES    = 2

logger.info(f"📦 DATABASE_URL configured: {bool(DATABASE_URL)}")

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
app = FastAPI(title="NutriScan API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SUPABASE AUTH ----------
SUPABASE_URL        = os.getenv("SUPABASE_URL", "https://zdmsfftfqnajanpbvcgn.supabase.co")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")  # Settings → API → JWT Secret

def get_user_id(authorization: Optional[str] = None) -> str:
    """Extract user ID from Supabase JWT. Falls back to default if no token (dev mode)."""
    if not authorization:
        logger.warning("⚠️ No authorization header — using default user (dev mode)")
        return "user_default"
    try:
        token = authorization.replace("Bearer ", "").strip()
        if not SUPABASE_JWT_SECRET:
            raise Exception("SUPABASE_JWT_SECRET not configured")
        payload = pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise Exception("No sub claim in JWT")
        return user_id
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={"error_type": "token_expired", "message": "Session expired. Please log in again."})
    except Exception as e:
        logger.warning(f"⚠️ JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Invalid or missing token"})

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
    fibre: Optional[float] = 0.0

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
# =============================================================================

KJ_PER_KCAL = 4.184
KJ_HEURISTIC_THRESHOLD = 900

def _parse_num(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d.]+", str(v))
    return float(m.group()) if m else 0.0

def _convert_kj_if_needed(calories_raw) -> float:
    val = _parse_num(calories_raw)
    if val > KJ_HEURISTIC_THRESHOLD:
        converted = round(val / KJ_PER_KCAL)
        logger.info(f"   🔄 kJ→kcal conversion: {val} → {converted}")
        return converted
    return val

def normalize_nutrition_section(section: dict) -> dict:
    if not section or not isinstance(section, dict):
        return section
    result = dict(section)
    if "calories" in result and result["calories"] is not None:
        result["calories"] = int(_convert_kj_if_needed(result["calories"]))
    return result

def normalize_extracted_data(data: dict) -> dict:
    if not data or not isinstance(data, dict):
        return data
    result = dict(data)
    if "per_serving" in result and result["per_serving"]:
        result["per_serving"] = normalize_nutrition_section(result["per_serving"])
    if "per_100g" in result and result["per_100g"]:
        result["per_100g"] = normalize_nutrition_section(result["per_100g"])
    return result


# =============================================================================
# DATABASE LAYER — PostgreSQL
# =============================================================================

def get_db():
    """Return a new psycopg2 connection. Caller must close it."""
    if not DATABASE_URL:
        raise Exception("DATABASE_URL environment variable not set")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to run on every startup."""
    try:
        conn = get_db()
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id         VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                s3_url     TEXT,
                raw_json   JSONB
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_records (
                image_id      VARCHAR PRIMARY KEY,
                user_id       VARCHAR NOT NULL,
                created_at    TIMESTAMP,
                raw_url       TEXT,
                processed_url TEXT,
                extracted_data JSONB
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                folder_id  VARCHAR PRIMARY KEY,
                user_id    VARCHAR NOT NULL,
                name       VARCHAR NOT NULL,
                created_at TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS folder_items (
                item_id    VARCHAR PRIMARY KEY,
                folder_id  VARCHAR NOT NULL,
                user_id    VARCHAR NOT NULL,
                image_id   VARCHAR,
                name       VARCHAR NOT NULL,
                nutrition  JSONB,
                created_at TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_goals (
                user_id    VARCHAR PRIMARY KEY,
                calories   DOUBLE PRECISION,
                protein    DOUBLE PRECISION,
                carbs      DOUBLE PRECISION,
                fat        DOUBLE PRECISION,
                fibre      DOUBLE PRECISION DEFAULT 0,
                updated_at TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_log (
                log_id     VARCHAR PRIMARY KEY,
                user_id    VARCHAR NOT NULL,
                date       VARCHAR NOT NULL,
                name       VARCHAR NOT NULL,
                servings   DOUBLE PRECISION,
                nutrition  JSONB,
                created_at TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ PostgreSQL database initialized (all tables)")
    except Exception as e:
        logger.error(f"❌ DB init failed: {e}")
        raise


@app.on_event("startup")
def startup():
    init_db()
    logger.info("\n📋 Registered Routes:")
    for route in app.routes:
        logger.info(f"   {getattr(route, 'methods', {'GET'})} {route.path}")


# =============================================================================
# STORAGE LAYER
# =============================================================================

def optimize_image(image_bytes: bytes, content_type: str = "image/jpeg") -> bytes:
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
                    Bucket=S3_BUCKET, Key=key, Body=body, ContentType="image/jpeg",
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
        "version": "3.2",
    }


# =============================================================================
# EXTRACTION ENDPOINTS
# =============================================================================

@app.post("/analyze-label")
async def analyze_label(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration", "retryable": False,
            "message": "Gemini API not configured",
        })

    user_id   = get_user_id(authorization)
    image_id  = str(uuid.uuid4())
    raw_bytes = await file.read()

    processed_bytes = optimize_image(raw_bytes, file.content_type or "image/jpeg")
    raw_url, processed_url = await upload_raw_and_processed(
        raw_bytes, processed_bytes, user_id, image_id, file.content_type or "image/jpeg"
    )

    image_part = types.Part.from_bytes(data=processed_bytes, mime_type="image/jpeg")
    contents   = [image_part, PROMPT_SINGLE]

    raw_text = await call_gemini_with_retry(contents, label=image_id[:8])

    try:
        result = parse_gemini_json(raw_text)
        result = normalize_extracted_data(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error", "retryable": True,
            "message": f"Failed to parse Gemini response: {str(e)}",
        })

    result["image_id"]      = image_id
    result["raw_url"]       = raw_url
    result["processed_url"] = processed_url

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO image_records (image_id, user_id, created_at, raw_url, processed_url, extracted_data) VALUES (%s, %s, %s, %s, %s, %s)",
            [image_id, user_id, datetime.now(), raw_url, processed_url, json.dumps(result)],
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"⚠️ DB insert failed (non-fatal): {e}")

    return result


@app.post("/analyze-labels")
async def analyze_labels(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(default=None),
):
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration", "retryable": False,
            "message": "Gemini API not configured",
        })
    if len(files) > 10:
        raise HTTPException(status_code=400, detail={
            "error_type": "too_many_files", "retryable": False,
            "message": "Maximum 10 images per batch",
        })

    user_id = get_user_id(authorization)

    processed_images = []
    image_ids        = []
    upload_tasks     = []

    for f in files:
        raw_bytes       = await f.read()
        processed_bytes = optimize_image(raw_bytes, f.content_type or "image/jpeg")
        image_id        = str(uuid.uuid4())
        image_ids.append(image_id)
        processed_images.append(processed_bytes)
        upload_tasks.append(upload_raw_and_processed(
            raw_bytes, processed_bytes, user_id, image_id, f.content_type or "image/jpeg"
        ))

    upload_results = await asyncio.gather(*upload_tasks)

    contents = []
    for pb in processed_images:
        contents.append(types.Part.from_bytes(data=pb, mime_type="image/jpeg"))
    contents.append(build_batch_prompt(len(files)))

    raw_text = await call_gemini_with_retry(contents, label=f"batch-{len(files)}")

    try:
        results = parse_gemini_json(raw_text)
        if not isinstance(results, list):
            results = [results]
        results = [normalize_extracted_data(r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error_type": "parse_error", "retryable": True,
            "message": f"Failed to parse batch response: {str(e)}",
        })

    try:
        conn = get_db()
        cur  = conn.cursor()
        for i, result in enumerate(results):
            raw_url, processed_url = upload_results[i] if i < len(upload_results) else ("", "")
            result["image_id"]      = image_ids[i]
            result["raw_url"]       = raw_url
            result["processed_url"] = processed_url
            cur.execute(
                "INSERT INTO image_records (image_id, user_id, created_at, raw_url, processed_url, extracted_data) VALUES (%s, %s, %s, %s, %s, %s)",
                [image_ids[i], user_id, datetime.now(), raw_url, processed_url, json.dumps(result)],
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"⚠️ Batch DB insert failed (non-fatal): {e}")

    return results


# =============================================================================
# FOLDER ENDPOINTS
# =============================================================================

@app.post("/folders")
async def create_folder(
    body: FolderCreate,
    authorization: Optional[str] = Header(default=None),
):
    user_id   = get_user_id(authorization)
    folder_id = str(uuid.uuid4())
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO folders (folder_id, user_id, name, created_at) VALUES (%s, %s, %s, %s)",
            [folder_id, user_id, body.name, datetime.now()],
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"folder_id": folder_id, "name": body.name}


@app.get("/folders")
async def list_folders(authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT folder_id, name FROM folders WHERE user_id = %s ORDER BY created_at DESC",
            [user_id],
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return [{"folder_id": r[0], "name": r[1]} for r in rows]


@app.get("/folders/{folder_id}")
async def get_folder(
    folder_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT folder_id, name FROM folders WHERE folder_id = %s AND user_id = %s",
            [folder_id, user_id],
        )
        folder = cur.fetchone()
        if not folder:
            cur.close(); conn.close()
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Folder not found"})
        cur.execute(
            "SELECT item_id, name, nutrition FROM folder_items WHERE folder_id = %s AND user_id = %s ORDER BY created_at DESC",
            [folder_id, user_id],
        )
        items = cur.fetchall()
        cur.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    return {
        "folder_id": folder[0],
        "name":      folder[1],
        "items": [
            {
                "item_id":   row[0],
                "name":      row[1],
                "nutrition": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            }
            for row in items
        ],
    }


@app.post("/folders/{folder_id}/items")
async def add_folder_item(
    folder_id: str,
    body: FolderItemCreate,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    item_id = str(uuid.uuid4())
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO folder_items (item_id, folder_id, user_id, image_id, name, nutrition, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [item_id, folder_id, user_id, body.image_id, body.name, json.dumps(body.nutrition), datetime.now()],
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"item_id": item_id, "name": body.name}


@app.delete("/folders/{folder_id}/items/{item_id}")
async def delete_folder_item(
    folder_id: str,
    item_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "DELETE FROM folder_items WHERE item_id = %s AND folder_id = %s AND user_id = %s",
            [item_id, folder_id, user_id],
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


@app.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("DELETE FROM folder_items WHERE folder_id = %s AND user_id = %s", [folder_id, user_id])
        cur.execute("DELETE FROM folders WHERE folder_id = %s AND user_id = %s", [folder_id, user_id])
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


# =============================================================================
# MACRO GOALS ENDPOINTS
# =============================================================================

@app.post("/goals")
async def set_goals(
    body: MacroGoal,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    fibre   = body.fibre or 0.0
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT user_id FROM user_goals WHERE user_id = %s", [user_id])
        existing = cur.fetchone()
        if existing:
            cur.execute(
                "UPDATE user_goals SET calories=%s, protein=%s, carbs=%s, fat=%s, fibre=%s, updated_at=%s WHERE user_id=%s",
                [body.calories, body.protein, body.carbs, body.fat, fibre, datetime.now(), user_id],
            )
        else:
            cur.execute(
                "INSERT INTO user_goals (user_id, calories, protein, carbs, fat, fibre, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [user_id, body.calories, body.protein, body.carbs, body.fat, fibre, datetime.now()],
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {
        "user_id": user_id, "calories": body.calories, "protein": body.protein,
        "carbs": body.carbs, "fat": body.fat, "fibre": fibre,
    }


@app.get("/goals")
async def get_goals(authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT calories, protein, carbs, fat, fibre FROM user_goals WHERE user_id = %s",
            [user_id],
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    if not row:
        return {"calories": 2000.0, "protein": 150.0, "carbs": 250.0, "fat": 65.0, "fibre": 30.0}
    fibre = row[4] if len(row) > 4 and row[4] is not None else 0.0
    return {"calories": row[0], "protein": row[1], "carbs": row[2], "fat": row[3], "fibre": fibre}


# =============================================================================
# DAILY LOG ENDPOINTS
# =============================================================================

@app.post("/log")
async def add_log_entry(
    body: LogEntry,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    log_id  = str(uuid.uuid4())
    today   = date.today().isoformat()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO daily_log (log_id, user_id, date, name, servings, nutrition, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [log_id, user_id, today, body.name, body.servings, json.dumps(body.nutrition), datetime.now()],
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"log_id": log_id, "date": today, "name": body.name, "servings": body.servings}


@app.get("/log")
async def get_daily_log(
    log_date: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    user_id     = get_user_id(authorization)
    target_date = log_date or date.today().isoformat()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT log_id, name, servings, nutrition FROM daily_log WHERE user_id = %s AND date = %s ORDER BY created_at",
            [user_id, target_date],
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    items  = []
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fibre": 0.0}

    for row in rows:
        log_id, name, servings, nutrition_raw = row
        nutrition = nutrition_raw if isinstance(nutrition_raw, dict) else json.loads(nutrition_raw)

        if nutrition.get("per_serving") and len(nutrition["per_serving"]) > 0:
            per_serving = nutrition["per_serving"]
        elif nutrition.get("per_100g") and len(nutrition["per_100g"]) > 0:
            per_serving = nutrition["per_100g"]
            logger.info(f"   ℹ️ [{name}] No per_serving — falling back to per_100g for log calculation")
        else:
            per_serving = nutrition

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
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT log_id FROM daily_log WHERE log_id = %s AND user_id = %s",
            [log_id, user_id],
        )
        if not cur.fetchone():
            cur.close(); conn.close()
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Log entry not found"})
        cur.execute(
            "UPDATE daily_log SET name=%s, servings=%s, nutrition=%s WHERE log_id=%s AND user_id=%s",
            [body.name, body.servings, json.dumps(body.nutrition), log_id, user_id],
        )
        conn.commit()
        cur.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"log_id": log_id, "name": body.name, "servings": body.servings}


@app.delete("/log/{log_id}")
async def delete_log_entry(
    log_id: str,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("DELETE FROM daily_log WHERE log_id = %s AND user_id = %s", [log_id, user_id])
        conn.commit()
        cur.close()
        conn.close()
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