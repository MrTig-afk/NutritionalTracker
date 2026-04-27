import os
import io
import json
import uuid
import asyncio
import time
import boto3
import psycopg2
import psycopg2.pool
import psycopg2.extras
import jwt as pyjwt
from collections import defaultdict
from datetime import datetime, date, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv
from typing import List, Optional
import logging
from PIL import Image
from pydantic import BaseModel, Field
import re
import threading
import urllib.request

try:
    from pywebpush import webpush, WebPushException
    _webpush_ok = True
except ImportError:
    _webpush_ok = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------- CONFIG ----------
DATABASE_URL   = os.getenv("DATABASE_URL")  # Set to Supabase connection string
PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.0-flash"
MAX_IMAGE_PX   = 1024
JPEG_QUALITY   = 60
GEMINI_TIMEOUT = 45
MAX_RETRIES    = 2
DAILY_LIMIT    = 10

NTFY_TOPIC       = os.getenv("NTFY_TOPIC", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "").replace("\\n", "\n")
VAPID_PUBLIC_KEY  = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_CLAIM       = os.getenv("VAPID_CLAIM", "mailto:theimpracticalguy007@gmail.com")

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

groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("✅ Groq client initialized")
else:
    logger.warning("⚠️ Groq API key missing — AI assistant disabled")

# ---------- CHAT RATE LIMITER ----------
# Sliding-window in-memory counters. Fine for single-process deployments.
_user_chat_log: dict  = defaultdict(list)  # user_id -> [timestamps]
_global_chat_log: list = []
USER_CHAT_RPM   = 3
GLOBAL_CHAT_RPM = 25

def _allow_chat(user_id: str) -> bool:
    now    = time.time()
    cutoff = now - 60
    _user_chat_log[user_id] = [t for t in _user_chat_log[user_id] if t > cutoff]
    while _global_chat_log and _global_chat_log[0] < cutoff:
        _global_chat_log.pop(0)
    if len(_user_chat_log[user_id]) >= USER_CHAT_RPM:
        return False
    if len(_global_chat_log) >= GLOBAL_CHAT_RPM:
        return False
    _user_chat_log[user_id].append(now)
    _global_chat_log.append(now)
    return True

# ---------- FASTAPI ----------
app = FastAPI(title="NutriScan API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nutritional-tracker-delta.vercel.app",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SUPABASE AUTH ----------
SUPABASE_URL        = os.getenv("SUPABASE_URL", "https://zdmsfftfqnajanpbvcgn.supabase.co")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

_jwks_client = pyjwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)

def get_user_id(authorization: Optional[str] = None) -> str:
    """Extract user ID from Supabase JWT. Supports both HS256 and RS256."""
    if not authorization:
        raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Missing authorization header"})
    try:
        token = authorization.replace("Bearer ", "").strip()
        header = pyjwt.get_unverified_header(token)
        alg    = header.get("alg", "HS256")
        logger.info(f"🔑 JWT alg: {alg}")

        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise Exception("SUPABASE_JWT_SECRET not configured")
            payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        else:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(token, signing_key.key, algorithms=[alg], options={"verify_aud": False})

        user_id = payload.get("sub")
        if not user_id:
            raise Exception("No sub claim in JWT")
        return user_id
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={"error_type": "token_expired", "message": "Session expired. Please log in again."})
    except Exception as e:
        logger.warning(f"⚠️ JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Invalid or missing token"})


def get_user_info(authorization: Optional[str] = None) -> tuple:
    """Extract user ID and email from Supabase JWT."""
    if not authorization:
        raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Missing authorization header"})
    try:
        token = authorization.replace("Bearer ", "").strip()
        header = pyjwt.get_unverified_header(token)
        alg    = header.get("alg", "HS256")
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise Exception("SUPABASE_JWT_SECRET not configured")
            payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        else:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(token, signing_key.key, algorithms=[alg], options={"verify_aud": False})
        user_id = payload.get("sub")
        if not user_id:
            raise Exception("No sub claim in JWT")
        return user_id, payload.get("email", "")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={"error_type": "token_expired", "message": "Session expired. Please log in again."})
    except Exception as e:
        logger.warning(f"⚠️ JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail={"error_type": "unauthorized", "message": "Invalid or missing token"})


def check_and_track(user_id: str, email: str, client_date: str = None):
    """Upsert user email and enforce daily rate limit."""
    today = client_date or date.today().isoformat()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO users (user_id, email, last_seen)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET email = EXCLUDED.email, last_seen = EXCLUDED.last_seen
        """, [user_id, email, datetime.now()])
        cur.execute(
            "SELECT count FROM api_usage WHERE user_id = %s AND date = %s",
            [user_id, today]
        )
        row = cur.fetchone()
        if row:
            if row[0] >= DAILY_LIMIT:
                cur.close(); release_db(conn)
                send_ntfy_alert("📊 Scan Limit Hit", f"User {user_id[:8]} hit the {DAILY_LIMIT}/day scan limit")
                send_push_to_user(user_id, "📊 Scan Limit Reached", f"You've used all {DAILY_LIMIT} scans for today. Come back tomorrow!")
                raise HTTPException(status_code=429, detail={
                    "error_type": "rate_limit_exceeded",
                    "retryable": False,
                    "message": f"You've used all {DAILY_LIMIT} scans for today. Come back tomorrow!",
                })
            cur.execute(
                "UPDATE api_usage SET count = count + 1 WHERE user_id = %s AND date = %s",
                [user_id, today]
            )
        else:
            cur.execute(
                "INSERT INTO api_usage (user_id, date, count) VALUES (%s, %s, 1)",
                [user_id, today]
            )
        conn.commit()
        cur.close()
        release_db(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"⚠️ Usage tracking failed (non-fatal): {e}")


# ---------- ALERT HELPERS ----------

def send_ntfy_alert(title: str, message: str, priority: int = 3):
    if not NTFY_TOPIC:
        return
    def _send():
        try:
            data = json.dumps({"topic": NTFY_TOPIC, "title": title, "message": message, "priority": priority}).encode()
            req = urllib.request.Request(
                "https://ntfy.sh",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug(f"ntfy send failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


def send_push_to_user(user_id: str, title: str, body: str):
    if not (_webpush_ok and VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY):
        return
    def _send():
        try:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("SELECT subscription_json FROM push_subscriptions WHERE user_id = %s", [user_id])
            rows = cur.fetchall()
            cur.close()
            release_db(conn)
            for row in rows:
                sub = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                webpush(
                    subscription_info=sub,
                    data=json.dumps({"title": title, "body": body}),
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims={"sub": VAPID_CLAIM},
                )
        except Exception as e:
            logger.debug(f"Push send failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


def _get_entry_calories(nutrition_raw, servings: float) -> float:
    if not nutrition_raw:
        return 0.0
    n = nutrition_raw if isinstance(nutrition_raw, dict) else json.loads(nutrition_raw)
    ps = n.get("per_serving") or n.get("per_100g") or n
    raw = _parse_num(ps.get("calories", 0)) * servings
    return round(raw / KJ_PER_KCAL, 1) if raw > KJ_HEURISTIC_THRESHOLD else round(raw, 1)


def _check_goal_and_push(user_id: str, today: str):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT calories FROM user_goals WHERE user_id = %s", [user_id])
        goal_row = cur.fetchone()
        if not goal_row or not goal_row[0]:
            cur.close(); release_db(conn); return
        goal_cal = goal_row[0]
        cur.execute("SELECT servings, nutrition FROM daily_log WHERE user_id = %s AND date = %s", [user_id, today])
        rows = cur.fetchall()
        cur.close()
        release_db(conn)
        total = sum(_get_entry_calories(r[1], r[0]) for r in rows)
        if total >= goal_cal:
            send_push_to_user(user_id, "🎯 Daily Goal Hit!", f"You've reached {round(total)} kcal — goal was {round(goal_cal)} kcal!")
    except Exception as e:
        logger.debug(f"Goal push check failed: {e}")


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
    log_date: Optional[str] = None

class ChatHistoryItem(BaseModel):
    role: str
    text: str

class ChatMessage(BaseModel):
    message: str = Field(..., max_length=2000)
    history: list[ChatHistoryItem] = []

class MealTemplateCreate(BaseModel):
    name: str

class MealTemplateItemCreate(BaseModel):
    name: str
    nutrition: dict
    servings: float = 1.0

class PushSubscriptionCreate(BaseModel):
    endpoint: str
    expirationTime: Optional[int] = None
    keys: dict

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

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise Exception("DATABASE_URL environment variable not set")
        _pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
        logger.info("✅ Connection pool initialised (min=1, max=5)")
    return _pool

def _reset_pool():
    global _pool
    if _pool:
        try:
            _pool.closeall()
        except Exception:
            pass
    _pool = None
    return get_pool()

def get_db():
    global _pool
    try:
        conn = get_pool().getconn()
        conn.cursor().execute("SELECT 1")
        conn.autocommit = False
        return conn
    except Exception:
        logger.debug("Pool stale — resetting (Neon cold start)")
        _reset_pool()
        conn = _pool.getconn()
        conn.autocommit = False
        return conn

def release_db(conn):
    get_pool().putconn(conn)


def init_db():
    """Create all tables if they don't exist. Safe to run on every startup."""
    try:
        conn = get_db()
        cur  = conn.cursor()

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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id   VARCHAR PRIMARY KEY,
                email     VARCHAR,
                last_seen TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                user_id VARCHAR NOT NULL,
                date    VARCHAR NOT NULL,
                count   INT DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS meal_templates (
                template_id VARCHAR PRIMARY KEY,
                user_id     VARCHAR NOT NULL,
                name        VARCHAR NOT NULL,
                created_at  TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS meal_template_items (
                item_id     VARCHAR PRIMARY KEY,
                template_id VARCHAR NOT NULL,
                user_id     VARCHAR NOT NULL,
                name        VARCHAR NOT NULL,
                nutrition   JSONB,
                servings    DOUBLE PRECISION DEFAULT 1,
                created_at  TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                sub_id            VARCHAR PRIMARY KEY,
                user_id           VARCHAR NOT NULL,
                endpoint          TEXT NOT NULL,
                subscription_json JSONB,
                created_at        TIMESTAMP,
                UNIQUE (user_id, endpoint)
            )
        """)

        conn.commit()
        cur.close()
        release_db(conn)
        logger.info("✅ PostgreSQL database initialized (all tables)")
    except Exception as e:
        logger.error(f"❌ DB init failed: {e}")
        raise


@app.on_event("startup")
def startup():
    try:
        init_db()
    except Exception as e:
        send_ntfy_alert("🔴 NutriScan Startup Failed", f"DB init error: {e}", priority=5)
        raise
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

@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    db_ok = False
    try:
        conn = get_db()
        conn.cursor().execute("SELECT 1")
        release_db(conn)
        db_ok = True
    except Exception:
        pass
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "db": "ok" if db_ok else "unavailable",
        "s3_configured": s3_client is not None,
        "gemini_configured": gemini_client is not None,
        "version": "3.2",
    }


# =============================================================================
# USAGE ENDPOINT
# =============================================================================

@app.get("/usage")
async def get_usage(authorization: Optional[str] = Header(default=None), client_date: Optional[str] = None):
    user_id = get_user_id(authorization)
    today   = client_date or date.today().isoformat()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT count FROM api_usage WHERE user_id = %s AND date = %s",
            [user_id, today]
        )
        row = cur.fetchone()
        cur.close()
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"used": row[0] if row else 0, "limit": DAILY_LIMIT, "date": today}


# =============================================================================
# EXTRACTION ENDPOINTS
# =============================================================================

@app.post("/analyze-label")
async def analyze_label(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    x_client_date: Optional[str] = Header(default=None),
):
    if not gemini_client:
        raise HTTPException(status_code=503, detail={
            "error_type": "configuration", "retryable": False,
            "message": "Gemini API not configured",
        })

    user_id, email = get_user_info(authorization)
    check_and_track(user_id, email, client_date=x_client_date)
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
        release_db(conn)
    except Exception as e:
        logger.warning(f"⚠️ DB insert failed (non-fatal): {e}")

    return result


@app.post("/analyze-labels")
async def analyze_labels(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(default=None),
    x_client_date: Optional[str] = Header(default=None),
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

    user_id, email = get_user_info(authorization)
    check_and_track(user_id, email, client_date=x_client_date)

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
        release_db(conn)
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
        release_db(conn)
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
        release_db(conn)
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
            cur.close(); release_db(conn)
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Folder not found"})
        cur.execute(
            "SELECT item_id, name, nutrition FROM folder_items WHERE folder_id = %s AND user_id = %s ORDER BY created_at DESC",
            [folder_id, user_id],
        )
        items = cur.fetchall()
        cur.close()
        release_db(conn)
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
                "nutrition": row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if row[2] is not None else {}),
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
        release_db(conn)
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
        release_db(conn)
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
        release_db(conn)
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
        release_db(conn)
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
        release_db(conn)
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
    today   = body.log_date or date.today().isoformat()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO daily_log (log_id, user_id, date, name, servings, nutrition, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [log_id, user_id, today, body.name, body.servings, json.dumps(body.nutrition), datetime.now()],
        )
        conn.commit()
        cur.close()
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    threading.Thread(target=_check_goal_and_push, args=(user_id, today), daemon=True).start()
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
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    items  = []
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fibre": 0.0}

    for row in rows:
        log_id, name, servings, nutrition_raw = row
        if nutrition_raw is None:
            continue
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


@app.get("/log/calendar")
async def get_log_calendar(
    year: int,
    month: int,
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_user_id(authorization)
    days_in_month = (date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)).day
    first = f"{year}-{str(month).zfill(2)}-01"
    last  = f"{year}-{str(month).zfill(2)}-{str(days_in_month).zfill(2)}"
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT DISTINCT date FROM daily_log WHERE user_id = %s AND date >= %s AND date <= %s ORDER BY date",
            [user_id, first, last],
        )
        rows = cur.fetchall()
        cur.close()
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    dates = [r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]) for r in rows]
    return {"dates": dates}


@app.get("/log/trends")
async def get_log_trends(
    time_range: str = Query("weekly", alias="range"),
    authorization: Optional[str] = Header(default=None),
):
    user_id    = get_user_id(authorization)
    days       = 30 if time_range == "monthly" else 7
    end_date   = date.today()
    start_date = end_date - timedelta(days=days - 1)

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "SELECT date, servings, nutrition FROM daily_log WHERE user_id = %s AND date >= %s AND date <= %s ORDER BY date, created_at",
            [user_id, start_date.isoformat(), end_date.isoformat()],
        )
        rows = cur.fetchall()
        cur.close()
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})

    daily = {
        (start_date + timedelta(i)).isoformat(): {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fibre": 0.0}
        for i in range(days)
    }

    for row in rows:
        row_date, servings, nutrition_raw = row
        date_key = row_date.isoformat() if hasattr(row_date, "isoformat") else str(row_date)
        if date_key not in daily:
            continue
        if nutrition_raw is None:
            continue
        nutrition = nutrition_raw if isinstance(nutrition_raw, dict) else json.loads(nutrition_raw)

        if nutrition.get("per_serving") and len(nutrition["per_serving"]) > 0:
            per_serving = nutrition["per_serving"]
        elif nutrition.get("per_100g") and len(nutrition["per_100g"]) > 0:
            per_serving = nutrition["per_100g"]
        else:
            per_serving = nutrition

        cal_val = _parse_num(per_serving.get("calories", 0))
        if cal_val > KJ_HEURISTIC_THRESHOLD:
            cal_val = round(cal_val / KJ_PER_KCAL, 1)

        daily[date_key]["calories"] += cal_val * servings
        daily[date_key]["protein"]  += _parse_num(per_serving.get("protein",       0)) * servings
        daily[date_key]["carbs"]    += _parse_num(per_serving.get("carbohydrates", 0)) * servings
        daily[date_key]["fat"]      += _parse_num(per_serving.get("fat",           0)) * servings
        daily[date_key]["fibre"]    += _parse_num(per_serving.get("fibre",         0)) * servings

    result = [
        {"date": d, **{k: round(v, 1) for k, v in vals.items()}}
        for d, vals in sorted(daily.items())
    ]
    return {"range": time_range, "data": result}


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
            cur.close(); release_db(conn)
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Log entry not found"})
        cur.execute(
            "UPDATE daily_log SET name=%s, servings=%s, nutrition=%s WHERE log_id=%s AND user_id=%s",
            [body.name, body.servings, json.dumps(body.nutrition), log_id, user_id],
        )
        conn.commit()
        cur.close()
        release_db(conn)
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
        release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


# =============================================================================
# MEAL TEMPLATE ENDPOINTS
# =============================================================================

@app.post("/meal-templates")
async def create_meal_template(body: MealTemplateCreate, authorization: Optional[str] = Header(default=None)):
    user_id     = get_user_id(authorization)
    template_id = str(uuid.uuid4())
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO meal_templates (template_id, user_id, name, created_at) VALUES (%s, %s, %s, %s)",
            [template_id, user_id, body.name, datetime.now()],
        )
        conn.commit(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"template_id": template_id, "name": body.name, "item_count": 0}


@app.get("/meal-templates")
async def list_meal_templates(authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            """SELECT mt.template_id, mt.name, COUNT(mti.item_id)
               FROM meal_templates mt
               LEFT JOIN meal_template_items mti ON mt.template_id = mti.template_id
               WHERE mt.user_id = %s
               GROUP BY mt.template_id, mt.name, mt.created_at
               ORDER BY mt.created_at DESC""",
            [user_id],
        )
        rows = cur.fetchall(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return [{"template_id": r[0], "name": r[1], "item_count": r[2]} for r in rows]


@app.get("/meal-templates/{template_id}")
async def get_meal_template(template_id: str, authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT template_id, name FROM meal_templates WHERE template_id = %s AND user_id = %s", [template_id, user_id])
        tmpl = cur.fetchone()
        if not tmpl:
            cur.close(); release_db(conn)
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Template not found"})
        cur.execute(
            "SELECT item_id, name, nutrition, servings FROM meal_template_items WHERE template_id = %s ORDER BY created_at",
            [template_id],
        )
        items = cur.fetchall(); cur.close(); release_db(conn)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {
        "template_id": tmpl[0],
        "name": tmpl[1],
        "items": [
            {"item_id": r[0], "name": r[1],
             "nutrition": r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {}),
             "servings": r[3]}
            for r in items
        ],
    }


@app.delete("/meal-templates/{template_id}")
async def delete_meal_template(template_id: str, authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM meal_template_items WHERE template_id = %s", [template_id])
        cur.execute("DELETE FROM meal_templates WHERE template_id = %s AND user_id = %s", [template_id, user_id])
        conn.commit(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


@app.post("/meal-templates/{template_id}/items")
async def add_meal_template_item(template_id: str, body: MealTemplateItemCreate, authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT template_id FROM meal_templates WHERE template_id = %s AND user_id = %s", [template_id, user_id])
        if not cur.fetchone():
            cur.close(); release_db(conn)
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Template not found"})
        item_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO meal_template_items (item_id, template_id, user_id, name, nutrition, servings, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [item_id, template_id, user_id, body.name, json.dumps(body.nutrition), body.servings, datetime.now()],
        )
        conn.commit(); cur.close(); release_db(conn)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"item_id": item_id, "name": body.name, "servings": body.servings}


@app.delete("/meal-templates/{template_id}/items/{item_id}")
async def delete_meal_template_item(template_id: str, item_id: str, authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM meal_template_items WHERE item_id = %s AND template_id = %s AND user_id = %s", [item_id, template_id, user_id])
        conn.commit(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"deleted": True}


@app.post("/meal-templates/{template_id}/log")
async def log_meal_template(template_id: str, log_date: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    today   = log_date or date.today().isoformat()
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT template_id FROM meal_templates WHERE template_id = %s AND user_id = %s", [template_id, user_id])
        if not cur.fetchone():
            cur.close(); release_db(conn)
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Template not found"})
        cur.execute("SELECT name, nutrition, servings FROM meal_template_items WHERE template_id = %s", [template_id])
        items = cur.fetchall()
        if not items:
            cur.close(); release_db(conn)
            return {"logged": 0}
        for name, nutrition, servings in items:
            log_id     = str(uuid.uuid4())
            nutri_json = json.dumps(nutrition) if isinstance(nutrition, dict) else (nutrition or "{}")
            cur.execute(
                "INSERT INTO daily_log (log_id, user_id, date, name, servings, nutrition, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [log_id, user_id, today, name, servings, nutri_json, datetime.now()],
            )
        conn.commit(); cur.close(); release_db(conn)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    threading.Thread(target=_check_goal_and_push, args=(user_id, today), daemon=True).start()
    return {"logged": len(items)}


# =============================================================================
# PUSH NOTIFICATION ENDPOINTS
# =============================================================================

@app.get("/push/vapid-key")
async def get_vapid_public_key():
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=503, detail={"error_type": "config_error", "message": "Push not configured."})
    return {"public_key": VAPID_PUBLIC_KEY}


@app.post("/push/subscribe")
async def push_subscribe(body: PushSubscriptionCreate, authorization: Optional[str] = Header(default=None)):
    user_id  = get_user_id(authorization)
    sub_id   = str(uuid.uuid4())
    sub_json = {"endpoint": body.endpoint, "expirationTime": body.expirationTime, "keys": body.keys}
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO push_subscriptions (sub_id, user_id, endpoint, subscription_json, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, endpoint) DO UPDATE SET subscription_json = EXCLUDED.subscription_json
        """, [sub_id, user_id, body.endpoint, json.dumps(sub_json), datetime.now()])
        conn.commit(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"subscribed": True}


@app.delete("/push/unsubscribe")
async def push_unsubscribe(authorization: Optional[str] = Header(default=None)):
    user_id = get_user_id(authorization)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM push_subscriptions WHERE user_id = %s", [user_id])
        conn.commit(); cur.close(); release_db(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_type": "db_error", "message": str(e)})
    return {"unsubscribed": True}


# =============================================================================
# AI CHAT ENDPOINT
# =============================================================================

_CHAT_SYSTEM = (
    "You are a smart, friendly nutrition assistant embedded in NutriScan. "
    "You have the user's goals, today's full food log with individual items and macros, "
    "today's totals, remaining macros, and their 7-day averages. "
    "Use plain text only — no tables, no checklists. You may use bold or bullet points sparingly. "
    "Be concise, short, and conversational. "
    "After your answer, briefly ask if they'd like help filling their remaining macros using foods "
    "they've logged before, then ask if they'd like to include something new instead."
)

@app.post("/chat")
async def chat(
    body: ChatMessage,
    authorization: Optional[str] = Header(default=None),
):
    if not groq_client:
        raise HTTPException(status_code=503, detail={"error_type": "config_error", "message": "AI assistant not configured."})

    user_id = get_user_id(authorization)

    if not _allow_chat(user_id):
        raise HTTPException(status_code=429, detail={"error_type": "rate_limited", "message": "Too many messages — wait a moment and try again."})

    # Fetch enriched context. Non-fatal if it fails.
    context = ""
    try:
        today_str  = date.today().isoformat()
        start_7    = (date.today() - timedelta(days=6)).isoformat()
        conn = get_db()
        cur  = conn.cursor()

        cur.execute("SELECT calories, protein, carbs, fat, fibre FROM user_goals WHERE user_id = %s", [user_id])
        goal_row = cur.fetchone()

        cur.execute(
            "SELECT name, servings, nutrition FROM daily_log WHERE user_id = %s AND date = %s ORDER BY created_at",
            [user_id, today_str],
        )
        log_rows = cur.fetchall()

        cur.execute(
            "SELECT date, servings, nutrition FROM daily_log WHERE user_id = %s AND date >= %s AND date <= %s ORDER BY date",
            [user_id, start_7, today_str],
        )
        trend_rows = cur.fetchall()

        cur.close()
        release_db(conn)

        def _per_serving(nutrition_raw):
            n = nutrition_raw if isinstance(nutrition_raw, dict) else json.loads(nutrition_raw)
            if n.get("per_serving") and len(n["per_serving"]) > 0:
                return n["per_serving"]
            if n.get("per_100g") and len(n["per_100g"]) > 0:
                return n["per_100g"]
            return n

        def _cal(ps, servings):
            v = _parse_num(ps.get("calories", 0)) * servings
            return round(v / KJ_PER_KCAL, 1) if v > KJ_HEURISTIC_THRESHOLD else round(v, 1)

        parts = []

        if goal_row:
            parts.append(
                f"Goals: {round(goal_row[0])}kcal / {round(goal_row[1])}g protein / "
                f"{round(goal_row[2])}g carbs / {round(goal_row[3])}g fat."
            )

        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        if log_rows:
            entries = []
            for name, servings, nutrition_raw in log_rows:
                if nutrition_raw is None:
                    continue
                ps = _per_serving(nutrition_raw)
                cal  = _cal(ps, servings)
                prot = round(_parse_num(ps.get("protein",       0)) * servings, 1)
                carb = round(_parse_num(ps.get("carbohydrates", 0)) * servings, 1)
                fat_ = round(_parse_num(ps.get("fat",           0)) * servings, 1)
                totals["calories"] += cal
                totals["protein"]  += prot
                totals["carbs"]    += carb
                totals["fat"]      += fat_
                entries.append(f"{name or 'Item'} (×{servings}s): {cal}kcal P{prot}g C{carb}g F{fat_}g")
            parts.append(f"Today's log: {'; '.join(entries)}.")

        parts.append(
            f"Today totals: {round(totals['calories'])}kcal / {round(totals['protein'])}g P / "
            f"{round(totals['carbs'])}g C / {round(totals['fat'])}g F."
        )
        if goal_row:
            parts.append(
                f"Remaining today: {round(goal_row[0] - totals['calories'])}kcal / "
                f"{round(goal_row[1] - totals['protein'])}g P / "
                f"{round(goal_row[2] - totals['carbs'])}g C / "
                f"{round(goal_row[3] - totals['fat'])}g F."
            )

        if trend_rows:
            daily_7: dict = {}
            for row_date, servings, nutrition_raw in trend_rows:
                dk = row_date.isoformat() if hasattr(row_date, "isoformat") else str(row_date)
                if dk not in daily_7:
                    daily_7[dk] = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
                if nutrition_raw is None:
                    continue
                ps = _per_serving(nutrition_raw)
                daily_7[dk]["calories"] += _cal(ps, servings)
                daily_7[dk]["protein"]  += _parse_num(ps.get("protein",       0)) * servings
                daily_7[dk]["carbs"]    += _parse_num(ps.get("carbohydrates", 0)) * servings
                daily_7[dk]["fat"]      += _parse_num(ps.get("fat",           0)) * servings
            days_with_data = [v for v in daily_7.values() if v["calories"] > 0]
            if days_with_data:
                n = len(days_with_data)
                parts.append(
                    f"7-day avg ({n} days logged): "
                    f"{round(sum(d['calories'] for d in days_with_data)/n)}kcal / "
                    f"{round(sum(d['protein']  for d in days_with_data)/n)}g P / "
                    f"{round(sum(d['carbs']    for d in days_with_data)/n)}g C / "
                    f"{round(sum(d['fat']      for d in days_with_data)/n)}g F."
                )

        context = " ".join(parts)
    except Exception:
        pass

    system = _CHAT_SYSTEM + (f"\n\nUser data: {context}" if context else "")

    api_messages = [{"role": "system", "content": system}]
    for h in body.history[-12:]:
        api_messages.append({"role": h.role, "content": h.text})
    api_messages.append({"role": "user", "content": body.message})

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=api_messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        err = str(e)
        if "429" in err:
            send_ntfy_alert("⚠️ Groq Rate Limited", "Groq returned 429 — chat temporarily unavailable")
            raise HTTPException(status_code=429, detail={"error_type": "rate_limited", "message": "Assistant is busy — try again in a moment."})
        raise HTTPException(status_code=500, detail={"error_type": "ai_error", "message": "Assistant unavailable."})


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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)