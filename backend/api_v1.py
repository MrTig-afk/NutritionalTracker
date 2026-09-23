"""NutriScan public API, /v1, plus the login-only token management routes.

Callers are scripts and chats, not browsers: personal access tokens
(`Authorization: Bearer nsk_live_...`) or a normal app login. Errors are RFC 9457
Problem Details carrying the app's usual `error_type`. Schema: api_v1.sql.

main.py imports this module LAST and then sets `m` to itself, so the helpers
below reach get_db, auth and alerts without a circular import (and without
loading a second copy of main when it runs as __main__).
"""
import hashlib
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

m = None  # the main module; set by main.py right after import

SCOPES = ("log:read", "log:write", "goals:read", "templates:read", "templates:write", "library:append")
TOKEN_PREFIX = "nsk_live_"
_TOKEN_RE = re.compile(r"^nsk_live_[A-Za-z0-9_-]{43}$")  # secrets.token_urlsafe(32) is 43 chars
MAX_TOKENS = 10
EXPIRY_DAYS = {"30d": 30, "90d": 90, "1y": 365, "never": None}
TOKEN_CACHE_SEC = 300
BURST_PER_MIN, WRITES_PER_MIN, DAILY_CAP = 20, 10, 200
BODY_CAP = 64 * 1024
MELBOURNE = ZoneInfo("Australia/Melbourne")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

router = APIRouter(prefix="/v1")
settings_router = APIRouter()


# ---------------------------------------------------------------- errors
class Problem(Exception):
    """Raise anywhere under /v1; rendered as application/problem+json."""

    def __init__(self, status: int, error_type: str, detail: str, headers: Optional[dict] = None, **extra):
        self.status, self.error_type, self.detail = status, error_type, detail
        self.headers, self.extra = headers or {}, extra


def problem_response(p: Problem) -> JSONResponse:
    body = {"type": f"urn:nutriscan:error:{p.error_type}", "title": HTTPStatus(p.status).phrase,
            "status": p.status, "detail": p.detail, "error_type": p.error_type, **p.extra}
    return JSONResponse(jsonable_encoder(body), status_code=p.status, headers=p.headers,
                        media_type="application/problem+json")


def from_http_exception(status: int, detail, headers=None) -> Problem:
    """The app's HTTPException(detail={"error_type", "message"}) in /v1's format."""
    if isinstance(detail, dict):
        return Problem(status, detail.get("error_type", "error"), detail.get("message", ""), headers)
    kind = {404: "not_found", 405: "method_not_allowed"}.get(status, "error")
    return Problem(status, kind, str(detail or HTTPStatus(status).phrase), headers)


def validation_problem(errors) -> Problem:
    """422 with the failing field paths. A batch change's index is part of its path."""
    fields = [{"field": ".".join(str(x) for x in e.get("loc", ())[1:]) or "body", "message": e.get("msg", "")}
              for e in errors]
    first = fields[0] if fields else {"field": "body", "message": "invalid request"}
    return Problem(422, "validation_error", f"{first['field']}: {first['message']}", errors=fields)


def budget_gate():
    """At 90% of the estimated free Neon month, /v1 pauses; the app keeps working."""
    if m.budget_used() >= 0.9:
        resume = m.neon_next_period_start(datetime.now(timezone.utc).date())
        retry = int(datetime(resume.year, resume.month, resume.day, tzinfo=timezone.utc).timestamp() - time.time())
        raise Problem(503, "api_paused_budget", f"The API is paused to protect the database budget until {resume.isoformat()}.",
                      headers={"Retry-After": str(max(retry, 1))}, resets_at=f"{resume.isoformat()}T00:00:00Z")


@contextmanager
def db(user_id: Optional[str]):
    """A cursor with Postgres RLS bound to `user_id` (always passed explicitly:
    a sync dependency's context does not reach a sync endpoint's thread).
    Commits when the block ends cleanly; any other error is a sanitized 500."""
    conn = None
    try:
        conn = m.get_db(user_id)
        cur = conn.cursor()
        yield cur
        conn.commit()
        cur.close()
    except (Problem, m.HTTPException):
        raise
    except Exception as e:
        raise m._db_error(e)
    finally:
        if conn:
            m.release_db(conn)


# ---------------------------------------------------------------- tokens
def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Caller:
    def __init__(self, user_id, token_id=None, name="app login", scopes=SCOPES):
        self.user_id, self.token_id, self.name, self.scopes = user_id, token_id, name, frozenset(scopes)


_token_cache: dict = {}   # token hash -> (row dict, fetched_at)
_cache_lock = threading.Lock()
_live_prefixes: Optional[set] = None   # None = not loaded yet: every well-formed token gets a lookup


def load_prefixes():
    """At startup: the prefixes of live tokens, so a guess that matches none of
    them is refused without waking Neon. Stale entries (revoked/expired) only
    cost a lookup; a new token adds its own prefix."""
    global _live_prefixes
    try:
        with db(None) as cur:
            cur.execute("SELECT * FROM api_token_prefixes()")
            _live_prefixes = {r[0] for r in cur.fetchall()}
    except Exception as e:
        m.logger.warning(f"token prefixes not loaded (run backend/api_v1.sql?): {e}")


def _lookup_token(h: str, user_agent: str) -> Optional[dict]:
    with _cache_lock:
        hit = _token_cache.get(h)
    if hit and time.time() - hit[1] < TOKEN_CACHE_SEC:
        return hit[0]
    with db(None) as cur:   # before the user is known; the function runs with owner rights
        cur.execute("SELECT token_id, user_id, name, scopes, expires_at, revoked_at FROM resolve_api_token(%s, %s)",
                    [h, (user_agent or "")[:120]])
        r = cur.fetchone()   # the block's commit keeps the last-used stamp
    row = None if not r else dict(zip(("token_id", "user_id", "name", "scopes", "expires_at", "revoked_at"), r))
    if row:
        with _cache_lock:
            if len(_token_cache) > 1000:
                _token_cache.clear()
            _token_cache[h] = (row, time.time())
    return row


def forget_token(token_id: str = None, user_id: str = None):
    """Revocation, account deletion and freezing take effect on the next call,
    not after the cache expires. For a whole user, unsaved usage counts go too,
    so a later flush cannot re-create rows for an erased account."""
    with _cache_lock:
        for h in [h for h, (row, _) in _token_cache.items()
                  if row["token_id"] == token_id or row["user_id"] == user_id]:
            del _token_cache[h]
    if user_id:
        with _limit_lock:
            for k in [k for k in _usage if k[0] == user_id]:
                del _usage[k]


def cached_user_id(authorization: str) -> Optional[str]:
    """The user behind a token that was resolved recently, without touching the DB
    (the delete-spree guard runs after the response)."""
    token = (authorization or "")[7:].strip()
    if not token.startswith(TOKEN_PREFIX):
        return None
    with _cache_lock:
        hit = _token_cache.get(token_hash(token))
    return hit[0]["user_id"] if hit else None


def resolve_caller(request: Request) -> Caller:
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise Problem(401, "unauthorized", "Send Authorization: Bearer <token>.")
    token = auth[7:].strip()
    if token.startswith(TOKEN_PREFIX):
        known = _live_prefixes is None or token[:12] in _live_prefixes
        row = _TOKEN_RE.match(token) and known and _lookup_token(token_hash(token), request.headers.get("user-agent"))
        if not row:
            ip = m._client_ip(request)
            if m._spike(f"patfail:{ip}", 20, 300):   # guessing tokens: same 10-minute box as other floods
                m._blocked[ip] = time.time() + 600
            raise Problem(401, "unauthorized", "Invalid token.")
        if row["revoked_at"]:
            raise Problem(401, "token_revoked", "This token was revoked. Make a new one in Settings.")
        exp = row["expires_at"]
        if exp and (exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)) <= datetime.now(timezone.utc):
            raise Problem(401, "token_expired", "This token has expired. Make a new one in Settings.")
        if row["user_id"] in m._frozen:
            raise Problem(423, "account_locked", "This account is locked after unusual activity.")
        return Caller(row["user_id"], row["token_id"], row["name"], row["scopes"])
    return Caller(m.get_user_id(auth))   # app login: every scope; raises 401/423 itself


# ---------------------------------------------------------------- limits
_calls: dict = defaultdict(deque)     # user_id -> deque[(ts, is_write)] within the last minute
_daily: dict = {}                     # user_id -> [melbourne date, count]
_usage: dict = defaultdict(lambda: [0, 0])  # (user_id, token_id, date) -> [reads, writes], flushed when Neon is awake
_limit_lock = threading.Lock()


def melbourne_today(now=None):
    return datetime.fromtimestamp(now or time.time(), MELBOURNE).date()


def next_melbourne_midnight(now=None) -> datetime:
    d = melbourne_today(now) + timedelta(days=1)
    return datetime(d.year, d.month, d.day, tzinfo=MELBOURNE).astimezone(timezone.utc)


def _zulu(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def take_request(caller: Caller, is_write: bool, now=None) -> dict:
    """Count one request against the caller's limits, or raise 429.
    Per user across all tokens; in memory, so checking never touches Neon.
    ponytail: in-memory like every other guard here; one Render process."""
    now = now or time.time()
    today = melbourne_today(now)
    reset = next_melbourne_midnight(now)
    with _limit_lock:
        q = _calls[caller.user_id]
        while q and q[0][0] <= now - 60:
            q.popleft()
        day = _daily.get(caller.user_id)
        if not day or day[0] != today:
            day = _daily[caller.user_id] = [today, 0]
        writes = sum(1 for _, w in q if w)
        if day[1] >= DAILY_CAP:
            limit, used, resets = DAILY_CAP, day[1], reset
        elif len(q) >= BURST_PER_MIN:
            limit, used, resets = BURST_PER_MIN, len(q), datetime.fromtimestamp(q[0][0] + 60, timezone.utc)
        elif is_write and writes >= WRITES_PER_MIN:
            limit, used, resets = WRITES_PER_MIN, writes, datetime.fromtimestamp(
                next(t for t, w in q if w) + 60, timezone.utc)
        else:
            q.append((now, is_write))
            day[1] += 1
            if caller.token_id:
                _usage[(caller.user_id, caller.token_id, today.isoformat())][1 if is_write else 0] += 1
            return {"RateLimit-Limit": str(DAILY_CAP), "RateLimit-Remaining": str(DAILY_CAP - day[1]),
                    "RateLimit-Reset": str(max(0, int(reset.timestamp() - now)))}
    retry = max(1, int(resets.timestamp() - now) + 1)
    raise Problem(429, "rate_limited", f"Over the limit of {limit}. Try again after {_zulu(resets)}.",
                  headers={"Retry-After": str(retry), "RateLimit-Limit": str(limit), "RateLimit-Remaining": "0"},
                  limit=limit, used=used, resets_at=_zulu(resets))


def requests_left(user_id: str) -> int:
    with _limit_lock:
        day = _daily.get(user_id)
    return DAILY_CAP - (day[1] if day and day[0] == melbourne_today() else 0)


def flush_usage():
    """Save per-token daily counts. Called only while Neon is already awake.
    Each row is written as its own user, so RLS (user_isolation) admits it."""
    with _limit_lock:
        pending = dict(_usage)
        _usage.clear()
    for (user_id, token_id, day), (reads, writes) in pending.items():
        try:
            with db(user_id) as cur:
                cur.execute("""
                    INSERT INTO api_token_usage (token_id, date, user_id, reads, writes) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (token_id, date) DO UPDATE
                    SET reads = api_token_usage.reads + EXCLUDED.reads, writes = api_token_usage.writes + EXCLUDED.writes
                """, [token_id, day, user_id, reads, writes])
        except Exception as e:
            m.logger.warning(f"api usage flush skipped: {e}")
            with _limit_lock:   # keep the counts for the next awake moment
                _usage[(user_id, token_id, day)][0] += reads
                _usage[(user_id, token_id, day)][1] += writes


# ---------------------------------------------------------------- dependencies
def need(*scopes: str, write: bool = False):
    """Route dependency: resolve the caller, check scopes, count the request."""
    def dep(request: Request) -> Caller:
        budget_gate()
        caller = resolve_caller(request)
        missing = [s for s in scopes if s not in caller.scopes]
        if missing:
            raise Problem(403, "insufficient_scope", f"This token lacks {', '.join(missing)}.")
        request.state.ratelimit = take_request(caller, write)
        request.state.caller = caller
        return caller
    return dep


async def v1_middleware(request: Request, call_next):
    """Body cap, JSON-only, and the headers every /v1 answer carries."""
    if not request.url.path.startswith("/v1/"):
        return await call_next(request)
    try:
        if request.method in ("POST", "PATCH", "PUT", "DELETE"):
            declared = request.headers.get("content-length")
            if declared is None and request.headers.get("transfer-encoding"):
                # a chunked body would be read whole before its size is known
                raise Problem(411, "length_required", "Send a Content-Length header.")
            if declared and declared.isdigit() and int(declared) > BODY_CAP:
                raise Problem(413, "payload_too_large", "Body over 64 KB.")
            body = await request.body()
            if len(body) > BODY_CAP:
                raise Problem(413, "payload_too_large", "Body over 64 KB.")
            ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if body and ctype != "application/json":
                raise Problem(415, "unsupported_media_type", "Send JSON with Content-Type: application/json.")
        response = await call_next(request)
    except Problem as p:
        response = problem_response(p)
    for k, v in (getattr(request.state, "ratelimit", None) or {}).items():
        response.headers.setdefault(k, v)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    if getattr(request.state, "caller", None) and request.state.caller.token_id and m.db_awake():
        threading.Thread(target=flush_usage, daemon=True).start()
    return response


# ---------------------------------------------------------------- /v1 routes
@router.get("/me")
def me(caller: Caller = Depends(need())):
    return {"token_name": caller.name, "scopes": sorted(caller.scopes),
            "requests_left_today": requests_left(caller.user_id),
            "resets_at": _zulu(next_melbourne_midnight())}


# ---------------------------------------------------------------- token management (app login only)
class TokenCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=40)
    scopes: list[Literal[SCOPES]] = Field(min_length=1, max_length=len(SCOPES))
    expires: Literal[tuple(EXPIRY_DAYS)]

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v):
        v = _CONTROL.sub("", v).strip()
        if not v:
            raise ValueError("name is empty")
        return v


def _admin_login(authorization: Optional[str]) -> str:
    """Token management accepts only a Supabase login - a PAT fails JWT
    verification here - so a leaked token can never mint another. Admin-only for now."""
    user_id = m.get_user_id(authorization)
    if not m.ADMIN_USER_ID or user_id != m.ADMIN_USER_ID:
        raise m.HTTPException(status_code=403, detail={"error_type": "feature_unavailable",
                              "message": "API tokens are not available on this account yet."})
    return user_id


def _token_view(r) -> dict:
    token_id, name, prefix, scopes, created_at, expires_at, revoked_at, last_used_at, last_user_agent = r
    return {"token_id": token_id, "name": name, "prefix": prefix, "scopes": list(scopes),
            "created_at": created_at, "expires_at": expires_at, "revoked_at": revoked_at,
            "last_used_at": last_used_at, "last_used_app": last_user_agent}


_TOKEN_COLS = "token_id, name, prefix, scopes, created_at, expires_at, revoked_at, last_used_at, last_user_agent"


@settings_router.post("/settings/api-tokens", status_code=201)
def create_token(body: TokenCreate, authorization: Optional[str] = Header(default=None)):
    user_id = _admin_login(authorization)
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    days = EXPIRY_DAYS[body.expires]
    expires_at = datetime.now(timezone.utc) + timedelta(days=days) if days else None
    with db(user_id) as cur:
        cur.execute("""SELECT count(*) FROM api_tokens WHERE user_id = %s AND revoked_at IS NULL
                       AND (expires_at IS NULL OR expires_at > now())""", [user_id])
        if cur.fetchone()[0] >= MAX_TOKENS:
            raise m.HTTPException(status_code=409, detail={"error_type": "token_limit",
                                  "message": f"You already have {MAX_TOKENS} active tokens. Revoke one first."})
        cur.execute(f"""INSERT INTO api_tokens (token_id, user_id, name, prefix, token_hash, scopes, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {_TOKEN_COLS}""",
                    [str(uuid.uuid4()), user_id, body.name, token[:12], token_hash(token), sorted(set(body.scopes)), expires_at])
        row = cur.fetchone()
    if _live_prefixes is not None:
        _live_prefixes.add(token[:12])
    m.send_push_to_user(user_id, "New API token", f"New API token '{body.name}' created. If you didn't make it, revoke it in Settings.")
    return {**_token_view(row), "token": token}


@settings_router.get("/settings/api-tokens")
def list_tokens(authorization: Optional[str] = Header(default=None)):
    user_id = _admin_login(authorization)
    with db(user_id) as cur:
        cur.execute(f"SELECT {_TOKEN_COLS} FROM api_tokens WHERE user_id = %s AND revoked_at IS NULL ORDER BY created_at DESC",
                    [user_id])
        return [_token_view(r) for r in cur.fetchall()]


@settings_router.delete("/settings/api-tokens/{token_id}")
def revoke_token(token_id: str, authorization: Optional[str] = Header(default=None)):
    user_id = _admin_login(authorization)
    with db(user_id) as cur:
        cur.execute("UPDATE api_tokens SET revoked_at = now() WHERE token_id = %s AND user_id = %s AND revoked_at IS NULL",
                    [token_id, user_id])
        done = cur.rowcount
    forget_token(token_id)
    if not done:
        raise m.HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Token not found"})
    return {"revoked": True}


def revoke_all(user_id: str):
    """Freezing an account also revokes its API tokens: the freeze's durable half
    is a Supabase ban, which a token never consults."""
    try:
        with db(user_id) as cur:
            cur.execute("UPDATE api_tokens SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL", [user_id])
    except Exception as e:
        m.logger.warning(f"token revoke on freeze failed for {user_id[:8]}: {e}")
    forget_token(user_id=user_id)   # after the commit, or a request in between re-caches the live row
