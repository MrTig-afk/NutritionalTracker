"""NutriScan public API, /v1, plus the login-only token management routes.

Callers are scripts and chats, not browsers: personal access tokens
(`Authorization: Bearer nsk_live_...`) or a normal app login. Errors are RFC 9457
Problem Details carrying the app's usual `error_type`. Schema: api_v1.sql.

main.py imports this module LAST and then sets `m` to itself, so the helpers
below reach get_db, auth and alerts without a circular import (and without
loading a second copy of main when it runs as __main__).
"""
import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from typing import Annotated, Literal, Optional, Union
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

import log_service

m = None  # the main module; set by main.py right after import

SCOPES = ("log:read", "log:write", "goals:read", "templates:read", "templates:write", "library:append", "library:read")
TOKEN_PREFIX = "nsk_live_"
_TOKEN_RE = re.compile(r"^nsk_live_[A-Za-z0-9_-]{43}$")  # secrets.token_urlsafe(32) is 43 chars
MAX_TOKENS = 10
EXPIRY_DAYS = {"30d": 30, "90d": 90, "1y": 365, "never": None}
TOKEN_CACHE_SEC = 300
BURST_PER_MIN, WRITES_PER_MIN, DAILY_CAP = 20, 10, 200
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
def db(user_id: Optional[str], commit: bool = True):
    """A cursor with Postgres RLS bound to `user_id` (always passed explicitly:
    a sync dependency's context does not reach a sync endpoint's thread).
    Commits when the block ends cleanly (unless commit=False: a preview, which
    release_db rolls back); any other error is a sanitized 500."""
    conn = None
    try:
        conn = m.get_db(user_id)
        cur = conn.cursor()
        yield cur
        if commit:
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


# Connected apps seen by /mcp: (user_id, client_id) -> [active, last touch]. In memory like the token cache:
# Render runs one process. Disconnect, allow and account deletion drop entries so the next call re-reads the row.
_apps: dict = {}
_apps_lock = threading.Lock()
_apps_gen = 0   # bumped by every forget: a gate lookup that overlapped one must not cache what it read


def forget_app(user_id: str, client_id: Optional[str] = None):
    global _apps_gen
    with _apps_lock:
        _apps_gen += 1
        for k in [k for k in _apps if k[0] == user_id and client_id in (None, k[1])]:
            del _apps[k]


APP_TOUCH_SECS, APPS_MAX = 60, 1000
CONNECT_PUSH = ("Claude connected to NutriScan", "Not you? Disconnect it in Settings.")


def app_known_disconnected(user_id: str, client_id: str) -> bool:
    with _apps_lock:
        hit = _apps.get((user_id, client_id))
    return bool(hit and not hit[0])


def connected_app_gate(user_id: str, client_id: str) -> None:
    """H7: a disconnected app gets the 401 challenge (claude.ai shows its own reconnect prompt). An unknown one is
    adopted with the Q17 push: that is how a connection made before this table existed gets its row."""
    key, now = (user_id, client_id), time.time()
    with _apps_lock:
        hit, gen = _apps.get(key), _apps_gen
    if hit and not hit[0]:
        raise _mcp_challenge()
    # Never wake a paused Neon for a lookup or a stamp: while paused, budget_gate refuses every tool call anyway.
    if (hit and now - hit[1] < APP_TOUCH_SECS) or m.budget_used() >= 0.9:
        return
    new = revoked = False
    with db(user_id) as cur:
        if not hit:
            cur.execute("SELECT revoked_at FROM connected_apps WHERE user_id = %s AND client_id = %s", key)
            row = cur.fetchone()
            revoked = bool(row and row[0] is not None)
            if row is None:
                cur.execute("INSERT INTO connected_apps (user_id, client_id) VALUES (%s, %s) "
                            "ON CONFLICT (user_id, client_id) DO NOTHING RETURNING user_id", key)
                new = cur.fetchone() is not None   # a racing first call inserted nothing: one push, not two
        if not revoked:
            cur.execute("UPDATE connected_apps SET last_used_at = now() "
                        "WHERE user_id = %s AND client_id = %s AND revoked_at IS NULL RETURNING 1", key)
            # No live row behind a cached "active": another instance disconnected it (Render overlaps the old and
            # new instance during a deploy), so this cache is stale.
            revoked = bool(hit) and cur.fetchone() is None
    with _apps_lock:
        if _apps_gen == gen:   # a disconnect or allow landed meanwhile: leave it uncached, the next call re-reads
            if len(_apps) >= APPS_MAX:
                _apps.clear()   # like the token cache: a full cache just re-reads rows
            _apps[key] = [not revoked, now]
    if revoked:
        raise _mcp_challenge()
    if new:
        m.send_push_to_user(user_id, *CONNECT_PUSH)


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
        forget_app(user_id)


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
        pending = row["user_id"] in m._deleting   # read before _purged, as main._account_gate does
        if row["user_id"] in m._purged:     # a token cached before the purge
            raise Problem(401, "account_deleted", "This account was deleted.")
        if pending:   # an API token reads nothing inside the 15 days (main._account_gate)
            raise Problem(423, "account_scheduled_for_deletion", "This account is being deleted.")
        return Caller(row["user_id"], row["token_id"], row["name"], row["scopes"])
    return Caller(m.get_user_id(auth, allow_client=True))   # app login: every scope; raises 401/423 itself


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
    In memory like every other guard here, which holds while Render runs one process."""
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
def _require(caller: Caller, *scopes: str):
    missing = [s for s in scopes if s not in caller.scopes]
    if missing:
        raise Problem(403, "insufficient_scope", f"This token lacks {', '.join(missing)}.")


def need(*scopes: str, write: bool = False):
    """Route dependency: resolve the caller, check scopes, count the request."""
    def dep(request: Request) -> Caller:
        budget_gate()
        caller = resolve_caller(request)
        _require(caller, *scopes)
        request.state.ratelimit = take_request(caller, write)
        request.state.caller = caller
        return caller
    return dep


async def v1_middleware(request: Request, call_next):
    """Body cap, JSON-only, and the headers every /v1 answer carries."""
    if not request.url.path.startswith("/v1/"):
        response = await call_next(request)
        if (request.method not in ("GET", "HEAD", "OPTIONS") and response.status_code < 400
                and request.url.path != "/mcp"):   # an MCP POST is a read: keep the cache get_context filled
            # a write through the app: the /v1 read cache must not serve the old numbers
            claims = m.claims_if_valid(request.headers.get("authorization", ""))
            invalidate(claims and claims.get("sub"))
        return response
    try:
        if request.method in ("POST", "PATCH", "PUT", "DELETE"):
            declared = request.headers.get("content-length")
            if declared is None and request.headers.get("transfer-encoding"):
                # a chunked body would be read whole before its size is known
                raise Problem(411, "length_required", "Send a Content-Length header.")
            if declared and declared.isdigit() and int(declared) > m.APP_BODY_CAP:
                raise Problem(413, "payload_too_large", "Body over 64 KB.")
            body = await request.body()
            if len(body) > m.APP_BODY_CAP:
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


# ---------------------------------------------------------------- reads (Phase 2)
# Everything here filters by caller.user_id AND runs
# with app.user_id bound to it (RLS), so another user's id is a plain 404.

MAX_RANGE_DAYS, MAX_ENTRY_DAYS, RESPONSE_CAP = 31, 7, 32 * 1024
_read_cache: dict = defaultdict(dict)   # user_id -> {request key: (body, built_at)}
_read_lock = threading.Lock()
# Bounds: a changed-outside-the-app row (an SQL fix, a restore) shows within
# READ_TTL_SEC, and a poller cannot grow one user's cache past READ_MAX_KEYS.
READ_TTL_SEC, READ_MAX_KEYS = 30 * 60, 40


_read_gen: dict = defaultdict(int)       # user_id -> bumped by every invalidate


def invalidate(user_id: Optional[str]):
    """Any write by this user, through /v1 or the app, drops their cached reads."""
    if user_id:
        with _read_lock:
            _read_cache.pop(user_id, None)
            _read_gen[user_id] += 1


def cached_read(caller: Caller, key: str, build):
    """The body for `key`, built at most once per write. In memory: fine while Render runs one process."""
    with _read_lock:
        hit = _read_cache[caller.user_id].get(key)
        gen = _read_gen[caller.user_id]
    if hit is not None and time.time() - hit[1] < READ_TTL_SEC:
        return hit[0]
    body = build()
    with _read_lock:
        if _read_gen[caller.user_id] == gen:   # a write landed while building: serve this once, keep nothing
            mine = _read_cache[caller.user_id]
            if len(mine) >= READ_MAX_KEYS:
                mine.clear()
            mine[key] = (body, time.time())
    return body


def etag_of(*parts) -> str:
    return '"' + hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16] + '"'


def _day(s: str, field: str):
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        raise Problem(422, "validation_error", f"{field}: use YYYY-MM-DD", errors=[{"field": field, "message": "use YYYY-MM-DD"}])


def _macros5(m5: dict) -> dict:
    return {"calories": round(m5["calories"], 1), "protein_g": round(m5["protein"], 1), "carbs_g": round(m5["carbs"], 1),
            "fat_g": round(m5["fat"], 1), "fibre_g": round(m5["fibre"], 1)}


def compact_entry(log_id, name, servings, nutrition, entry_date) -> dict:
    n = log_service.load_nutrition(nutrition)
    ps = log_service.per_serving_section(n)
    return {"log_id": log_id, "name": name, "portion": ps.get("size"), "servings": servings,
            **_macros5(log_service.entry_macros(n, servings)), "source": n.get("_source", "app"),
            "etag": etag_of(name, servings, n, str(entry_date))}


def _full_macros(n: dict, servings) -> dict:
    """The detail view: the five main macros plus sugars, saturated fat and sodium."""
    ps = log_service.per_serving_section(n)
    s = float(servings or 0)
    sodium = ps.get("sodium")
    sodium_mg = log_service._parse_num(sodium) * (1000 if isinstance(sodium, str) and "mg" not in sodium.lower() else 1)
    return {**_macros5(log_service.entry_macros(n, s)),
            "sugars_g": round(log_service._parse_num(ps.get("sugars")) * s, 1),
            "sat_fat_g": round(log_service._parse_num(ps.get("saturated_fat")) * s, 1),
            "sodium_mg": round(sodium_mg * s, 1)}


def _rows(caller: Caller, sql: str, params: list):
    with db(caller.user_id) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


GOALS_SQL = "SELECT calories, protein, carbs, fat, fibre FROM user_goals WHERE user_id = %s"


def goals_from(rows) -> dict:
    g = rows[0] if rows else (2000.0, 150.0, 250.0, 65.0, 30.0)   # the app's defaults for a user who never set goals
    return {"calories": g[0], "protein_g": g[1], "carbs_g": g[2], "fat_g": g[3], "fibre_g": g[4] or 0.0}


def _goals(caller: Caller) -> dict:
    return goals_from(_rows(caller, GOALS_SQL, [caller.user_id]))


def _meals(entries_rows) -> tuple:
    """Rows (log_id, name, servings, nutrition, date) -> (meals, by_label, totals).
    One block per logged meal (its _meal_group); ungrouped rows share a block per
    label ("Other" when unlabelled). Order: first appearance."""
    blocks, index, totals = [], {}, dict.fromkeys(log_service.MACRO_KEYS, 0.0)
    by_label: dict = {}
    for log_id, name, servings, nutrition, d in entries_rows:
        n = log_service.load_nutrition(nutrition)
        label = n.get("_meal_label") or "Other"
        gid = n.get("_meal_group")
        key = ("g", gid) if gid else ("l", label)
        if key not in index:
            index[key] = len(blocks)
            blocks.append({"label": label, "group_id": gid, "subtotal": dict.fromkeys(log_service.MACRO_KEYS, 0.0), "entries": []})
        b = blocks[index[key]]
        m5 = log_service.entry_macros(n, servings)
        for k in log_service.MACRO_KEYS:
            b["subtotal"][k] += m5[k]
            totals[k] += m5[k]
            by_label.setdefault(label, dict.fromkeys(log_service.MACRO_KEYS, 0.0))[k] += m5[k]
        b["entries"].append(compact_entry(log_id, name, servings, n, d))
    for b in blocks:
        b["subtotal"] = _macros5(b["subtotal"])
        b["etag"] = etag_of([e["etag"] for e in b["entries"]])   # If-Match for deleting the whole meal
    return blocks, {k: _macros5(v) for k, v in by_label.items()}, totals


def _remaining(goals: dict, totals: dict) -> dict:
    t = _macros5(totals)
    return {k: round(goals[k] - t[k], 1) for k in t}


def _day_rows(caller: Caller, start: date, end: date):
    return _rows(caller, """SELECT log_id, name, servings, nutrition, date FROM daily_log
                            WHERE user_id = %s AND date >= %s AND date <= %s ORDER BY date, created_at""",
                 [caller.user_id, start.isoformat(), end.isoformat()])


def _templates(caller: Caller) -> list:
    rows = _rows(caller, """SELECT t.template_id, t.name, i.item_id, i.name, i.servings, i.nutrition
                            FROM meal_templates t LEFT JOIN meal_template_items i
                              ON i.template_id = t.template_id AND i.user_id = t.user_id
                            WHERE t.user_id = %s ORDER BY t.created_at, i.created_at""", [caller.user_id])
    out, index = [], {}
    for tid, tname, iid, iname, servings, nutrition in rows:
        if tid not in index:
            index[tid] = len(out)
            out.append({"template_id": tid, "name": tname, "totals": dict.fromkeys(log_service.MACRO_KEYS, 0.0), "items": []})
        if iid is None:
            continue
        t = out[index[tid]]
        n = log_service.load_nutrition(nutrition)
        m5 = log_service.entry_macros(n, servings)
        for k in log_service.MACRO_KEYS:
            t["totals"][k] += m5[k]
        t["items"].append({"item_id": iid, "name": iname, "portion": log_service.per_serving_section(n).get("size"),
                           "servings": servings, **_macros5(m5)})
    for t in out:
        t["totals"] = _macros5(t["totals"])
    return out


def _usual_foods(caller: Caller, today: date) -> list:
    """The 20 most recently logged distinct foods (by name), per 1 serving as logged."""
    rows = _rows(caller, """SELECT name, servings, nutrition FROM (
                              SELECT DISTINCT ON (lower(name)) name, servings, nutrition, created_at FROM daily_log
                              WHERE user_id = %s AND date >= %s ORDER BY lower(name), created_at DESC) u
                            ORDER BY created_at DESC LIMIT 20""",
                 [caller.user_id, (today - timedelta(days=90)).isoformat()])
    out = []
    for name, servings, nutrition in rows:
        n = log_service.load_nutrition(nutrition)
        out.append({"name": name, "portion": log_service.per_serving_section(n).get("size"),
                    **_macros5(log_service.entry_macros(n, 1))})
    return out


def _trend_days(rows, start: date, end: date) -> tuple:
    """Per-day totals over [start, end] plus averages over the days that have entries."""
    days = {(start + timedelta(i)).isoformat(): dict.fromkeys(log_service.MACRO_KEYS, 0.0)
            for i in range((end - start).days + 1)}
    logged = set()
    for _, _, servings, nutrition, d in rows:
        k = str(d)
        if k not in days:
            continue
        logged.add(k)
        for mk, v in log_service.entry_macros(nutrition, servings).items():
            days[k][mk] += v
    per_day = [{"date": d, **_macros5(v)} for d, v in days.items()]
    n = max(len(logged), 1)
    avg = _macros5({k: sum(days[d][k] for d in logged) / n for k in log_service.MACRO_KEYS})
    return per_day, {**avg, "days_logged": len(logged)}


def _capped(body: dict) -> dict:
    if len(json.dumps(body, default=str)) > RESPONSE_CAP:
        raise Problem(422, "validation_error", "That answer would be over 32 KB. Ask for a narrower range.",
                      errors=[{"field": "range", "message": "response over 32 KB"}])
    return body


@router.get("/context")
def context(date_: str = Query(alias="date"), include: str = "",
            caller: Caller = Depends(need("log:read", "goals:read", "templates:read"))):
    """The one call a conversation starts with."""
    day = _day(date_, "date")

    def build():
        goals = _goals(caller)
        want_trends = "trends" in include.split(",")
        rows = _day_rows(caller, day - timedelta(days=6) if want_trends else day, day)
        today_rows = [r for r in rows if str(r[4]) == day.isoformat()]
        meals, by_label, totals = _meals(today_rows)
        body = {"date": day.isoformat(), "goals": goals, "totals": _macros5(totals),
                "remaining": _remaining(goals, totals), "meals": meals, "by_label": by_label,
                "templates": _templates(caller), "usual_foods": _usual_foods(caller, day)}
        if want_trends:
            per_day, avg = _trend_days(rows, day - timedelta(days=6), day)
            body["trends"] = {"days": per_day, "average": avg}
        return _capped(body)
    return cached_read(caller, f"context:{day}:{'trends' if 'trends' in include.split(',') else ''}", build)


@router.get("/days")
def days(from_: str = Query(alias="from"), to: str = Query(...), include: str = "",
         limit: int = Query(MAX_RANGE_DAYS, ge=1, le=MAX_RANGE_DAYS), cursor: Optional[str] = None,
         caller: Caller = Depends(need("log:read"))):
    start, end = _day(cursor or from_, "cursor" if cursor else "from"), _day(to, "to")
    if end < start:
        raise Problem(422, "validation_error", "to: must not be before from", errors=[{"field": "to", "message": "before from"}])
    with_entries = "entries" in include.split(",")
    span = min(limit, MAX_ENTRY_DAYS if with_entries else MAX_RANGE_DAYS)
    if (end - start).days + 1 > MAX_RANGE_DAYS and not cursor:
        raise Problem(422, "validation_error", f"range: at most {MAX_RANGE_DAYS} days",
                      errors=[{"field": "range", "message": f"at most {MAX_RANGE_DAYS} days"}])
    page_end = min(end, start + timedelta(days=span - 1))

    def build():
        rows = _day_rows(caller, start, page_end)
        per_day, avg = _trend_days(rows, start, page_end)
        if with_entries:
            by_date = defaultdict(list)
            for r in rows:
                by_date[str(r[4])].append(compact_entry(*r))
            for d in per_day:
                d["entries"] = by_date.get(d["date"], [])
        nxt = page_end + timedelta(days=1)
        return _capped({"from": start.isoformat(), "to": page_end.isoformat(), "days": per_day, "average": avg,
                        "next_cursor": nxt.isoformat() if nxt <= end else None})
    return cached_read(caller, f"days:{start}:{page_end}:{with_entries}", build)


def _not_found(what: str) -> Problem:
    return Problem(404, "not_found", f"{what} not found.")   # also for someone else's id: never 403


def entry_detail(log_id, name, servings, nutrition, entry_date) -> dict:
    n = log_service.load_nutrition(nutrition)
    ps = log_service.per_serving_section(n)
    return {"log_id": log_id, "date": str(entry_date), "name": name, "portion": ps.get("size"), "servings": servings,
            "macros": _full_macros(n, servings), "per_100g": n.get("per_100g") or None,
            "group_id": n.get("_meal_group"), "label": n.get("_meal_label"), "source": n.get("_source", "app"),
            "etag": etag_of(name, servings, n, str(entry_date))}


def _etagged(body: dict) -> JSONResponse:
    return JSONResponse(jsonable_encoder(body), headers={"ETag": body["etag"]})


@router.get("/entries/{log_id}")
def get_entry(log_id: str, caller: Caller = Depends(need("log:read"))):
    def build():
        r = _rows(caller, "SELECT log_id, name, servings, nutrition, date FROM daily_log WHERE log_id = %s AND user_id = %s",
                  [log_id, caller.user_id])
        if not r:
            raise _not_found("Entry")
        return entry_detail(*r[0])
    return _etagged(cached_read(caller, f"entry:{log_id}", build))


@router.get("/meals/{group_id}")
def get_meal(group_id: str, caller: Caller = Depends(need("log:read"))):
    def build():
        rows = _rows(caller, """SELECT log_id, name, servings, nutrition, date FROM daily_log
                                WHERE user_id = %s AND nutrition->>'_meal_group' = %s ORDER BY created_at""",
                     [caller.user_id, group_id])
        if not rows:
            raise _not_found("Meal")
        items = [entry_detail(*r) for r in rows]
        totals = dict.fromkeys(log_service.MACRO_KEYS, 0.0)
        for r in rows:
            for k, v in log_service.entry_macros(r[3], r[2]).items():
                totals[k] += v
        return {"group_id": group_id, "label": items[0]["label"], "date": items[0]["date"],
                "totals": _macros5(totals), "items": items, "etag": etag_of([i["etag"] for i in items])}
    return _etagged(cached_read(caller, f"meal:{group_id}", build))


@router.get("/goals")
def get_goals(caller: Caller = Depends(need("goals:read"))):
    return cached_read(caller, "goals", lambda: _goals(caller))


# ---------------------------------------------------------------- the Library (one for the app and Claude)
LIBRARY_LIMIT = 50
ASK_SQL = "SELECT prefs->>'ask_before_saving_foods' FROM notification_prefs WHERE user_id = %s"
PREFS_MERGE_SQL = """INSERT INTO notification_prefs (user_id, prefs, updated_at) VALUES (%s, %s, now())
    ON CONFLICT (user_id) DO UPDATE SET prefs = COALESCE(notification_prefs.prefs, '{}'::jsonb) || EXCLUDED.prefs,
        updated_at = EXCLUDED.updated_at"""


def ask_before_saving(rows) -> bool:
    """"Ask before saving new foods to my Library": on unless the user turned it off (PRD change 2026-09-25)."""
    return not (rows and rows[0][0] == "false")


def library(caller: Caller, q: str) -> dict:
    """Library foods whose name contains q (literally, any case), with per-serving numbers, and the setting."""
    q = _CONTROL.sub("", q).strip()
    like = "%" + re.sub(r"([\\%_])", r"\\\1", q) + "%"
    with db(caller.user_id, commit=False) as cur:
        cur.execute(f"""SELECT i.item_id, f.name, i.name, i.nutrition FROM folder_items i
                        JOIN folders f ON f.folder_id = i.folder_id AND f.user_id = i.user_id
                        WHERE i.user_id = %s AND i.name ILIKE %s ORDER BY lower(i.name) LIMIT {LIBRARY_LIMIT + 1}""",
                    [caller.user_id, like])
        rows = cur.fetchall()
        cur.execute(ASK_SQL, [caller.user_id])
        ask = ask_before_saving(cur.fetchall())
    items = []
    for item_id, folder, name, raw in rows[:LIBRARY_LIMIT]:
        n = log_service.load_nutrition(raw)
        ps = log_service.per_serving_section(n)   # the same section entry_macros reads the numbers from
        items.append({"item_id": item_id, "name": name, "folder": folder,
                      "portion": str(ps.get("size") or ("100 g" if ps is n.get("per_100g") else "")),
                      "per_serving": _macros5(log_service.entry_macros(n, 1))})
    return {"items": items, "truncated": len(rows) > LIBRARY_LIMIT, "ask_before_saving": ask}


@router.get("/library")
def get_library(q: str = Query("", max_length=80), caller: Caller = Depends(need("library:read"))):
    return library(caller, q)


def template_detail(cur, user_id: str, template_id: str) -> Optional[dict]:
    cur.execute("SELECT name FROM meal_templates WHERE template_id = %s AND user_id = %s", [template_id, user_id])
    t = cur.fetchone()
    if not t:
        return None
    cur.execute("""SELECT item_id, name, servings, nutrition FROM meal_template_items
                   WHERE template_id = %s AND user_id = %s ORDER BY created_at""", [template_id, user_id])
    items = cur.fetchall()
    out_items, totals = [], dict.fromkeys(log_service.MACRO_KEYS, 0.0)
    for iid, name, servings, nutrition in items:
        n = log_service.load_nutrition(nutrition)
        for k, v in log_service.entry_macros(n, servings).items():
            totals[k] += v
        out_items.append({"item_id": iid, "name": name, "portion": log_service.per_serving_section(n).get("size"),
                          "servings": servings, "macros": _full_macros(n, servings)})
    return {"template_id": template_id, "name": t[0], "totals": _macros5(totals), "items": out_items,
            "etag": etag_of(t[0], [(i[1], i[2], log_service.load_nutrition(i[3])) for i in items])}


def template_body(template_id: str, caller: Caller) -> dict:
    def build():
        with db(caller.user_id) as cur:
            t = template_detail(cur, caller.user_id, template_id)
        if not t:
            raise _not_found("Template")
        return t
    return cached_read(caller, f"template:{template_id}", build)


@router.get("/templates/{template_id}")
def get_template(template_id: str, caller: Caller = Depends(need("templates:read"))):
    return _etagged(template_body(template_id, caller))


# ---------------------------------------------------------------- writes
# One engine behind every write route: a list of changes applied in ONE
# transaction, so a failing change rolls back all of them. A preview runs the
# very same writes and then rolls back, so what the chat shows is exactly what
# the commit will do. Nothing a chat sends is trusted: every field is typed,
# bounded, control characters are stripped, and unknown fields are rejected.
MAX_ITEMS, MAX_BATCH, MAX_KCAL = 25, 10, 5000
IDEMPOTENCY_SEC = 24 * 3600
LIBRARY_FOLDER = "From Claude"


def _text(v: str) -> str:
    v = _CONTROL.sub("", v).strip()
    if not v:
        raise ValueError("empty once control characters are removed")
    return v


def _log_date(v: str) -> str:
    d = date.fromisoformat(v)
    # Melbourne runs ahead of UTC, so "today" there can be UTC tomorrow.
    if d > datetime.now(timezone.utc).date() + timedelta(days=1):
        raise ValueError("date is in the future")
    return d.isoformat()


Name = Annotated[str, Field(min_length=1, max_length=80), AfterValidator(_text)]
Label = Annotated[str, Field(min_length=1, max_length=40), AfterValidator(_text)]
MealLabel = Annotated[str, Field(min_length=1, max_length=80), AfterValidator(_text)]   # a logged meal's name
Portion = Annotated[str, Field(min_length=1, max_length=40), AfterValidator(_text)]
Servings = Annotated[float, Field(gt=0, le=100)]
LogDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(_log_date)]
Id = Annotated[str, Field(min_length=1, max_length=64)]
ETag = Annotated[str, Field(min_length=1, max_length=64)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Macros(Strict):
    calories: float = Field(ge=0, le=MAX_KCAL)
    protein_g: float = Field(ge=0, le=1000)
    carbs_g: float = Field(ge=0, le=1000)
    fat_g: float = Field(ge=0, le=1000)
    fibre_g: float = Field(ge=0, le=1000)
    sugars_g: Optional[float] = Field(None, ge=0, le=1000)
    sat_fat_g: Optional[float] = Field(None, ge=0, le=1000)
    sodium_mg: Optional[float] = Field(None, ge=0, le=100000)


class Item(Strict):
    name: Name
    portion: Optional[Portion] = None
    servings: Servings = 1
    macros: Macros
    save_to_library: bool = False


class LogEntryChange(Item):
    type: Literal["log_entry"] = "log_entry"
    date: LogDate


class LogMealChange(Strict):
    type: Literal["log_meal"] = "log_meal"
    date: LogDate
    label: MealLabel
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEMS)


class ItemChange(Strict):
    item_id: Id
    portion: Optional[Portion] = None
    servings: Optional[Servings] = None
    macros: Optional[Macros] = None


class TemplateLog(Strict):
    date: LogDate
    changes: list[ItemChange] = Field(default_factory=list, max_length=MAX_ITEMS)
    add: list[Item] = Field(default_factory=list, max_length=MAX_ITEMS)
    remove: list[Id] = Field(default_factory=list, max_length=MAX_ITEMS)


class LogTemplateChange(TemplateLog):
    type: Literal["log_template"] = "log_template"
    template_id: Id


class EntryPatch(Strict):
    name: Optional[Name] = None
    portion: Optional[Portion] = None
    servings: Optional[Servings] = None
    macros: Optional[Macros] = None
    date: Optional[LogDate] = None


class UpdateEntryChange(EntryPatch):
    type: Literal["update_entry"] = "update_entry"
    log_id: Id
    if_match: ETag


class DeleteEntryChange(Strict):
    type: Literal["delete_entry"] = "delete_entry"
    log_id: Id
    if_match: ETag


class DeleteMealChange(Strict):
    type: Literal["delete_meal"] = "delete_meal"
    group_id: Id
    if_match: ETag


class CreateTemplateChange(Strict):
    type: Literal["create_template"] = "create_template"
    name: Label
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEMS)


class TemplatePatch(Strict):
    name: Optional[Label] = None
    items: Optional[list[Item]] = Field(None, min_length=1, max_length=MAX_ITEMS)


class UpdateTemplateChange(TemplatePatch):
    type: Literal["update_template"] = "update_template"
    template_id: Id
    if_match: ETag


class SaveFoodChange(Strict):
    type: Literal["save_food"] = "save_food"
    name: Name
    portion: Optional[Portion] = None
    macros: Macros


Change = Annotated[Union[LogEntryChange, LogMealChange, LogTemplateChange, UpdateEntryChange, DeleteEntryChange,
                         DeleteMealChange, CreateTemplateChange, UpdateTemplateChange, SaveFoodChange],
                   Field(discriminator="type")]


class Batch(Strict):
    changes: list[Change] = Field(min_length=1, max_length=MAX_BATCH)


SCOPE_OF = {"log_entry": "log:write", "log_meal": "log:write", "log_template": "log:write",
            "update_entry": "log:write", "delete_entry": "log:write", "delete_meal": "log:write",
            "create_template": "templates:write", "update_template": "templates:write", "save_food": "library:append"}


def _needs_library(c) -> bool:
    items = [c] if isinstance(c, Item) else (getattr(c, "items", None) or []) + (getattr(c, "add", None) or [])
    return c.type == "save_food" or any(getattr(i, "save_to_library", False) for i in items)


def _check_scopes(caller: Caller, changes):
    for i, c in enumerate(changes):
        for scope in {SCOPE_OF[c.type]} | ({"library:append"} if _needs_library(c) else set()):
            if scope not in caller.scopes:
                raise Problem(403, "insufficient_scope", f"changes[{i}]: this token lacks {scope}.", change_index=i)


def _g(x: float) -> str:
    return f"{round(x, 1):g} g"


def app_nutrition(portion: Optional[str], macros: Macros, caller: Optional[Caller] = None) -> dict:
    """The app's own shape, so the Tracker shows it with no frontend change.
    `_kcal` marks the calories as kcal (the chat states them), which switches off
    the 'over 900 must be kJ' guess for this entry."""
    ps = {"size": portion or "1 serving", "calories": round(macros.calories, 1), "protein": _g(macros.protein_g),
          "carbohydrates": _g(macros.carbs_g), "fat": _g(macros.fat_g), "fibre": _g(macros.fibre_g)}
    if macros.sugars_g is not None:
        ps["sugars"] = _g(macros.sugars_g)
    if macros.sat_fat_g is not None:
        ps["saturated_fat"] = _g(macros.sat_fat_g)
    if macros.sodium_mg is not None:
        ps["sodium"] = f"{round(macros.sodium_mg, 1):g} mg"
    n = {"per_serving": ps, "_source": "claude", "_kcal": True}
    if caller and caller.token_id:
        n["_token_id"] = caller.token_id
    return n


def _kp(n, servings) -> tuple:
    m5 = log_service.entry_macros(n, servings)
    return m5["calories"], m5["protein"]


def _delta(before: tuple, after: tuple) -> dict:
    return {"kcal_change": round(after[0] - before[0], 1), "protein_change": round(after[1] - before[1], 1)}


class _Run:
    """One request's changes against one transaction."""

    def __init__(self, cur, caller: Caller):
        self.cur, self.caller, self.uid = cur, caller, caller.user_id
        self.diff, self.dates, self.affected, self.deletes = [], set(), [], 0
        self.logged_dates = set()   # days that gained an entry: the only ones a goal push is about
        self.library = {"saved": [], "skipped_existing": []}
        self._lib_names, self._folder = None, None

    def q(self, sql, params):
        self.cur.execute(sql, params)
        return self.cur.fetchall() if self.cur.description else []

    # -- entries
    def insert_entry(self, d: str, name: str, servings: float, n: dict) -> dict:
        log_id = str(uuid.uuid4())
        self.cur.execute("INSERT INTO daily_log (log_id, user_id, date, name, servings, nutrition, created_at) "
                         "VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp())", [log_id, self.uid, d, name, servings, json.dumps(n)])
        self.dates.add(d)
        self.logged_dates.add(d)
        self.affected.append(log_id)
        kcal, prot = _kp(n, servings)
        return {"log_id": log_id, "name": name, "servings": servings,
                "contribution": {"calories": round(kcal, 1), "protein": round(prot, 1)},
                "etag": etag_of(name, servings, n, d)}

    def entry(self, log_id: str, lock: bool = True):
        rows = self.q("SELECT log_id, name, servings, nutrition, date FROM daily_log WHERE log_id = %s AND user_id = %s"
                      + (" FOR UPDATE" if lock else ""), [log_id, self.uid])
        if not rows:
            raise _not_found("Entry")
        return rows[0]

    # -- library
    def save_to_library(self, name: str, portion: Optional[str], macros: Macros):
        """Skip if the name is anywhere in the Library (any case); otherwise add it
        to the 'From Claude' folder, creating that folder on first use."""
        if self._lib_names is None:
            self._lib_names = {r[0] for r in self.q("SELECT lower(name) FROM folder_items WHERE user_id = %s", [self.uid])}
        if name.lower() in self._lib_names:
            self.library["skipped_existing"].append(name)
            return
        if self._folder is None:
            rows = self.q("SELECT folder_id FROM folders WHERE user_id = %s AND name = %s ORDER BY created_at LIMIT 1",
                          [self.uid, LIBRARY_FOLDER])
            self._folder = rows[0][0] if rows else str(uuid.uuid4())
            if not rows:
                self.cur.execute("INSERT INTO folders (folder_id, user_id, name, created_at) VALUES (%s, %s, %s, now())",
                                 [self._folder, self.uid, LIBRARY_FOLDER])
        n = app_nutrition(portion, macros)
        n.pop("_source", None)
        self.cur.execute("INSERT INTO folder_items (item_id, folder_id, user_id, image_id, name, nutrition, created_at) "
                         "VALUES (%s, %s, %s, NULL, %s, %s, now())",
                         [str(uuid.uuid4()), self._folder, self.uid, name, json.dumps(n)])
        self._lib_names.add(name.lower())
        self.library["saved"].append(name)
        self.diff.append({"op": "add", "kind": "library_food", "name": name})

    def log_items(self, d: str, items, group: Optional[tuple] = None) -> list:
        out = []
        for it in items:
            if it.save_to_library:
                self.save_to_library(it.name, it.portion, it.macros)
            n = app_nutrition(it.portion, it.macros, self.caller)
            if group:
                n["_meal_group"], n["_meal_label"] = group
            e = self.insert_entry(d, it.name, it.servings, n)
            self.diff.append({"op": "add", "kind": "entry", "name": it.name, "date": d,
                              **_delta((0, 0), _kp(n, it.servings))})
            out.append(e)
        return out

    # -- one method per change type; each returns its result
    def log_entry(self, c: LogEntryChange):
        return {"date": c.date, "entry": self.log_items(c.date, [c])[0]}

    def log_meal(self, c: LogMealChange):
        gid = str(uuid.uuid4())
        return {"date": c.date, "meal": {"group_id": gid, "label": c.label,
                                         "items": self.log_items(c.date, c.items, (gid, c.label))}}

    def log_template(self, c: LogTemplateChange):
        t = self.q("SELECT name FROM meal_templates WHERE template_id = %s AND user_id = %s", [c.template_id, self.uid])
        if not t:
            raise _not_found("Template")
        rows = self.q("""SELECT item_id, name, servings, nutrition FROM meal_template_items
                         WHERE template_id = %s AND user_id = %s ORDER BY created_at""", [c.template_id, self.uid])
        ids = {r[0] for r in rows}
        for field, wanted in (("changes", [ch.item_id for ch in c.changes]), ("remove", c.remove)):
            for j, iid in enumerate(wanted):
                if iid not in ids:
                    raise Problem(422, "validation_error", f"{field}[{j}].item_id: not an item of this template",
                                  errors=[{"field": f"{field}.{j}.item_id", "message": "not an item of this template"}])
        changes = {ch.item_id: ch for ch in c.changes}
        gid, label = str(uuid.uuid4()), t[0][0]
        items = []
        for iid, name, servings, nutrition in rows:
            base = log_service.load_nutrition(nutrition)
            if iid in c.remove:
                self.diff.append({"op": "omit", "kind": "template_item", "name": name,
                                  **_delta(_kp(base, servings), (0, 0))})
                continue
            n, s, ch = dict(base), servings, changes.get(iid)
            edits = {}
            if ch:
                old_portion = log_service.per_serving_section(base).get("size")
                if ch.portion and not ch.macros:
                    raise Problem(422, "validation_error", "changes: a new portion needs its macros",
                                  errors=[{"field": "changes.macros", "message": "required when portion changes"}])
                if ch.macros:
                    keep = {k: v for k, v in base.items() if k.startswith("_") and k != "_kcal"}
                    n = {**keep, **app_nutrition(ch.portion or old_portion, ch.macros, self.caller)}
                    if (ch.portion or old_portion) != old_portion:
                        edits["portion"] = [old_portion, ch.portion]
                if ch.servings:
                    s = ch.servings
                    if s != servings:
                        edits["servings"] = [servings, s]
            n.update({"_meal_group": gid, "_meal_label": label, "_source": "claude"})
            if self.caller.token_id:
                n["_token_id"] = self.caller.token_id
            e = self.insert_entry(c.date, name, s, n)
            row = {"op": "add", "kind": "entry", "name": name, "date": c.date, **_delta((0, 0), _kp(n, s))}
            if ch:
                row.update(changes_from_template=edits, **{"vs_template_" + k: v for k, v in
                                                           _delta(_kp(base, servings), _kp(n, s)).items()})
            self.diff.append(row)
            items.append(e)
        items += self.log_items(c.date, c.add, (gid, label))
        return {"date": c.date, "meal": {"group_id": gid, "label": label, "template_id": c.template_id, "items": items}}

    def update_entry(self, c: UpdateEntryChange):
        log_id, name, servings, nutrition, d = self.entry(c.log_id)
        n = log_service.load_nutrition(nutrition)
        if etag_of(name, servings, n, str(d)) != c.if_match:
            raise Problem(412, "precondition_failed", "The entry changed since you read it. Read it again.")
        if c.portion and not c.macros:
            raise Problem(422, "validation_error", "portion: a new portion needs its macros",
                          errors=[{"field": "macros", "message": "required when portion changes"}])
        before = entry_detail(log_id, name, servings, n, d)
        new_n = n
        if c.macros:
            keep = {k: v for k, v in n.items() if k.startswith("_")}
            new_n = {**keep, **app_nutrition(c.portion or log_service.per_serving_section(n).get("size"), c.macros, self.caller)}
        new_name, new_s, new_d = c.name or name, c.servings or servings, c.date or str(d)
        if new_d != str(d):   # a meal is one day's meal: an entry moved to another day leaves it
            new_n = {k: v for k, v in new_n.items() if k not in ("_meal_group", "_meal_label")}
        self.cur.execute("UPDATE daily_log SET name = %s, servings = %s, nutrition = %s, date = %s WHERE log_id = %s AND user_id = %s",
                         [new_name, new_s, json.dumps(new_n), new_d, log_id, self.uid])
        self.dates.update({str(d), new_d})
        self.affected.append(log_id)
        after = entry_detail(log_id, new_name, new_s, new_n, new_d)
        fields = {k: [before[k], after[k]] for k in ("name", "portion", "servings", "date") if before[k] != after[k]}
        self.diff.append({"op": "update", "kind": "entry", "name": new_name, "changes": fields,
                          **_delta(_kp(n, servings), _kp(new_n, new_s))})
        return {"date": new_d, "before": before, "after": after}

    def delete_entry(self, c: DeleteEntryChange):
        log_id, name, servings, nutrition, d = self.entry(c.log_id)
        n = log_service.load_nutrition(nutrition)
        if etag_of(name, servings, n, str(d)) != c.if_match:
            raise Problem(412, "precondition_failed", "The entry changed since you read it. Read it again.")
        self.cur.execute("DELETE FROM daily_log WHERE log_id = %s AND user_id = %s", [log_id, self.uid])
        self.dates.add(str(d))
        self.affected.append(log_id)
        self.deletes += 1
        self.diff.append({"op": "remove", "kind": "entry", "name": name, **_delta(_kp(n, servings), (0, 0))})
        return {"date": str(d), "before": entry_detail(log_id, name, servings, n, d), "deleted": True}

    def delete_meal(self, c: DeleteMealChange):
        rows = self.q("""SELECT log_id, name, servings, nutrition, date FROM daily_log
                         WHERE user_id = %s AND nutrition->>'_meal_group' = %s ORDER BY created_at FOR UPDATE""",
                      [self.uid, c.group_id])
        if not rows:
            raise _not_found("Meal")
        items = [entry_detail(*r) for r in rows]
        if etag_of([i["etag"] for i in items]) != c.if_match:
            raise Problem(412, "precondition_failed", "The meal changed since you read it. Read it again.")
        self.cur.execute("DELETE FROM daily_log WHERE user_id = %s AND nutrition->>'_meal_group' = %s", [self.uid, c.group_id])
        for r in rows:
            self.dates.add(str(r[4]))
            self.affected.append(r[0])
            self.diff.append({"op": "remove", "kind": "entry", "name": r[1], **_delta(_kp(r[3], r[2]), (0, 0))})
        self.deletes += 1   # a whole meal counts as one delete towards the spree freeze
        return {"date": items[0]["date"], "before": {"group_id": c.group_id, "label": items[0]["label"], "items": items},
                "deleted": True}

    def _template_items(self, template_id: str, items):
        for it in items:
            if it.save_to_library:
                self.save_to_library(it.name, it.portion, it.macros)
            n = app_nutrition(it.portion, it.macros)
            self.cur.execute("INSERT INTO meal_template_items (item_id, template_id, user_id, name, nutrition, servings, created_at) "
                             "VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp())",
                             [str(uuid.uuid4()), template_id, self.uid, it.name, json.dumps(n), it.servings])

    def create_template(self, c: CreateTemplateChange):
        tid = str(uuid.uuid4())
        self.cur.execute("INSERT INTO meal_templates (template_id, user_id, name, created_at) VALUES (%s, %s, %s, now())",
                         [tid, self.uid, c.name])
        self._template_items(tid, c.items)
        self.affected.append(tid)
        after = template_detail(self.cur, self.uid, tid)
        for it in after["items"]:
            self.diff.append({"op": "add", "kind": "template_item", "template": c.name, "name": it["name"],
                              "portion": it["portion"], "servings": it["servings"],
                              "kcal_change": it["macros"]["calories"], "protein_change": it["macros"]["protein_g"]})
        return {"template": after}

    def update_template(self, c: UpdateTemplateChange):
        self.q("SELECT 1 FROM meal_templates WHERE template_id = %s AND user_id = %s FOR UPDATE", [c.template_id, self.uid])
        before = template_detail(self.cur, self.uid, c.template_id)
        if not before:
            raise _not_found("Template")
        if before["etag"] != c.if_match:
            raise Problem(412, "precondition_failed", "The template changed since you read it. Read it again.")
        if c.name:
            self.cur.execute("UPDATE meal_templates SET name = %s WHERE template_id = %s AND user_id = %s",
                             [c.name, c.template_id, self.uid])
        if c.items is not None:
            self.cur.execute("DELETE FROM meal_template_items WHERE template_id = %s AND user_id = %s", [c.template_id, self.uid])
            self._template_items(c.template_id, c.items)
        after = template_detail(self.cur, self.uid, c.template_id)
        self.affected.append(c.template_id)
        old = {i["name"].lower(): i for i in before["items"]}
        new = {i["name"].lower(): i for i in after["items"]}
        if before["name"] != after["name"]:
            self.diff.append({"op": "update", "kind": "template", "name": after["name"], "changes": {"name": [before["name"], after["name"]]}})
        for key in list(old) + [k for k in new if k not in old]:
            o, nw = old.get(key), new.get(key)
            kp = lambda i: (i["macros"]["calories"], i["macros"]["protein_g"]) if i else (0, 0)   # noqa: E731
            if o and nw:
                fields = {f: [o[f], nw[f]] for f in ("portion", "servings") if o[f] != nw[f]}
                if not fields and kp(o) == kp(nw):
                    continue
                row = {"op": "update", "changes": fields}
            else:
                row = {"op": "add" if nw else "remove"}
            self.diff.append({**row, "kind": "template_item", "template": after["name"], "name": (nw or o)["name"],
                              **_delta(kp(o), kp(nw))})
        return {"before": before, "after": after}

    def save_food(self, c: SaveFoodChange):
        self.save_to_library(c.name, c.portion, c.macros)
        return {"name": c.name, "saved": c.name in self.library["saved"]}

    def day_summary(self) -> dict:
        if not self.dates:
            return {}
        goal = goals_from(self.q(GOALS_SQL, [self.uid]))
        rows = self.q("SELECT date, servings, nutrition FROM daily_log WHERE user_id = %s AND date = ANY(%s)",
                      [self.uid, sorted(self.dates)])
        out = {}
        for d in sorted(self.dates):
            t = _macros5(log_service.sum_macros((s, n) for dd, s, n in rows if str(dd) == d))
            out[d] = {"totals_after": t, "remaining_after": {k: round(goal[k] - t[k], 1) for k in t}}
        return out


def _request_hash(path: str, payload: dict) -> str:
    return hashlib.sha256(json.dumps([path, payload], sort_keys=True, default=str).encode()).hexdigest()


def run_changes(request: Request, caller: Caller, changes: list, preview: bool, idempotency_key: Optional[str],
                success_status: int = 200) -> tuple:
    """Apply `changes` in one transaction. Returns (status, body).
    Real POSTs need an Idempotency-Key: the same key within 24 hours replays
    the first answer; the same key with a different body is 409."""
    _check_scopes(caller, changes)
    payload = [c.model_dump(mode="json") for c in changes]
    key = None
    if not preview and request.method == "POST":
        key = (idempotency_key or "").strip()
        if not key or len(key) > 255:
            raise Problem(422, "validation_error", "Idempotency-Key header required (1-255 characters) on a real write.",
                          errors=[{"field": "Idempotency-Key", "message": "required"}])
    rhash = _request_hash(request.url.path, payload)
    replay = None
    with db(caller.user_id, commit=not preview) as cur:
        if key:
            cur.execute("DELETE FROM api_idempotency WHERE user_id = %s AND key = %s AND created_at < now() - interval '24 hours'",
                        [caller.user_id, key])
            cur.execute("""INSERT INTO api_idempotency (user_id, key, request_hash, status_code, response)
                           VALUES (%s, %s, %s, 0, '{}') ON CONFLICT (user_id, key) DO NOTHING RETURNING 1""",
                        [caller.user_id, key, rhash])
            if not cur.fetchone():
                # Taken. A concurrent first attempt holds the row lock until it
                # commits, so this read sees its finished answer.
                cur.execute("SELECT request_hash, status_code, response FROM api_idempotency WHERE user_id = %s AND key = %s",
                            [caller.user_id, key])
                old = cur.fetchone()
                if old[0] != rhash:
                    raise Problem(409, "idempotency_conflict", "This Idempotency-Key was used for a different request.")
                if not old[1]:
                    raise Problem(409, "idempotency_conflict", "A request with this Idempotency-Key is still running.",
                                  in_progress=True)
                replay = (old[1], old[2] if isinstance(old[2], dict) else json.loads(old[2]))
        if replay is None:
            run = _Run(cur, caller)
            results = []
            for i, c in enumerate(changes):
                try:
                    results.append({"type": c.type, **getattr(run, c.type)(c)})
                except Problem as p:
                    p.detail = f"changes[{i}]: {p.detail}" if len(changes) > 1 else p.detail
                    p.extra.setdefault("change_index", i)
                    raise
            body = {"preview": preview, "results": results, "diff": run.diff, "days": run.day_summary(),
                    "library": run.library}
            if not preview:
                cur.execute("""INSERT INTO api_audit (audit_id, user_id, token_id, method, path, status, affected_ids)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                            [str(uuid.uuid4()), caller.user_id, caller.token_id, request.method, request.url.path[:200],
                             success_status, json.dumps(run.affected)])
                if key:
                    cur.execute("UPDATE api_idempotency SET status_code = %s, response = %s WHERE user_id = %s AND key = %s",
                                [success_status, json.dumps(jsonable_encoder(body)), caller.user_id, key])
    if replay:
        return replay
    if not preview:
        invalidate(caller.user_id)
        today = melbourne_today().isoformat()
        for d in run.logged_dates & {today}:   # like the app: only logging to today can newly hit the goal
            threading.Thread(target=m._check_goal_and_push, args=(caller.user_id, d), daemon=True).start()
        if request.method != "DELETE":   # DELETE routes are counted by abuse_guard, like the app's
            for _ in range(run.deletes):
                if caller.user_id not in m._frozen and m._spike(f"del:{caller.user_id}", 60, 600):
                    m.freeze_user(caller.user_id, "60 deletes in 10 minutes (possible hijacked token)")
    return success_status if not preview else 200, body


def _single(status: int, body: dict, location: Optional[str] = None) -> JSONResponse:
    """A one-change route answers with that change's result at the top level."""
    r = body["results"][0] if body.get("results") else {}
    d = r.get("date")
    out = {"preview": body["preview"], **{k: v for k, v in r.items() if k != "type"}, "diff": body["diff"],
           "library": body["library"]}
    if d and d in body["days"]:
        out["day_totals_after"] = body["days"][d]["totals_after"]
        out["remaining_after"] = body["days"][d]["remaining_after"]
    if len(body["days"]) > 1:
        out["days"] = body["days"]
    headers = {"Location": location} if location and status == 201 else {}
    return JSONResponse(jsonable_encoder(out), status_code=status, headers=headers)


def _if_match(value: Optional[str]) -> str:
    if not value:
        raise Problem(422, "validation_error", "If-Match header required: send the ETag you read.",
                      errors=[{"field": "If-Match", "message": "required"}])
    # Cloudflare (in front of Render) weakens ETag to W/"..." when it compresses a
    # response, so the header a client read back carries a W/ our etags never have.
    return value.strip().removeprefix("W/")


PreviewQ = Query(False, alias="preview")
WRITE = need(write=True)   # scopes are checked per change


@router.post("/batch")
def batch(body: Batch, request: Request, preview: bool = PreviewQ,
          idempotency_key: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    status, out = run_changes(request, caller, body.changes, preview, idempotency_key)
    return JSONResponse(jsonable_encoder(out), status_code=status)


def _change(model, **fields):
    """Build a change from path ids and headers. Their length limits are
    checked here, not by FastAPI, so a failure must still be a 422, never a 500."""
    try:
        return model(**fields)
    except ValidationError as e:
        raise validation_problem([{**err, "loc": ("request", *err["loc"])} for err in e.errors()])


def _created(status: int, out: dict, kind: str, id_key: str, path: str) -> JSONResponse:
    new_id = ((out.get("results") or [{}])[0].get(kind) or {}).get(id_key)
    return _single(status, out, f"/v1/{path}/{new_id}")


@router.post("/templates", status_code=201)
def post_template(body: CreateTemplateChange, request: Request, preview: bool = PreviewQ,
                  idempotency_key: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    status, out = run_changes(request, caller, [body], preview, idempotency_key, 201)
    return _created(status, out, "template", "template_id", "templates")


@router.patch("/templates/{template_id}")
def patch_template(template_id: str, body: TemplatePatch, request: Request, preview: bool = PreviewQ,
                   if_match: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    change = _change(UpdateTemplateChange, template_id=template_id, if_match=_if_match(if_match),
                     **body.model_dump(exclude_none=True))
    return _single(*run_changes(request, caller, [change], preview, None))


@router.post("/templates/{template_id}/log", status_code=201)
def post_template_log(template_id: str, body: TemplateLog, request: Request, preview: bool = PreviewQ,
                      idempotency_key: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    change = _change(LogTemplateChange, template_id=template_id, **body.model_dump())
    status, out = run_changes(request, caller, [change], preview, idempotency_key, 201)
    return _created(status, out, "meal", "group_id", "meals")


@router.post("/meals", status_code=201)
def post_meal(body: LogMealChange, request: Request, preview: bool = PreviewQ,
              idempotency_key: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    status, out = run_changes(request, caller, [body], preview, idempotency_key, 201)
    return _created(status, out, "meal", "group_id", "meals")


@router.post("/entries", status_code=201)
def post_entry(body: LogEntryChange, request: Request, preview: bool = PreviewQ,
               idempotency_key: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    status, out = run_changes(request, caller, [body], preview, idempotency_key, 201)
    return _created(status, out, "entry", "log_id", "entries")


@router.patch("/entries/{log_id}")
def patch_entry(log_id: str, body: EntryPatch, request: Request, preview: bool = PreviewQ,
                if_match: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    change = _change(UpdateEntryChange, log_id=log_id, if_match=_if_match(if_match), **body.model_dump(exclude_none=True))
    return _single(*run_changes(request, caller, [change], preview, None))


@router.delete("/entries/{log_id}")
def delete_entry(log_id: str, request: Request, preview: bool = PreviewQ,
                 if_match: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    change = _change(DeleteEntryChange, log_id=log_id, if_match=_if_match(if_match))
    return _single(*run_changes(request, caller, [change], preview, None))


@router.delete("/meals/{group_id}")
def delete_meal(group_id: str, request: Request, preview: bool = PreviewQ,
                if_match: Optional[str] = Header(None), caller: Caller = Depends(WRITE)):
    change = _change(DeleteMealChange, group_id=group_id, if_match=_if_match(if_match))
    return _single(*run_changes(request, caller, [change], preview, None))


def openapi_v1() -> dict:
    """The published contract: /v1 only (the app's own routes stay unlisted).
    backend/openapi-v1.json is this, written out; a test fails when they drift."""
    from fastapi.openapi.utils import get_openapi
    return get_openapi(title="NutriScan API", version="1", routes=router.routes,
                       description="Public API v1. Auth: `Authorization: Bearer nsk_live_...` (a personal access "
                                   "token) or an app login. Errors: RFC 9457 application/problem+json with "
                                   "`error_type`. Writes: `?preview=true` first, then the real write with an "
                                   "`Idempotency-Key` (POST) or `If-Match` (PATCH/DELETE).")


# ---------------------------------------------------------------- token management (app login only)
class TokenCreate(Strict):
    name: Label
    scopes: list[Literal[SCOPES]] = Field(min_length=1, max_length=len(SCOPES))
    expires: Literal[tuple(EXPIRY_DAYS)]


def _admin_login(authorization: Optional[str], allow_deleting: bool = False) -> str:
    """Token management accepts only a Supabase login - a PAT fails JWT
    verification here - so a leaked token can never mint another. Admin-only for now.
    allow_deleting: taking access away (revoke, disconnect) still works inside the 15 days."""
    user_id = m.get_user_id(authorization, allow_deleting=allow_deleting)
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
    user_id = _admin_login(authorization, allow_deleting=True)
    with db(user_id) as cur:
        cur.execute("UPDATE api_tokens SET revoked_at = now() WHERE token_id = %s AND user_id = %s AND revoked_at IS NULL",
                    [token_id, user_id])
        done = cur.rowcount
    forget_token(token_id)
    if not done:
        raise m.HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Token not found"})
    return {"revoked": True}


# ---------------------------------------------------------------- connected apps (app login only, owner only)
class AppConnect(Strict):
    client_id: str = Field(min_length=1, max_length=64)


class AppRename(Strict):
    name: Label


_APP_COLS = "id, client_id, name, connected_at, last_used_at"


def _app_view(r) -> dict:
    return dict(zip(("id", "client_id", "name", "connected_at", "last_used_at"), (str(r[0]), *r[1:])))


def _app_missing():
    return m.HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Connection not found"})


def _app_uuid(app_id: str) -> str:
    """A path id that is not a uuid is simply not found; a real one keeps the primary-key lookup."""
    try:
        return str(uuid.UUID(app_id))
    except ValueError:
        raise _app_missing()


@settings_router.get("/settings/connected-apps")
def list_apps(authorization: Optional[str] = Header(default=None)):
    """Also the Allow page's first question: a 403 means this account sees H4b's 'still in testing' line."""
    user_id = _admin_login(authorization)
    with db(user_id) as cur:
        cur.execute(f"SELECT {_APP_COLS} FROM connected_apps WHERE user_id = %s AND revoked_at IS NULL "
                    "ORDER BY connected_at DESC", [user_id])
        return [_app_view(r) for r in cur.fetchall()]


@settings_router.post("/settings/connected-apps", status_code=201)
def connect_app(body: AppConnect, authorization: Optional[str] = Header(default=None)):
    """Called by the Allow page once Supabase approved: creates the row or revives a disconnected one (the push goes
    only then). Also called for every live grant when Supabase skips the Allow page, so a live grant always has an
    active row."""
    user_id = _admin_login(authorization)
    with db(user_id) as cur:
        cur.execute("""INSERT INTO connected_apps (user_id, client_id) VALUES (%s, %s)
                       ON CONFLICT (user_id, client_id)
                       DO UPDATE SET revoked_at = NULL, connected_at = now() WHERE connected_apps.revoked_at IS NOT NULL
                       RETURNING id""", [user_id, body.client_id])
        changed = cur.fetchone() is not None
    forget_app(user_id, body.client_id)
    if changed:
        m.send_push_to_user(user_id, *CONNECT_PUSH)
    return {"connected": True}


@settings_router.patch("/settings/connected-apps/{app_id}")
def rename_app(app_id: str, body: AppRename, authorization: Optional[str] = Header(default=None)):
    user_id = _admin_login(authorization)
    app_uuid = _app_uuid(app_id)   # a malformed id is a 404 before any connection is borrowed
    with db(user_id) as cur:
        cur.execute(f"""UPDATE connected_apps SET name = %s WHERE id = %s AND user_id = %s
                        AND revoked_at IS NULL RETURNING {_APP_COLS}""", [body.name, app_uuid, user_id])
        row = cur.fetchone()
    if not row:
        raise _app_missing()
    return _app_view(row)


class MealRename(Strict):
    label: MealLabel


@settings_router.patch("/log/meals/{group_id}")
def rename_meal(group_id: str, body: MealRename, authorization: Optional[str] = Header(default=None)):
    """Rename one logged meal from the app (Artifact v9 M5). The name lives on the meal's log rows only, so the
    template it was logged from keeps its own name."""
    user_id = m.get_user_id(authorization)
    with db(user_id) as cur:
        cur.execute("""UPDATE daily_log SET nutrition = jsonb_set(nutrition, '{_meal_label}', to_jsonb(%s::text))
                       WHERE user_id = %s AND nutrition->>'_meal_group' = %s RETURNING log_id""",
                    [body.label, user_id, group_id])
        renamed = cur.fetchall()
    if not renamed:
        raise m.HTTPException(status_code=404, detail={"error_type": "not_found", "message": "Meal not found"})
    return {"group_id": group_id, "label": body.label}


@settings_router.delete("/settings/connected-apps/{app_id}")
def disconnect_app(app_id: str, authorization: Optional[str] = Header(default=None)):
    """What cuts access: the gate re-reads the row on the next call and sends the 401 challenge."""
    user_id = _admin_login(authorization, allow_deleting=True)
    app_uuid = _app_uuid(app_id)   # a malformed id is a 404 before any connection is borrowed
    with db(user_id) as cur:
        cur.execute("""UPDATE connected_apps SET revoked_at = now() WHERE id = %s AND user_id = %s
                       AND revoked_at IS NULL RETURNING client_id""", [app_uuid, user_id])
        row = cur.fetchone()
    if not row:
        raise _app_missing()
    forget_app(user_id, row[0])
    return {"disconnected": True}


@settings_router.get("/settings/admin/health")
def admin_health(authorization: Optional[str] = Header(default=None)):
    """The Admin panel's API health, from memory only: reading it never wakes Neon."""
    user_id = _admin_login(authorization)
    used = m.budget_used()
    return {"requests_left_today": requests_left(user_id), "resets_at": _zulu(next_melbourne_midnight()),
            "neon_budget_percent": round(used * 100, 1), "api_paused": used >= 0.9,
            "budget_period_resets": m.neon_next_period_start(datetime.now(timezone.utc).date()).isoformat()}


@settings_router.post("/settings/admin/test-alert")
def admin_test_alert(authorization: Optional[str] = Header(default=None)):
    """Sends through the in-memory device list, the path a 'Neon is down' alert takes."""
    _admin_login(authorization)
    # No device count: the send thread reloads an empty list first, so a count
    # taken here could say 0 for an alert that then arrives.
    m._admin_push("Test alert", "NutriScan admin alerts reach this device.")
    return {"sent": True}


def revoke_all(user_id: str):
    """Freezing an account also revokes its API tokens: the freeze's durable half
    is a Supabase ban, which a token never consults."""
    try:
        with db(user_id) as cur:
            cur.execute("UPDATE api_tokens SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL", [user_id])
    except Exception as e:
        m.logger.warning(f"token revoke on freeze failed for {user_id[:8]}: {e}")
    forget_token(user_id=user_id)   # after the commit, or a request in between re-caches the live row


# ---------------------------------------------------------------- /mcp: a minimal remote MCP server for claude.ai
# On its own router, off `router`, so openapi_v1() (router.routes only) is unaffected. Revocation on the next call
# is NOT handled here yet: a connector token stays valid until exp (up to 1h) once minted, because nothing looks up
# the Supabase session.
MCP_URL = "https://nutritionaltracker.onrender.com/mcp"          # PRM resource: must equal the pasted URL exactly
MCP_PRM_URL = "https://nutritionaltracker.onrender.com/.well-known/oauth-protected-resource/mcp"
MCP_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")         # first = the one we offer
# The app icon, per the MCP spec's serverInfo.icons. claude.ai shows a generic icon for every custom connector today
# (anthropics/claude-ai-mcp#152); this is here so ours appears once it reads the field.
MCP_SERVER_INFO = {"name": "nutriscan", "title": "NutriScan", "version": "1", "icons": [
    {"src": "https://nutritional-tracker-delta.vercel.app/icon-512.png", "mimeType": "image/png", "sizes": ["512x512"]}]}
MCP_ORIGINS = ("https://claude.ai", "https://claude.com")
CONNECTOR_SCOPES = SCOPES   # PRD: one access level, every "Read + log" scope; named so it is never app-login by accident
TODAY_HINT = "YYYY-MM-DD. Leave out for today (Australia/Melbourne); never ask the user for the date."

MCP_TOOL = {
    "name": "get_context", "title": "Read a day's food log",
    "description": "The user's NutriScan food log for one day: entries grouped by meal, the day's totals, goals and "
                   "what is left, their meal templates and the foods they log most. Energy is kcal. Read-only. "
                   "Call it first when the user talks about what they ate. Foods the user saved are in their "
                   "Library: search_library. Say the date in words (\"Thu 24 Sep\") "
                   "when you report a day. If more than one entry matches what the user means, list them and ask; "
                   "never guess.",
    "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"date": {"type": "string",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}$", "description": TODAY_HINT}}},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}}

TOOL_RULES = (" PREVIEW ONLY: this saves nothing and returns a confirm_code. Show the user the change as a short table "
              "(food, amount, kcal, protein, carbs, fat, date in words like \"Thu 24 Sep\") and ask yes or no. Only "
              "after an explicit yes, call confirm_change with the code; never call it without the user's yes. If "
              "more than one entry matches what the user means, list them and ask; never guess.")
LIBRARY_RULE = (" A food the user names: call search_library first (its numbers, and ask_before_saving). For a food "
                "not in the Library yet, while ask_before_saving is true: include save_to_library in this preview and "
                "ask in the same preview message whether to save it to the Library too; if the user says yes to the "
                "rest but no to the Library, preview again without save_to_library (their answer covers it) and "
                "confirm that. While ask_before_saving is false: include save_to_library and do not ask about the "
                "Library; the preview and the user's yes before confirm_change still apply.")
SAVES_TO_LIBRARY = {"log_meal", "log_template", "save_template", "update_template"}   # items can carry save_to_library
MEAL_RULE = (" Everything the user logs through you is a meal: a dish or several foods become one meal with each part "
             "its own item, and a single food is a meal of one item. label is the meal's name: suggest it yourself "
             "(one food: its name; a dish the user named: that dish; several foods: Breakfast, Lunch, Dinner or Snack "
             "by the time of day in Melbourne). Put everything in ONE preview message, starting \"I'll log this as a "
             "meal called <label> for <date in words>\" and ending \"Say yes, or tell me a different name.\" Never ask "
             "for the name or the date on their own. If the user answers yes with a different name, preview again "
             "with that label and confirm it without asking again (their answer covers it).")

# Claude logs through log_meal only (PRD change 2026-09-25: every Claude log is a named meal); /v1 keeps log_entry.
WRITE_TOOLS = {   # tool name -> (/v1 change model, title, what it does)
    "log_meal": (LogMealChange, "Log a meal", "Log what the user ate, as one named meal on a date." + MEAL_RULE),
    "log_template": (LogTemplateChange, "Log a saved meal",
                     "Log one of the user's meal templates on a date, optionally changing, adding or removing items. "
                     "Item ids come from get_template."),
    "edit_entry": (UpdateEntryChange, "Change a logged food",
                   "Change one logged entry (servings, portion, macros, name or date). log_id and if_match (the "
                   "entry's etag) come from get_context."),
    "delete_entry": (DeleteEntryChange, "Delete a logged food",
                     "Delete one logged entry. log_id and if_match (the entry's etag) come from get_context."),
    "delete_meal": (DeleteMealChange, "Delete a whole meal",
                    "Delete every item of one logged meal. Before previewing, ask the user: the whole meal or one "
                    "item? For one item use delete_entry. group_id and if_match (the meal's etag) come from get_context."),
    "save_template": (CreateTemplateChange, "Save a meal template", "Save a new meal template to log again later."),
    "update_template": (UpdateTemplateChange, "Change a meal template",
                        "Rename a meal template or replace its items. template_id and if_match (its etag) come from "
                        "get_template."),
    "save_to_library": (SaveFoodChange, "Save a food to the Library",
                        "Save a food to the user's Library for later. Call it only when the user asked to save the "
                        "food, or said yes to saving it."),
}


def _write_schema(model) -> dict:
    s = model.model_json_schema()
    s["properties"].pop("type", None)   # fixed per tool
    if "date" in s.get("required", []):   # _preview fills in today
        s["required"].remove("date")
        s["properties"]["date"]["description"] = TODAY_HINT
    return s


def _annotations(name: str) -> dict:
    """Only confirm_change writes. A preview saves nothing (its transaction rolls back), so claude.ai need not ask
    before one: the owner gets a single prompt, for the save (owner 2026-09-25)."""
    saves = name == "confirm_change"
    return {"readOnlyHint": not saves, "destructiveHint": saves, "idempotentHint": False, "openWorldHint": False}


GET_TEMPLATE_TOOL = {
    "name": "get_template", "title": "Read a meal template",
    "description": "One of the user's meal templates with its items (item ids) and its etag. Read-only.",
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["template_id"],
                    "properties": {"template_id": {"type": "string", "minLength": 1, "maxLength": 64}}},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}}
SEARCH_LIBRARY_TOOL = {
    "name": "search_library", "title": "Search the Library",
    "description": "The user's Library (the same one the NutriScan app uses): saved foods whose name contains the "
                   "query, with per-serving numbers (kcal). Search by one key word (\"milk\"), not the whole phrase; "
                   "if truncated is true, search narrower. Use these numbers when logging a food that is in it. If "
                   "more than one food matches what the user means, list them and ask which; never guess. "
                   "ask_before_saving: while true, ask the user before saving a new food to the Library. If they say "
                   "not to ask anymore, call stop_asking_before_saving. Read-only.",
    "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
        "query": {"type": "string", "maxLength": 80, "description": "Part of a food name. Leave out for all foods."}}},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}}
STOP_ASKING_TOOL = {
    "name": "stop_asking_before_saving", "title": "Stop asking before saving foods",
    "description": "Turn off the user's setting \"Ask before saving new foods to my Library\". Call it only when the "
                   "user tells you not to ask anymore, then tell them they can turn it back on in Settings > "
                   "Library. It changes nothing else, and nothing can turn the setting back on from here.",
    "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}}
CONFIRM_TOOL = {
    "name": "confirm_change", "title": "Save a previewed change",
    "description": "Save exactly the change a preview tool showed, using its confirm_code. Call this only after "
                   "the user said yes to that preview. Codes work once and expire after 10 minutes.",
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["code"],
                    "properties": {"code": {"type": "string", "minLength": 1, "maxLength": 64}}},
    "annotations": _annotations("confirm_change")}
MCP_TOOLS = [MCP_TOOL, GET_TEMPLATE_TOOL, SEARCH_LIBRARY_TOOL, STOP_ASKING_TOOL,
             *({"name": n, "title": t, "description": d + TOOL_RULES + (LIBRARY_RULE if n in SAVES_TO_LIBRARY else ""),
                "inputSchema": _write_schema(mdl),
                "annotations": _annotations(n)} for n, (mdl, t, d) in WRITE_TOOLS.items()),
             CONFIRM_TOOL]
TOOL_NAMES = {t["name"] for t in MCP_TOOLS}

mcp_router = APIRouter()


def mcp_issuer() -> str:
    return f"{m.SUPABASE_URL.rstrip('/')}/auth/v1"


@mcp_router.get("/.well-known/oauth-protected-resource/mcp")
@mcp_router.get("/.well-known/oauth-protected-resource")
def protected_resource() -> dict:
    return {"resource": MCP_URL, "authorization_servers": [mcp_issuer()], "scopes_supported": ["email"],
            "bearer_methods_supported": ["header"], "resource_name": "NutriScan"}


def _mcp_challenge() -> Problem:
    return Problem(401, "unauthorized", "Connect through claude.ai to use NutriScan.",
                   headers={"WWW-Authenticate": f'Bearer resource_metadata="{MCP_PRM_URL}"'})


def mcp_caller(request: Request) -> tuple:
    """(Caller, client_id) for a valid connector bearer token, or raises the 401 challenge.
    What gets the 401: a missing token, a bad one, an expired one, an app login (no client_id), and a PAT."""
    auth = request.headers.get("authorization") or ""
    claims = m.claims_if_valid(auth) if auth.startswith("Bearer ") else None
    aud = claims.get("aud") if claims else None
    aud_ok = aud == "authenticated" or (isinstance(aud, list) and "authenticated" in aud)
    if not (claims and claims.get("client_id") and claims.get("iss") == mcp_issuer() and aud_ok):
        m.logger.info(f"mcp 401 method={request.method} proto={request.headers.get('mcp-protocol-version', '-')} "
                      f"ua={(request.headers.get('user-agent') or '')[:60]}")
        raise _mcp_challenge()
    try:
        user_id = m.get_user_id(auth, allow_client=True)   # the frozen (423) and owner-only (403) checks
    except m.HTTPException as e:
        raise from_http_exception(e.status_code, e.detail) from e
    return Caller(user_id, name="Claude", scopes=CONNECTOR_SCOPES), claims["client_id"]


# One-time confirm codes, in memory: this holds while Render runs one process (like the limiter); move them to
# Postgres if it ever runs more than one. A restart just expires pending codes, which is safe.
PENDING_TTL, PENDING_MAX = 600, 20
_pending: dict = {}   # code -> (user_id, client_id, [change], created_at, template etag or None)
_pending_lock = threading.Lock()
NEXT_STEP = ("Show the user this change as a short table (food, amount, kcal, protein, carbs, fat, date in words) and "
             "ask yes or no. Only on yes, call confirm_change with confirm_code.")


def _template_etag(caller: Caller, template_id: str) -> Optional[str]:
    """Read straight from the DB (not the read cache): the confirm must see an edit made seconds ago."""
    with db(caller.user_id) as cur:
        t = template_detail(cur, caller.user_id, template_id)
    return t and t["etag"]


def _preview(request: Request, caller: Caller, client_id: str, name: str, args: dict) -> str:
    take_request(caller, True)   # as /v1's ?preview=true (need(write=True)): the write runs, then rolls back
    model = WRITE_TOOLS[name][0]
    if "date" not in args and (f := model.model_fields.get("date")) and f.is_required():
        args = {**args, "date": melbourne_today().isoformat()}   # only where a date is required: an edit keeps its own
    try:
        change = model.model_validate(args)
    except ValidationError as e:   # validation_problem drops loc[0] (FastAPI's "body"): stand in for it
        raise validation_problem([{**err, "loc": ("arguments", *err["loc"])} for err in e.errors()])
    # log_template carries no if_match, so remember the template the preview uses; confirm refuses if it moved.
    # Read before the preview: an edit landing in between then shows as a mismatch, never as a silent swap.
    guard = _template_etag(caller, change.template_id) if change.type == "log_template" else None
    _, body = run_changes(request, caller, [change], True, None)
    code, now = secrets.token_urlsafe(16), time.time()
    with _pending_lock:
        for k in [k for k, v in _pending.items() if now - v[3] > PENDING_TTL]:
            del _pending[k]
        mine = sorted((v[3], k) for k, v in _pending.items() if v[0] == caller.user_id)
        for _, k in mine[:max(0, len(mine) - PENDING_MAX + 1)]:
            del _pending[k]
        _pending[code] = (caller.user_id, client_id, [change], now, guard)
    return json.dumps({"preview": jsonable_encoder(body), "confirm_code": code, "expires_in_minutes": 10,
                       "next": NEXT_STEP}, default=str)


def _confirm(request: Request, caller: Caller, client_id: str, code) -> str:
    take_request(caller, True)
    # The code stays until it expires: the idempotency key (the code) makes a retried confirm replay the saved
    # answer instead of saving twice, e.g. when the first answer was lost on the way back to claude.ai.
    entry = _pending.get(code) if isinstance(code, str) else None
    if (not entry or entry[0] != caller.user_id or entry[1] != client_id
            or time.time() - entry[3] > PENDING_TTL):
        raise Problem(422, "confirmation_invalid", "This confirmation expired or is not valid. Ask again.")
    if entry[4] and _template_etag(caller, entry[2][0].template_id) != entry[4]:
        raise Problem(412, "precondition_failed", "This changed since the preview. Ask again.")
    try:
        _, body = run_changes(request, caller, entry[2], False, code)   # the code is the idempotency key
    except Problem as p:
        if p.extra.get("in_progress"):   # the same code confirmed twice at once: the first one is saving it
            raise Problem(409, p.error_type, "This change is already being saved. Do not save it again; check the "
                                             "log in a moment.") from p
        if p.status in (404, 409, 412):
            raise Problem(p.status, p.error_type, "This changed since the preview. Ask again.") from p
        raise
    if entry[4]:   # saved: a retry must replay the answer, not re-check a template edited since
        with _pending_lock:
            if code in _pending:
                _pending[code] = (*entry[:4], None)
    return json.dumps({"saved": True, **jsonable_encoder(body)}, default=str)


def _call_tool(request: Request, caller: Caller, client_id: str, name: str, args: dict) -> str:
    """The JSON text of one tool's result; raises Problem/HTTPException on failure."""
    if name == "get_context":
        take_request(caller, False)
        body = context(date_=args.get("date") or melbourne_today().isoformat(), include="", caller=caller)
    elif name == "get_template":
        tid = args.get("template_id")
        if not isinstance(tid, str) or not 1 <= len(tid) <= 64:
            raise Problem(422, "validation_error", "template_id: 1-64 characters.")
        take_request(caller, False)
        body = template_body(tid, caller)
    elif name == "search_library":
        _require(caller, "library:read")
        q = args.get("query", "")
        if not isinstance(q, str) or len(q) > 80:
            raise Problem(422, "validation_error", "query: up to 80 characters.")
        take_request(caller, False)
        body = library(caller, q)
    elif name == "stop_asking_before_saving":   # the one setting Claude may change, and only this way (PRD)
        _require(caller, "library:append")
        if args:   # e.g. a call meant to turn asking back on: refuse rather than turn it off
            raise Problem(422, "validation_error", "This tool takes no arguments and only turns asking off.")
        take_request(caller, True)
        with db(caller.user_id) as cur:
            cur.execute(PREFS_MERGE_SQL, [caller.user_id, json.dumps({"ask_before_saving_foods": False})])
            cur.execute("""INSERT INTO api_audit (audit_id, user_id, token_id, method, path, status, affected_ids)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",   # like every other Claude write (run_changes)
                        [str(uuid.uuid4()), caller.user_id, caller.token_id, request.method, request.url.path[:200],
                         200, json.dumps(["setting:ask_before_saving_foods=false"])])
        body = {"ask_before_saving": False}
    elif name in WRITE_TOOLS:
        return _preview(request, caller, client_id, name, args)
    else:   # confirm_change: the dispatcher already refused any name outside TOOL_NAMES
        return _confirm(request, caller, client_id, args.get("code"))
    return json.dumps(body, default=str)


def _mcp_rpc_result(id_, result) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "result": result})


def _mcp_rpc_error(id_, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


@mcp_router.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def mcp(request: Request):
    # The body is read raw and parsed only after the origin and auth checks: a declared JSON body would
    # 422 before them, and a tokenless probe would never see the 401 challenge. Capped at 64 KB upstream.
    raw = await request.body()
    return await run_in_threadpool(_mcp, request, raw)   # sync work: JWKS fetch, Postgres


def _mcp(request: Request, raw: bytes):
    origin = request.headers.get("origin")
    if origin and origin not in MCP_ORIGINS:
        m.logger.info(f"mcp origin refused: {origin[:100]}")
        raise Problem(403, "forbidden_origin", "This origin may not call /mcp.")
    caller, client_id = mcp_caller(request)
    if request.method != "POST":
        return Response(status_code=405, headers={"Allow": "POST"})
    if app_known_disconnected(caller.user_id, client_id):   # memory only: protocol calls never wake Neon
        raise _mcp_challenge()
    version = request.headers.get("mcp-protocol-version")
    if version and version not in MCP_VERSIONS:
        m.logger.info(f"mcp version refused: {version[:40]}")   # claude.ai tries newer versions, then falls back
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32022,
                             "message": "Unsupported protocol version", "data": {"supported": list(MCP_VERSIONS)}}},
                            status_code=400)
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):   # bad JSON, bad UTF-8, or nesting past the recursion limit
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                            status_code=400)
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}},
                            status_code=400)
    method = payload.get("method")
    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
    log_line = f"mcp rpc={method} proto={version or '-'} client={client_id} user={caller.user_id[:8]}"
    if method == "initialize":
        info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        log_line += f" req_proto={params.get('protocolVersion')} client_info={info.get('name')}/{info.get('version')}"
    m.logger.info(log_line)
    if "id" not in payload:   # a notification, e.g. notifications/initialized
        return Response(status_code=202)
    id_ = payload.get("id")
    if method == "initialize":
        proto = params.get("protocolVersion")
        return _mcp_rpc_result(id_, {"protocolVersion": proto if proto in MCP_VERSIONS else MCP_VERSIONS[0],
                                     "capabilities": {"tools": {}}, "serverInfo": MCP_SERVER_INFO})
    if method == "ping":
        return _mcp_rpc_result(id_, {})
    if method == "tools/list":
        return _mcp_rpc_result(id_, {"tools": MCP_TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOL_NAMES:
            return _mcp_rpc_error(id_, -32602, f"Unknown tool: {name}")
        args = params.get("arguments")
        if args is not None and not isinstance(args, dict):   # never quietly {}: [true] is not "no arguments"
            return _mcp_rpc_error(id_, -32602, "Tool arguments must be an object.")
        args = args or {}
        mode = ("confirm" if name == "confirm_change" else "preview" if name in WRITE_TOOLS
                else "setting" if name == "stop_asking_before_saving" else "read")
        where = f"mcp tool={name} mode={mode} client={client_id} user={caller.user_id[:8]}"   # never arguments
        try:
            connected_app_gate(caller.user_id, client_id)   # before budget_gate: it skips the DB while paused
            budget_gate()
            text = _call_tool(request, caller, client_id, name, args)
            m.logger.info(f"{where} outcome=ok")
            return _mcp_rpc_result(id_, {"content": [{"type": "text", "text": text}], "isError": False})
        except (Problem, m.HTTPException) as e:   # db() raises HTTPException 500: still a tool error, not transport
            if isinstance(e, Problem) and e.status == 401:
                raise   # a disconnected app: the 401 challenge is what makes claude.ai offer to reconnect
            p = e if isinstance(e, Problem) else from_http_exception(e.status_code, e.detail)
            m.logger.info(f"{where} outcome={p.error_type}")
            return _mcp_rpc_result(id_, {"content": [{"type": "text", "text": p.detail}], "isError": True})
    return _mcp_rpc_error(id_, -32601, "Method not found")
