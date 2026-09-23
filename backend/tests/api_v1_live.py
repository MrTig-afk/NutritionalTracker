"""Hand test for /v1 against a RUNNING server with a REAL token. Not part of the
unit suite (no test_ prefix): it writes to whatever database that server uses.

    NUTRI_API_BASE=https://<backend>  NUTRI_API_TOKEN=nsk_live_...  venv/Scripts/python backend/tests/api_v1_live.py

Optional, each unlocks more rows of the Safety table:
    NUTRI_API_TOKEN_GOALS_ONLY   a token with only goals:read      -> 403 insufficient_scope
    NUTRI_API_TOKEN_REVOKED      a token you revoked               -> 401 token_revoked
    NUTRI_TEMPLATE_ID            one of your meal templates        -> log_template preview

Everything it logs is dated 2000-01-01 and deleted again at the end. Paced at
one request every 3.2 s to stay under the 20-a-minute limit, except the burst
check at the very end (which spends the rest of the minute). About 45 requests.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get("NUTRI_API_BASE", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("NUTRI_API_TOKEN", "")
DAY = "2000-01-01"
MACROS = {"calories": 260, "protein_g": 5.4, "carbs_g": 56, "fat_g": 0.6, "fibre_g": 0.8}
results, sent = [], [0]


def call(method, path, body=None, token=None, headers=None, raw=None, pace=True):
    if pace:
        time.sleep(3.2)
    sent[0] += 1
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Authorization": f"Bearer {token or TOKEN}", "User-Agent": "nutriscan-live-test"}
    if body is not None and raw is None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, json.loads(e.read() or b"{}")


def check(name, got, want_status, want_type=None):
    status, _, body = got
    ok = status == want_status and (want_type is None or body.get("error_type") == want_type)
    results.append((ok, name, status, body.get("error_type")))
    print(("PASS " if ok else "FAIL ") + f"{name}: {status} {body.get('error_type', '')}")
    return got


def main():
    if not TOKEN:
        sys.exit("set NUTRI_API_TOKEN")
    key = lambda: str(uuid.uuid4())   # noqa: E731

    # --- reads
    check("me", call("GET", "/v1/me"), 200)
    check("goals", call("GET", "/v1/goals"), 200)
    check("days", call("GET", f"/v1/days?from={DAY}&to=2000-01-07"), 200)
    check("days over 31", call("GET", "/v1/days?from=2000-01-01&to=2000-03-01"), 422, "validation_error")

    # --- the plan-then-log conversation: exactly 3 requests
    before = sent[0]
    _, _, ctx = check("context", call("GET", f"/v1/context?date={DAY}&include=trends"), 200)
    meal = {"date": DAY, "label": "Live test", "items": [{"name": "Live test rice", "portion": "200 g", "macros": MACROS}]}
    check("preview meal", call("POST", "/v1/meals?preview=true", meal), 200)
    k = key()
    _, h, logged = check("log meal", call("POST", "/v1/meals", meal, headers={"Idempotency-Key": k}), 201)
    results.append((sent[0] - before == 3, "plan-then-log takes 3 requests", sent[0] - before, None))
    gid = logged.get("meal", {}).get("group_id")
    log_id = logged.get("meal", {}).get("items", [{}])[0].get("log_id")
    print(f"      Location: {h.get('location')}")

    check("retry same key -> same meal", call("POST", "/v1/meals", meal, headers={"Idempotency-Key": k}), 201)
    check("same key, other body", call("POST", "/v1/meals", {**meal, "label": "Other"}, headers={"Idempotency-Key": k}),
          409, "idempotency_conflict")
    _, eh, entry = check("entry detail", call("GET", f"/v1/entries/{log_id}"), 200)
    check("meal detail", call("GET", f"/v1/meals/{gid}"), 200)
    etag = eh.get("etag")
    check("patch stale etag", call("PATCH", f"/v1/entries/{log_id}", {"servings": 2}, headers={"If-Match": '"stale"'}),
          412, "precondition_failed")
    _, _, patched = check("patch", call("PATCH", f"/v1/entries/{log_id}", {"servings": 2}, headers={"If-Match": etag}), 200)

    # --- errors
    check("no token", call("GET", "/v1/me", token="nope"), 401, "unauthorized")
    check("bad token", call("GET", "/v1/me", token="nsk_live_" + "x" * 43), 401, "unauthorized")
    if os.environ.get("NUTRI_API_TOKEN_REVOKED"):
        check("revoked token", call("GET", "/v1/me", token=os.environ["NUTRI_API_TOKEN_REVOKED"]), 401, "token_revoked")
    if os.environ.get("NUTRI_API_TOKEN_GOALS_ONLY"):
        check("missing scope", call("GET", f"/v1/context?date={DAY}", token=os.environ["NUTRI_API_TOKEN_GOALS_ONLY"]),
              403, "insufficient_scope")
    check("other id -> 404", call("GET", "/v1/entries/00000000-0000-0000-0000-000000000000"), 404, "not_found")
    check("body over 64 KB", call("POST", "/v1/batch", raw=b"{" + b" " * 70000 + b"}",
                                  headers={"Content-Type": "application/json"}), 413, "payload_too_large")
    check("not JSON", call("POST", "/v1/batch", raw=b"hi", headers={"Content-Type": "text/plain"}),
          415, "unsupported_media_type")
    check("unknown field", call("POST", "/v1/entries?preview=true", {"date": DAY, "name": "x", "macros": MACROS, "user_id": "x"}),
          422, "validation_error")
    check("future date", call("POST", "/v1/entries?preview=true", {"date": "2999-01-01", "name": "x", "macros": MACROS}),
          422, "validation_error")
    change = {"type": "log_entry", "date": DAY, "name": "x", "macros": MACROS}
    check("batch over 10", call("POST", "/v1/batch?preview=true", {"changes": [change] * 11}), 422, "validation_error")
    check("missing key", call("POST", "/v1/entries", {"date": DAY, "name": "x", "macros": MACROS}), 422, "validation_error")
    check("no If-Match", call("DELETE", f"/v1/entries/{log_id}"), 422, "validation_error")
    if os.environ.get("NUTRI_TEMPLATE_ID"):
        check("template log preview", call("POST", f"/v1/templates/{os.environ['NUTRI_TEMPLATE_ID']}/log?preview=true",
                                           {"date": DAY}), 200)

    # --- clean up (whole meal)
    _, mh, _ = call("GET", f"/v1/meals/{gid}")
    check("delete meal", call("DELETE", f"/v1/meals/{gid}", headers={"If-Match": mh.get("etag")}), 200)
    check("gone", call("GET", f"/v1/meals/{gid}"), 404, "not_found")

    # --- burst limit last: spends the rest of this minute
    got = None
    for _ in range(25):
        got = call("GET", "/v1/me", pace=False)
        if got[0] == 429:
            break
    status, headers, body = got
    check("burst -> 429", got, 429, "rate_limited")
    print(f"      Retry-After {headers.get('retry-after')}  resets_at {body.get('resets_at')}")

    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed, {sent[0]} requests sent")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
