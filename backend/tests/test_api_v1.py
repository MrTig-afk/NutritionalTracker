"""/v1 and token management, without a database.

Run:  venv/Scripts/python -m unittest backend.tests.test_api_v1 -v
FakeConn answers each query from a script (see test_phase0). The integration
tests against a real Postgres live in test_api_v1_db.py and need
NUTRI_TEST_DATABASE_URL.
"""
import hashlib
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

import main
import api_v1
from . import ALERTS
from .test_phase0 import FakeConn, ImmediateThread

TOKEN = "nsk_live_" + "A" * 43
HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
ALL = list(api_v1.SCOPES)


def token_row(scopes=ALL, expires_at=None, revoked_at=None, user="user-1"):
    return ("tok-1", user, "Claude", scopes, expires_at, revoked_at)


class V1Case(unittest.TestCase):
    """Fresh limits and caches per test; the DB is whatever FakeConn the test sets."""

    def setUp(self):
        for d in (api_v1._token_cache, api_v1._calls, api_v1._daily, api_v1._usage, api_v1._read_cache, api_v1._apps):
            d.clear()
        main._blocked.clear()
        main._event_windows.clear()   # the flood counter: the whole suite comes from one test client
        p = mock.patch.object(api_v1, "_live_prefixes", None)
        p.start()
        self.addCleanup(p.stop)
        self.conn = FakeConn([("resolve_api_token", [token_row()])])
        for target, value in (("get_db", lambda *a, **k: self.conn), ("release_db", lambda c: None),
                              ("_budget", {**main._budget, "seconds": 0.0, "awake_until": 0.0})):
            p = mock.patch.object(main, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def get(self, path, token=TOKEN, **kw):
        return self.client.get(path, headers={"Authorization": f"Bearer {token}", **kw.pop("headers", {})}, **kw)


class Tokens(V1Case):
    def test_valid_token_reaches_me(self):
        r = self.get("/v1/me")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["token_name"], "Claude")
        self.assertEqual(r.json()["requests_left_today"], 199)
        self.assertEqual(r.headers["RateLimit-Limit"], "200")
        self.assertEqual(r.headers["RateLimit-Remaining"], "199")
        self.assertEqual(r.headers["Cache-Control"], "no-store")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        lookup = [p for s, p in self.conn.executed if "resolve_api_token" in s][0]
        self.assertEqual(lookup[0], HASH)   # only the hash ever reaches the database

    def test_token_check_is_cached(self):
        self.get("/v1/me")
        self.get("/v1/me")
        self.assertEqual(len([s for s, _ in self.conn.executed if "resolve_api_token" in s]), 1)

    def test_revoked_and_expired(self):
        self.conn.script = [("resolve_api_token", [token_row(revoked_at=datetime.now(timezone.utc))])]
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "token_revoked"))
        self.assertEqual(r.headers["content-type"], "application/problem+json")
        api_v1._token_cache.clear()
        self.conn.script = [("resolve_api_token", [token_row(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))])]
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "token_expired"))

    def test_revoke_takes_effect_on_the_next_call_despite_the_cache(self):
        self.assertEqual(self.get("/v1/me").status_code, 200)
        self.conn.script = [("UPDATE api_tokens SET revoked_at", []),
                            ("resolve_api_token", [token_row(revoked_at=datetime.now(timezone.utc))])]
        with mock.patch.object(main, "get_user_id", lambda a=None, **k: "admin-1"), \
             mock.patch.object(main, "ADMIN_USER_ID", "admin-1"), \
             mock.patch.object(api_v1, "forget_token", wraps=api_v1.forget_token) as forget:
            self.client.delete("/settings/api-tokens/tok-1", headers={"Authorization": "Bearer login"})
        forget.assert_called_once_with("tok-1")
        self.assertEqual(self.get("/v1/me").json()["error_type"], "token_revoked")

    def test_malformed_token_never_reaches_the_database(self):
        r = self.get("/v1/me", token="nsk_live_short")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "unauthorized"))
        self.assertFalse(self.conn.executed)

    def test_unknown_token(self):
        self.conn.script = [("resolve_api_token", [])]
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "unauthorized"))

    def test_no_header(self):
        r = self.client.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "unauthorized"))

    def test_frozen_account(self):
        with mock.patch.object(main, "_frozen", {"user-1"}):
            r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (423, "account_locked"))

    def test_frozen_account_message_names_the_contact_email(self):
        want = "This account is locked after unusual activity. Email kaushiknaru2002@gmail.com to restore it."
        with mock.patch.object(main, "_frozen", {"user-1"}), \
             mock.patch.object(main, "verify_claims", lambda a: {"sub": "user-1", "email": "u@example.com"}):
            for fn in (main.get_user_id, main.get_user_info):
                with self.assertRaises(main.HTTPException) as cm:
                    fn("Bearer login")
                self.assertEqual((cm.exception.status_code, cm.exception.detail["message"]), (423, want), fn.__name__)

    def test_token_is_refused_on_app_routes(self):
        # PATs only reach /v1: the old routes verify a Supabase JWT, which a PAT is not
        for method, path in (("get", "/log"), ("delete", "/account"), ("get", "/settings/api-tokens")):
            r = getattr(self.client, method)(path, headers={"Authorization": f"Bearer {TOKEN}"})
            self.assertEqual(r.status_code, 401, path)

    def test_scope_is_enforced(self):
        self.conn.script = [("resolve_api_token", [token_row(scopes=["goals:read"])])]
        from starlette.requests import Request
        request = Request({"type": "http", "method": "GET", "path": "/v1/x",
                           "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
                           "query_string": b"", "client": ("1.2.3.4", 1), "state": {}})
        with self.assertRaises(api_v1.Problem) as ctx:
            api_v1.need("log:read")(request)
        self.assertEqual((ctx.exception.status, ctx.exception.error_type), (403, "insufficient_scope"))
        self.assertIsInstance(api_v1.need("goals:read")(request), api_v1.Caller)   # positive control


class TokenManagement(V1Case):
    def admin(self, user="admin-1"):
        p1 = mock.patch.object(main, "get_user_id", lambda a=None, **k: user)
        p2 = mock.patch.object(main, "ADMIN_USER_ID", "admin-1")
        for p in (p1, p2):
            p.start()
            self.addCleanup(p.stop)

    def create(self, **body):
        return self.client.post("/settings/api-tokens", headers={"Authorization": "Bearer login"},
                                json={"name": "Claude", "scopes": ["log:read"], "expires": "90d", **body})

    def test_other_accounts_get_feature_unavailable(self):
        self.admin(user="someone-else")
        r = self.create()
        self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (403, "feature_unavailable"))
        self.assertEqual(self.client.get("/settings/api-tokens", headers={"Authorization": "Bearer x"}).status_code, 403)

    def test_create_shows_the_token_once_and_stores_only_its_hash(self):
        self.admin()
        now = datetime.now(timezone.utc)
        self.conn.script = [("SELECT count(*)", [(0,)]),
                            ("INSERT INTO api_tokens", [("t", "Claude", "nsk_live_abc", ["log:read"], now, None, None, None, None)])]
        ALERTS.clear()
        r = self.create()
        self.assertEqual(r.status_code, 201)
        token = r.json()["token"]
        self.assertRegex(token, r"^nsk_live_[A-Za-z0-9_-]{43}$")
        insert = [p for s, p in self.conn.executed if s.startswith("INSERT INTO api_tokens")][0]
        self.assertEqual(insert[3], token[:12])
        self.assertEqual(insert[4], hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn(token, [str(x) for x in insert])
        self.assertTrue(abs((insert[6] - now).days - 90) <= 1)
        self.assertEqual(ALERTS, [("New API token", "New API token 'Claude' created. If you didn't make it, revoke it in Settings.")])

    def test_limit_of_ten(self):
        self.admin()
        self.conn.script = [("SELECT count(*)", [(10,)])]
        r = self.create()
        self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (409, "token_limit"))

    def test_bad_bodies(self):
        self.admin()
        for body in ({"scopes": []}, {"scopes": ["admin"]}, {"expires": "5y"}, {"name": "\x00\x01"},
                     {"name": "x" * 41}, {"extra": 1}):
            self.assertEqual(self.create(**body).status_code, 422, body)


class Limits(V1Case):
    def test_21st_request_in_a_minute_is_429(self):
        for _ in range(20):
            self.assertEqual(self.get("/v1/me").status_code, 200)
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (429, "rate_limited"))
        body = r.json()
        self.assertEqual((body["limit"], body["used"]), (20, 20))
        self.assertRegex(body["resets_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertTrue(1 <= int(r.headers["Retry-After"]) <= 61)

    def test_writes_cap_at_ten_a_minute(self):
        caller = api_v1.Caller("user-1", "tok-1")
        for _ in range(10):
            api_v1.take_request(caller, True)
        api_v1.take_request(caller, False)   # reads still pass
        with self.assertRaises(api_v1.Problem) as ctx:
            api_v1.take_request(caller, True)
        self.assertEqual(ctx.exception.extra["limit"], 10)

    def test_daily_cap_resets_at_melbourne_midnight(self):
        caller = api_v1.Caller("user-1")
        t = datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc).timestamp()   # 23:00 in Melbourne (AEST)
        for i in range(200):
            api_v1.take_request(caller, False, now=t + i * 4)   # spread out: under the burst limit
        with self.assertRaises(api_v1.Problem) as ctx:
            api_v1.take_request(caller, False, now=t + 900)
        self.assertEqual(ctx.exception.extra["resets_at"], "2026-09-24T14:00:00Z")   # 00:00 Melbourne
        api_v1.take_request(caller, False, now=t + 3700)   # past midnight: a new day

    def test_limits_are_per_user_across_tokens(self):
        for i in range(20):
            api_v1.take_request(api_v1.Caller("user-1", f"tok-{i % 2}"), False)
        with self.assertRaises(api_v1.Problem):
            api_v1.take_request(api_v1.Caller("user-1", "tok-3"), False)
        api_v1.take_request(api_v1.Caller("user-2", "tok-9"), False)   # someone else is unaffected


class Budget(V1Case):
    def test_90_percent_pauses_v1_but_not_the_app(self):
        main._budget["seconds"] = 0.9 * 100 / 0.25 * 3600
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (503, "api_paused_budget"))
        self.assertGreater(int(r.headers["Retry-After"]), 0)
        self.assertEqual(self.client.get("/health").status_code, 200)
        main._budget["seconds"] = 0.89 * 100 / 0.25 * 3600   # positive control
        self.assertEqual(self.get("/v1/me").status_code, 200)

    def test_awake_time_is_the_union_of_five_minute_windows(self):
        with mock.patch.object(main, "_save_budget", lambda: None), \
             mock.patch.object(main.threading, "Thread", mock.Mock()):
            main._budget["saved_at"] = time.time() + 10**6
            t = 1_790_000_000.0
            main._note_db_use(t)            # wakes: +300
            main._note_db_use(t + 60)       # still awake: extends by 60
            main._note_db_use(t + 1000)     # asleep again: +300
        self.assertEqual(main._budget["seconds"], 660)

    def test_70_percent_alerts_once(self):
        alerts = []
        with mock.patch.object(main, "notify_admin", lambda k, t, msg: alerts.append(k)), \
             mock.patch.object(main.threading, "Thread", mock.Mock()):
            main._budget.update(period=main.neon_period_start(datetime.now(timezone.utc).date()),
                                seconds=0.7 * 100 / 0.25 * 3600, alerted_70=False, saved_at=time.time())
            main._note_db_use()
            main._note_db_use()
        self.assertEqual(alerts, ["neon_budget_70"])

    def test_period_boundaries(self):
        with mock.patch.object(main, "NEON_PERIOD_START_DAY", 15):
            self.assertEqual(main.neon_period_start(datetime(2026, 9, 14).date()).isoformat(), "2026-08-15")
            self.assertEqual(main.neon_period_start(datetime(2026, 9, 15).date()).isoformat(), "2026-09-15")
            self.assertEqual(main.neon_next_period_start(datetime(2026, 12, 20).date()).isoformat(), "2027-01-15")


class Remediation(V1Case):
    """Phase 1 review findings, each pinned."""

    def test_usage_is_flushed_as_its_own_user(self):
        # P1-CR-1: as __system__ the RLS policy would admit nothing
        seen = []
        with mock.patch.object(main, "get_db", lambda uid=None: seen.append(uid) or self.conn):
            api_v1.take_request(api_v1.Caller("user-1", "tok-1"), False)
            api_v1.take_request(api_v1.Caller("user-1", "tok-1"), True)
            api_v1.flush_usage()
        self.assertEqual(seen, ["user-1"])
        insert = [p for s, p in self.conn.executed if s.startswith("INSERT INTO api_token_usage")][0]
        self.assertEqual(insert[2:], ["user-1", 1, 1])
        self.assertFalse(api_v1._usage)

    def test_failed_flush_keeps_the_counts(self):
        api_v1.take_request(api_v1.Caller("user-1", "tok-1"), False)
        with mock.patch.object(main, "get_db", mock.Mock(side_effect=RuntimeError("down"))):
            api_v1.flush_usage()
        self.assertEqual(list(api_v1._usage.values()), [[1, 0]])

    def test_account_wipe_skips_missing_api_tables(self):
        # P1-CR-2 (the wipe now runs from the day-15 purge; forgetting tokens: test_account_deletion)
        self.conn.script = [("FOR UPDATE", [(1,)]), ("to_regclass", [(None,)])]
        main._wipe_account_rows("user-1")
        self.assertFalse([s for s, _ in self.conn.executed if s.startswith("DELETE FROM")])   # every table "missing"
        # positive control: tables present -> rows deleted
        self.conn.executed.clear()
        self.conn.script = [("FOR UPDATE", [(1,)]), ("to_regclass", [("x",)])]
        main._wipe_account_rows("user-1")
        deleted = [s for s, _ in self.conn.executed if s.startswith("DELETE FROM")]
        self.assertIn("DELETE FROM api_tokens WHERE user_id = %s", deleted)

    def test_freezing_revokes_tokens(self):
        # P1-SEC-1: the Supabase ban is the durable half of a freeze, and a token never asks Supabase
        self.get("/v1/me")
        with mock.patch.object(main.threading, "Thread", ImmediateThread), \
             mock.patch.object(main, "SUPABASE_SERVICE_ROLE_KEY", ""), \
             mock.patch.object(main, "_frozen", set()):
            main.freeze_user("user-1", "test")
        self.assertFalse(api_v1._token_cache)
        revoke = [p for s, p in self.conn.executed if s.startswith("UPDATE api_tokens SET revoked_at")]
        self.assertEqual(revoke, [["user-1"]])

    def test_guesses_matching_no_live_prefix_never_reach_the_database(self):
        # P1-D2-2: a guess must not be able to keep Neon awake
        api_v1._live_prefixes = {"nsk_live_XYZ"}
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "unauthorized"))
        self.assertFalse(self.conn.executed)
        api_v1._live_prefixes = {TOKEN[:12]}   # positive control: a live prefix is looked up
        self.assertEqual(self.get("/v1/me").status_code, 200)

    def test_prefixes_load_from_the_database(self):
        self.conn.script = [("api_token_prefixes", [("nsk_live_abc",), ("nsk_live_def",)])]
        api_v1.load_prefixes()
        self.assertEqual(api_v1._live_prefixes, {"nsk_live_abc", "nsk_live_def"})

    def test_freeze_forgets_the_cache_only_after_the_revoke(self):
        # P1-D2-3
        order = []
        with mock.patch.object(api_v1, "forget_token", lambda **k: order.append("forget")),              mock.patch.object(self.conn, "commit", lambda: order.append("commit")):
            api_v1.revoke_all("user-1")
        self.assertEqual(order, ["commit", "forget"])

    def test_account_delete_drops_unsaved_usage(self):
        # P1-D2-4
        api_v1.take_request(api_v1.Caller("user-1", "tok-1"), False)
        api_v1.take_request(api_v1.Caller("user-2", "tok-2"), False)
        api_v1.forget_token(user_id="user-1")
        self.assertEqual([k[0] for k in api_v1._usage], ["user-2"])

    def test_chunked_body_needs_a_length(self):
        # P1-SEC-3
        def gen():
            yield b"{}"
        r = self.client.post("/v1/me", headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                             content=gen())
        self.assertEqual((r.status_code, r.json()["error_type"]), (411, "length_required"))

    def test_budget_save_never_clears_a_stored_alert(self):
        # P1-CR-4
        main._budget.update(period=datetime(2026, 9, 1).date(), alerted_70=False)
        main._save_budget()
        sql = [s for s, _ in self.conn.executed if "INSERT INTO ops_budget" in s][0]
        self.assertIn("alerted_70 = ops_budget.alerted_70 OR EXCLUDED.alerted_70", sql)


class Hygiene(V1Case):
    def test_oversize_and_non_json_bodies(self):
        h = {"Authorization": f"Bearer {TOKEN}"}
        r = self.client.post("/v1/me", headers=h, content=b"{" + b" " * (64 * 1024) + b"}")
        self.assertEqual((r.status_code, r.json()["error_type"]), (413, "payload_too_large"))
        r = self.client.post("/v1/me", headers={**h, "Content-Type": "text/plain"}, content=b"hi")
        self.assertEqual((r.status_code, r.json()["error_type"]), (415, "unsupported_media_type"))

    def test_v1_uses_the_one_app_body_cap(self):
        # One 64 KB rule for app and /v1 writes (INT-2): changing it in main changes /v1 too.
        with mock.patch.object(main, "APP_BODY_CAP", 10):
            r = self.client.post("/v1/me", headers={"Authorization": f"Bearer {TOKEN}"}, content=b'{"a": "0123456789"}')
        self.assertEqual((r.status_code, r.json()["error_type"]), (413, "payload_too_large"))

    def test_no_browser_origin_on_v1(self):
        pre = {"Origin": "https://nutritional-tracker-delta.vercel.app", "Access-Control-Request-Method": "GET"}
        self.assertNotIn("access-control-allow-origin", self.client.options("/v1/me", headers=pre).headers)
        self.assertIn("access-control-allow-origin", self.client.options("/log", headers=pre).headers)   # control

    def test_unknown_path_is_a_problem(self):
        r = self.get("/v1/nope")
        self.assertEqual((r.status_code, r.headers["content-type"], r.json()["error_type"]),
                         (404, "application/problem+json", "not_found"))

    def test_account_wipe_covers_the_api_tables(self):
        for t in ("api_tokens", "api_token_usage", "api_idempotency", "api_audit"):
            self.assertIn(t, main._ACCOUNT_TABLES)

    def test_published_openapi_matches_the_code(self):
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openapi-v1.json")
        with open(path, encoding="utf-8") as f:
            published = json.load(f)
        self.assertEqual(published, json.loads(json.dumps(api_v1.openapi_v1())),
                         "regenerate backend/openapi-v1.json from api_v1.openapi_v1()")
        routes = sum(len(ops) for ops in published["paths"].values())
        self.assertEqual(routes, 17)   # the PRD's route table, plus GET /v1/library (PRD change 2026-09-25)
        self.assertFalse([p for p in published["paths"] if not p.startswith("/v1/")])

    def test_no_internal_fields_in_me(self):
        self.assertEqual(set(self.get("/v1/me").json()), {"token_name", "scopes", "requests_left_today", "resets_at"})


if __name__ == "__main__":
    unittest.main()
