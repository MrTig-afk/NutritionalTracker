"""The /mcp connector: the OAuth-token guard on app routes, and the minimal MCP server.

Run:  venv/Scripts/python -m unittest backend.tests.test_mcp -v
No network, no paid API: HS256 JWTs are minted locally with PyJWT against a test
secret, exercising the real verify_claims/get_user_id.
"""
import json
import time
import unittest
from unittest import mock

import jwt as pyjwt

import main
import api_v1
from .test_api_v1 import V1Case, TOKEN, token_row
from .test_api_v1_reads import kcal, log_row, LABEL

ISS = f"{main.SUPABASE_URL.rstrip('/')}/auth/v1"


def token(sub="admin-1", client_id="c1", aud="authenticated", iss=ISS, exp_in=600, **extra):
    claims = {"sub": sub, "aud": aud, "iss": iss, "exp": int(time.time()) + exp_in, **extra}
    if client_id is not None:
        claims["client_id"] = client_id
    return pyjwt.encode(claims, "test-secret", algorithm="HS256")


class WithConnectorAuth:
    """Every guard/mcp test needs a known signing secret and a known owner id."""

    def setUp(self):
        super().setUp()
        for target, value in (("SUPABASE_JWT_SECRET", "test-secret"), ("ADMIN_USER_ID", "admin-1")):
            p = mock.patch.object(main, target, value)
            p.start()
            self.addCleanup(p.stop)
        api_v1._apps[("admin-1", "c1")] = [True, time.time()]   # a known, active connection: the gate stays quiet

    def rpc(self, body, tok=None, headers=None):
        h = dict(headers or {})
        h["Authorization"] = f"Bearer {tok if tok is not None else token()}"
        return self.client.post("/mcp", headers=h, json=body)

    def call_tool(self, args=None, name="get_context", tok=None, id_=1, headers=None):
        return self.rpc({"jsonrpc": "2.0", "id": id_, "method": "tools/call",
                         "params": {"name": name, "arguments": args or {}}}, tok=tok, headers=headers)


class WithReadsFixture:
    """The same conn script as test_api_v1_reads.Reads (not a TestCase: mixed in
    for its fixture data only, so this file does not re-run Reads' own tests)."""

    def setUp(self):
        super().setUp()
        self.day_rows = [
            log_row("l1", "Firm tofu", 1.0, kcal(360, 42, "Meal 1", "g1")),
            log_row("l2", "Rice", 1.5, kcal(130, 3, "Meal 1", "g1")),
            log_row("l3", "Granola", 1.0, LABEL),
            log_row("l4", "Coffee", 1.0, kcal(90, 8, "Coffee")),
        ]
        self.conn.script = [
            ("resolve_api_token", [token_row()]),
            ("FROM user_goals", [(1800, 120, 200, 60, 30)]),
            ("FROM meal_templates t LEFT JOIN", [("t1", "Meal 1", "i1", "Firm tofu", 1.0, kcal(360, 42)),
                                                 ("t1", "Meal 1", "i2", "Rice", 1.0, kcal(130, 3))]),
            ("DISTINCT ON", [("Rice", 1.5, kcal(130, 3)), ("Granola", 1.0, LABEL)]),
            ("FROM daily_log", self.day_rows),
        ]


class Guard(WithConnectorAuth, V1Case):
    """A connected app's token (client_id present) is refused off /v1 and /mcp."""

    def test_client_token_refused_on_app_routes(self):
        h = {"Authorization": f"Bearer {token()}"}
        for method, path in (("get", "/log"), ("delete", "/account"),
                             ("get", "/settings/energy-unit"), ("get", "/settings/api-tokens")):
            r = getattr(self.client, method)(path, headers=h)
            self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (403, "connected_app_not_allowed"), path)
        with self.assertRaises(main.HTTPException) as cm:
            main.get_user_info(f"Bearer {token()}")
        self.assertEqual((cm.exception.status_code, cm.exception.detail["error_type"]), (403, "connected_app_not_allowed"))

    def test_v1_accepts_the_owner_client_token(self):
        r = self.client.get("/v1/me", headers={"Authorization": f"Bearer {token()}"})
        self.assertEqual(r.status_code, 200)

    def test_v1_refuses_a_non_owner_client_token(self):
        r = self.client.get("/v1/me", headers={"Authorization": f"Bearer {token(sub='someone')}"})
        self.assertEqual((r.status_code, r.json()["error_type"]), (403, "feature_unavailable"))

    def test_app_login_is_unaffected(self):
        # positive control: no client_id at all behaves exactly as before the guard
        h = {"Authorization": f"Bearer {token(client_id=None)}"}
        self.conn.script = [("FROM daily_log", [])]
        self.assertEqual(self.client.get("/log", headers=h).status_code, 200)
        self.assertEqual(self.client.get("/v1/me", headers=h).status_code, 200)


class Challenge(WithConnectorAuth, V1Case):
    """The 401 challenge on /mcp, and what does NOT satisfy it."""

    WWW = f'Bearer resource_metadata="{api_v1.MCP_PRM_URL}"'

    def test_no_token(self):
        for method in ("post", "get"):
            r = getattr(self.client, method)("/mcp")
            self.assertEqual(r.status_code, 401, method)
            self.assertEqual(r.headers["www-authenticate"], self.WWW, method)

    def test_app_login_is_not_a_connector(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, tok=token(client_id=None))
        self.assertEqual(r.status_code, 401)

    def test_pat_is_not_a_connector(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, tok=TOKEN)
        self.assertEqual(r.status_code, 401)

    def test_wrong_issuer(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, tok=token(iss="https://evil.example/auth/v1"))
        self.assertEqual(r.status_code, 401)

    def test_expired(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, tok=token(exp_in=-10))
        self.assertEqual(r.status_code, 401)


class PRM(V1Case):
    def test_protected_resource_metadata(self):
        want = {"resource": api_v1.MCP_URL, "authorization_servers": [api_v1.mcp_issuer()],
                "scopes_supported": ["email"], "bearer_methods_supported": ["header"], "resource_name": "NutriScan"}
        for path in ("/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"):
            self.assertEqual(self.client.get(path).json(), want, path)


class Initialize(WithConnectorAuth, V1Case):
    def test_requested_version_is_echoed(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-ai", "version": "1.0"}}})
        self.assertEqual(r.status_code, 200)
        result = r.json()["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "nutriscan")
        self.assertEqual(result["serverInfo"]["icons"][0]["src"], "https://nutritional-tracker-delta.vercel.app/icon-512.png")
        self.assertNotIn("mcp-session-id", {k.lower() for k in r.headers})

    def test_unknown_version_falls_back_to_the_first_offered(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
        self.assertEqual(r.json()["result"]["protocolVersion"], "2025-11-25")


class NotificationAndPing(WithConnectorAuth, V1Case):
    def test_notification_gives_202_with_empty_body(self):
        r = self.rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual((r.status_code, r.content), (202, b""))

    def test_ping(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(r.json()["result"], {})


class ToolsList(WithConnectorAuth, V1Case):
    def test_one_tool_shaped_correctly(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = r.json()["result"]["tools"]
        t = next(x for x in tools if x["name"] == "get_context")   # the read tool among the write tools
        self.assertIn("title", t)
        self.assertIs(t["annotations"]["readOnlyHint"], True)
        self.assertIs(t["annotations"]["destructiveHint"], False)
        self.assertIn("list them and ask", t["description"])


class ToolsCall(WithConnectorAuth, WithReadsFixture, V1Case):
    def test_get_context_reuses_v1_and_counts_only_the_call(self):
        r = self.call_tool({"date": "2026-09-24"})
        result = r.json()["result"]
        self.assertFalse(result["isError"])
        text = json.loads(result["content"][0]["text"])
        v1 = self.get("/v1/context?date=2026-09-24").json()
        self.assertEqual(text["totals"], v1["totals"])
        self.assertEqual(api_v1._daily["admin-1"][1], 1)
        self.rpc({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self.rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        self.assertEqual(api_v1._daily["admin-1"][1], 1)


class ToolsCallErrors(WithConnectorAuth, WithReadsFixture, V1Case):
    def test_bad_date_is_a_validation_isError(self):
        r = self.call_tool({"date": "24/09"})
        result = r.json()["result"]
        self.assertTrue(result["isError"])
        self.assertIn("YYYY-MM-DD", result["content"][0]["text"])

    def test_over_the_limit_is_an_isError(self):
        caller = api_v1.Caller("admin-1")
        for _ in range(20):
            api_v1.take_request(caller, False)
        result = self.call_tool().json()["result"]
        self.assertTrue(result["isError"])
        self.assertIn("Over the limit", result["content"][0]["text"])

    def test_unknown_tool(self):
        r = self.call_tool(name="delete_everything")
        self.assertEqual(r.json()["error"]["code"], -32602)

    def test_unknown_method(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "nope"})
        self.assertEqual(r.json()["error"]["code"], -32601)

    def test_list_body_is_invalid_request(self):
        r = self.client.post("/mcp", headers={"Authorization": f"Bearer {token()}"}, json=[1, 2, 3])
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, -32600))

    def test_malformed_json_is_a_parse_error_after_auth(self):
        bad = {"Content-Type": "application/json"}
        r = self.client.post("/mcp", content=b"{bad", headers=bad)
        self.assertEqual(r.status_code, 401)   # the challenge first, never FastAPI's 422
        self.assertIn("resource_metadata", r.headers["www-authenticate"])
        r = self.client.post("/mcp", content=b"{bad", headers={**bad, "Authorization": f"Bearer {token()}"})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, -32700))

    def test_deep_nesting_is_a_parse_error(self):
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {token()}"}
        r = self.client.post("/mcp", content=b"[" * 60000, headers=h)   # RecursionError, not a ValueError
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, -32700))

    def test_non_object_client_info_is_served(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18", "clientInfo": "x"}})
        self.assertEqual(r.status_code, 200)

    def test_db_failure_is_an_isError(self):
        down = main.HTTPException(500, {"error_type": "db_error", "message": "A database error occurred. Please try again."})
        with mock.patch.object(api_v1, "context", side_effect=down):
            result = self.call_tool({"date": "2026-09-24"}).json()["result"]
        self.assertTrue(result["isError"])
        self.assertIn("database error", result["content"][0]["text"])


class ProtocolHeader(WithConnectorAuth, V1Case):
    def test_unsupported_protocol_version(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"MCP-Protocol-Version": "2026-07-28"})
        self.assertEqual(r.status_code, 400)
        err = r.json()["error"]
        self.assertEqual(err["code"], -32022)
        self.assertEqual(err["data"]["supported"], list(api_v1.MCP_VERSIONS))

    def test_supported_protocol_version_is_served(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"MCP-Protocol-Version": "2025-06-18"})
        self.assertEqual(r.status_code, 200)


class Origin(WithConnectorAuth, V1Case):
    def test_unknown_origin_refused(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_claude_origin_allowed(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Origin": "https://claude.ai"})
        self.assertEqual(r.status_code, 200)


class ReadCacheSurvives(WithConnectorAuth, WithReadsFixture, V1Case):
    def test_second_identical_call_touches_the_database_no_further(self):
        self.call_tool({"date": "2026-09-24"}, id_=1)
        n = len(self.conn.executed)
        self.call_tool({"date": "2026-09-24"}, id_=2)
        self.assertEqual(len(self.conn.executed), n)


class FrozenAndNonOwner(WithConnectorAuth, V1Case):
    def test_frozen_owner_gives_423(self):
        with mock.patch.object(main, "_frozen", {"admin-1"}):
            r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(r.status_code, 423)

    def test_non_owner_client_token_gives_403(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, tok=token(sub="someone"))
        self.assertEqual(r.status_code, 403)


class MethodNotAllowed(WithConnectorAuth, V1Case):
    def test_get_after_auth_is_405(self):
        r = self.client.get("/mcp", headers={"Authorization": f"Bearer {token()}"})
        self.assertEqual((r.status_code, r.headers.get("allow")), (405, "POST"))

    def test_get_without_a_token_is_401_not_405(self):
        self.assertEqual(self.client.get("/mcp").status_code, 401)


class LogLine(WithConnectorAuth, WithReadsFixture, V1Case):
    def test_log_line_has_no_pii(self):
        with self.assertLogs(main.logger, "INFO") as cm:
            self.call_tool({"date": "2026-09-24"}, headers={"MCP-Protocol-Version": "2025-06-18"})
        lines = "\n".join(cm.output)
        self.assertIn("proto=2025-06-18", lines)
        self.assertIn("client=c1", lines)
        self.assertNotIn("2026-09-24", lines)
        self.assertNotIn("Firm tofu", lines)


if __name__ == "__main__":
    unittest.main()
