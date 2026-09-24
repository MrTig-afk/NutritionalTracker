"""Connected apps (lane H, Part B): the table's account-delete hook, the /mcp gate, the app routes.

Run:  venv/Scripts/python -m unittest backend.tests.test_connected_apps -v
No network: FakeConn scripts the DB; send_push_to_user is patched.
"""
import time
from datetime import datetime, timezone
import unittest
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case
from .test_mcp import WithConnectorAuth, token


class AccountDelete(unittest.TestCase):
    def test_connected_apps_is_an_account_table(self):
        self.assertIn("connected_apps", main._ACCOUNT_TABLES)

    def test_forgetting_a_user_forgets_their_apps(self):
        self.addCleanup(api_v1._apps.clear)
        api_v1._apps[("u1", "c1")] = [True, time.time()]
        api_v1._apps[("u2", "c1")] = [True, time.time()]
        api_v1.forget_token(user_id="u1")
        self.assertNotIn(("u1", "c1"), api_v1._apps)
        self.assertIn(("u2", "c1"), api_v1._apps)


NOW = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
A1 = "6f1c2d3e-0000-4000-8000-000000000001"
ROW = (A1, "c1", "Claude", NOW, NOW)

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}


class Gate(WithConnectorAuth, V1Case):
    def setUp(self):
        super().setUp()
        api_v1._apps.clear()   # these tests start with nothing cached
        p = mock.patch.object(main, "send_push_to_user")
        self.push = p.start()
        self.addCleanup(p.stop)

    def stmts(self, needle):
        return [s for s, _ in self.conn.executed if needle in s]

    def test_unknown_connection_is_adopted_with_a_push(self):
        self.conn.script = [("SELECT revoked_at FROM connected_apps", []), ("INSERT INTO connected_apps", [("admin-1",)])]
        r = self.call_tool({})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(self.stmts("INSERT INTO connected_apps")), 1)
        self.push.assert_called_once_with("admin-1", "Claude connected to NutriScan", "Not you? Disconnect it in Settings.")

    def test_revoked_connection_gets_the_challenge_and_nothing_runs(self):
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(time.time(),)])]
        r = self.call_tool({})
        self.assertEqual(r.status_code, 401)
        self.assertIn("resource_metadata", r.headers["WWW-Authenticate"])
        self.assertEqual(self.stmts("FROM daily_log"), [])
        self.push.assert_not_called()
        r = self.call_tool({})   # cached as revoked: refused again without a query
        self.assertEqual(r.status_code, 401)
        self.assertEqual(len(self.stmts("SELECT revoked_at FROM connected_apps")), 1)

    def test_active_connection_is_touched_at_most_once_a_minute(self):
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(None,)])]
        self.call_tool({})
        self.call_tool({})
        self.assertEqual(len(self.stmts("SET last_used_at")), 1)
        api_v1._apps[("admin-1", "c1")][1] -= api_v1.APP_TOUCH_SECS + 1
        self.call_tool({})
        self.assertEqual(len(self.stmts("SET last_used_at")), 2)
        self.push.assert_not_called()

    def test_no_touch_while_the_budget_is_paused(self):
        api_v1._apps[("admin-1", "c1")] = [True, time.time() - 61]
        with mock.patch.object(main, "budget_used", return_value=0.95):
            self.call_tool({})
        self.assertEqual(self.stmts("connected_apps"), [])

    def test_a_racing_first_call_that_inserted_nothing_sends_no_push(self):
        self.conn.script = [("SELECT revoked_at FROM connected_apps", []), ("INSERT INTO connected_apps", [])]
        self.assertEqual(self.call_tool({}).status_code, 200)
        self.push.assert_not_called()

    def test_disconnect_during_the_lookup_is_not_overwritten_by_a_stale_active(self):
        def active_then_disconnected(sql, params):
            api_v1.forget_app("admin-1", "c1")   # disconnect_app commits and forgets while this lookup runs
            return [(None,)]
        self.conn.script = [("SELECT revoked_at FROM connected_apps", active_then_disconnected)]
        self.call_tool({})
        self.assertNotIn(("admin-1", "c1"), api_v1._apps)   # the next call re-reads the row
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(NOW,)])]
        self.assertEqual(self.call_tool({}).status_code, 401)

    def test_allow_during_the_lookup_is_not_overwritten_by_a_stale_revoked(self):
        def revoked_then_revived(sql, params):
            api_v1.forget_app("admin-1", "c1")   # connect_app revives and forgets while this lookup runs
            return [(NOW,)]
        self.conn.script = [("SELECT revoked_at FROM connected_apps", revoked_then_revived)]
        self.assertEqual(self.call_tool({}).status_code, 401)
        self.assertNotIn(("admin-1", "c1"), api_v1._apps)
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(None,)])]
        self.assertEqual(self.call_tool({}).status_code, 200)

    def test_cold_cache_does_not_wake_a_paused_neon(self):   # budget_gate then refuses the tool
        with mock.patch.object(main, "budget_used", return_value=0.95):
            r = self.call_tool({})
        self.assertEqual(r.json()["result"]["isError"], True)
        self.assertEqual(self.stmts("connected_apps"), [])

    def test_two_claude_accounts_are_two_rows(self):
        self.conn.script = [("SELECT revoked_at FROM connected_apps", []), ("INSERT INTO connected_apps", [("admin-1",)])]
        self.call_tool({}, tok=token(client_id="c1"))
        self.call_tool({}, tok=token(client_id="c2"))
        self.assertEqual(self.push.call_count, 2)
        api_v1.forget_app("admin-1", "c1")
        self.assertIn(("admin-1", "c2"), api_v1._apps)


    def test_protocol_calls_never_touch_the_database(self):
        for body in (INIT, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 3, "method": "ping"}):
            self.assertEqual(self.rpc(body).status_code, 200)
        self.assertEqual(self.stmts("connected_apps"), [])
        self.push.assert_not_called()

    def test_a_connection_known_to_be_disconnected_is_refused_even_on_initialize(self):
        api_v1._apps[("admin-1", "c1")] = [False, time.time()]
        self.assertEqual(self.rpc(INIT).status_code, 401)

    def test_a_database_failure_in_the_gate_is_a_tool_error_not_a_500(self):
        def down(sql, params):
            raise RuntimeError("neon unreachable")
        self.conn.script = [("SELECT revoked_at FROM connected_apps", down)]
        r = self.call_tool({})
        self.assertEqual(r.status_code, 200)
        self.assertIs(r.json()["result"]["isError"], True)
        self.assertNotIn(("admin-1", "c1"), api_v1._apps)

    def test_the_cache_is_bounded(self):
        for i in range(api_v1.APPS_MAX + 1):
            api_v1._apps[("u", str(i))] = [True, time.time()]
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(None,)])]
        self.call_tool({})
        self.assertLessEqual(len(api_v1._apps), 1)


class Routes(WithConnectorAuth, V1Case):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(main, "send_push_to_user")
        self.push = p.start()
        self.addCleanup(p.stop)

    def call(self, method, path, sub="admin-1", **kw):
        login = token(sub=sub, client_id=None)   # an app login: no client_id
        return self.client.request(method, path, headers={"Authorization": f"Bearer {login}"}, **kw)

    def test_everything_is_owner_only(self):
        for method, path, body in (("GET", "/settings/connected-apps", None),
                                   ("POST", "/settings/connected-apps", {"client_id": "c9"}),
                                   ("PATCH", f"/settings/connected-apps/{A1}", {"name": "Work"}),
                                   ("DELETE", f"/settings/connected-apps/{A1}", None)):
            r = self.call(method, path, sub="someone-else", json=body)
            self.assertEqual(r.status_code, 403, path)
            self.assertEqual(r.json()["detail"]["error_type"], "feature_unavailable")
        self.assertEqual(self.conn.executed, [])

    def test_a_connector_token_cannot_manage_connections(self):
        r = self.client.get("/settings/connected-apps", headers={"Authorization": f"Bearer {token()}"})
        self.assertEqual(r.status_code, 403)

    def test_list_shows_active_rows(self):
        self.conn.script = [("FROM connected_apps", [ROW])]
        r = self.call("GET", "/settings/connected-apps")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()[0]["name"], "Claude")
        self.assertEqual(r.json()[0]["client_id"], "c1")
        self.assertIn("revoked_at IS NULL", [s for s, _ in self.conn.executed if "FROM connected_apps" in s][0])

    def test_allow_revives_pushes_and_forgets_the_cache(self):
        api_v1._apps[("admin-1", "c1")] = [False, time.time()]
        self.conn.script = [("INSERT INTO connected_apps", [(A1,)])]
        r = self.call("POST", "/settings/connected-apps", json={"client_id": "c1"})
        self.assertEqual(r.status_code, 201)
        sql = [s for s, _ in self.conn.executed if "INSERT INTO connected_apps" in s][0]
        self.assertIn("revoked_at = NULL", sql)
        self.assertIn("WHERE connected_apps.revoked_at IS NOT NULL", sql)
        self.assertNotIn(("admin-1", "c1"), api_v1._apps)
        self.push.assert_called_once_with("admin-1", "Claude connected to NutriScan", "Not you? Disconnect it in Settings.")

    def test_allow_on_an_already_active_connection_sends_no_push(self):
        self.conn.script = [("INSERT INTO connected_apps", [])]   # the DO UPDATE's WHERE matched nothing
        self.assertEqual(self.call("POST", "/settings/connected-apps", json={"client_id": "c1"}).status_code, 201)
        self.push.assert_not_called()

    def test_client_id_bounds(self):
        for bad in ("", "x" * 65):
            self.assertEqual(self.call("POST", "/settings/connected-apps", json={"client_id": bad}).status_code, 422)

    def test_rename_bounds_and_trim(self):
        self.conn.script = [("UPDATE connected_apps SET name", [ROW])]
        self.assertEqual(self.call("PATCH", f"/settings/connected-apps/{A1}", json={"name": "x" * 41}).status_code, 422)
        self.assertEqual(self.call("PATCH", f"/settings/connected-apps/{A1}", json={"name": ""}).status_code, 422)
        r = self.call("PATCH", f"/settings/connected-apps/{A1}", json={"name": "  Work Claude  "})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Work Claude", [p for s, p in self.conn.executed if "SET name" in s][0])

    def test_rename_or_disconnect_of_a_missing_row_is_404(self):
        self.conn.script = []
        for app_id in ("zz", A1):   # not a uuid (no query at all), and a uuid with no row
            self.assertEqual(self.call("PATCH", f"/settings/connected-apps/{app_id}", json={"name": "A"}).status_code, 404)
            self.assertEqual(self.call("DELETE", f"/settings/connected-apps/{app_id}").status_code, 404)
        self.assertEqual(len(self.conn.executed), 2)
        self.assertIn("WHERE id = %s", self.conn.executed[0][0])

    def test_a_malformed_id_does_not_borrow_a_database_connection(self):
        with mock.patch.object(main, "get_db", side_effect=AssertionError("no connection for a 404")):
            self.assertEqual(self.call("DELETE", "/settings/connected-apps/zz").status_code, 404)
            self.assertEqual(self.call("PATCH", "/settings/connected-apps/zz", json={"name": "A"}).status_code, 404)

    def test_disconnect_revokes_and_forgets_the_cache(self):
        api_v1._apps[("admin-1", "c1")] = [True, time.time()]
        self.conn.script = [("SET revoked_at = now()", [("c1",)])]
        r = self.call("DELETE", f"/settings/connected-apps/{A1}")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(("admin-1", "c1"), api_v1._apps)
        self.push.assert_not_called()

    def test_a_disconnected_claude_is_refused_on_its_next_call(self):
        self.conn.script = [("SET revoked_at = now()", [("c1",)])]
        self.call("DELETE", f"/settings/connected-apps/{A1}")
        self.conn.script = [("SELECT revoked_at FROM connected_apps", [(NOW,)])]
        self.assertEqual(self.call_tool({}).status_code, 401)
