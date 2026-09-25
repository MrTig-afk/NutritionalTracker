"""One Library for the app and Claude: Claude reads it (library:read, /v1/library, search_library), and
"Ask before saving new foods to my Library" (on by default; the only setting Claude may change, and only off).

Run:  venv/Scripts/python -m unittest backend.tests.test_claude_library -v
No network: FakeConn scripts the DB.
"""
import json
import unittest
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case, token_row
from .test_mcp import WithConnectorAuth
from .test_phase0 import FakeConn, route

MILK = {"_kcal": True, "per_serving": {"size": "250 mL", "calories": 125, "protein": "9.3 g",
                                       "carbohydrates": "14.3 g", "fat": "3.3 g", "fibre": "0 g"}}
OATS = {"_kcal": True, "per_100g": {"calories": 380, "protein": "13 g"}}
LIB = [("i1", "Dairy", "Sungold Milk", MILK), ("i2", "From Claude", "Oat milk", OATS)]


def tool_text(r):
    body = r.json()["result"]
    return body["isError"], body["content"][0]["text"]


class Scope(unittest.TestCase):
    def test_claude_may_read_the_library(self):
        self.assertIn("library:read", api_v1.SCOPES)
        self.assertIn("library:read", api_v1.CONNECTOR_SCOPES)


class V1Library(V1Case):
    def test_matches_with_per_serving_numbers_and_the_setting(self):
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", LIB),
                            ("FROM notification_prefs", [])]
        r = self.get("/v1/library?q=milk")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["items"][0], {"item_id": "i1", "name": "Sungold Milk", "folder": "Dairy", "portion": "250 mL",
                                            "per_serving": {"calories": 125, "protein_g": 9.3, "carbs_g": 14.3,
                                                            "fat_g": 3.3, "fibre_g": 0}})
        self.assertEqual(body["items"][1]["portion"], "100 g")
        self.assertEqual((body["truncated"], body["ask_before_saving"]), (False, True))   # on by default

    def test_the_query_is_matched_literally(self):
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", [])]
        self.get("/v1/library?q=50%_x")
        params = [p for s, p in self.conn.executed if "FROM folder_items" in s][0]
        self.assertEqual(params, ["user-1", "%50\\%\\_x%"])

    def test_folders_are_matched_to_the_owner_too(self):
        # CL-10: defence in depth beside RLS, as every other folder query does
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", [])]
        self.get("/v1/library")
        sql = [s for s, _ in self.conn.executed if "FROM folder_items" in s][0]
        self.assertIn("f.user_id = i.user_id", sql)

    def test_control_characters_and_spaces_are_dropped_from_the_query(self):
        # CL-6: a NUL must not reach the database (a 500)
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", [])]
        self.assertEqual(self.get("/v1/library?q=%20mi%00lk%20").status_code, 200)
        params = [p for s, p in self.conn.executed if "FROM folder_items" in s][0]
        self.assertEqual(params[1], "%milk%")

    def test_capped_and_says_so(self):
        rows = [(f"i{n}", "F", f"Food {n}", MILK) for n in range(api_v1.LIBRARY_LIMIT + 1)]
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", rows)]
        body = self.get("/v1/library").json()
        self.assertEqual((len(body["items"]), body["truncated"]), (api_v1.LIBRARY_LIMIT, True))
        sql = [s for s, _ in self.conn.executed if "FROM folder_items" in s][0]
        self.assertIn(f"LIMIT {api_v1.LIBRARY_LIMIT + 1}", sql)

    def test_needs_the_scope(self):
        self.conn.script = [("resolve_api_token", [token_row(scopes=["log:read"])])]
        r = self.get("/v1/library")
        self.assertEqual((r.status_code, r.json()["error_type"]), (403, "insufficient_scope"))

    def test_the_setting_off(self):
        self.conn.script = [("resolve_api_token", [token_row()]), ("FROM folder_items", []),
                            ("FROM notification_prefs", [("false",)])]
        self.assertFalse(self.get("/v1/library").json()["ask_before_saving"])


class Connector(WithConnectorAuth, V1Case):
    def test_the_tools_are_listed(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        self.assertTrue(tools["search_library"]["annotations"]["readOnlyHint"])
        stop = tools["stop_asking_before_saving"]
        self.assertEqual(stop["inputSchema"]["properties"], {})   # nothing to pass: it can only turn asking off
        self.assertFalse(stop["annotations"]["destructiveHint"])
        for name in ("log_food", "log_meal", "log_template", "save_template", "update_template"):
            d = tools[name]["description"]   # CL-3: the save question rides in the one preview, never a second prompt
            self.assertIn("ask_before_saving", d, name)
            self.assertIn("same preview", d, name)
            self.assertIn("preview again without", d, name)   # CL2-1: "log it, not the Library" has a path
            self.assertNotIn("save it without asking", d, name)   # CL2-4: the yes before confirm_change always stands
        own = tools["save_to_library"]["description"]   # CL2-2: its own rule, no field it does not have
        self.assertNotIn("same preview", own)
        self.assertIn("asked to save", own)
        self.assertIn("one key word", tools["search_library"]["description"])   # CL2-10
        for name in ("edit_entry", "delete_entry", "delete_meal"):   # CL-7: cannot save to the Library
            self.assertNotIn("ask_before_saving", tools[name]["description"], name)

    def test_search_library(self):
        self.conn.script = [("FROM folder_items", LIB[:1]), ("FROM notification_prefs", [])]
        is_err, text = tool_text(self.call_tool({"query": "sungold"}, name="search_library"))
        self.assertFalse(is_err)
        body = json.loads(text)
        self.assertEqual((body["items"][0]["name"], body["ask_before_saving"]), ("Sungold Milk", True))
        params = [p for s, p in self.conn.executed if "FROM folder_items" in s][0]
        self.assertEqual(params, ["admin-1", "%sungold%"])

    def test_stop_asking_takes_no_arguments(self):
        # CL-5: a call meant to turn asking back on must not quietly turn it off
        is_err, text = tool_text(self.call_tool({"ask_before_saving": True}, name="stop_asking_before_saving"))
        self.assertTrue(is_err)
        self.assertFalse([s for s, _ in self.conn.executed if "notification_prefs" in s])

    def test_arguments_must_be_an_object(self):
        # CL2-8: [true] meant "turn it on" must not become {} and turn it off
        r = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "stop_asking_before_saving", "arguments": [True]}})
        self.assertEqual(r.json()["error"]["code"], -32602)
        self.assertFalse([s for s, _ in self.conn.executed if "notification_prefs" in s])
        is_err, _ = tool_text(self.call_tool({"query": 0}, name="search_library"))
        self.assertTrue(is_err)

    def test_the_tools_check_their_scopes(self):
        # CL-4: safe today only because the connector holds every scope
        with mock.patch.object(api_v1, "CONNECTOR_SCOPES", ("log:read",)):
            for name, args in (("search_library", {}), ("stop_asking_before_saving", {})):
                is_err, text = tool_text(self.call_tool(args, name=name))
                self.assertTrue(is_err, name)
                self.assertIn("lacks", text, name)
        self.assertFalse([s for s, _ in self.conn.executed if "folder_items" in s or "notification_prefs" in s])

    def test_stop_asking_turns_it_off_and_only_off(self):
        self.conn.script = [("INSERT INTO notification_prefs", [])]
        is_err, text = tool_text(self.call_tool({}, name="stop_asking_before_saving"))
        self.assertFalse(is_err)
        self.assertEqual(json.loads(text), {"ask_before_saving": False})
        writes = [p for s, p in self.conn.executed if "INSERT INTO notification_prefs" in s]
        self.assertEqual(writes, [["admin-1", json.dumps({"ask_before_saving_foods": False})]])
        audit = [p for s, p in self.conn.executed if "INSERT INTO api_audit" in s]   # CL2-7: like every Claude write
        self.assertEqual(len(audit), 1)
        self.assertIn("ask_before_saving_foods", audit[0][-1])
        self.assertEqual(self.conn.commits, 1)


class OtherSettingsKeepIt(unittest.TestCase):
    def test_saving_notification_settings_keeps_the_library_setting(self):
        # CL-1: that upsert replaces the prefs and carried over only the energy unit
        conn = FakeConn([])
        client = route(self, conn)
        client.put("/settings/notifications", headers={"Authorization": "Bearer x"}, json={"prefs": {"enabled": True}})
        sql = [s for s, _ in conn.executed if "INSERT INTO notification_prefs" in s][0]
        self.assertIn("'ask_before_saving_foods', notification_prefs.prefs->'ask_before_saving_foods'", sql)

    def test_the_energy_unit_uses_the_one_merge(self):
        # CL-9
        conn = FakeConn([])
        route(self, conn).put("/settings/energy-unit", headers={"Authorization": "Bearer x"}, json={"unit": "kJ"})
        sql = [s for s, _ in conn.executed if "INSERT INTO notification_prefs" in s][0]
        self.assertEqual(sql, " ".join(api_v1.PREFS_MERGE_SQL.split()))


class AppSetting(unittest.TestCase):
    def test_on_by_default_and_the_app_can_turn_it_back_on(self):
        conn = FakeConn([("FROM notification_prefs", [])])
        client = route(self, conn)
        h = {"Authorization": "Bearer x"}
        self.assertEqual(client.get("/settings/library", headers=h).json(), {"ask_before_saving": True})
        conn.script = [("INSERT INTO notification_prefs", [])]
        self.assertEqual(client.put("/settings/library", headers=h, json={"ask_before_saving": True}).json(),
                         {"ask_before_saving": True})
        self.assertEqual([p[1] for s, p in conn.executed if "INSERT INTO notification_prefs" in s],
                         [json.dumps({"ask_before_saving_foods": True})])
        self.assertEqual(client.put("/settings/library", headers=h, json={"ask_before_saving": "yes", "x": 1}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
