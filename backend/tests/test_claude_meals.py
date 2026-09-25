"""Everything Claude logs is a named meal, and a logged meal can be renamed in the app (PRD change 2026-09-25,
Artifact v9 lane M).

Run:  venv/Scripts/python -m unittest backend.tests.test_claude_meals -v
No network: FakeConn scripts the DB.
"""
import json
import unittest
from datetime import date
from unittest import mock

import api_v1
from .test_mcp_writes import PreviewBase, result
from .test_api_v1_writes import TODAY
from .test_account_deletion import Case, login
from .test_phase0 import FakeConn, route


class MealTool(PreviewBase):
    def tools(self):
        return {t["name"]: t for t in self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()["result"]["tools"]}

    def previewed(self, args, name="log_meal"):
        is_err, text = result(self.call_tool(args, name=name))
        self.assertFalse(is_err, text)
        return api_v1._pending[json.loads(text)["confirm_code"]][2][0]

    def test_the_date_is_optional_and_never_asked(self):
        t = self.tools()
        schema = t["log_meal"]["inputSchema"]
        self.assertNotIn("date", schema["required"])
        self.assertIn("never ask", schema["properties"]["date"]["description"])
        self.assertIn("never ask", t["get_context"]["inputSchema"]["properties"]["date"]["description"])

    def test_a_missing_date_is_today_in_melbourne(self):
        args = self.banana()
        del args["date"]
        with mock.patch.object(api_v1, "melbourne_today", return_value=date(2026, 9, 20)):
            self.assertEqual(self.previewed(args).date, "2026-09-20")
        self.assertEqual(self.previewed(self.banana()).date, TODAY)   # positive control: a given date is kept

    def test_an_edit_keeps_its_own_date(self):
        # only a change that needs a date gets today: filling one in would move the entry
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        change = self.previewed({"log_id": "l1", "if_match": api_v1.etag_of(*row[1:]), "servings": 2}, name="edit_entry")
        self.assertIsNone(change.date)

    def test_meal_names_up_to_80_characters(self):
        self.assertEqual(self.previewed(self.banana(label="x" * 80)).label, "x" * 80)
        is_err, text = result(self.call_tool(self.banana(label="x" * 81), name="log_meal"))
        self.assertTrue(is_err)
        self.assertTrue(text.startswith("label:"), text)

    def test_the_meal_rule_is_in_the_description(self):
        d = self.tools()["log_meal"]["description"]
        for phrase in ("a single food is a meal of one item", "Say yes, or tell me a different name",
                       "Never ask for the name or the date"):
            self.assertIn(phrase, d)


class Rename(unittest.TestCase):
    H = {"Authorization": "Bearer x"}

    def test_renames_this_users_meal_rows_only(self):
        conn = FakeConn([("UPDATE daily_log", [("l1",), ("l2",)])])
        r = route(self, conn).patch("/log/meals/g1", headers=self.H, json={"label": "Tomato gnocchi night"})
        self.assertEqual((r.status_code, r.json()), (200, {"group_id": "g1", "label": "Tomato gnocchi night"}))
        sql, params = next((s, p) for s, p in conn.executed if "UPDATE daily_log" in s)
        self.assertIn("jsonb_set(nutrition, '{_meal_label}'", sql)
        self.assertIn("WHERE user_id = %s AND nutrition->>'_meal_group' = %s", sql)
        self.assertEqual(params, ["Tomato gnocchi night", "user-1", "g1"])
        self.assertEqual(conn.commits, 1)
        self.assertFalse([s for s, _ in conn.executed if "meal_templates" in s])   # the template keeps its name

    def test_unknown_meal_is_not_found(self):
        r = route(self, FakeConn([])).patch("/log/meals/nope", headers=self.H, json={"label": "Dinner"})
        self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (404, "not_found"))

    def test_names_are_checked_before_the_database(self):
        conn = FakeConn([("UPDATE daily_log", [("l1",)])])
        client = route(self, conn)
        for body in ({"label": "x" * 81}, {"label": " \x00 "}, {"label": "Dinner", "x": 1}, {}):
            self.assertEqual(client.patch("/log/meals/g1", headers=self.H, json=body).status_code, 422, body)
        self.assertEqual(conn.executed, [])
        self.assertEqual(client.patch("/log/meals/g1", headers=self.H, json={"label": "x" * 80}).status_code, 200)


class RenameWhilePending(Case):
    def test_refused_while_the_account_is_being_deleted(self):
        self.pending()
        r = self.client.patch("/log/meals/g1", headers=login(), json={"label": "Dinner"})
        self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (423, "account_scheduled_for_deletion"))
        self.assertFalse([s for s, _ in self.conn.executed if "daily_log" in s])
        api_v1.m._deleting.clear()   # positive control: the same request reaches the database once nothing is pending
        self.client.patch("/log/meals/g1", headers=login(), json={"label": "Dinner"})
        self.assertTrue([s for s, _ in self.conn.executed if "UPDATE daily_log" in s])


if __name__ == "__main__":
    unittest.main()
