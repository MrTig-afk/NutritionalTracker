"""Every calorie is stored in kcal (PRD "Change 2026-09-24", Phase 0b).

Run:  venv/Scripts/python -m unittest backend.tests.test_kcal_storage -v
No database: the app's write routes run against FakeConn from test_phase0.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import main  # noqa: E402
import log_service  # noqa: E402
from .test_phase0 import FakeConn, route  # noqa: E402

TYPED_950 = {"per_serving": {"size": "1 plate", "calories": 950, "protein": "40g"}}


def stored(conn, table):
    """The nutrition JSON written by the one INSERT/UPDATE into `table`."""
    sql, params = next((s, p) for s, p in conn.executed if table in s and ("INSERT" in s or "UPDATE" in s))
    return json.loads(next(p for p in params if isinstance(p, str) and p.startswith("{")))


class NewWritesAreKcal(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(main, "_check_goal_and_push", lambda *a: None)
        p.start()
        self.addCleanup(p.stop)

    def test_a_typed_950_stays_950(self):
        conn = FakeConn()
        r = route(self, conn).post("/log", json={"name": "Big meal", "servings": 1, "nutrition": TYPED_950})
        self.assertEqual(r.status_code, 200, r.text)
        n = stored(conn, "daily_log")
        self.assertIs(n["_kcal"], True)
        self.assertEqual(log_service.entry_macros(n, 1)["calories"], 950)   # was read as 227 before

    def test_a_value_labelled_kj_is_converted(self):
        conn = FakeConn()
        route(self, conn).post("/log", json={"name": "Label", "servings": 1,
                                             "nutrition": {"per_serving": {"calories": "1500 kJ"}}})
        self.assertEqual(stored(conn, "daily_log")["per_serving"]["calories"], 358.5)

    def test_edit_library_and_template_items_are_tagged(self):
        conn = FakeConn([("SELECT log_id FROM daily_log", [("l1",)]),
                         ("SELECT template_id FROM meal_templates", [("t1",)])])
        client = route(self, conn)
        client.put("/log/l1", json={"name": "Big meal", "servings": 1, "nutrition": TYPED_950})
        client.post("/folders/f1/items", json={"name": "Big meal", "image_id": "", "nutrition": TYPED_950})
        client.post("/meal-templates/t1/items", json={"name": "Big meal", "nutrition": TYPED_950})
        for table in ("UPDATE daily_log", "folder_items", "meal_template_items"):
            n = stored(conn, table)
            self.assertEqual((n["_kcal"], n["per_serving"]["calories"]), (True, 950), table)

    def test_logging_a_legacy_template_settles_its_items(self):
        # An untagged template item predates kcal storage: its 1500 was always shown as kJ.
        conn = FakeConn([("SELECT name FROM meal_templates", [("Breakfast",)]),
                         ("FROM meal_template_items", [("Oats", {"per_serving": {"calories": 1500}}, 1.0),
                                                       ("Eggs", {"_kcal": True, "per_serving": {"calories": 950}}, 1.0)])])
        r = route(self, conn).post("/meal-templates/t1/log?log_date=2026-09-24")
        self.assertEqual(r.json(), {"logged": 2})
        rows = [json.loads(p[5]) for s, p in conn.executed if "INSERT INTO daily_log" in s]
        self.assertEqual([(n["_kcal"], n["per_serving"]["calories"], n["_meal_label"]) for n in rows],
                         [(True, 358.5, "Breakfast"), (True, 950, "Breakfast")])


class Hardening(unittest.TestCase):
    """CodeRabbit on PR #37."""

    def test_a_client_kcal_tag_does_not_skip_settling(self):
        conn = FakeConn()
        with mock.patch.object(main, "_check_goal_and_push", lambda *a: None):
            route(self, conn).post("/log", json={"name": "x", "servings": 1,
                                                 "nutrition": {"_kcal": True, "per_serving": {"calories": "1500 kJ"}}})
        self.assertEqual(stored(conn, "daily_log")["per_serving"]["calories"], 358.5)

    def test_migration_preview_survives_a_malformed_section(self):
        import migrate_kcal
        self.assertEqual(migrate_kcal._cals({"per_serving": "junk", "per_100g": {"calories": 600}}), [None, 600])

    def test_migration_settles_a_string_kcal_tag(self):
        # Pre-kcal code stored client JSON as sent; "true" as a string is not our boolean tag.
        import migrate_kcal
        row = ("l1", "2026-09-01", "Pie", {"_kcal": "true", "per_serving": {"calories": 1500}})
        conn = FakeConn([("FROM daily_log", [row])])
        out = migrate_kcal.plan(conn.cursor(), set())
        self.assertIn("IS DISTINCT FROM 'true'::jsonb", conn.executed[0][0])
        self.assertEqual([(t, new["_kcal"], after) for t, _, _, _, _, after, new in out], [("daily_log", True, [358.5, None])])


if __name__ == "__main__":
    unittest.main()
