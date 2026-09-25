"""Owner-only Admin panel routes and the per-account energy unit (PRD changes 2026-09-24).

Run:  venv/Scripts/python -m unittest backend.tests.test_admin_settings -v
No database: FakeConn from test_phase0. The merge SQL was checked against a
real Postgres branch when written; here the tests pin what the routes send.
"""
import unittest
from unittest import mock

import main
from . import ALERTS
from .test_api_v1 import V1Case


class Admin(V1Case):
    def login(self, user="admin-1"):
        for p in (mock.patch.object(main, "get_user_id", lambda a=None, **k: user),
                  mock.patch.object(main, "ADMIN_USER_ID", "admin-1")):
            p.start()
            self.addCleanup(p.stop)

    def test_other_accounts_get_feature_unavailable(self):
        self.login(user="someone-else")
        h = {"Authorization": "Bearer login"}
        for r in (self.client.get("/settings/admin/health", headers=h), self.client.post("/settings/admin/test-alert", headers=h)):
            self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (403, "feature_unavailable"))

    def test_health_reads_memory_only(self):
        self.login()
        main._budget["seconds"] = 0.5 * 100 / 0.25 * 3600   # half the free month
        body = self.client.get("/settings/admin/health", headers={"Authorization": "Bearer login"}).json()
        self.assertEqual((body["neon_budget_percent"], body["api_paused"], body["requests_left_today"]), (50.0, False, 200))
        self.assertEqual(self.conn.executed, [])   # never wakes Neon

    def test_test_alert_uses_the_admin_push(self):
        self.login()
        ALERTS.clear()
        r = self.client.post("/settings/admin/test-alert", headers={"Authorization": "Bearer login"})
        self.assertEqual(r.json(), {"sent": True})
        self.assertEqual(ALERTS, [("Test alert", "NutriScan admin alerts reach this device.")])
        self.assertEqual(self.conn.executed, [])


class EnergyUnit(V1Case):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(main, "get_user_id", lambda a=None, **k: "user-1")
        p.start()
        self.addCleanup(p.stop)
        self.h = {"Authorization": "Bearer login"}

    def test_default_is_kcal(self):
        self.conn.script = []
        self.assertEqual(self.client.get("/settings/energy-unit", headers=self.h).json(), {"unit": "kcal"})
        self.conn.script = [("prefs->>'energy_unit'", [("kJ",)])]
        self.assertEqual(self.client.get("/settings/energy-unit", headers=self.h).json(), {"unit": "kJ"})

    def test_set_merges_into_the_prefs_row(self):
        r = self.client.put("/settings/energy-unit", headers=self.h, json={"unit": "kJ"})
        self.assertEqual(r.json(), {"unit": "kJ"})
        sql, params = next((s, p) for s, p in self.conn.executed if "INSERT INTO notification_prefs" in s)
        self.assertIn("|| EXCLUDED.prefs", sql)
        self.assertEqual(params[1], '{"energy_unit": "kJ"}')

    def test_only_kcal_or_kj(self):
        for body in ({"unit": "cal"}, {"unit": "KJ"}, {"unit": "kJ", "extra": 1}, {}):
            self.assertEqual(self.client.put("/settings/energy-unit", headers=self.h, json=body).status_code, 422, body)

    def test_saving_notifications_replaces_only_the_reminder_keys(self):
        # every other setting in the row survives, including ones added later (CL2-6)
        with mock.patch.object(main, "_mark_schedule_dirty", lambda: None):
            self.client.put("/settings/notifications", headers=self.h, json={"prefs": {"meal_morning": True}})
        sql, params = next((s, p) for s, p in self.conn.executed if "INSERT INTO notification_prefs" in s)
        # COALESCE: a NULL prefs row must not swallow the new reminders (NP-1)
        self.assertIn("(COALESCE(notification_prefs.prefs, '{}'::jsonb) - %s::text[]) || EXCLUDED.prefs", sql)
        self.assertEqual(set(params[3]), set(main.NOTIF_PREF_KEYS) | set(main.NOTIF_TIME_DEFAULTS))


if __name__ == "__main__":
    unittest.main()
