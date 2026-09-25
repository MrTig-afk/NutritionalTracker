"""/v1 reads (context, days, detail routes), without a database.

Run:  venv/Scripts/python -m unittest backend.tests.test_api_v1_reads -v
"""
import unittest
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case, token_row


def log_row(log_id, name, servings, nutrition, day="2026-09-24"):
    return (log_id, name, servings, nutrition, day)


def kcal(cal, p=0, label=None, group=None):
    n = {"_kcal": True, "_source": "claude",
         "per_serving": {"size": "100 g", "calories": cal, "protein": f"{p} g", "carbohydrates": "10 g", "fat": "5 g", "fibre": "2 g"}}
    if label:
        n["_meal_label"] = label
    if group:
        n["_meal_group"] = group
    return n


LABEL = {"per_serving": {"size": "40 g", "calories": 1250, "protein": "6g", "carbohydrates": "30g", "fat": "4g"}}   # a kJ label


class Reads(V1Case):
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

    def test_context_groups_by_meal_and_matches_the_tracker(self):
        body = self.get("/v1/context?date=2026-09-24").json()
        self.assertEqual([(b["label"], len(b["entries"])) for b in body["meals"]],
                         [("Meal 1", 2), ("Other", 1), ("Coffee", 1)])
        self.assertEqual(body["by_label"]["Meal 1"]["calories"], 555.0)
        self.conn.script = [("FROM daily_log", [r[:4] for r in self.day_rows])]   # the Tracker's row shape
        with mock.patch.object(main, "get_user_id", lambda a=None, **k: "user-1"):
            tracker = self.client.get("/log?log_date=2026-09-24", headers={"Authorization": "Bearer login"}).json()["totals"]
        for k in ("calories", "protein", "carbs", "fat", "fibre"):
            self.assertAlmostEqual(body["totals"][k if k == "calories" else k + "_g"], tracker[k], delta=0.1)
        self.assertEqual(body["remaining"]["calories"], round(1800 - body["totals"]["calories"], 1))
        self.assertEqual(body["templates"][0]["totals"]["calories"], 490.0)
        self.assertEqual([f["name"] for f in body["usual_foods"]], ["Rice", "Granola"])
        self.assertTrue(all(b["etag"] for b in body["meals"]))

    def test_compact_entries_carry_only_the_documented_fields(self):
        entry = self.get("/v1/context?date=2026-09-24").json()["meals"][0]["entries"][0]
        self.assertEqual(set(entry), {"log_id", "name", "portion", "servings", "calories", "protein_g", "carbs_g",
                                      "fat_g", "fibre_g", "source", "etag"})
        self.assertEqual(entry["source"], "claude")

    def test_typical_context_is_under_10_kb(self):
        self.day_rows[:] = [log_row(f"l{i}", f"Food number {i}", 1.0, kcal(200, 10, f"Meal {i % 3}", f"g{i % 3}"))
                            for i in range(12)]
        self.conn.script[2] = ("FROM meal_templates t LEFT JOIN",
                               [(f"t{t}", f"Template {t}", f"i{t}{j}", f"Ingredient {j}", 1.0, kcal(100, 5))
                                for t in range(5) for j in range(5)])
        self.conn.script[3] = ("DISTINCT ON", [(f"Usual {i}", 1.0, kcal(150, 7)) for i in range(20)])
        with_trends = self.get("/v1/context?date=2026-09-24&include=trends")
        self.assertEqual(with_trends.status_code, 200)
        self.assertLess(len(with_trends.content), 12 * 1024)
        self.assertLess(len(self.get("/v1/context?date=2026-09-24").content), 10 * 1024)

    def test_second_identical_read_does_not_touch_the_database(self):
        self.get("/v1/context?date=2026-09-24")
        n = len(self.conn.executed)
        self.get("/v1/context?date=2026-09-24")
        self.assertEqual(len(self.conn.executed), n)

    def test_a_write_in_the_app_clears_the_cache(self):
        self.get("/v1/context?date=2026-09-24")
        n = len(self.conn.executed)
        with mock.patch.object(main, "get_user_id", lambda a=None, **k: "user-1"), \
             mock.patch.object(main, "claims_if_valid", lambda a: {"sub": "user-1"}):
            self.client.post("/log", headers={"Authorization": "Bearer login"},
                             json={"name": "Apple", "servings": 1, "nutrition": {}, "log_date": "2026-09-24"})
            n_after_write = len(self.conn.executed)
        self.get("/v1/context?date=2026-09-24")
        self.assertGreater(len(self.conn.executed), n_after_write)
        self.assertGreater(n_after_write, n)

    def test_a_read_racing_a_write_is_not_cached(self):
        # P24-CR-3: the write invalidates while the read is still building
        caller = api_v1.Caller("user-1")

        def build():
            api_v1.invalidate("user-1")
            return {"old": True}
        self.assertEqual(api_v1.cached_read(caller, "k", build), {"old": True})
        self.assertNotIn("k", api_v1._read_cache.get("user-1", {}))
        api_v1.cached_read(caller, "k", lambda: {"new": True})   # positive control: no race, cached
        self.assertEqual(api_v1._read_cache["user-1"]["k"][0], {"new": True})

    def test_cache_expires_and_is_bounded(self):
        # P24-D2-1
        caller = api_v1.Caller("user-1")
        api_v1.cached_read(caller, "k", lambda: {"v": 1})
        with mock.patch.object(api_v1.time, "time", lambda: 10**12):   # far past the TTL
            self.assertEqual(api_v1.cached_read(caller, "k", lambda: {"v": 2}), {"v": 2})
        for i in range(api_v1.READ_MAX_KEYS + 5):
            api_v1.cached_read(caller, f"key{i}", lambda: {})
        self.assertLessEqual(len(api_v1._read_cache["user-1"]), api_v1.READ_MAX_KEYS)

    def test_include_junk_does_not_mint_cache_entries(self):
        # P24-SEC-1
        for inc in ("trends", "trends,x", "trends,xx", "x"):
            self.get(f"/v1/context?date=2026-09-24&include={inc}")
        self.assertEqual(sorted(api_v1._read_cache["user-1"]), ["context:2026-09-24:", "context:2026-09-24:trends"])

    def test_reads_bind_rls_to_the_caller(self):
        users = []
        with mock.patch.object(main, "get_db", lambda uid=None: users.append(uid) or self.conn):
            self.get("/v1/context?date=2026-09-24")
        self.assertEqual(set(users) - {None}, {"user-1"})
        self.assertEqual(users.count(None), 1)   # only the token lookup runs before the user is known

    def test_context_needs_all_three_scopes(self):
        self.conn.script[0] = ("resolve_api_token", [token_row(scopes=["log:read", "goals:read"])])
        r = self.get("/v1/context?date=2026-09-24")
        self.assertEqual((r.status_code, r.json()["error_type"]), (403, "insufficient_scope"))

    def test_bad_date(self):
        r = self.get("/v1/context?date=24-09-2026")
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))

    def test_days_range_and_paging(self):
        r = self.get("/v1/days?from=2026-08-01&to=2026-09-24")
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))
        body = self.get("/v1/days?from=2026-09-10&to=2026-09-24&include=entries").json()
        self.assertEqual((body["from"], body["to"], body["next_cursor"]), ("2026-09-10", "2026-09-16", "2026-09-17"))
        body = self.get("/v1/days?from=2026-09-10&to=2026-09-24&include=entries&cursor=2026-09-17").json()
        self.assertEqual((body["to"], body["next_cursor"]), ("2026-09-23", "2026-09-24"))
        body = self.get("/v1/days?from=2026-09-18&to=2026-09-24").json()
        self.assertEqual(len(body["days"]), 7)
        self.assertEqual(body["average"]["days_logged"], 1)
        self.assertIsNone(body["next_cursor"])

    def test_entry_detail_etag_and_404(self):
        self.conn.script.insert(1, ("WHERE log_id = %s AND user_id = %s", [self.day_rows[0]]))
        r = self.get("/v1/entries/l1")
        self.assertEqual(r.headers["ETag"], r.json()["etag"])
        self.assertEqual((r.json()["macros"]["calories"], r.json()["group_id"]), (360, "g1"))
        self.conn.script[1] = ("WHERE log_id = %s AND user_id = %s", [])   # another user's id looks missing
        r = self.get("/v1/entries/theirs")
        self.assertEqual((r.status_code, r.json()["error_type"]), (404, "not_found"))

    def test_meal_detail(self):
        self.conn.script.insert(1, ("_meal_group", self.day_rows[:2]))
        r = self.get("/v1/meals/g1")
        body = r.json()
        self.assertEqual((body["label"], len(body["items"]), body["totals"]["calories"]), ("Meal 1", 2, 555.0))
        self.assertEqual(r.headers["ETag"], body["etag"])

    def test_goals_default_when_never_set(self):
        self.conn.script[1] = ("FROM user_goals", [])
        self.assertEqual(self.get("/v1/goals").json()["calories"], 2000.0)

    def test_template_detail(self):
        self.conn.script.insert(1, ("SELECT name FROM meal_templates", [("Meal 1",)]))
        self.conn.script.insert(2, ("FROM meal_template_items", [("i1", "Firm tofu", 1.0, kcal(360, 42))]))
        r = self.get("/v1/templates/t1")
        self.assertEqual((r.json()["name"], r.json()["totals"]["protein_g"]), ("Meal 1", 42.0))
        self.assertEqual(r.headers["ETag"], r.json()["etag"])

    def test_sodium_in_grams_is_converted(self):
        self.assertEqual(api_v1._full_macros({"per_serving": {"sodium": "0.5 g"}}, 1)["sodium_mg"], 500)
        self.assertEqual(api_v1._full_macros({"per_serving": {"sodium": "120mg"}}, 2)["sodium_mg"], 240)


if __name__ == "__main__":
    unittest.main()
