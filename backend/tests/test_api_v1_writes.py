"""/v1 writes (the change engine and its routes), without a database.

Run:  venv/Scripts/python -m unittest backend.tests.test_api_v1_writes -v
FakeConn records every statement and every commit, so "nothing saved" is
checked as "no commit happened" and "the template is unchanged" as "no
statement touched meal_templates except to read it".
"""
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case, TOKEN, token_row
from .test_phase0 import ImmediateThread


def kcal_n(cal, p, size, **tags):
    return {"per_serving": {"size": size, "calories": cal, "protein": f"{p} g", "carbohydrates": "10 g",
                            "fat": "1 g", "fibre": "1 g"}, "_kcal": True, **tags}


MACROS = {"calories": 260, "protein_g": 5.4, "carbs_g": 56, "fat_g": 0.6, "fibre_g": 0.8}
TODAY = datetime.now(timezone.utc).date().isoformat()


class Writes(V1Case):
    def setUp(self):
        super().setUp()
        self.idem = {}   # key -> (hash, status, response): a tiny api_idempotency
        self.template_rows = [("i-tofu", "Firm tofu", 1.0, kcal_n(360, 42, "300 g")),
                              ("i-rice", "Rice", 1.0, kcal_n(195, 4, "150 g"))]
        self.conn.script = [
            ("resolve_api_token", [token_row()]),
            ("INSERT INTO api_idempotency", self._idem_insert),
            ("SELECT request_hash, status_code, response FROM api_idempotency", self._idem_select),
            ("UPDATE api_idempotency", self._idem_update),
            ("SELECT name FROM meal_templates", [("Meal 1",)]),
            ("FROM meal_template_items", self.template_rows),
            ("FROM user_goals", [(1800, 120, 200, 60, 30)]),
            ("SELECT lower(name) FROM folder_items", [("firm tofu",)]),
        ]
        p = mock.patch.object(main, "_check_goal_and_push", lambda *a: None)
        p.start()
        self.addCleanup(p.stop)
        self.get("/v1/me")   # resolve the token now: its lookup commits (the last-used stamp)
        self.conn.commits = 0
        self.conn.executed.clear()

    def _idem_insert(self, sql, params):
        uid, key, rhash = params
        if key in self.idem:
            return []
        self.idem[key] = (rhash, 0, "{}")
        return [(1,)]

    def _idem_select(self, sql, params):
        return [self.idem[params[1]]]

    def _idem_update(self, sql, params):
        status, response, uid, key = params
        self.idem[key] = (self.idem[key][0], status, response)
        return []

    def post(self, path, body, key="k1", preview=False, **headers):
        h = {"Authorization": f"Bearer {TOKEN}", **headers}
        if key:
            h["Idempotency-Key"] = key
        return self.client.post(path + ("?preview=true" if preview else ""), headers=h, json=body)

    def statements(self, prefix):
        return [(s, p) for s, p in self.conn.executed if s.startswith(prefix)]

    # ---- the PRD's done-when for Phase 3
    def test_meal_1_but_200g_rice(self):
        body = {"date": TODAY, "changes": [{"item_id": "i-rice", "portion": "200 g", "macros": MACROS}]}
        r = self.post("/v1/templates/t1/log", body, preview=True)
        self.assertEqual(r.status_code, 200)
        out = r.json()
        rice = [d for d in out["diff"] if d["name"] == "Rice"][0]
        self.assertEqual(rice["changes_from_template"], {"portion": ["150 g", "200 g"]})
        self.assertEqual(rice["vs_template_kcal_change"], 65.0)
        self.assertTrue(out["preview"])
        self.assertEqual(self.conn.commits, 0)   # a preview saves nothing
        # then the real write: a grouped "via Claude" meal, template untouched
        self.conn.executed.clear()
        r = self.post("/v1/templates/t1/log", body)
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.headers["Location"], f"/v1/meals/{r.json()['meal']['group_id']}")
        inserts = self.statements("INSERT INTO daily_log")
        self.assertEqual(len(inserts), 2)
        tags = [json.loads(p[5]) for _, p in inserts]
        self.assertEqual({t["_meal_group"] for t in tags}, {r.json()["meal"]["group_id"]})
        self.assertEqual({t["_meal_label"] for t in tags}, {"Meal 1"})
        self.assertEqual({t["_source"] for t in tags}, {"claude"})
        self.assertEqual(tags[1]["per_serving"]["size"], "200 g")
        touched = [s for s, _ in self.conn.executed if "meal_templates" in s or "meal_template_items" in s]
        self.assertTrue(all(s.startswith("SELECT") for s in touched))   # the template itself is unchanged
        self.assertEqual(r.json()["day_totals_after"]["calories"], 0)   # the fake day query returns nothing
        self.assertEqual(self.conn.commits, 1)

    def test_a_failing_change_saves_nothing(self):
        body = {"changes": [
            {"type": "log_entry", "date": TODAY, "name": "Apple", "macros": MACROS},
            {"type": "delete_entry", "log_id": "missing", "if_match": '"x"'},
        ]}
        r = self.post("/v1/batch", body)
        self.assertEqual((r.status_code, r.json()["error_type"], r.json()["change_index"]), (404, "not_found", 1))
        self.assertTrue(r.json()["detail"].startswith("changes[1]:"))
        self.assertEqual(self.conn.commits, 0)
        self.assertFalse(self.statements("INSERT INTO api_audit"))

    def test_same_key_twice_gives_one_set_of_rows(self):
        body = {"date": TODAY, "name": "Apple", "macros": MACROS}
        first = self.post("/v1/entries", body)
        self.assertEqual(first.status_code, 201)
        second = self.post("/v1/entries", body)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.json()["entry"]["log_id"], first.json()["entry"]["log_id"])
        self.assertEqual(len(self.statements("INSERT INTO daily_log")), 1)

    def test_same_key_different_body_is_409(self):
        self.post("/v1/entries", {"date": TODAY, "name": "Apple", "macros": MACROS})
        r = self.post("/v1/entries", {"date": TODAY, "name": "Pear", "macros": MACROS})
        self.assertEqual((r.status_code, r.json()["error_type"]), (409, "idempotency_conflict"))

    def test_real_post_needs_a_key_but_a_preview_does_not(self):
        body = {"date": TODAY, "name": "Apple", "macros": MACROS}
        r = self.post("/v1/entries", body, key=None)
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))
        self.assertEqual(self.post("/v1/entries", body, key=None, preview=True).status_code, 200)

    def test_existing_library_name_is_skipped(self):
        body = {"date": TODAY, "label": "Meal 1", "items": [
            {"name": "Firm Tofu", "macros": MACROS, "save_to_library": True},
            {"name": "Nutritional yeast", "macros": MACROS, "save_to_library": True}]}
        out = self.post("/v1/meals", body).json()
        self.assertEqual(out["library"], {"saved": ["Nutritional yeast"], "skipped_existing": ["Firm Tofu"]})
        folder = self.statements("INSERT INTO folders")
        self.assertEqual(folder[0][1][2], "From Claude")
        item = self.statements("INSERT INTO folder_items")
        self.assertEqual(len(item), 1)
        self.assertIn("VALUES (%s, %s, %s, NULL,", item[0][0])   # no image: folder_items.image_id is NULL

    def test_library_needs_its_scope(self):
        self.conn.script[0] = ("resolve_api_token", [token_row(scopes=["log:write"])])
        api_v1._token_cache.clear()
        r = self.post("/v1/entries", {"date": TODAY, "name": "Apple", "macros": MACROS, "save_to_library": True})
        self.assertEqual((r.status_code, r.json()["error_type"]), (403, "insufficient_scope"))
        self.assertEqual(self.post("/v1/entries", {"date": TODAY, "name": "Apple", "macros": MACROS}).status_code, 201)

    def test_write_marks_kcal_so_950_stays_950(self):
        out = self.post("/v1/entries", {"date": TODAY, "name": "Big bowl", "macros": {**MACROS, "calories": 950}}).json()
        self.assertEqual(out["entry"]["contribution"]["calories"], 950)
        n = json.loads(self.statements("INSERT INTO daily_log")[0][1][5])
        self.assertIs(n["_kcal"], True)
        self.assertEqual(n["_token_id"], "tok-1")

    # ---- edits and deletes
    def entry_row(self, name="Rice", servings=1.0):
        return ("l1", name, servings, kcal_n(195, 4, "150 g", _meal_group="g1", _meal_label="Meal 1"), TODAY)

    def test_update_with_a_stale_etag_is_412(self):
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [self.entry_row()]))
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": '"stale"'},
                              json={"servings": 2})
        self.assertEqual((r.status_code, r.json()["error_type"]), (412, "precondition_failed"))
        self.assertEqual(self.conn.commits, 0)

    def test_update_keeps_the_meal_tags_and_returns_before_and_after(self):
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        etag = api_v1.etag_of(row[1], row[2], row[3], row[4])
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": etag},
                              json={"portion": "200 g", "macros": MACROS})
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        self.assertEqual((out["before"]["portion"], out["after"]["portion"]), ("150 g", "200 g"))
        n = json.loads(self.statements("UPDATE daily_log")[0][1][2])
        self.assertEqual((n["_meal_group"], n["_meal_label"], n["_kcal"]), ("g1", "Meal 1", True))
        self.assertEqual(out["diff"][0]["kcal_change"], 65.0)

    def test_weak_etag_from_the_proxy_still_matches(self):
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        etag = "W/" + api_v1.etag_of(*row[1:])   # what Cloudflare hands back after compressing
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": etag},
                              json={"servings": 2})
        self.assertEqual(r.status_code, 200, r.text)

    def test_new_portion_needs_new_macros(self):
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        etag = api_v1.etag_of(*row[1:])
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": etag},
                              json={"portion": "200 g"})
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))

    def test_patch_needs_if_match(self):
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}"}, json={"servings": 2})
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))

    def test_delete_whole_meal_or_one_ingredient(self):
        rows = [self.entry_row(), ("l2", "Tofu", 1.0, kcal_n(360, 42, "300 g", _meal_group="g1"), TODAY)]
        self.conn.script.insert(1, ("nutrition->>'_meal_group' = %s ORDER BY", rows))
        meal_etag = api_v1.etag_of([api_v1.entry_detail(*r)["etag"] for r in rows])
        r = self.client.delete("/v1/meals/g1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": meal_etag})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["before"]["items"]), 2)
        self.assertEqual(len(self.statements("DELETE FROM daily_log")), 1)
        # one ingredient
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [rows[0]]))
        r = self.client.delete("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}",
                                                          "If-Match": api_v1.etag_of(*rows[0][1:])})
        self.assertEqual(r.status_code, 200, r.text)

    def test_deletes_count_towards_the_spree_freeze(self):
        frozen = []
        rows = [self.entry_row()]
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", rows))
        etag = api_v1.etag_of(*rows[0][1:])
        change = {"type": "delete_entry", "log_id": "l1", "if_match": etag}
        with mock.patch.object(api_v1, "take_request", lambda *a, **k: {}), \
             mock.patch.object(main, "freeze_user", lambda uid, why: frozen.append(uid)), \
             mock.patch.object(main, "_event_windows", {}):
            for i in range(6):
                self.post("/v1/batch", {"changes": [change] * 10}, key=f"k{i}")
            self.assertEqual(frozen, ["user-1"])   # the 60th delete in 10 minutes
            # and a DELETE route is counted by abuse_guard through the token cache
            main._event_windows.clear()
            frozen.clear()
            for _ in range(60):
                self.client.delete("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": etag})
            self.assertEqual(frozen, ["user-1"])

    # ---- templates
    def test_create_and_edit_template(self):
        created = {}

        def items_after(sql, params):
            replaced = any(s.startswith("DELETE FROM meal_template_items") for s, _ in self.conn.executed)
            return created["rows"] if replaced else self.template_rows
        self.conn.script[5] = ("FROM meal_template_items", items_after)
        r = self.post("/v1/templates", {"name": "Meal 3", "items": [{"name": "Dal", "portion": "250 g", "macros": MACROS}]})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertTrue(r.headers["Location"].startswith("/v1/templates/"))
        before = api_v1.etag_of("Meal 1", [(i[1], i[2], i[3]) for i in self.template_rows])
        created["rows"] = self.template_rows + [("i-peas", "Peas", 1.0, kcal_n(60, 4, "80 g"))]
        r = self.client.patch("/v1/templates/t1", headers={"Authorization": f"Bearer {TOKEN}", "If-Match": before},
                              json={"items": [{"name": "Firm tofu", "portion": "300 g", "macros": MACROS},
                                              {"name": "Rice", "portion": "150 g", "macros": MACROS},
                                              {"name": "Peas", "portion": "80 g", "macros": MACROS}]})
        self.assertEqual(r.status_code, 200, r.text)
        ops = {(d["op"], d["name"]) for d in r.json()["diff"]}
        self.assertIn(("add", "Peas"), ops)

    # ---- validation
    def test_future_dates_and_batch_size_and_unknown_fields(self):
        far = (datetime.now(timezone.utc).date() + timedelta(days=3)).isoformat()
        r = self.post("/v1/entries", {"date": far, "name": "Apple", "macros": MACROS})
        self.assertEqual((r.status_code, r.json()["errors"][0]["field"]), (422, "date"))
        tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()   # Melbourne may be there already
        self.assertEqual(self.post("/v1/entries", {"date": tomorrow, "name": "Apple", "macros": MACROS}, key="k9").status_code, 201)
        change = {"type": "log_entry", "date": TODAY, "name": "Apple", "macros": MACROS}
        r = self.post("/v1/batch", {"changes": [change] * 11})
        self.assertEqual((r.status_code, r.json()["errors"][0]["field"]), (422, "changes"))
        r = self.post("/v1/entries", {"date": TODAY, "name": "Apple", "macros": MACROS, "user_id": "someone-else"})
        self.assertEqual(r.status_code, 422)
        r = self.post("/v1/batch", {"changes": [{**change, "macros": {**MACROS, "calories": 5001}}]})
        self.assertEqual(r.json()["errors"][0]["field"], "changes.0.log_entry.macros.calories")
        r = self.post("/v1/entries", {"date": TODAY, "name": "\x00\x07", "macros": MACROS})
        self.assertEqual(r.status_code, 422)

    def test_meal_item_cap(self):
        items = [{"name": f"f{i}", "macros": MACROS} for i in range(26)]
        r = self.post("/v1/meals", {"date": TODAY, "label": "Meal 1", "items": items})
        self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"))

    # ---- cycle 3 review findings, each pinned
    def test_overlong_if_match_or_id_is_422_not_500(self):
        # P24-CR-1: the limits live on the change model, built inside the handler
        h = {"Authorization": f"Bearer {TOKEN}", "If-Match": '"' + "x" * 80 + '"'}
        for method, path in (("delete", "/v1/entries/l1"), ("delete", "/v1/meals/g1"), ("delete", "/v1/entries/" + "y" * 70)):
            r = getattr(self.client, method)(path, headers=h)
            self.assertEqual((r.status_code, r.json()["error_type"]), (422, "validation_error"), path)
        r = self.client.patch("/v1/templates/t1", headers=h, json={"name": "x"})
        self.assertEqual(r.status_code, 422)

    def test_goal_push_only_after_logging(self):
        # P24-CR-2
        pushed = []
        rows = [("l1", "Rice", 1.0, kcal_n(195, 4, "150 g"), TODAY)]
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", rows))
        with mock.patch.object(main, "_check_goal_and_push", lambda uid, d: pushed.append(d)), \
             mock.patch.object(api_v1.threading, "Thread", ImmediateThread):
            self.client.delete("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}",
                                                          "If-Match": api_v1.etag_of(*rows[0][1:])})
            self.assertEqual(pushed, [])
            mel_today = api_v1.melbourne_today()
            self.post("/v1/entries", {"date": mel_today.isoformat(), "name": "Apple", "macros": MACROS})
            self.assertEqual(pushed, [mel_today.isoformat()])
            # P24-D2-3: a late snack logged to an earlier day never pushes
            self.post("/v1/entries", {"date": (mel_today - timedelta(days=1)).isoformat(), "name": "Snack",
                                      "macros": MACROS}, key="k-past")
        self.assertEqual(pushed, [mel_today.isoformat()])

    def test_moving_an_entry_to_another_day_takes_it_out_of_its_meal(self):
        # P24-D2-2: a meal is one day's meal, so the context ETag always matches delete_meal's
        row = ("l1", "Rice", 1.0, kcal_n(195, 4, "150 g", _meal_group="g1", _meal_label="Meal 1"), "2026-09-20")
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        r = self.client.patch("/v1/entries/l1", headers={"Authorization": f"Bearer {TOKEN}",
                                                          "If-Match": api_v1.etag_of(*row[1:])}, json={"date": "2026-09-21"})
        self.assertEqual(r.status_code, 200, r.text)
        n = json.loads(self.statements("UPDATE daily_log")[0][1][2])
        self.assertNotIn("_meal_group", n)
        self.assertNotIn("_meal_label", n)

    def test_macros_only_change_shows_its_difference_from_the_template(self):
        # P24-D2-4
        body = {"date": TODAY, "changes": [{"item_id": "i-rice", "macros": {**MACROS, "calories": 300}}]}
        rice = [d for d in self.post("/v1/templates/t1/log", body, preview=True).json()["diff"] if d["name"] == "Rice"][0]
        self.assertEqual(rice["vs_template_kcal_change"], 105.0)

    def test_preview_yes_is_not_counted_either(self):
        # P24-D2-5: every spelling FastAPI reads as true is a preview
        rows = [("l1", "Rice", 1.0, kcal_n(195, 4, "150 g"), TODAY)]
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", rows))
        h = {"Authorization": f"Bearer {TOKEN}", "If-Match": api_v1.etag_of(*rows[0][1:])}
        with mock.patch.object(main, "_event_windows", {}) as windows:
            for flag in ("yes", "on", "t", "y", "True"):
                self.client.delete(f"/v1/entries/l1?preview={flag}", headers=h)
            self.assertNotIn("del:user-1", windows)

    def test_preview_delete_is_not_counted_towards_the_freeze(self):
        # P24-CR-4
        rows = [("l1", "Rice", 1.0, kcal_n(195, 4, "150 g"), TODAY)]
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", rows))
        h = {"Authorization": f"Bearer {TOKEN}", "If-Match": api_v1.etag_of(*rows[0][1:])}
        with mock.patch.object(main, "_event_windows", {}) as windows:
            self.client.delete("/v1/entries/l1?preview=true", headers=h)
            self.assertNotIn("del:user-1", windows)
            self.client.delete("/v1/entries/l1", headers=h)   # positive control
            self.assertIn("del:user-1", windows)

    def test_preview_flag_on_an_app_route_still_counts(self):
        # P24-D2-SEC-1: app routes ignore ?preview and really delete, so they must still be counted
        with mock.patch.object(main, "_event_windows", {}) as windows, \
             mock.patch.object(main, "get_user_id", lambda a=None: "user-1"), \
             mock.patch.object(main, "claims_if_valid", lambda a: {"sub": "user-1"}):
            r = self.client.delete("/log/l1?preview=true", headers={"Authorization": "Bearer login"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("del:user-1", windows)

    def test_real_write_invalidates_the_read_cache_and_audits(self):
        api_v1._read_cache["user-1"]["context:x"] = {"stale": True}
        self.post("/v1/entries", {"date": TODAY, "name": "Apple", "macros": MACROS})
        self.assertNotIn("user-1", api_v1._read_cache)
        audit = self.statements("INSERT INTO api_audit")
        self.assertEqual(audit[0][1][3:6], ["POST", "/v1/entries", 201])


if __name__ == "__main__":
    unittest.main()
