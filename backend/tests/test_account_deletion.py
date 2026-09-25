"""Account deletion with a 15-day grace period: the lock, the routes, the reload and the daily purge.

Run:  venv/Scripts/python -m unittest backend.tests.test_account_deletion -v
No network: FakeConn scripts the DB; app logins are HS256 JWTs minted against a test secret,
so the real get_user_id runs.
"""
import os
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case
from .test_mcp import WithConnectorAuth, token
from .test_phase0 import ImmediateThread
from . import REAL_SEND_PUSH, ALERTS

WHEN = datetime(2026, 10, 10, 1, 0, tzinfo=timezone.utc)
SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "account_deletions.sql")


def login(sub="user-1", **extra):
    return {"Authorization": f"Bearer {token(sub=sub, client_id=None, **extra)}"}


class Sql(unittest.TestCase):
    def test_file_is_idempotent_and_locked_down(self):
        with open(SQL, encoding="utf-8") as f:
            sql = f.read()
        for want in ("CREATE TABLE IF NOT EXISTS account_deletions", "FORCE  ROW LEVEL SECURITY",
                     "FUNCTION pending_account_deletions()", "FUNCTION due_account_deletions()",
                     "REVOKE ALL ON FUNCTION pending_account_deletions() FROM PUBLIC",
                     "REVOKE ALL ON FUNCTION due_account_deletions() FROM PUBLIC",
                     "ADD COLUMN IF NOT EXISTS purged_at timestamptz",
                     "DROP FUNCTION IF EXISTS pending_account_deletions()",
                     "DROP FUNCTION IF EXISTS due_account_deletions()",
                     "BEGIN;", "COMMIT;",   # C2D2-7: the drop and re-create are one step in any editor
                     "d.purged_at IS NULL OR d.purged_at < now() - interval '8 days'",   # C2D2-9: only rows with work
                     "FUNCTION purge_user_recycle_bin()",
                     "WHERE user_id = NULLIF(current_setting('app.user_id', true), '')",
                     "REVOKE ALL ON FUNCTION purge_user_recycle_bin() FROM PUBLIC"):
            self.assertIn(want, sql)
        self.assertNotIn("recycle_bin_trg", sql)

    def test_the_marker_is_not_wiped_with_the_account(self):
        # the purge removes it last, only once the login is gone
        self.assertNotIn("account_deletions", main._ACCOUNT_TABLES)


class Case(WithConnectorAuth, V1Case):
    def setUp(self):
        super().setUp()
        for target, value in (("_deleting", {}), ("_purged", set()), ("_reload_db_uses", [-1]),   # -1: never reloaded
                              ("_deletion_loaded", [True])):
            p = mock.patch.object(main, target, value)
            p.start()
            self.addCleanup(p.stop)

    def pending(self, user="user-1"):
        main._deleting[user] = WHEN


class Lock(Case):
    def test_view_is_allowed_while_pending(self):
        self.pending()
        r = self.client.get("/goals", headers=login())
        self.assertEqual(r.status_code, 200)

    def test_writes_are_refused_while_pending(self):
        self.pending()
        p = mock.patch.object(main, "groq_client", mock.Mock())   # /chat answers 503 before auth without a model
        p.start()
        self.addCleanup(p.stop)
        for method, path, kw in (("post", "/log", {"json": {"name": "x", "servings": 1, "nutrition": {}}}),
                                 ("post", "/chat", {"json": {"message": "hi"}}),
                                 ("put", "/settings/energy-unit", {"json": {"unit": "kJ"}}),
                                 ("delete", "/log/abc", {})):
            r = getattr(self.client, method)(path, headers=login(), **kw)
            self.assertEqual(r.status_code, 423, path)
            self.assertEqual(r.json()["detail"]["error_type"], "account_scheduled_for_deletion", path)
        self.assertFalse([s for s, _ in self.conn.executed if "daily_log" in s])

    def test_other_accounts_are_untouched(self):
        self.pending("someone-else")
        self.conn.script = [("INSERT INTO daily_log", [])]
        r = self.client.post("/log", headers=login(), json={"name": "x", "servings": 1, "nutrition": {}})
        self.assertEqual(r.status_code, 200)

    def test_api_token_is_refused_even_for_a_read(self):
        self.pending()
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (423, "account_scheduled_for_deletion"))

    def test_a_purged_accounts_api_token_is_refused(self):
        # C2D2-4: a token cached before the purge must not resolve in the moment before forget_token runs
        main._purged.add("user-1")
        r = self.get("/v1/me")
        self.assertEqual((r.status_code, r.json()["error_type"]), (401, "account_deleted"))

    def test_the_purge_marks_purged_before_it_unmarks_pending(self):
        # C2D2-3: the gate reads _deleting then _purged without a lock; this order leaves no gap between them
        class Watch(dict):
            def pop(self, key, *default):
                assert key in main._purged, "popped from _deleting before it was added to _purged"
                return super().pop(key, *default)
        with mock.patch.object(main, "_deleting", Watch({"user-1": WHEN})):
            main._set_purged("user-1")
        self.assertIn("user-1", main._purged)

    def test_app_login_on_v1_is_refused_even_for_a_read(self):
        self.pending()
        r = self.client.get("/v1/me", headers=login())
        self.assertEqual((r.status_code, r.json()["error_type"]), (423, "account_scheduled_for_deletion"))

    def test_connector_is_refused(self):
        self.pending("admin-1")
        r = self.call_tool({})
        self.assertEqual((r.status_code, r.json()["error_type"]), (423, "account_scheduled_for_deletion"))
        self.assertFalse([s for s, _ in self.conn.executed if "daily_log" in s])

    def test_frozen_still_wins(self):
        self.pending()
        with mock.patch.object(main, "_frozen", {"user-1"}):
            r = self.client.get("/goals", headers=login())
        self.assertEqual(r.json()["detail"]["error_type"], "account_locked")

    def test_outside_a_request_counts_as_a_write(self):
        self.pending()
        with mock.patch.object(main, "verify_claims", lambda a: {"sub": "user-1"}):
            with self.assertRaises(main.HTTPException) as cm:
                main.get_user_id("Bearer x")
            self.assertEqual(cm.exception.status_code, 423)
            self.assertEqual(main.get_user_id("Bearer x", allow_deleting=True), "user-1")


def fresh(**extra):
    return login(amr=[{"method": "otp", "timestamp": int(time.time())}], **extra)


class Routes(Case):
    def test_delete_schedules_and_locks_at_once(self):
        self.get("/v1/me")
        self.assertTrue(api_v1._token_cache)
        self.conn.script = [("INSERT INTO account_deletions", [(WHEN,)])]
        r = self.client.delete("/account", headers=fresh())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"scheduled": True, "delete_after": "2026-10-10T01:00:00+00:00"})
        ins = [p for s, p in self.conn.executed if "INSERT INTO account_deletions" in s][0]
        self.assertEqual(ins[0], "user-1")
        self.assertAlmostEqual(ins[1], datetime.now(timezone.utc) + timedelta(days=15), delta=timedelta(minutes=1))
        self.assertFalse([s for s, _ in self.conn.executed if s.startswith("DELETE FROM")])   # nothing wiped yet
        self.assertEqual(main._deleting["user-1"], WHEN)
        self.assertFalse(api_v1._token_cache)
        self.assertEqual(self.client.post("/log", headers=login(), json={"name": "x", "servings": 1,
                                                                          "nutrition": {}}).status_code, 423)

    def test_asking_again_keeps_the_first_date(self):
        self.pending()
        self.conn.script = [("INSERT INTO account_deletions", [(WHEN,)])]
        r = self.client.delete("/account", headers=fresh())
        self.assertEqual(r.json()["delete_after"], "2026-10-10T01:00:00+00:00")
        sql = [s for s, _ in self.conn.executed if "INSERT INTO account_deletions" in s][0]
        self.assertIn("ON CONFLICT (user_id) DO UPDATE SET user_id = EXCLUDED.user_id RETURNING delete_after", sql)

    def test_stale_sign_in_still_needs_reauth(self):
        r = self.client.delete("/account", headers=login())
        self.assertEqual(r.json()["detail"]["error_type"], "reauth_required")
        self.assertEqual(main._deleting, {})

    def test_status(self):
        r = self.client.get("/account/deletion", headers=login())
        self.assertEqual((r.status_code, r.json()), (200, {"delete_after": None}))
        self.pending()
        r = self.client.get("/account/deletion", headers=login())
        self.assertEqual(r.json(), {"delete_after": "2026-10-10T01:00:00+00:00"})
        self.assertFalse(self.conn.executed)   # CR-AD-10: the app asks on every load; never a DB round trip

    def test_restore_needs_a_fresh_sign_in(self):
        self.pending()
        r = self.client.post("/account/restore", headers=login())
        self.assertEqual(r.json()["detail"]["error_type"], "reauth_required")
        self.assertIn("user-1", main._deleting)
        self.assertFalse(self.conn.executed)

    def test_restore_unlocks(self):
        self.pending()
        self.conn.script = [("DELETE FROM account_deletions", [("user-1",)])]
        r = self.client.post("/account/restore", headers=fresh())
        self.assertEqual(r.json(), {"restored": True})
        self.assertNotIn("user-1", main._deleting)
        self.conn.script = [("INSERT INTO daily_log", [])]
        self.assertEqual(self.client.post("/log", headers=login(), json={"name": "x", "servings": 1,
                                                                          "nutrition": {}}).status_code, 200)

    def test_restore_with_nothing_pending(self):
        self.conn.script = [("DELETE FROM account_deletions", [])]
        self.assertEqual(self.client.post("/account/restore", headers=fresh()).json(), {"restored": False})

    def test_restore_is_refused_once_the_date_has_passed(self):
        # CR-AD-3: after delete_after (or a purge that wiped the rows but not the login) nothing is left to keep
        self.pending()
        self.conn.script = [("DELETE FROM account_deletions", [])]   # the row exists but is due: WHERE excludes it
        r = self.client.post("/account/restore", headers=fresh())
        self.assertEqual(r.json(), {"restored": False})
        sql = [s for s, _ in self.conn.executed if "DELETE FROM account_deletions" in s][0]
        self.assertIn("delete_after > clock_timestamp()", sql)   # D2-3: re-checked after waiting on the purge's lock
        self.assertIn("user-1", main._deleting)   # still locked: tomorrow's purge finishes it

    def test_push_unsubscribe_still_works_while_pending(self):
        # CR-AD-5: taking something away is allowed inside the 15 days
        self.pending()
        self.conn.script = [("DELETE FROM push_subscriptions", [])]
        self.assertEqual(self.client.delete("/push/unsubscribe", headers=login()).status_code, 200)

    def test_connector_and_api_token_cannot_restore(self):
        self.pending("admin-1")
        r = self.client.post("/account/restore", headers={"Authorization": f"Bearer {token()}"})
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/account/restore", headers={"Authorization": f"Bearer {api_v1.TOKEN_PREFIX}{'A' * 43}"})
        self.assertEqual(r.status_code, 401)
        self.assertIn("admin-1", main._deleting)

    def test_owner_can_still_disconnect_claude_while_pending(self):
        self.pending("admin-1")
        self.conn.script = [("UPDATE connected_apps SET revoked_at", [("c1",)])]
        r = self.client.delete("/settings/connected-apps/6f1c2d3e-0000-4000-8000-000000000001",
                               headers=login("admin-1"))
        self.assertEqual(r.json(), {"disconnected": True})


class Reload(Case):
    def test_reload_replaces_the_map(self):
        self.pending("restored-elsewhere")
        self.conn.script = [("pending_account_deletions", [("user-2", WHEN, None)])]
        main._load_deleting()
        self.assertEqual(main._deleting, {"user-2": WHEN})

    def test_reload_learns_purges_made_elsewhere(self):
        # CR2-5: another instance purged it: this one locks it out and drops its cached tokens too
        self.pending("user-3")
        with mock.patch.object(api_v1, "forget_token") as forget:
            self.conn.script = [("pending_account_deletions", [("user-3", WHEN, WHEN)])]
            main._load_deleting()
            main._load_deleting()   # already known: not forgotten twice
        self.assertEqual((main._deleting, main._purged), ({}, {"user-3"}))
        forget.assert_called_once_with(user_id="user-3")

    def test_until_the_first_load_works_every_tick_tries_again(self):
        # C2D2-1: a failed startup load must not leave the lock open until something else uses the database
        main._deletion_loaded[0] = False
        with mock.patch.object(main, "get_db", mock.Mock(side_effect=RuntimeError("down"))):
            main._deletion_tick()   # asleep, yet it tries
        self.assertFalse(main._deletion_loaded[0])
        self.conn.script = [("pending_account_deletions", [("user-2", WHEN, None)])]
        main._deletion_tick()
        self.assertEqual((main._deleting, main._deletion_loaded[0]), ({"user-2": WHEN}, True))

    def test_an_unreadable_list_is_reported(self):
        # C2D2-2: e.g. the backend deployed before account_deletions.sql was re-applied: the lock is open
        main._deletion_loaded[0] = False   # a fresh start
        with mock.patch.object(main, "get_db", mock.Mock(side_effect=RuntimeError("column purged_at does not exist"))), \
             mock.patch.object(main, "_admin_alert_last", {}):
            ALERTS.clear()
            main._load_deleting()
        self.assertTrue([t for t, _ in ALERTS if t == "Account deletion lock not loaded"])

    def test_reload_failure_keeps_what_it_had(self):
        self.pending()
        with mock.patch.object(main, "get_db", mock.Mock(side_effect=RuntimeError("down"))):
            main._load_deleting()
        self.assertEqual(main._deleting, {"user-1": WHEN})

    def test_tick_never_wakes_the_database(self):
        # asleep: no query; awake: reload
        main._deletion_tick()
        self.assertFalse(self.conn.executed)
        with mock.patch.dict(main._budget, {"awake_until": time.time() + 60}):
            self.conn.script = [("pending_account_deletions", [("user-2", WHEN, None)])]
            main._deletion_tick()
        self.assertEqual(main._deleting, {"user-2": WHEN})

    def test_tick_does_not_keep_the_database_awake_by_itself(self):
        # CR-AD-1: the reload is DB use too; only use by something else since the last reload earns another
        real_get_db = lambda *a, **k: (main._note_db_use(), self.conn)[1]
        self.conn.script = [("pending_account_deletions", [])]
        with mock.patch.object(main, "get_db", real_get_db), mock.patch.object(main, "_save_budget", lambda: None), \
             mock.patch.dict(main._budget, {"awake_until": time.time() + 60, "saved_at": time.time()}):
            main._deletion_tick()
            main._deletion_tick()
            main._deletion_tick()
            self.assertEqual(len(self.conn.executed), 1)
            main.get_db()   # a real request used the database
            main._deletion_tick()
            self.assertEqual(len([s for s, _ in self.conn.executed if "pending_account_deletions" in s]), 2)

    def test_a_failed_reload_does_not_keep_the_database_awake(self):
        # D2-1: the failed attempt's own connection counts as use; it must not earn another attempt
        def boom(sql, params):
            raise RuntimeError("relation account_deletions does not exist")
        real_get_db = lambda *a, **k: (main._note_db_use(), self.conn)[1]
        self.conn.script = [("pending_account_deletions", boom)]
        with mock.patch.object(main, "get_db", real_get_db), mock.patch.object(main, "_save_budget", lambda: None), \
             mock.patch.dict(main._budget, {"awake_until": time.time() + 60, "saved_at": time.time()}):
            main._deletion_tick()
            main._deletion_tick()
        self.assertEqual(len(self.conn.executed), 1)

    def test_a_restore_during_the_reload_is_not_undone(self):
        # CR-AD-6: the SELECT saw the row, then Keep my account landed; the stale result must not re-lock
        self.pending()
        def rows(sql, params):
            main._drop_deleting("user-1")
            return [("user-1", WHEN, None)]
        self.conn.script = [("pending_account_deletions", rows)]
        main._load_deleting()
        self.assertNotIn("user-1", main._deleting)
        self.conn.script = [("pending_account_deletions", [("user-2", WHEN, None)])]
        main._load_deleting()   # positive control: an undisturbed reload applies
        self.assertEqual(main._deleting, {"user-2": WHEN})

    def test_background_failures_do_not_raise_the_endpoint_db_alert(self):
        # CR-AD-9
        def boom(sql, params):
            raise RuntimeError("relation account_deletions does not exist")
        self.conn.script = [("account_deletions", boom)]
        with mock.patch.object(main, "_db_error", mock.Mock()) as db_error:
            main._load_deleting()
            self.assertFalse(main._purge_due_accounts())
        db_error.assert_not_called()


class PurgeSchedule(Case):
    # CR-AD-7: a purge that could not read the due list retries within the hour, not tomorrow
    def test_success_waits_a_day_failure_an_hour(self):
        now = time.time()
        with mock.patch.object(main, "_purge_due_accounts", lambda: True):
            self.assertAlmostEqual(main._purge_when_due(0.0), now + 86400, delta=5)
        with mock.patch.object(main, "_purge_due_accounts", lambda: False):
            self.assertAlmostEqual(main._purge_when_due(0.0), now + 3600, delta=5)
        with mock.patch.object(main, "_purge_due_accounts", mock.Mock()) as purge:
            self.assertEqual(main._purge_when_due(now + 100), now + 100)
        purge.assert_not_called()


class Purge(Case):
    def setUp(self):
        super().setUp()
        for name in ("user-1", "user-2"):
            self.pending(name)
        self.conn.script = [("due_account_deletions", [("user-1", None), ("user-2", None)]), ("FOR UPDATE", [(1,)]),
                            ("to_regclass", [("x",)]), ("SET purged_at", [(1,)])]
        self.logins = []
        p = mock.patch.object(main, "_delete_login", lambda uid: self.logins.append(uid) or True)
        p.start()
        self.addCleanup(p.stop)

    def test_purges_every_table_then_the_login_then_stamps_the_marker(self):
        api_v1._apps[("user-1", "c9")] = [True, time.time()]
        main._purge_due_accounts()
        sql = [s for s, _ in self.conn.executed]
        wipe = [s for s in sql if s.startswith("DELETE FROM") and "account_deletions" not in s]
        self.assertEqual(len(wipe), 2 * len(main._ACCOUNT_TABLES))
        self.assertIn("SELECT set_config('app.skip_bin', '1', true)", sql)
        self.assertEqual(self.logins, ["user-1", "user-2"])
        stamp = [(s, p) for s, p in self.conn.executed if "SET purged_at" in s]
        self.assertEqual([p for _, p in stamp], [["user-1"], ["user-2"]])
        self.assertIn("purged_at IS NULL", stamp[0][0])
        self.assertFalse([s for s in sql if s.startswith("DELETE FROM account_deletions")])
        self.assertEqual((main._deleting, main._purged), ({}, {"user-1", "user-2"}))
        self.assertNotIn(("user-1", "c9"), api_v1._apps)

    def test_the_wipe_empties_the_recycle_bin_too(self):
        # D2-6: rows the user deleted earlier are "permanently deleted" with the account, not 30 days later
        main._purge_due_accounts()
        sql = [s for s, _ in self.conn.executed]
        self.assertEqual(sql.count("SELECT purge_user_recycle_bin()"), 2)

    def test_a_purged_login_that_is_still_open_cannot_write_it_back(self):
        # D2-2, CR51-1, CR2-1: the access token outlives the deleted login; the lock has to survive a restart,
        # so the tombstone is the marker row (purged_at), and a purged account is gone, not "pending"
        main._purge_due_accounts()
        for held in (main._deleting, main._purged):   # a restart: rebuilt from the database
            held.clear()
        past = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
        self.conn.script.insert(0, ("pending_account_deletions", [("user-1", past, past)]))
        main._load_deleting()
        for method, path, kw in (("get", "/goals", {}), ("get", "/account/deletion", {}),
                                 ("post", "/log", {"json": {"name": "x", "servings": 1, "nutrition": {}}})):
            r = getattr(self.client, method)(path, headers=login(), **kw)
            self.assertEqual((r.status_code, r.json()["detail"]["error_type"]), (401, "account_deleted"), path)
        self.assertFalse([s for s, _ in self.conn.executed if "INSERT INTO daily_log" in s])
        self.assertEqual(self.client.get("/goals", headers=login("user-3")).status_code, 200)   # control

    def test_an_old_tombstone_goes_after_one_last_sweep(self):
        # CR2-2, CR2-6, C2D2-1: the function returns a tombstone only once it is 8 days old; the rows get one
        # last sweep (anything written meanwhile goes too), then the marker; no Supabase call, no second alert
        main._purged.add("user-1")
        self.conn.script = [("due_account_deletions", [("user-1", WHEN)]), ("FOR UPDATE", [(1,)]), ("to_regclass", [("x",)]),
                            ("DELETE FROM account_deletions", [("user-1",)])]
        with mock.patch.object(main, "_admin_alert_last", {}):
            ALERTS.clear()
            main._purge_due_accounts()
        sql = [s for s, _ in self.conn.executed]
        self.assertEqual(len([s for s in sql if s.startswith("DELETE FROM") and "account_deletions" not in s]),
                         len(main._ACCOUNT_TABLES))
        gone = [s for s in sql if s.startswith("DELETE FROM account_deletions")]
        self.assertEqual(len(gone), 1)
        self.assertIn("purged_at IS NOT NULL", gone[0])
        self.assertEqual(self.logins, [])
        self.assertFalse([t for t, _ in ALERTS if t == "🗑️ Account Deleted"])
        self.assertNotIn("user-1", main._purged)
        self.assertIn("user-2", main._deleting)   # not due: untouched

    def test_a_failed_tombstone_cleanup_says_what_it_is(self):
        # C2D2-5: the account is already deleted; the alert must not say it is still due
        def boom(sql, params):
            raise RuntimeError("neon blip")
        self.conn.script = [("due_account_deletions", [("user-1", WHEN)]), ("FOR UPDATE", boom)]
        with mock.patch.object(main, "_admin_alert_last", {}):
            ALERTS.clear()
            main._purge_due_accounts()
        titles = [t for t, _ in ALERTS]
        self.assertIn("Deleted-account cleanup failed", titles)
        self.assertNotIn("Account delete failed", titles)

    def test_stamped_elsewhere_still_forgets_this_instances_tokens(self):
        # C2D2-4
        self.conn.script = [("due_account_deletions", [("user-1", None)]), ("FOR UPDATE", [(1,)]),
                            ("to_regclass", [("x",)]), ("SET purged_at", [])]
        with mock.patch.object(api_v1, "forget_token") as forget:
            main._purge_due_accounts()
        forget.assert_called_with(user_id="user-1")
        self.assertIn("user-1", main._purged)

    def test_each_account_gets_its_own_alert(self):
        # D2-10: the per-event cooldown must not swallow the second account
        with mock.patch.object(main, "_admin_alert_last", {}):
            ALERTS.clear()
            main._purge_due_accounts()
        self.assertEqual(len([t for t, _ in ALERTS if t == "🗑️ Account Deleted"]), 2)

    def test_a_failed_wipe_is_not_reported_as_an_endpoint_db_error(self):
        # D2-8
        def boom(sql, params):
            raise RuntimeError("permission denied for table api_audit")
        self.conn.script = [("due_account_deletions", [("user-1", None)]), ("FOR UPDATE", boom)]
        with mock.patch.object(main, "_db_error", mock.Mock()) as db_error, \
             mock.patch.object(main, "_admin_alert_last", {}):
            ALERTS.clear()
            main._purge_due_accounts()
        db_error.assert_not_called()
        self.assertIn("permission denied", " ".join(m for _, m in ALERTS))

    def test_login_failure_keeps_the_marker_for_tomorrow(self):
        main._delete_login = lambda uid: uid != "user-1"
        main._purge_due_accounts()
        marker = [p for s, p in self.conn.executed if "SET purged_at" in s]
        self.assertEqual(marker, [["user-2"]])
        self.assertIn("user-1", main._deleting)

    def test_one_failure_does_not_stop_the_next(self):
        real = main._wipe_account_rows
        def wipe(uid):
            if uid == "user-1":
                raise RuntimeError("boom")
            return real(uid)
        with mock.patch.object(main, "_wipe_account_rows", wipe):
            main._purge_due_accounts()
        self.assertEqual(self.logins, ["user-2"])
        self.assertIn("user-1", main._deleting)

    def test_an_account_kept_meanwhile_is_left_alone(self):
        # CR-AD-2: the list was read, then Keep my account removed the marker; the wipe re-checks it under a lock
        self.conn.script = [("due_account_deletions", [("user-1", None)]), ("FOR UPDATE", []), ("to_regclass", [("x",)])]
        main._purge_due_accounts()
        sql = [s for s, _ in self.conn.executed]
        lock = [s for s in sql if "FOR UPDATE" in s][0]
        self.assertIn("delete_after <= now()", lock)
        self.assertFalse([s for s in sql if s.startswith("DELETE FROM")])
        self.assertEqual(self.logins, [])

    def test_nothing_due_touches_nothing_else(self):
        self.conn.script = [("due_account_deletions", [])]
        main._purge_due_accounts()
        self.assertEqual([s for s, _ in self.conn.executed if "due_account_deletions" not in s], [])


class Login(unittest.TestCase):
    def test_gone_already_counts_as_done(self):
        import urllib.error
        err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with mock.patch.object(main, "SUPABASE_SERVICE_ROLE_KEY", "k"), \
             mock.patch.object(main.urllib.request, "urlopen", mock.Mock(side_effect=err)):
            self.assertTrue(main._delete_login("user-1"))
        with mock.patch.object(main, "SUPABASE_SERVICE_ROLE_KEY", "k"), \
             mock.patch.object(main.urllib.request, "urlopen", mock.Mock(side_effect=OSError("down"))):
            self.assertFalse(main._delete_login("user-1"))
        with mock.patch.object(main, "SUPABASE_SERVICE_ROLE_KEY", ""), \
             mock.patch.object(main, "notify_admin") as alert:
            self.assertTrue(main._delete_login("user-1"))   # CR-AD-4: no key: best effort as before, not a forever-retry
        alert.assert_called_once()


class PushWhilePending(Case):
    # SR-AD-1 / CR-AD-5: no reminder, summary or goal push to an account inside the 15 days
    def test_no_push_reaches_a_pending_account(self):
        with mock.patch.object(main, "VAPID_KEY", object()), mock.patch.object(main, "VAPID_PUBLIC_KEY", "k"), \
             mock.patch.object(main.threading, "Thread", ImmediateThread):
            self.pending()
            REAL_SEND_PUSH("user-1", "Morning reminder", "Log your first meal")
            self.assertFalse(self.conn.executed)
            REAL_SEND_PUSH("user-2", "Morning reminder", "Log your first meal")   # positive control
        self.assertTrue([s for s, _ in self.conn.executed if "push_subscriptions" in s])


if __name__ == "__main__":
    unittest.main()
