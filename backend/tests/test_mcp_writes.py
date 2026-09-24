"""/mcp write tools (Part A of the full connector). Run:
venv/Scripts/python -m unittest backend.tests.test_mcp_writes -v

A write tool only previews and hands back a one-time code; confirm_change saves
exactly that preview. FakeConn counts commits, so "nothing saved" is "no commit".
"""
import json
import time
import unittest
from unittest import mock

import main
import api_v1
from .test_api_v1 import V1Case
from .test_api_v1_writes import WritesFixture, MACROS, TODAY
from .test_mcp import WithConnectorAuth, token


def result(r):
    body = r.json()["result"]
    return body["isError"], body["content"][0]["text"]


class Registry(WithConnectorAuth, V1Case):
    def tools(self):
        return {t["name"]: t for t in self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()["result"]["tools"]}

    def test_all_tools_listed_with_titles(self):
        t = self.tools()
        self.assertEqual(set(t), {"get_context", "get_template", "log_food", "log_meal", "log_template", "edit_entry",
                                  "delete_entry", "delete_meal", "save_template", "update_template",
                                  "save_to_library", "confirm_change"})
        for name, tool in t.items():
            self.assertTrue(tool["title"], name)
            self.assertLessEqual(len(name), 64)
            self.assertEqual(tool["inputSchema"]["type"], "object", name)

    def test_destructive_flags(self):
        t = self.tools()
        destructive = {n for n, x in t.items() if x["annotations"]["destructiveHint"]}
        self.assertEqual(destructive, {"delete_entry", "delete_meal", "confirm_change"})
        self.assertTrue(t["get_template"]["annotations"]["readOnlyHint"])
        self.assertFalse(t["log_food"]["annotations"]["readOnlyHint"])

    def test_write_schema_has_no_type_field(self):
        self.assertNotIn("type", self.tools()["log_food"]["inputSchema"]["properties"])

    def test_preview_rule_is_in_every_write_description(self):
        t = self.tools()
        for name in ("log_food", "log_meal", "log_template", "edit_entry", "delete_entry", "delete_meal",
                     "save_template", "update_template", "save_to_library"):
            self.assertIn("confirm_change", t[name]["description"], name)
            self.assertIn("list them and ask", t[name]["description"], name)
        self.assertIn("whole meal or one item", t["delete_meal"]["description"])


class GetTemplate(WithConnectorAuth, WritesFixture, V1Case):
    def test_returns_items_and_etag(self):
        is_err, text = result(self.call_tool({"template_id": "t1"}, name="get_template"))
        self.assertFalse(is_err, text)
        body = json.loads(text)
        self.assertEqual([i["item_id"] for i in body["items"]], ["i-tofu", "i-rice"])
        self.assertTrue(body["etag"])

    def test_bad_id_is_an_isError(self):
        is_err, text = result(self.call_tool({"template_id": "x" * 65}, name="get_template"))
        self.assertTrue(is_err)
        self.assertIn("template_id", text)


class PreviewBase(WithConnectorAuth, WritesFixture, V1Case):
    """Shared setup; no tests of its own, so subclasses do not re-run each other's tests."""

    def setUp(self):
        super().setUp()
        api_v1._pending.clear()
        self.conn.commits = 0
        self.conn.executed.clear()

    def banana(self, **over):
        return {"date": TODAY, "name": "Banana", "portion": "1 medium", "macros": MACROS, **over}


class Preview(PreviewBase):
    def test_preview_saves_nothing_and_returns_a_code(self):
        is_err, text = result(self.call_tool(self.banana(), name="log_food"))
        self.assertFalse(is_err, text)
        body = json.loads(text)
        self.assertTrue(body["preview"]["preview"])
        self.assertEqual(self.conn.commits, 0)
        self.assertIn(body["confirm_code"], api_v1._pending)
        uid, cid, changes, _, _ = api_v1._pending[body["confirm_code"]]
        self.assertEqual((uid, cid, changes[0].type), ("admin-1", "c1", "log_entry"))
        self.assertIn("confirm_change", body["next"])

    def test_bad_arguments_are_readable(self):
        for args, field in (({"date": TODAY, "name": "Banana"}, "macros"),                      # missing
                            ({**self.banana(), "extra": 1}, "extra"),                          # extra field
                            ({**self.banana(), "macros": {"calories": "lots"}}, "macros.calories")):   # wrong type
            is_err, text = result(self.call_tool(args, name="log_food"))
            self.assertTrue(is_err, args)
            self.assertTrue(text.startswith(field + ":"), text)
            self.assertNotIn("Traceback", text)
        self.assertEqual(api_v1._pending, {})

    def test_preview_counts_as_a_write(self):
        # /v1's ?preview=true goes through need(write=True): a connector preview is held to the same write limit
        now = time.time()
        api_v1._calls["admin-1"].extend((now, True) for _ in range(api_v1.WRITES_PER_MIN))
        is_err, text = result(self.call_tool(self.banana(), name="log_food"))
        self.assertTrue(is_err)
        self.assertIn("Over the limit of 10", text)

    def test_codes_are_bounded(self):
        for _ in range(api_v1.PENDING_MAX + 5):
            self.call_tool(self.banana(), name="log_food")
            api_v1._calls.clear()   # stay under the per-minute limiter for this test
            api_v1._daily.clear()
        self.assertEqual(sum(1 for v in api_v1._pending.values() if v[0] == "admin-1"), api_v1.PENDING_MAX)

    def test_expired_codes_are_swept_on_the_next_preview(self):
        api_v1._pending["old"] = ("admin-1", "c1", [], time.time() - api_v1.PENDING_TTL - 1, None)
        self.call_tool(self.banana(), name="log_food")
        self.assertNotIn("old", api_v1._pending)


class Confirm(PreviewBase):
    def preview(self, args=None, name="log_food", tok=None):
        is_err, text = result(self.call_tool(args or self.banana(), name=name, tok=tok))
        self.assertFalse(is_err, text)
        return json.loads(text)["confirm_code"]

    def confirm(self, code, tok=None):
        return result(self.call_tool({"code": code}, name="confirm_change", tok=tok))

    def test_confirm_saves_exactly_the_preview_with_the_badge(self):
        code = self.preview()
        is_err, text = self.confirm(code)
        self.assertFalse(is_err, text)
        self.assertTrue(json.loads(text)["saved"])
        self.assertEqual(self.conn.commits, 1)
        params = self.statements("INSERT INTO daily_log")[0][1]
        self.assertIn("Banana", params)
        n = json.loads(params[5])
        self.assertEqual(n["_source"], "claude")

    def test_retry_replays_instead_of_saving_twice(self):
        code = self.preview()
        self.conn.executed.clear()   # the preview ran (and rolled back) its own INSERT
        first = self.confirm(code)
        second = self.confirm(code)   # e.g. the first answer was lost and claude.ai retried
        self.assertFalse(second[0], second[1])
        self.assertEqual(json.loads(second[1]), json.loads(first[1]))
        self.assertEqual(len(self.statements("INSERT INTO daily_log")), 1)

    def test_code_expires(self):
        code = self.preview()
        api_v1._apps[("admin-1", "c1")] = [True, time.time() + api_v1.PENDING_TTL + 1]   # gate quiet at the moved clock
        with mock.patch.object(api_v1.time, "time", return_value=time.time() + api_v1.PENDING_TTL + 1):
            is_err, text = self.confirm(code)
        self.assertTrue(is_err)
        self.assertIn("expired or is not valid", text)
        self.assertEqual(self.conn.commits, 0)

    def test_other_client_cannot_use_code(self):
        code = self.preview()
        api_v1._apps[("admin-1", "c2")] = [True, time.time()]   # a known second connection: the gate stays quiet
        is_err, _ = self.confirm(code, tok=token(client_id="c2"))
        self.assertTrue(is_err)
        self.assertEqual(self.conn.commits, 0)
        is_err, text = self.confirm(code)   # still usable by the connection that previewed it
        self.assertFalse(is_err, text)

    def test_unknown_or_bad_code(self):
        for code in ("nope", ""):
            is_err, _ = self.confirm(code)
            self.assertTrue(is_err, code)
        r = self.call_tool({"code": 5}, name="confirm_change")
        self.assertTrue(r.json()["result"]["isError"])
        self.assertEqual(self.conn.commits, 0)

    def test_confirm_counts_as_a_write(self):
        code = self.preview()
        before = api_v1._daily["admin-1"][1]
        self.confirm(code)
        self.assertEqual(api_v1._daily["admin-1"][1], before + 1)

    def test_stale_confirm(self):
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        code = self.preview({"log_id": "l1", "if_match": api_v1.etag_of(*row[1:]), "servings": 2}, name="edit_entry")
        self.conn.script[1] = ("FROM daily_log WHERE log_id", [self.entry_row(servings=3.0)])   # changed in the app
        is_err, text = self.confirm(code)
        self.assertTrue(is_err)
        self.assertIn("changed since the preview", text)
        self.assertEqual(self.conn.commits, 0)

    def test_template_edited_after_the_preview_is_refused(self):
        code = self.preview({"template_id": "t1", "date": TODAY}, name="log_template")
        self.conn.executed.clear()
        self.template_rows[1] = ("i-oats", "Oats", 1.0, self.template_rows[1][3])   # rice -> oats, in the app
        is_err, text = self.confirm(code)
        self.assertTrue(is_err)
        self.assertIn("changed since the preview", text)
        self.assertEqual(self.statements("INSERT INTO daily_log"), [])

    def test_template_edited_during_the_preview_is_refused(self):
        real = api_v1.run_changes

        def edit_mid_preview(*a, **k):   # the app saves an edit while the preview transaction runs
            out = real(*a, **k)
            self.template_rows[1] = ("i-oats", "Oats", 1.0, self.template_rows[1][3])
            return out
        with mock.patch.object(api_v1, "run_changes", edit_mid_preview):
            code = self.preview({"template_id": "t1", "date": TODAY}, name="log_template")
        is_err, text = self.confirm(code)
        self.assertTrue(is_err)
        self.assertIn("changed since the preview", text)

    def test_template_retry_after_save_replays(self):
        code = self.preview({"template_id": "t1", "date": TODAY}, name="log_template")
        first = self.confirm(code)
        self.assertFalse(first[0], first[1])
        self.template_rows[1] = ("i-oats", "Oats", 1.0, self.template_rows[1][3])   # edited after the save
        self.conn.executed.clear()
        second = self.confirm(code)   # claude.ai retries a lost answer
        self.assertFalse(second[0], second[1])
        self.assertEqual(json.loads(second[1]), json.loads(first[1]))
        self.assertEqual(self.statements("INSERT INTO daily_log"), [])

    def test_unchanged_template_is_logged(self):
        code = self.preview({"template_id": "t1", "date": TODAY}, name="log_template")
        is_err, text = self.confirm(code)
        self.assertFalse(is_err, text)

    def test_confirmed_deletes_count(self):
        row = self.entry_row()
        self.conn.script.insert(1, ("FROM daily_log WHERE log_id", [row]))
        code = self.preview({"log_id": "l1", "if_match": api_v1.etag_of(*row[1:])}, name="delete_entry")
        with mock.patch.object(main, "_spike", return_value=False) as spike:
            is_err, text = self.confirm(code)
        self.assertFalse(is_err, text)
        spike.assert_any_call("del:admin-1", 60, 600)


class Logging(PreviewBase):
    def test_tool_line_has_outcome_and_no_content(self):
        with self.assertLogs(main.logger, "INFO") as logs:
            self.call_tool(self.banana(), name="log_food")
        line = next(l for l in logs.output if "mcp tool=log_food" in l)
        self.assertIn("mode=preview", line)
        self.assertIn("outcome=ok", line)
        self.assertNotIn("Banana", " ".join(logs.output))
        self.assertNotIn(TODAY, " ".join(logs.output))

    def test_failed_tool_logs_its_error_type(self):
        with self.assertLogs(main.logger, "INFO") as logs:
            self.call_tool({"code": "nope"}, name="confirm_change")
        line = next(l for l in logs.output if "mcp tool=confirm_change" in l)
        self.assertIn("mode=confirm", line)
        self.assertIn("outcome=confirmation_invalid", line)

    def test_refused_version_is_logged(self):
        with self.assertLogs(main.logger, "INFO") as logs:
            self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"MCP-Protocol-Version": "2026-07-28"})
        self.assertTrue(any("mcp version refused: 2026-07-28" in l for l in logs.output))

    def test_origin_is_truncated(self):
        with self.assertLogs(main.logger, "INFO") as logs:
            self.rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Origin": "https://e.example/" + "a" * 500})
        line = next(l for l in logs.output if "origin refused" in l)
        self.assertLess(len(line), 200)
