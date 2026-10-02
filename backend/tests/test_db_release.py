"""Every connection taken with get_db() is handed back in a finally that covers all the work done with it.

A route that released only on success leaked its connection whenever a query failed, until the pool filled
(2026-10-02). This reads the source, so a new route that forgets the finally fails here, not in production.

Run:  venv/Scripts/python -m unittest backend.tests.test_db_release -v
"""
import ast
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, "..", f) for f in ("main.py", "api_v1.py")]


def _calls(node, name):   # get_db(...) and m.get_db(...)
    f = getattr(node, "func", None)
    return isinstance(node, ast.Call) and (getattr(f, "id", None) == name or getattr(f, "attr", None) == name)


def _releases_in_finally(t):
    return isinstance(t, ast.Try) and any(_calls(c, "release_db") for s in t.finalbody for c in ast.walk(s))


def _check(tree):
    """(function, line) of every get_db() not covered, and of every release_db() outside a finally."""
    bad, parent = [], {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parent[c] = p
    for node in ast.walk(tree):
        if _calls(node, "release_db"):
            n, ok = node, False
            while n in parent and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                p = parent[n]
                if isinstance(p, ast.Try) and any(n is s for s in p.finalbody):
                    ok = True
                n = p
            if not ok:
                bad.append(("release_db outside a finally", node.lineno))
        if not _calls(node, "get_db") or isinstance(parent.get(node), ast.Return):
            continue
        stmt = node
        while not isinstance(stmt, ast.stmt):
            stmt = parent[stmt]
        covered, n = False, stmt
        while n in parent and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            p = parent[n]
            if _releases_in_finally(p) and any(n is s for s in p.body):
                covered = True   # inside the try whose finally releases
            n = p
        block = next((getattr(parent[stmt], f) for f in ("body", "orelse", "finalbody")
                      if stmt in getattr(parent[stmt], f, [])), [])
        nxt = block[block.index(stmt) + 1] if stmt in block and block.index(stmt) + 1 < len(block) else None
        if not covered and not _releases_in_finally(nxt):   # or: conn = get_db() directly followed by that try
            bad.append(("get_db not covered by a releasing finally", node.lineno))
    return bad


class ReleaseInFinally(unittest.TestCase):
    def test_every_connection_is_released_in_a_finally_covering_its_use(self):
        for path in FILES:
            with self.subTest(file=os.path.basename(path)):
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(_check(ast.parse(f.read())), [])

    def test_the_check_catches_a_query_between_get_db_and_the_try(self):
        leaky = ("def f():\n    conn = get_db()\n    conn.cursor().execute('x')\n    try:\n        pass\n"
                 "    finally:\n        release_db(conn)\n")
        self.assertEqual(_check(ast.parse(leaky)), [("get_db not covered by a releasing finally", 2)])


if __name__ == "__main__":
    unittest.main()
