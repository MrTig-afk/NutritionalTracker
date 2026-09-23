"""The app's startup hook runs under whatever FastAPI Render installs (requirements
are unpinned). A deploy once died here: startup listed app.routes and read
route.path, which FastAPI 0.141's included routers do not have.

Run:  venv/Scripts/python -m unittest backend.tests.test_startup -v
No database: every loader startup calls is stubbed.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


class StartupTest(unittest.TestCase):
    def test_startup_completes(self):
        with mock.patch.object(main, "init_db"), mock.patch.object(main, "_load_admin_subs"), \
                mock.patch.object(main, "_load_budget"), mock.patch.object(main.api_v1, "load_prefixes"), \
                mock.patch.object(main, "_purge_recycle_bin"), mock.patch.object(main, "MEAL_REMINDERS_ENABLED", False):
            with TestClient(main.app) as c:
                self.assertEqual(c.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
