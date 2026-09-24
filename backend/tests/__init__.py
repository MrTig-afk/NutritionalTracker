"""Runs before any test module imports main.

No real database: main.py loads backend/.env at import, and load_dotenv never
overrides a variable that is already set, so pinning DATABASE_URL here wins.
It is empty unless NUTRI_TEST_DATABASE_URL names a throwaway database.

No real alerts, ever: importing main loads the real admin push config. Tests
that exercise the abuse guards (the IP burst cap, the delete-spike freeze)
reach notify_admin, which used to send a genuine push to the maintainer's
phone on every local run. CI never saw it because .env is gitignored there, so
it only ever hit whoever ran the suite locally. The outbound channels are
stubbed rather than notify_admin itself, so its cooldown and bookkeeping still
run and stay under test. A test suite must not be able to page a human.
"""
import os
import sys

os.environ["DATABASE_URL"] = os.environ.get("NUTRI_TEST_DATABASE_URL", "")
# Same trick for the Supabase admin key: DELETE /account and freeze_user call
# the real admin API with it, and a test must never reach production.
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import main  # noqa: E402

ALERTS = []                           # (title, message), if a test ever wants them
REAL_ADMIN_PUSH = main._admin_push    # test_phase0 tests the real one
main._admin_push = lambda title, message: ALERTS.append((title, message))
main.send_push_to_user = lambda user_id, title, message: ALERTS.append((title, message))

# No real model calls either: importing main builds live Groq/Gemini clients from
# backend/.env, and a /chat test without its own stub once reached api.groq.com.
# Tests that need a model patch in a stand-in, as they already do.
main.groq_client = None
main.gemini_client = None
