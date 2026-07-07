-- ============================================================================
-- Row-Level Security (RLS) for NutriScan
-- ----------------------------------------------------------------------------
-- Defense-in-depth for tenant isolation. The app already filters every query
-- by user_id, but a single forgotten WHERE clause would leak data. These
-- policies make Postgres itself reject any row that is not the current user's.
--
-- How it ties to the app:
--   On each request the API binds the authenticated user to a transaction-local
--   GUC `app.user_id` (see get_db() -> set_config('app.user_id', uid, true)).
--   Every policy below compares user_id against current_setting('app.user_id').
--
-- IMPORTANT — Neon's `neondb_owner` role has the BYPASSRLS attribute, which
-- bypasses RLS even with FORCE ROW LEVEL SECURITY (verified empirically).
-- RLS therefore only takes effect when the app connects as the dedicated
-- runtime role `nutriscan_app` created below (roles created via SQL default
-- to NOBYPASSRLS). Until the app's DATABASE_URL is switched to nutriscan_app,
-- applying this file changes nothing for the running app — so it is safe to
-- run at any time. Sequencing:
--   1. Run this file in the Neon SQL editor (replace CHANGE_ME first).
--   2. Deploy the backend code that sets app.user_id.
--   3. Switch DATABASE_URL on Render to connect as nutriscan_app.
-- Switching before step 2 would fail closed (every query sees 0 rows).
--
-- neondb_owner keeps full access (BYPASSRLS), so the Neon SQL editor remains
-- a working admin console. Any NEW table must be re-covered by re-running the
-- policy block below with the table added to the list.
--
-- Fail-closed: if app.user_id is unset, current_setting(..., true) returns NULL,
-- so `user_id = NULL` is false and no rows are visible.
--
-- Run in the Neon SQL editor as neondb_owner. Idempotent (the role's password
-- is only set on first creation).
-- ============================================================================

-- ---------- 1) Least-privilege runtime role (no BYPASSRLS) ----------
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nutriscan_app') THEN
    CREATE ROLE nutriscan_app LOGIN PASSWORD 'CHANGE_ME';  -- set a real secret before running
  END IF;
END $$;

GRANT USAGE, CREATE ON SCHEMA public TO nutriscan_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nutriscan_app;
ALTER DEFAULT PRIVILEGES FOR ROLE neondb_owner IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nutriscan_app;

-- ---------- 2) Enable RLS + per-user isolation policy on every table ----------
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'users', 'api_usage', 'image_records', 'folders', 'folder_items',
    'daily_log', 'user_goals', 'meal_templates', 'meal_template_items',
    'push_subscriptions'
  ]
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY;', t);
    EXECUTE format('ALTER TABLE %I FORCE  ROW LEVEL SECURITY;', t);
    EXECUTE format('DROP POLICY IF EXISTS user_isolation ON %I;', t);
    EXECUTE format($f$
      CREATE POLICY user_isolation ON %I
        USING      (user_id = current_setting('app.user_id', true))
        WITH CHECK (user_id = current_setting('app.user_id', true));
    $f$, t);
  END LOOP;
END $$;

-- ---------- 3) System read access for scheduled jobs ----------
-- The meal-reminder scheduler runs with app.user_id = '__system__' (a real
-- Supabase JWT sub is always a UUID, so no user can occupy this value) and
-- needs cross-user READ on exactly these two tables. Policies are OR'd, so
-- this adds to user_isolation without widening writes (SELECT only).
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['push_subscriptions', 'daily_log']
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS system_read ON %I;', t);
    EXECUTE format($f$
      CREATE POLICY system_read ON %I FOR SELECT
        USING (current_setting('app.user_id', true) = '__system__');
    $f$, t);
  END LOOP;
END $$;

-- Verify:
--   SELECT tablename, policyname FROM pg_policies WHERE schemaname = 'public';
--   -- as nutriscan_app, with no GUC set, this must return 0:
--   SELECT count(*) FROM daily_log;
--
-- Rollback (if needed):
--   DO $$ DECLARE t text; BEGIN
--     FOREACH t IN ARRAY ARRAY['users','api_usage','image_records','folders',
--       'folder_items','daily_log','user_goals','meal_templates',
--       'meal_template_items','push_subscriptions']
--     LOOP
--       EXECUTE format('DROP POLICY IF EXISTS user_isolation ON %I;', t);
--       EXECUTE format('DROP POLICY IF EXISTS system_read ON %I;', t);
--       EXECUTE format('ALTER TABLE %I NO FORCE ROW LEVEL SECURITY;', t);
--       EXECUTE format('ALTER TABLE %I DISABLE  ROW LEVEL SECURITY;', t);
--     END LOOP;
--   END $$;
--   REVOKE ALL ON ALL TABLES IN SCHEMA public FROM nutriscan_app;
--   REVOKE ALL ON SCHEMA public FROM nutriscan_app;
--   DROP ROLE IF EXISTS nutriscan_app;
