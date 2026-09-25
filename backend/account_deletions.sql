-- ============================================================================
-- Account deletion with a 15-day grace period (PRD change 2026-09-25)
-- ----------------------------------------------------------------------------
-- A row here means: this account asked to be deleted and is view-only until
-- delete_after, when the daily clean-up deletes everything. "Keep my account"
-- deletes the row. Run in the Neon SQL editor as neondb_owner, AFTER
-- rls_policies.sql and recycle_bin.sql. Idempotent. Deploy order: take a Neon backup branch, run
-- this, THEN deploy the backend that reads it.
-- ============================================================================

BEGIN;   -- one step: the functions are dropped and re-created below

CREATE TABLE IF NOT EXISTS account_deletions (
  user_id      varchar PRIMARY KEY,
  requested_at timestamptz NOT NULL DEFAULT now(),
  delete_after timestamptz NOT NULL
);

-- Set by the day-15 purge: the row then stays as a tombstone (the account is gone and its old access tokens are
-- refused, across restarts) until no access token from before the purge can still be valid.
ALTER TABLE account_deletions ADD COLUMN IF NOT EXISTS purged_at timestamptz;

GRANT SELECT, INSERT, UPDATE, DELETE ON account_deletions TO nutriscan_app;

ALTER TABLE account_deletions ENABLE ROW LEVEL SECURITY;
ALTER TABLE account_deletions FORCE  ROW LEVEL SECURITY;
DROP POLICY IF EXISTS user_isolation ON account_deletions;
CREATE POLICY user_isolation ON account_deletions
  USING      (user_id = NULLIF(current_setting('app.user_id', true), ''))
  WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), ''));
-- No recycle-bin trigger: a restored or purged marker is not user data.

-- The server reads across users (the lock map, the daily purge): owner rights,
-- like purge_api_rows. They return ids and dates only.
-- Dropped first: CREATE OR REPLACE cannot change a function's result columns.
DROP FUNCTION IF EXISTS pending_account_deletions();
CREATE FUNCTION pending_account_deletions()
RETURNS TABLE (user_id varchar, delete_after timestamptz, purged_at timestamptz)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT d.user_id, d.delete_after, d.purged_at FROM account_deletions d;
$$;
REVOKE ALL ON FUNCTION pending_account_deletions() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION pending_account_deletions() TO nutriscan_app;

DROP FUNCTION IF EXISTS due_account_deletions();
CREATE FUNCTION due_account_deletions()
RETURNS TABLE (user_id varchar, purged_at timestamptz)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  -- work to do: due and not purged yet, or a tombstone old enough to go (8 days: past any Supabase access
  -- token's life, which is configurable up to a week)
  SELECT d.user_id, d.purged_at FROM account_deletions d
  WHERE d.delete_after <= now() AND (d.purged_at IS NULL OR d.purged_at < now() - interval '8 days');
$$;
REVOKE ALL ON FUNCTION due_account_deletions() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION due_account_deletions() TO nutriscan_app;

-- Day 15 empties the account's recycle-bin rows too (things it deleted earlier), so "permanently deleted"
-- does not mean "30 days later". Bound to the transaction's app.user_id, never a parameter: the app role can
-- only ever empty the bin of the account it is acting as. Needs recycle_bin.sql.
CREATE OR REPLACE FUNCTION purge_user_recycle_bin() RETURNS int
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE n int;
BEGIN
  DELETE FROM recycle_bin WHERE user_id = NULLIF(current_setting('app.user_id', true), '');
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;
REVOKE ALL ON FUNCTION purge_user_recycle_bin() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION purge_user_recycle_bin() TO nutriscan_app;

COMMIT;

-- Verify:
--   SELECT tablename, policyname FROM pg_policies WHERE tablename = 'account_deletions';
--   -- as nutriscan_app with no GUC set: 0 rows, yet the functions still resolve:
--   SELECT count(*) FROM account_deletions;
--   SELECT * FROM pending_account_deletions();
