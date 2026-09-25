-- ============================================================================
-- Connected apps (lane H, Part B): one row per Claude connection
-- ----------------------------------------------------------------------------
-- Run in the Neon SQL editor as neondb_owner, AFTER rls_policies.sql,
-- recycle_bin.sql and api_v1.sql. Idempotent. Deploy order: take a Neon backup
-- branch, run this, THEN deploy the backend that reads it.
-- ============================================================================

CREATE TABLE IF NOT EXISTS connected_apps (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      varchar NOT NULL,
  client_id    varchar(64) NOT NULL,
  name         varchar(40) NOT NULL DEFAULT 'Claude',
  connected_at timestamptz NOT NULL DEFAULT now(),
  last_used_at timestamptz,
  revoked_at   timestamptz,
  UNIQUE (user_id, client_id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON connected_apps TO nutriscan_app;

ALTER TABLE connected_apps ENABLE ROW LEVEL SECURITY;
ALTER TABLE connected_apps FORCE  ROW LEVEL SECURITY;
DROP POLICY IF EXISTS user_isolation ON connected_apps;
CREATE POLICY user_isolation ON connected_apps
  USING      (user_id = NULLIF(current_setting('app.user_id', true), ''))
  WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), ''));

-- Kept 30 days after delete, like every other user table.
DROP TRIGGER IF EXISTS recycle_bin_trg ON connected_apps;
CREATE TRIGGER recycle_bin_trg BEFORE DELETE ON connected_apps FOR EACH ROW EXECUTE FUNCTION to_recycle_bin();
