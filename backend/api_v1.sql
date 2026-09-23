-- ============================================================================
-- Public API v1: personal access tokens, idempotency, audit, usage, budget
-- ----------------------------------------------------------------------------
-- Run in the Neon SQL editor as neondb_owner, AFTER rls_policies.sql and
-- recycle_bin.sql (they create nutriscan_app and to_recycle_bin()). Idempotent.
-- This file puts the same user_isolation policy on its own four per-user tables
-- itself, so rls_policies.sql never names a table that may not exist yet.
--
-- Deploy order: run this BEFORE the backend code that uses it goes live.
--
-- Why the tables are created here and not in main.py init_db(): in production
-- the app connects as nutriscan_app, and a table it created would be OWNED by
-- it. resolve_api_token() must be owned by neondb_owner to run with owner
-- rights, and keeping every API table under one owner keeps the grants simple.
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_tokens (
  token_id        varchar PRIMARY KEY,
  user_id         varchar NOT NULL,
  name            varchar(40) NOT NULL,
  prefix          varchar(12) NOT NULL,
  token_hash      char(64) NOT NULL UNIQUE,      -- sha256 hex of the full token
  scopes          text[] NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  expires_at      timestamptz,                    -- NULL = never
  revoked_at      timestamptz,
  last_used_at    timestamptz,
  last_user_agent varchar(120)
);
CREATE INDEX IF NOT EXISTS api_tokens_user_idx ON api_tokens (user_id);

CREATE TABLE IF NOT EXISTS api_token_usage (
  token_id varchar NOT NULL,
  date     varchar NOT NULL,                        -- Melbourne-local YYYY-MM-DD
  user_id  varchar NOT NULL,
  reads    int NOT NULL DEFAULT 0,
  writes   int NOT NULL DEFAULT 0,
  PRIMARY KEY (token_id, date)
);

CREATE TABLE IF NOT EXISTS api_idempotency (
  user_id      varchar NOT NULL,
  key          varchar(255) NOT NULL,
  request_hash char(64) NOT NULL,
  status_code  int NOT NULL,
  response     jsonb NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS api_audit (
  audit_id     varchar PRIMARY KEY,
  user_id      varchar NOT NULL,
  token_id     varchar,
  method       varchar(8) NOT NULL,
  path         varchar(200) NOT NULL,
  status       int NOT NULL,
  affected_ids jsonb,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS api_audit_user_idx ON api_audit (user_id, created_at);

-- Not user data: one row per Neon billing period, written only by the server.
-- No RLS (there is no user_id to isolate on); nutriscan_app reads and writes it
-- through the default grants from rls_policies.sql.
CREATE TABLE IF NOT EXISTS ops_budget (
  period_start date PRIMARY KEY,
  awake_seconds double precision NOT NULL DEFAULT 0,
  alerted_70    boolean NOT NULL DEFAULT false,
  paused        boolean NOT NULL DEFAULT false,
  updated_at    timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON api_tokens, api_token_usage, api_idempotency, api_audit, ops_budget TO nutriscan_app;

-- Same per-user isolation as rls_policies.sql, for this file's tables.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['api_tokens', 'api_token_usage', 'api_idempotency', 'api_audit'] LOOP
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

-- The prefixes (first 12 characters, 3 of them random) of every token, revoked
-- and expired included, so those still get their precise 401. The server keeps
-- them in memory so a guessed token that matches no prefix is refused without a
-- query: guesses must not be able to keep Neon awake.
CREATE OR REPLACE FUNCTION api_token_prefixes()
RETURNS SETOF varchar
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  SELECT DISTINCT prefix FROM api_tokens;
$$;
REVOKE ALL ON FUNCTION api_token_prefixes() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION api_token_prefixes() TO nutriscan_app;

-- ---------- Token lookup before the user is known ----------
-- RLS hides api_tokens rows unless app.user_id already names their owner, but a
-- token has to be looked up to FIND the owner. This function runs with owner
-- rights and returns only what the resolver needs for one exact hash - never a
-- list, never another user's row by any other key. It also stamps last-used,
-- which the resolver calls only on a token-cache miss (at most every 5 minutes).
CREATE OR REPLACE FUNCTION resolve_api_token(p_hash text, p_user_agent text)
RETURNS TABLE (token_id varchar, user_id varchar, name varchar, scopes text[],
               expires_at timestamptz, revoked_at timestamptz)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  UPDATE api_tokens t
     SET last_used_at = now(), last_user_agent = left(p_user_agent, 120)
   WHERE t.token_hash = p_hash AND t.revoked_at IS NULL;
  RETURN QUERY
    SELECT t.token_id, t.user_id, t.name, t.scopes, t.expires_at, t.revoked_at
      FROM api_tokens t WHERE t.token_hash = p_hash;
END $$;
REVOKE ALL ON FUNCTION resolve_api_token(text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION resolve_api_token(text, text) TO nutriscan_app;

-- A deleted token row is kept 30 days like every other user table. (Audit and
-- idempotency rows expire by design, so they are not binned.)
DROP TRIGGER IF EXISTS recycle_bin_trg ON api_audit;   -- an earlier draft of this file binned audit rows
DROP TRIGGER IF EXISTS recycle_bin_trg ON api_tokens;
CREATE TRIGGER recycle_bin_trg BEFORE DELETE ON api_tokens FOR EACH ROW EXECUTE FUNCTION to_recycle_bin();

-- Retention, called daily by the server. Cross-user, so it runs with owner
-- rights; the windows are fixed here so the app role cannot shorten them.
CREATE OR REPLACE FUNCTION purge_api_rows() RETURNS void
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  DELETE FROM api_idempotency WHERE created_at < now() - interval '24 hours';
  DELETE FROM api_audit WHERE created_at < now() - interval '90 days';
$$;
REVOKE ALL ON FUNCTION purge_api_rows() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION purge_api_rows() TO nutriscan_app;

-- Verify:
--   SELECT tablename, policyname FROM pg_policies WHERE tablename LIKE 'api_%';
--   -- as nutriscan_app with no GUC set: 0 rows, yet the function still resolves:
--   SELECT count(*) FROM api_tokens;
--   SELECT * FROM resolve_api_token('<sha256 hex>', 'psql');
