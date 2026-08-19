-- ============================================================================
-- Recycle bin for NutriScan — every deleted user row is kept 30 days
-- ----------------------------------------------------------------------------
-- Why: a hijacked session (or a fat finger) can delete data in one tap. The
-- app freezes the account on a delete spree (see abuse_guard in main.py), but
-- whatever was deleted before the freeze must still be recoverable. Doing it
-- with a BEFORE DELETE trigger means every delete path — the 7 routes,
-- DELETE /account, anything added later — is covered with zero Python.
--
-- Isolation: the runtime role `nutriscan_app` gets NO privileges on
-- recycle_bin. The trigger and the purge run as SECURITY DEFINER (owned by
-- neondb_owner), so the API can write to the bin through the trigger and purge
-- old rows through purge_recycle_bin(), but can never read or empty it. Only
-- neondb_owner (Neon SQL editor / backend/.env) can restore.
--
-- Run ONCE in the Neon SQL editor as neondb_owner. Idempotent.
-- ============================================================================

CREATE TABLE IF NOT EXISTS recycle_bin (
  id         bigserial PRIMARY KEY,
  tbl        text        NOT NULL,
  user_id    varchar,
  row        jsonb       NOT NULL,
  deleted_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS recycle_bin_user_idx ON recycle_bin (user_id, deleted_at);

-- rls_policies.sql granted default privileges on new tables to nutriscan_app;
-- take them back for this one table.
REVOKE ALL ON recycle_bin FROM nutriscan_app;
REVOKE ALL ON SEQUENCE recycle_bin_id_seq FROM nutriscan_app;

CREATE OR REPLACE FUNCTION to_recycle_bin() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  INSERT INTO recycle_bin (tbl, user_id, row)
  VALUES (TG_TABLE_NAME, OLD.user_id, to_jsonb(OLD));
  RETURN OLD;
END $$;

-- Retention is fixed inside the function on purpose: the app role may call it
-- (daily, from the scheduler) but cannot shorten the window.
CREATE OR REPLACE FUNCTION purge_recycle_bin() RETURNS int
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE n int;
BEGIN
  DELETE FROM recycle_bin WHERE deleted_at < now() - interval '30 days';
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;
REVOKE ALL ON FUNCTION purge_recycle_bin() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION purge_recycle_bin() TO nutriscan_app;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'daily_log', 'folders', 'folder_items', 'meal_templates', 'meal_template_items',
    'user_goals', 'image_records', 'notification_prefs', 'users'
  ]
  LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS recycle_bin_trg ON %I;', t);
    EXECUTE format('CREATE TRIGGER recycle_bin_trg BEFORE DELETE ON %I FOR EACH ROW EXECUTE FUNCTION to_recycle_bin();', t);
  END LOOP;
END $$;

-- Verify:
--   SELECT tgrelid::regclass, tgname FROM pg_trigger WHERE tgname = 'recycle_bin_trg';
--   -- as nutriscan_app this must fail with "permission denied":
--   SELECT count(*) FROM recycle_bin;
--
-- Restore (as neondb_owner; one statement per table, newest copy wins):
--   INSERT INTO daily_log
--   SELECT (jsonb_populate_record(NULL::daily_log, row)).*
--   FROM recycle_bin WHERE user_id = '<uuid>' AND tbl = 'daily_log'
--   ON CONFLICT DO NOTHING;
--
-- Rollback:
--   DO $$ DECLARE t text; BEGIN
--     FOREACH t IN ARRAY ARRAY['daily_log','folders','folder_items','meal_templates',
--       'meal_template_items','user_goals','image_records','notification_prefs','users']
--     LOOP EXECUTE format('DROP TRIGGER IF EXISTS recycle_bin_trg ON %I;', t); END LOOP;
--   END $$;
--   DROP FUNCTION IF EXISTS to_recycle_bin(); DROP FUNCTION IF EXISTS purge_recycle_bin();
--   DROP TABLE IF EXISTS recycle_bin;
