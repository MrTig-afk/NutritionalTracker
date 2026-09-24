"""One-time migration: settle every stored calorie value to kcal (PRD "Change
2026-09-24: every calorie is stored in kcal").

Rows in daily_log, meal_template_items and folder_items that carry no `_kcal`
tag get it; a per-serving or per-100 g calorie value over 900 (or labelled kJ)
is divided by 4.184 first - exactly what the screen shows today, so nothing a
user sees changes. Already-tagged rows are left alone, so it is safe to re-run
(do: once before the deploy that tags new writes, once right after it).

    DATABASE_URL=<owner connection> python backend/migrate_kcal.py            # preview, writes nothing
    DATABASE_URL=<owner connection> python backend/migrate_kcal.py --apply    # one transaction
    ... --exclude "Big pasta" --exclude "Pizza"   # really kcal: tag without converting

Run as neondb_owner (RLS hides other users' rows from the app role). Take a
Neon backup branch of production before --apply.

Stale tabs: after the deploy, the app stores any untagged number it is sent as
kcal. A PWA tab opened before the migration could send back an old kJ value it
loaded then. That can only happen if some stored value WAS converted, so if the
preview converts nothing (as on 2026-09-24: 0 of 375 values over 900) there is
nothing for a stale tab to carry; otherwise ask users to reload after the deploy.
"""
import argparse
import json
import os
import sys

import psycopg2

from log_service import load_nutrition, settle_kcal

TABLES = {"daily_log": ("log_id", "date"), "meal_template_items": ("item_id", "template_id"),
          "folder_items": ("item_id", "folder_id")}


def _cals(n: dict) -> list:
    return [(n.get(k) or {}).get("calories") for k in ("per_serving", "per_100g")] if (
        isinstance(n.get("per_serving"), dict) or isinstance(n.get("per_100g"), dict)) else [n.get("calories")]


def plan(cur, exclude: set) -> list:
    """(table, id, where, name, before, after, new nutrition) for every untagged row."""
    out = []
    for table, (id_col, where_col) in TABLES.items():
        cur.execute(f"SELECT {id_col}, {where_col}, name, nutrition FROM {table} "   # names come from TABLES, not input
                    "WHERE COALESCE(nutrition->>'_kcal', '') <> 'true' FOR UPDATE")
        for rid, where, name, raw in cur.fetchall():
            n = load_nutrition(raw)
            new = settle_kcal(n, legacy_guess=(name or "").lower() not in exclude)
            out.append((table, rid, str(where), name, _cals(n), _cals(new), new))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the changes (default: preview only)")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="a food name whose big number really is kcal (case-insensitive, repeatable)")
    a = ap.parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set")
    conn = psycopg2.connect(url)
    try:
        cur = conn.cursor()
        rows = plan(cur, {x.lower() for x in a.exclude})
        changed = [r for r in rows if r[4] != r[5]]
        for table in TABLES:
            print(f"{table}: {sum(r[0] == table for r in rows)} untagged rows, "
                  f"{sum(r[0] == table for r in changed)} with calories converted")
        for table, _, where, name, before, after, _ in changed:
            print(f"  {table} | {where} | {name} | {before} -> {after}")
        if not a.apply:
            conn.rollback()
            print("PREVIEW ONLY: nothing written. Re-run with --apply to write.")
            return 0
        for table, rid, *_, new in rows:
            cur.execute(f"UPDATE {table} SET nutrition = %s WHERE {TABLES[table][0]} = %s", [json.dumps(new), rid])
        conn.commit()
        print(f"APPLIED: {len(rows)} rows tagged, {len(changed)} converted.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
