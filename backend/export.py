"""GET /export: the user's own data as a download, any time (PRD change 2026-09-25); with the account-deletion
change it also works during the 15 days after deleting, being a read.

XLSX: four sheets. CSV: the same four tables, one file each, in a zip. Numbers come from the helper the /v1 API
uses (energy in kcal, old kJ-looking values converted as the app does); every number column is a total for the
row, servings included."""
import csv
import io
import re
import threading
import time
import zipfile
from collections import deque
from datetime import date
from typing import Literal, Optional

import openpyxl
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from openpyxl.cell import WriteOnlyCell

import api_v1
import log_service

m = None   # main, set by main.py right after importing this module (as api_v1.m): importing it here would
           # load a second copy of main when main.py runs as __main__
router = APIRouter()

MACROS = ["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)", "Sodium (mg)"]
LOG_CAP, ITEM_CAP = 50000, 10000
PER_WINDOW, WINDOW = 5, 600                  # exports per user per 10 minutes: a whole-account read each
_recent: dict = {}                           # user_id -> deque of export times
_recent_lock = threading.Lock()
_building = threading.BoundedSemaphore(2)    # at most two files in memory at once on the one Render process
BUILD_WAIT = 20                              # seconds a third export waits for a slot before a 503
_UNWRITABLE = re.compile(r"[\x00-\x1f\x7f\ufffe\uffff]")   # openpyxl refuses the first, XML 1.0 forbids the rest


def _clean(v):
    """Characters a spreadsheet file cannot hold out (a pasted name can carry one). Excel's 32,767-character
    cell limit needs nothing here: openpyxl truncates (test_a_cell_never_exceeds_the_spreadsheet_limit)."""
    return _UNWRITABLE.sub(" ", v).strip() if isinstance(v, str) else v


def _csv_cell(v):
    """CSV has no cell types: text a spreadsheet would run as a formula gets a leading ' (OWASP CSV injection)."""
    v = _clean(v)
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def _numbers(n, servings) -> list:
    f = api_v1._full_macros(n, servings)
    return [f["calories"], f["protein_g"], f["carbs_g"], f["fat_g"], f["fibre_g"], f["sodium_mg"]]


def _size(n: dict) -> str:
    if n.get("per_serving"):
        return str(n["per_serving"].get("size") or "")
    return "100 g" if n.get("per_100g") else ""   # the numbers are per 100 g times servings then


def _day(v):
    """A real date for the spreadsheet (filters, charts); anything unparseable stays as it was."""
    try:
        return date.fromisoformat(v)
    except (TypeError, ValueError):
        return v


def _capped(rows: list, cap: int, note: str) -> tuple:
    """(rows kept, note row or []): one row past the cap is read only to know whether anything was cut."""
    return (rows[:cap], [[note]]) if len(rows) > cap else (rows, [])


def tables(cur, user_id: str) -> list:
    """[(name, header, rows)] for the four tables, in sheet order."""
    cur.execute("SELECT date, name, servings, nutrition FROM daily_log WHERE user_id = %s "
                f"ORDER BY date DESC, created_at DESC LIMIT {LOG_CAP + 1}", [user_id])   # the cap keeps the newest
    kept, log_note = _capped(cur.fetchall(), LOG_CAP, f"Only the newest {LOG_CAP} entries are included.")
    log = []
    for day, name, servings, raw in reversed(kept):
        n = log_service.load_nutrition(raw)
        log.append([_day(day), n.get("_meal_label") or "", name, servings, _size(n), *_numbers(n, servings),
                    "Yes" if n.get("_source") == "claude" else "No"])
    cur.execute(api_v1.GOALS_SQL, [user_id])
    g = api_v1.goals_from(cur.fetchall())
    cur.execute(f"""SELECT t.name, i.name, i.servings, i.nutrition FROM meal_templates t
                    LEFT JOIN meal_template_items i ON i.template_id = t.template_id
                    WHERE t.user_id = %s ORDER BY t.name, t.template_id, i.created_at LIMIT {ITEM_CAP + 1}""", [user_id])
    kept, templates_note = _capped(cur.fetchall(), ITEM_CAP, f"Only the first {ITEM_CAP} template items are included.")
    templates = [[t, food or "", *([servings, *_numbers(log_service.load_nutrition(raw), servings)]
                                   if food is not None else [])]   # None: a template with no items
                 for t, food, servings, raw in kept]
    cur.execute(f"""SELECT f.name, i.name, i.nutrition FROM folders f
                    LEFT JOIN folder_items i ON i.folder_id = f.folder_id
                    WHERE f.user_id = %s ORDER BY f.name, f.folder_id, i.name LIMIT {ITEM_CAP + 1}""", [user_id])
    kept, library_note = _capped(cur.fetchall(), ITEM_CAP, f"Only the first {ITEM_CAP} Library foods are included.")
    library = []
    for folder, food, raw in kept:
        n = log_service.load_nutrition(raw)
        library.append([folder, food or "", *([_size(n), *_numbers(n, 1)] if food is not None else [])])
    return [
        ("Food log", ["Date", "Meal", "Food", "Servings", "Serving size", *MACROS, "Via Claude"], log + log_note),
        ("Goals", MACROS[:5], [[g["calories"], g["protein_g"], g["carbs_g"], g["fat_g"], g["fibre_g"]]]),
        ("Meal templates", ["Template", "Food", "Servings", *MACROS], templates + templates_note),
        ("Library", ["Folder", "Food", "Serving size", *MACROS], library + library_note),
    ]


def to_xlsx(data: list) -> bytes:
    wb = openpyxl.Workbook(write_only=True)
    for name, header, rows in data:
        ws = wb.create_sheet(name)
        ws.append(header)
        for row in rows:
            cells = []
            for v in map(_clean, row):
                if isinstance(v, str):   # an explicit text cell: "=..." is shown, never run
                    v = WriteOnlyCell(ws, value=v)
                    v.data_type = "s"
                cells.append(v)
            ws.append(cells)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def to_csv_zip(data: list) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, header, rows in data:
            text = io.StringIO()
            w = csv.writer(text)
            w.writerow(header)
            w.writerows([_csv_cell(v) for v in row] for row in rows)
            z.writestr(name.lower().replace(" ", "-") + ".csv", "﻿" + text.getvalue())   # BOM: Excel reads UTF-8
    return out.getvalue()


FORMATS = {"xlsx": (to_xlsx, "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
           "csv": (to_csv_zip, "zip", "application/zip")}


def _allowed(user_id: str):
    """The slot's timestamp if this export may run (hand it back with _release on failure), else None."""
    now = time.time()
    with _recent_lock:
        for uid in [u for u, q in _recent.items() if q and now - q[-1] > WINDOW]:   # forget idle users
            del _recent[uid]
        q = _recent.setdefault(user_id, deque())
        while q and now - q[0] > WINDOW:
            q.popleft()
        if len(q) >= PER_WINDOW:
            return None
        q.append(now)
        return now


def _release(user_id: str, stamp: float):
    """A failed export does not count: the user gets the slot back."""
    with _recent_lock:
        try:
            _recent[user_id].remove(stamp)
        except (KeyError, ValueError):
            pass


@router.get("/export")
def export(format: Literal["xlsx", "csv"] = Query(...), authorization: Optional[str] = Header(default=None)):
    user_id = m.get_user_id(authorization)
    stamp = _allowed(user_id)
    if stamp is None:
        raise HTTPException(status_code=429, detail={"error_type": "rate_limited",
                            "message": "Too many exports. Try again in a few minutes."})
    write, ext, media = FORMATS[format]
    if not _building.acquire(timeout=BUILD_WAIT):   # never park a shared worker thread for long
        _release(user_id, stamp)
        raise HTTPException(status_code=503, headers={"Retry-After": "30"}, detail={
            "error_type": "busy", "message": "Exports are busy right now. Try again in a minute."})
    try:
        with api_v1.db(user_id, commit=False) as cur:
            data = tables(cur, user_id)
        body = write(data)
    except Exception:
        _release(user_id, stamp)
        raise
    finally:
        _building.release()
    return Response(body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="nutriscan-export-{api_v1.melbourne_today().isoformat()}.{ext}"',
        "Cache-Control": "no-store"})
