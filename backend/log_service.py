"""Food-log arithmetic shared by the app routes and /v1.

One place decides what an entry is worth, so the Tracker, Trends, the goal push
and the assistant cannot drift apart. Imports nothing from main.py.
"""
import json
import re

KJ_PER_KCAL = 4.184
KJ_HEURISTIC_THRESHOLD = 900
MACRO_KEYS = ("calories", "protein", "carbs", "fat", "fibre")
_NUTRIENT_OF = {"protein": "protein", "carbs": "carbohydrates", "fat": "fat", "fibre": "fibre"}


def _parse_num(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d.]+", str(v))
    try:
        return float(m.group()) if m else 0.0
    except ValueError:  # "." alone, or "1.2.3"
        return 0.0


def load_nutrition(nutrition_raw) -> dict:
    if not nutrition_raw:
        return {}
    n = nutrition_raw if isinstance(nutrition_raw, dict) else json.loads(nutrition_raw)
    return n if isinstance(n, dict) else {}


def per_serving_section(n: dict) -> dict:
    """per_serving if filled, else per_100g, else the object itself."""
    if n.get("per_serving"):
        return n["per_serving"]
    if n.get("per_100g"):
        return n["per_100g"]
    return n


def entry_macros(nutrition_raw, servings) -> dict:
    """Unrounded {calories, protein, carbs, fat, fibre} for one log row.

    A per-serving calories value over 900 is assumed to be kJ (scanned labels)
    unless the entry carries `_kcal: true`, which API writes always set because
    the caller states kcal explicitly.
    """
    n = load_nutrition(nutrition_raw)
    ps = per_serving_section(n)
    s = float(servings or 0)
    cal = _parse_num(ps.get("calories", 0))
    if cal > KJ_HEURISTIC_THRESHOLD and n.get("_kcal") is not True:
        cal = cal / KJ_PER_KCAL
    out = {"calories": cal * s}
    for k, src in _NUTRIENT_OF.items():
        out[k] = _parse_num(ps.get(src, 0)) * s
    return out


def sum_macros(rows) -> dict:
    """rows: iterable of (servings, nutrition_raw). Unrounded totals."""
    t = dict.fromkeys(MACRO_KEYS, 0.0)
    for servings, nutrition_raw in rows:
        m = entry_macros(nutrition_raw, servings)
        for k in MACRO_KEYS:
            t[k] += m[k]
    return t


def rounded(d: dict, ndigits: int = 1) -> dict:
    return {k: round(v, ndigits) for k, v in d.items()}


if __name__ == "__main__":
    # a chat entry at 950 kcal stays 950; an untagged label at 950 is read as kJ
    assert entry_macros({"_kcal": True, "per_serving": {"calories": 950}}, 1)["calories"] == 950
    assert round(entry_macros({"per_serving": {"calories": 950}}, 1)["calories"], 1) == 227.1
    # the heuristic looks at the per-serving value, not the multiplied one
    assert entry_macros({"per_serving": {"calories": 500}}, 2)["calories"] == 1000
    print("ok")
