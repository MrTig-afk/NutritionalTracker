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


# "1,500" as well as "1500.5"; never starts mid-number, so "15,00" matches "15" rather than "00"
_NUM = r"(?<![\d,.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"   # ".5" too


def _parse_num(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(_NUM, str(v))
    try:
        return float(m.group().replace(",", "")) if m else 0.0
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


_KCAL_NUM = re.compile(_NUM + r"\s*(?:kcal|kilocalories?)", re.I)
_KJ_NUM = re.compile(_NUM + r"\s*(?:kj|kilojoules?)", re.I)


def settle_kcal(nutrition, legacy_guess: bool) -> dict:
    """Store-time settlement: a copy whose calories are kcal, tagged `_kcal: true`.

    A value with a unit uses the number next to that unit: kcal if present,
    else kJ converted ("358 kcal / 1500 kJ" is 358, "1500 kJ" is 358.5). `legacy_guess` is for
    untagged data already in the database (template items copied into the log,
    the one-time migration): there a bare number over 900 is read as kJ, exactly
    as entry_macros() shows it today. A number from a client is taken as kcal.
    """
    n = load_nutrition(nutrition)
    if n.get("_kcal") is True and legacy_guess:   # our own stored tag; a client's tag is not trusted
        return n
    out = dict(n)
    sections = [k for k in ("per_serving", "per_100g") if isinstance(out.get(k), dict)] or [None]
    for k in sections:
        sec = out if k is None else dict(out[k])
        v = sec.get("calories")
        s = v if isinstance(v, str) else ""
        kcal, kj = _KCAL_NUM.search(s), _KJ_NUM.search(s)
        if kcal:
            sec["calories"] = float(kcal.group(1).replace(",", ""))
        elif kj:
            sec["calories"] = round(float(kj.group(1).replace(",", "")) / KJ_PER_KCAL, 1)
        elif legacy_guess and _parse_num(v) > KJ_HEURISTIC_THRESHOLD:
            sec["calories"] = round(_parse_num(v) / KJ_PER_KCAL, 1)
        if k:
            out[k] = sec
    out["_kcal"] = True
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
    # store-time settlement: typed 950 stays kcal; a legacy untagged 950 becomes 227.1; "1500 kJ" always converts
    assert settle_kcal({"per_serving": {"calories": 950}}, False) == {"per_serving": {"calories": 950}, "_kcal": True}
    assert settle_kcal({"per_serving": {"calories": 950}}, True)["per_serving"]["calories"] == 227.1
    assert settle_kcal({"per_serving": {"calories": "1500 kJ"}}, False)["per_serving"]["calories"] == 358.5
    assert settle_kcal({"per_serving": {"calories": "358 kcal / 1500 kJ"}}, False)["per_serving"]["calories"] == 358.0
    assert settle_kcal({"per_serving": {"calories": "1500kJ (358 kcal)"}}, True)["per_serving"]["calories"] == 358.0
    assert settle_kcal({"per_serving": {"calories": "1,500 kJ"}}, False)["per_serving"]["calories"] == 358.5
    assert settle_kcal({"per_serving": {"calories": "1,050 kcal"}}, False)["per_serving"]["calories"] == 1050.0
    # a client's _kcal tag is not trusted (its unit strings still settle); our own stored tag is
    assert settle_kcal({"_kcal": True, "per_serving": {"calories": "1500 kJ"}}, False)["per_serving"]["calories"] == 358.5
    assert settle_kcal({"_kcal": True, "per_serving": {"calories": 1500}}, True)["per_serving"]["calories"] == 1500
    # bare comma-grouped numbers read in full
    assert _parse_num("1,500") == 1500 and _parse_num("15,00") == 15 and _parse_num("250 kcal") == 250
    assert _parse_num(".5 g") == 0.5 and _parse_num("0.5g") == 0.5 and _parse_num("1.2.3") == 1.2
    assert entry_macros(settle_kcal({"per_serving": {"calories": 950}}, True), 1)["calories"] == 227.1
    print("ok")
