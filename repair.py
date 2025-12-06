import json
import re

# ------------------------------
# JSON Repair function (all strings, strict schema)
# ------------------------------
SCHEMA = {
    "nutritional_information": {
        "serving_size": "",
        "per_serving": {
            "calories": "",
            "protein": "",
            "saturated_fat": "",
            "trans_fat": "",
            "carbohydrate": "",
            "fiber": "",
            "cholesterol": "",
            "sodium": ""
        },
        "per_100g": {
            "calories": "",
            "protein": "",
            "saturated_fat": "",
            "trans_fat": "",
            "carbohydrate": ""
        }
    }
}


def repair_output(text: str):
    """
    Force model output into the predefined schema.
    - All values are strings.
    - Captures per_serving and per_100g blocks.
    """
    result = json.loads(json.dumps(SCHEMA))  # deep copy

    # Normalize quotes
    text = text.replace("“", '"').replace("”", '"').replace("'", '"')

    # Extract serving_size
    m = re.search(r'serving_size"\s*:\s*"([^"]*)"', text)
    if m:
        result["nutritional_information"]["serving_size"] = m.group(1)

    # Extract per_serving block (before per_100g)
    per_serving_text = text.split('per_100g')[0]
    kv_pairs = re.findall(r'([\w\-]+)"?\s*:\s*"?([\w\.\% ]*?)"?[,}]', per_serving_text)
    for k, v in kv_pairs:
        if k in result["nutritional_information"]["per_serving"]:
            result["nutritional_information"]["per_serving"][k] = str(v).strip()

    # Extract per_100g block (after per_100g)
    per_100g_text = text.split('per_100g')[-1]
    kv_pairs_100g = re.findall(r'([\w\-]+)"?\s*:\s*"?([\w\.\% ]*?)"?[,}]', per_100g_text)
    for k, v in kv_pairs_100g:
        if k in result["nutritional_information"]["per_100g"]:
            result["nutritional_information"]["per_100g"][k] = str(v).strip()

    return result
