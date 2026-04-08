import os
import json
import base64
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
# This will now correctly pull the key from your .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

IMAGE_PATH = r"D:\Projects\OCR\20009878.jpg"
MODEL_ID = "google/gemma-4-26b-a4b-it:free"

PROMPT = (
    "Analyze the nutrition label image. Extract values for both the 'per 130g' (serving) "
    "and the 'per 100g' columns. "
    "Return ONLY a valid JSON object with this exact structure: "
    "{"
    "  'per_serving': {'size': '130g', 'calories': int, 'fat': 'string', 'protein': 'string'},"
    "  'per_100g': {'calories': int, 'fat': 'string', 'protein': 'string'}"
    "}"
    "Ensure 'per_serving' appears first in the JSON. Output numbers as integers where possible."
)

# Initialize the OpenRouter client with a clean URL string
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def encode_image(image_path):
    """Converts the image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_extraction():
    if not OPENROUTER_API_KEY:
        return {"status": "failed", "error": "API Key missing. Check your .env file.", "model": MODEL_ID}

    print(f"🚀 Sending image to {MODEL_ID} via OpenRouter...")
    
    if not os.path.exists(IMAGE_PATH):
        return {"status": "failed", "error": f"Image not found at {IMAGE_PATH}", "model": MODEL_ID}

    base64_image = encode_image(IMAGE_PATH)
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        elapsed_time = round(time.time() - start_time, 2)
        raw_content = response.choices[0].message.content
        
        # Clean up Markdown formatting if the model includes it
        clean_json_str = raw_content.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_json_str)
        
        return {
            "status": "success",
            "time": elapsed_time,
            "model": MODEL_ID,
            "data": parsed_json
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "model": MODEL_ID
        }

def save_and_display(result):
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"hierarchical_extraction_{timestamp}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    print("-" * 50)
    if result["status"] == "success":
        print(f"✅ EXTRACTION SUCCESSFUL ({result['time']}s)")
        print(json.dumps(result["data"], indent=2))
    else:
        print(f"❌ EXTRACTION FAILED")
        print(f"Error: {result['error']}")
    print("-" * 50)
    print(f"Full log saved to: {filename}")

if __name__ == "__main__":
    final_result = run_extraction()
    save_and_display(final_result)