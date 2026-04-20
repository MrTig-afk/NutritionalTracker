import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- LOAD MODEL ---
print("Loading OCR model...")
model = ocr_predictor(pretrained=True)

def clean_json(doctr_result):
    result_dict = doctr_result.export()
    
    for page in result_dict.get("pages", []):
        page.pop("confidence", None)
        page.pop("geometry", None)
        page.pop("objectness_score", None)
        
        for block in page.get("blocks", []):
            block.pop("geometry", None)
            block.pop("confidence", None)
            
            for line in block.get("lines", []):
                line.pop("confidence", None)
                line.pop("geometry", None)
                line.pop("objectness_score", None)
                
                for word in line.get("words", []):
                    word.pop("confidence", None)
                    word.pop("geometry", None)
                    word.pop("objectness_score", None)
                    word.pop("crop_orientation", None)
                    
    return result_dict


# --- HARDCODED IMAGE PATH ---
image_path = r"D:\Projects\Data Engineering Projects\NutritionDE\20009878.jpg"

# --- RUN OCR ---
try:
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    
    cleaned = clean_json(result)
    
    print("\n--- OCR OUTPUT ---\n")
    print(json.dumps(cleaned, indent=2))

except Exception as e:
    print(f"Error: {e}")