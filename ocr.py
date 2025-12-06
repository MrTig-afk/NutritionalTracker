# ocr.py
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Initialize OCR model once
model = ocr_predictor(pretrained=True)

def run_ocr(image_path):
    """
    Run OCR on an image file saved on disk and return cleaned JSON.
    """
    doc = DocumentFile.from_images(image_path)  # file path
    result = model(doc)
    result_dict = result.export()

    # Clean unwanted fields
    for page in result_dict.get("pages", []):
        page.pop("confidence", None)
        page.pop("geometry", None)
        page.pop("objectness_score", None)
        for block in page.get("blocks", []):
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
