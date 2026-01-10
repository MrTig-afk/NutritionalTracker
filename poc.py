import os
import json
import torch
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
from ocr import run_ocr
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from repair import repair_output
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
if not MODEL_DIR:
    raise ValueError("Please set the path to the Model Weights.")

SAVE_DIR = os.getenv("SAVE_DIR")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = LongT5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("POC: Screenshot → Nutrition Info")

uploaded_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg", "heic"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)

    # CRITICAL FIXES FOR IPHONE
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img.thumbnail((1000, 1000))

    use_cropper = st.checkbox("Crop Image", value=True)

    if use_cropper:
        cropped_img = st_cropper(img, aspect_ratio=None, box_color="red")
        st.caption("Tip: If the cropper freezes, uncheck 'Crop Image'.")
    else:
        cropped_img = img
        st.image(img, caption="Using Full Screenshot", width=400)

    barcode = st.text_input("Enter barcode (digits only)", max_chars=13)

    if st.button("Run Model"):
        safe_name = barcode if (barcode and barcode.isdigit()) else "temp_screenshot"
        filename = f"{safe_name}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        cropped_img.save(save_path, format="JPEG", quality=85)

        ocr_output = run_ocr(save_path)

        st.info("🤖 Processing Nutrition Data...")
        input_text = json.dumps(ocr_output, separators=(",", ":"))
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        repaired = repair_output(decoded)

        st.subheader("Result")
        st.json(repaired)