import os
import json
import torch
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from dotenv import load_dotenv

# Import your local helper modules
from ocr import run_ocr
from repair import repair_output

load_dotenv()

# ------------------------------
# Configuration
# ------------------------------

# ✅ UPDATED: Your Hugging Face Repo ID
HF_MODEL_ID = "MrTig/NutritinalTracker-checkpoints"

# Set up a temporary save directory for the cropped images
SAVE_DIR = os.getenv("SAVE_DIR", "temp_images")
os.makedirs(SAVE_DIR, exist_ok=True)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load tokenizer + model (from Hugging Face)
# ------------------------------
@st.cache_resource
def load_model():
    """
    Loads the model and tokenizer from the Hugging Face Hub.
    Results are cached so this only runs once.
    """
    st.info(f"Loading model from Hugging Face Hub: {HF_MODEL_ID}...")
    try:
        # This will download config.json, model.safetensors, etc. automatically
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = LongT5ForConditionalGeneration.from_pretrained(HF_MODEL_ID).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Could not find repo: {HF_MODEL_ID}. Please check if the repo is private (requires token) or if the name is spelled correctly.")
        st.stop()

tokenizer, model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("POC: Image → Nutritional Information")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)

    # --------------------------
    # Rotation controls
    # --------------------------
    if "rotation" not in st.session_state:
        st.session_state.rotation = 0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("↺ Rotate Left"):
            st.session_state.rotation = (st.session_state.rotation - 90) % 360
    with col2:
        if st.button("↻ Rotate Right"):
            st.session_state.rotation = (st.session_state.rotation + 90) % 360
    with col3:
        if st.button("⟳ Reset"):
            st.session_state.rotation = 0

    rotated_img = img.rotate(st.session_state.rotation, expand=True)

    # --------------------------
    # Crop after rotation
    # --------------------------
    st.write("Crop the image to focus on the nutritional table:")
    cropped_img = st_cropper(rotated_img, aspect_ratio=None)
    st.image(cropped_img, caption="Preview", width=400)

    # --------------------------
    # Barcode input
    # --------------------------
    barcode = st.text_input("Enter barcode (digits only, max 13 digits)")

    # --------------------------
    # Run model
    # --------------------------
    if st.button("Run Model"):
        # Basic validation
        if barcode and (not barcode.isdigit() or len(barcode) > 13):
            st.error("❌ Invalid barcode: must be digits only and ≤13 digits.")
        else:
            safe_barcode = barcode if barcode else "unknown_item"

            # Save cropped image locally for OCR processing
            filename = f"{safe_barcode}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cropped_img.save(save_path, format="JPEG")

            # Run OCR
            st.info("🔍 Running OCR on image...")
            try:
                ocr_output = run_ocr(save_path)
            except Exception as e:
                st.error(f"OCR Failed: {e}")
                st.stop()

            # Model inference
            st.info("🤖 Processing with AI Model...")
            input_text = json.dumps(ocr_output, separators=(",", ":"))

            # Tokenize with truncation to ensure we don't exceed model limits
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048 
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    num_beams=4,
                    early_stopping=True
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Repair JSON
            repaired = repair_output(decoded)
            st.subheader("Model Output")
            st.json(repaired)