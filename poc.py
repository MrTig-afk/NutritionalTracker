import os
import io
import json
import torch
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from dotenv import load_dotenv
from ocr import run_ocr
from repair import repair_output

# IMPORTANT: For HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass # HEIC support optional on Android

load_dotenv()

HF_MODEL_ID = "MrTig/NutritinalTracker-checkpoints"
SAVE_DIR = os.getenv("SAVE_DIR", "temp_images")
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    with st.spinner("Initializing model..."):
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = LongT5ForConditionalGeneration.from_pretrained(HF_MODEL_ID).to(device)
        model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------
# Image Processing
# ------------------------------
def process_uploaded_image(uploaded_file):
    try:
        # CRITICAL: Reset file pointer for Android camera uploads
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        
        # Auto-rotate (Critical for iPhone/Android)
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too huge (Prevents crash on mobile RAM)
        max_dimension = 2000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# ------------------------------
# Display Helper
# ------------------------------
def display_nutrition_data(data, cropped_image):
    if not isinstance(data, dict):
        st.error("Invalid data")
        return
    
    if "nutritional_information" in data:
        data = data["nutritional_information"]

    per_100g = data.get("per_100g", {})
    per_serving = data.get("per_serving", {})
    serving_size = data.get("serving_size", "-")

    rows = [
        ("Calories", ["calories"]),
        ("Total Fat", ["fat", "total_fat"]),
        ("Saturated Fat", ["saturated_fat", "saturated"]),
        ("Trans Fat", ["trans_fat", "trans"]),
        ("Cholesterol", ["cholesterol"]),
        ("Sodium", ["sodium"]),
        ("Carbohydrates", ["carbohydrate", "carbohydrates"]),
        ("Dietary Fiber", ["fiber", "dietary_fiber", "fibre"]),
        ("Sugars", ["sugar", "sugars"]),
        ("Protein", ["protein"]),
    ]

    rows_html = ""
    for label, keys in rows:
        val_100g = "-"
        val_serving = "-"
        for k in keys:
            if k in per_100g and per_100g[k] is not None:
                val_100g = per_100g[k]
                if label == "Calories" and "kcal" not in str(val_100g).lower():
                    val_100g = f"{val_100g}kCal"
            if k in per_serving and per_serving[k] is not None:
                val_serving = per_serving[k]
                if label == "Calories" and "kcal" not in str(val_serving).lower():
                    val_serving = f"{val_serving}kCal"
        rows_html += f"<tr><td>{label}</td><td>{val_100g}</td><td>{val_serving}</td></tr>"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(cropped_image, caption="Cropped Label")
    with col2:
        st.markdown(f"""
        <style>
            .nutrition-table {{ width: 100%; border-collapse: collapse; font-family: Arial; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }}
            .nutrition-table th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; text-align: center; }}
            .nutrition-table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
            .nutrition-table tr:nth-child(even) {{ background: #f8f8f8; }}
        </style>
        <div style="margin-bottom:15px;"><strong>Serving Size:</strong> {serving_size}</div>
        <table class="nutrition-table">
            <thead><tr><th>Macros</th><th>Per 100g</th><th>Per Serving</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1,1])
        with c1:
            st.download_button("📥 JSON", json.dumps(data, indent=2), "nutrition.json", "application/json", use_container_width=True)
        with c2:
            if st.button("🔄 Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# ------------------------------
# Main UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Nutritional Tracker")
st.title("🍎 Nutritional Information Extractor")

# Crash Prevention
if "last_processed_time" in st.session_state:
    del st.session_state["last_processed_time"]

# Initialize State
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data", "zoom_level"]:
    if key not in st.session_state:
        st.session_state[key] = None if "image" in key else (False if "confirmed" in key else 1.0)

# ------------------------------
# FILE UPLOADER (Native)
# ------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload or take a photo", 
    type=["png", "jpg", "jpeg", "heic", "heif", "webp"],
    accept_multiple_files=False,
    key="file_uploader"
)

# ------------------------------
# LOGIC
# ------------------------------
if uploaded_file:
    # Stable signature check
    file_signature = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    
    if st.session_state.last_processed_file != file_signature:
        with st.spinner("📸 Processing image..."):
            processed_img = process_uploaded_image(uploaded_file)
            
            if processed_img:
                st.session_state.original_image = processed_img
                # Reset downstream steps
                st.session_state.crop_confirmed = False
                st.session_state.cropped_image = None
                st.session_state.results_data = None
                st.session_state.rotation = 0
                st.session_state.zoom_level = 1.0
                
                # Mark as processed
                st.session_state.last_processed_file = file_signature
                st.session_state.upload_counter += 1
                
                st.success("✅ Image uploaded!")
                st.rerun()
            else:
                st.error("Failed to process image.")

    # ------------------------------
    # App runs if image is loaded
    # ------------------------------
    if st.session_state.original_image:
        img = st.session_state.original_image

        # --- STEP 1: CROP ---
        if not st.session_state.crop_confirmed:
            st.subheader("1️⃣ Crop Label")
            
            # --- REPLACED BUTTONS WITH SLIDER CONTROLS ---
            control_col1, control_col2 = st.columns([2, 1])
            
            with control_col1:
                # Slider for Zoom
                zoom_val = st.slider(
                    "🔍 Zoom Level", 
                    min_value=1.0, 
                    max_value=3.0, 
                    value=st.session_state.get("zoom_level", 1.0),
                    step=0.1,
                    key="zoom_slider"
                )
                # Update session state if slider moved
                if zoom_val != st.session_state.get("zoom_level", 1.0):
                    st.session_state.zoom_level = zoom_val
                    st.rerun()

            with control_col2:
                # Rotation Controls (Compact)
                rot_col1, rot_col2, rot_col3 = st.columns(3)
                if rot_col1.button("↺", help="Rotate Left"):
                    st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                    st.rerun()
                if rot_col2.button("↻", help="Rotate Right"):
                    st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                    st.rerun()
                if rot_col3.button("🗑️", help="Clear Image"):
                    st.session_state.last_processed_file = None
                    st.session_state.original_image = None
                    st.rerun()
            
            # Apply Rotation
            rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
            
            # Apply Zoom (Scaling)
            zoom = st.session_state.get("zoom_level", 1.0)
            if zoom != 1.0:
                new_size = tuple(int(dim * zoom) for dim in rotated_img.size)
                # Use Resampling.NEAREST for speed during crop preview, or LANCZOS for quality
                display_img = rotated_img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                display_img = rotated_img

            c1, c2 = st.columns([1, 1])
            with c1:
                # Add a hint for mobile users about the slider
                st.caption("📱 Use the slider above to zoom. Drag corners to crop.")
                
                # Unique key prevents crop box from jumping around
                cropped_img = st_cropper(
                    display_img, 
                    realtime_update=True, 
                    box_color='#FF0000', 
                    return_type="image",
                    key=f"crop_{st.session_state.upload_counter}_{st.session_state.get('rotation',0)}_{zoom}"
                )
                
                # If zoomed in, we need to ensure the quality of the cropped image isn't just the upscaled version
                # However, for simplicity in this flow, using the upscaled crop is usually sufficient for visual verification
                st.session_state.cropped_image = cropped_img
            
            with c2:
                st.image(cropped_img, caption="Preview")
                if st.button("✅ Confirm Crop", type="primary", use_container_width=True):
                    st.session_state.crop_confirmed = True
                    st.rerun()

        # --- STEP 2: EXTRACT ---
        elif st.session_state.crop_confirmed and not st.session_state.results_data:
            st.subheader("2️⃣ Extract Data")
            c1, c2 = st.columns([1,1])
            with c1:
                st.image(st.session_state.cropped_image, caption="Selection")
                if st.button("✏️ Edit", use_container_width=True):
                    st.session_state.crop_confirmed = False
                    st.rerun()
            with c2:
                barcode = st.text_input("Barcode (13 digits)", placeholder="Scan or type...")
                if st.button("🚀 Extract", type="primary", use_container_width=True):
                    if len(barcode) == 13 and barcode.isdigit():
                        with st.spinner("Analyzing..."):
                            save_path = os.path.join(SAVE_DIR, f"{barcode}.jpg")
                            
                            # Save the image. If it was zoomed, we might want to scale it back down 
                            # if it's too massive, but generally sending the zoomed crop is fine for OCR.
                            st.session_state.cropped_image.save(save_path, format="JPEG")
                            
                            try:
                                ocr_output = run_ocr(save_path)
                                input_text = json.dumps(ocr_output, separators=(",", ":"))
                                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                                with torch.no_grad():
                                    outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4, early_stopping=True)
                                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                st.session_state.results_data = repair_output(decoded)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.error("Invalid barcode")

        # --- STEP 3: RESULTS ---
        elif st.session_state.results_data:
            st.subheader("3️⃣ Results")
            display_nutrition_data(st.session_state.results_data, st.session_state.cropped_image)

else:
    st.info("👆 Upload an image to start.")