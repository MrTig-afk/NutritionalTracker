import os
import io
import json
import torch
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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
    print("Warning: pillow-heif not installed. HEIC images may not work on iPhone.")

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
# Image Processing & Quality Booster
# ------------------------------
def process_image(image_file, source_type="upload"):
    try:
        # Reset file pointer
        image_file.seek(0)
        file_bytes = image_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        
        # 1. Fix Orientation (iPhone/Android rotation issues)
        image = ImageOps.exif_transpose(image)
        
        # 2. Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        
        # 3. SMART ENHANCEMENT LOGIC
        if source_type == "camera" or max(width, height) < 1200:
            # Case A: Low Quality / Webcam (st.camera_input)
            # Upscale and Sharpen to make text readable for OCR
            
            # Upscale 2x
            new_size = (int(width * 2), int(height * 2))
            image = image.resize(new_size, Image.Resampling.BICUBIC)
            
            # Sharpen edges (Clarifies text)
            image = image.filter(ImageFilter.SHARPEN)
            
            # Enhance Contrast (Separates text from background)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.4)  # +40% contrast
            
            # Enhance Sharpness again slightly
            enhancer_sharp = ImageEnhance.Sharpness(image)
            image = enhancer_sharp.enhance(1.5)

        elif max(width, height) > 2000:
            # Case B: Huge Image (Native Camera Upload)
            # Resize down slightly to save RAM/Processing time
            ratio = 2000 / max(width, height)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        return image

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# ------------------------------
# Display Data Helper
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
# Mobile Crop Helper
# ------------------------------
def mobile_crop_helper(image):
    st.markdown("""<style>.stCropper > div { touch-action: pan-x pan-y pinch-zoom !important; }</style>""", unsafe_allow_html=True)
    
    st.markdown("**🔍 Zoom (Mobile):**")
    z1, z2, z3 = st.columns(3)
    if z1.button("➕ In", use_container_width=True):
        st.session_state.zoom_level = min(st.session_state.get("zoom_level", 1.0) + 0.2, 3.0)
        st.rerun()
    if z2.button("➖ Out", use_container_width=True):
        st.session_state.zoom_level = max(st.session_state.get("zoom_level", 1.0) - 0.2, 0.5)
        st.rerun()
    if z3.button("🔄 Reset", use_container_width=True):
        st.session_state.zoom_level = 1.0
        st.rerun()
    
    zoom = st.session_state.get("zoom_level", 1.0)
    if zoom != 1.0:
        new_size = tuple(int(dim * zoom) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

# ------------------------------
# Main UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Nutritional Tracker")
st.title("🍎 Nutritional Information Extractor")

# Crash Prevention: Clear legacy time variables if they exist
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
# INPUT METHOD SELECTION
# ------------------------------
input_method = st.radio(
    "Choose Input Method:", 
    ["📤 Upload (Highest Quality)", "📸 In-App Camera (Stable)"],
    horizontal=True,
    help="Use 'In-App Camera' if your browser crashes when taking photos."
)

if input_method == "📤 Upload (Highest Quality)":
    st.info("💡 **Android Tip:** If the app reloads/crashes, take the photo FIRST, then select 'Files/Gallery' here.")
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["png", "jpg", "jpeg", "heic", "heif", "webp"],
        key="file_uploader"
    )
    source = "upload"
else:
    st.warning("⚡ **Stabilized Mode:** Uses webcam. We will automatically sharpen the image for better results.")
    uploaded_file = st.camera_input("Capture Label", key="camera_input")
    source = "camera"

# ------------------------------
# PROCESSING LOGIC
# ------------------------------
if uploaded_file:
    # Signature check to prevent random re-processing
    file_signature = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    
    if st.session_state.last_processed_file != file_signature:
        with st.spinner("Processing & Enhancing Image..."):
            processed_img = process_image(uploaded_file, source_type=source)
            
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
                
                st.success("✅ Image loaded!")
                st.rerun()
            else:
                st.error("Failed to process image.")

    # ------------------------------
    # App Logic (Runs if image is loaded)
    # ------------------------------
    if st.session_state.original_image:
        img = st.session_state.original_image

        # --- STEP 1: CROP ---
        if not st.session_state.crop_confirmed:
            st.subheader("1️⃣ Crop Label")
            
            r1, r2, r3, r4 = st.columns(4)
            if r1.button("↺ Left", use_container_width=True):
                st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                st.rerun()
            if r2.button("↻ Right", use_container_width=True):
                st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                st.rerun()
            if r3.button("Reset", use_container_width=True):
                st.session_state.rotation = 0
                st.rerun()
            if r4.button("🗑️ Clear", use_container_width=True):
                st.session_state.last_processed_file = None
                st.session_state.original_image = None
                st.rerun()
            
            rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                zoomed_img = mobile_crop_helper(rotated_img)
                cropped_img = st_cropper(
                    zoomed_img, 
                    realtime_update=True, 
                    box_color='#FF0000', 
                    return_type="image",
                    key=f"crop_{st.session_state.upload_counter}_{st.session_state.get('rotation',0)}_{st.session_state.get('zoom_level',1)}"
                )
                
                zoom = st.session_state.get("zoom_level", 1.0)
                if zoom != 1.0:
                    original_size = tuple(int(dim / zoom) for dim in cropped_img.size)
                    cropped_img = cropped_img.resize(original_size, Image.Resampling.LANCZOS)
                
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
    st.info("👆 Choose an input method to start.")