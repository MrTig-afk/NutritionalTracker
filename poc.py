import os
import io
import json
import torch
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
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
    pass

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
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if huge
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
# Helper: Pan/Shift Image
# ------------------------------
def shift_image(image, x_offset, y_offset):
    new_img = Image.new("RGB", image.size, (255, 255, 255))
    new_img.paste(image, (x_offset, y_offset))
    return new_img

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

# Custom CSS to fix Mobile D-Pad Layout and Scrolling
st.markdown("""
<style>
    /* Force columns in the D-Pad to stay side-by-side on mobile */
    div[data-testid="column"] {
        width: auto !important;
        flex: 1 1 auto !important;
        min-width: 0px !important;
    }
    
    /* Make buttons bigger and touch-friendly */
    div.stButton > button {
        width: 100%;
        height: 50px;
        font-size: 20px;
        touch-action: manipulation;
    }

    /* Add padding to the cropper container to allow scrolling */
    .cropper-container {
        padding: 10px 0px;
        margin-bottom: 20px;
        border-bottom: 2px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Cleanup
if "last_processed_time" in st.session_state:
    del st.session_state["last_processed_time"]

# Initialize State
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

# Pan State
if "pan_x" not in st.session_state:
    st.session_state.pan_x = 0
if "pan_y" not in st.session_state:
    st.session_state.pan_y = 0

for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if "image" in key else (False if "confirmed" in key else 1.0)

# ------------------------------
# FILE UPLOADER
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
    file_signature = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    
    if st.session_state.last_processed_file != file_signature:
        with st.spinner("Processing..."):
            processed_img = process_uploaded_image(uploaded_file)
            
            if processed_img:
                st.session_state.original_image = processed_img
                st.session_state.crop_confirmed = False
                st.session_state.cropped_image = None
                st.session_state.results_data = None
                st.session_state.rotation = 0
                st.session_state.pan_x = 0
                st.session_state.pan_y = 0
                
                st.session_state.last_processed_file = file_signature
                st.session_state.upload_counter += 1
                
                st.success("✅ Image uploaded!")
                st.rerun()
            else:
                st.error("Failed to process image.")

    # ------------------------------
    # App Runs
    # ------------------------------
    if st.session_state.original_image:
        img = st.session_state.original_image

        # --- STEP 1: CROP ---
        if not st.session_state.crop_confirmed:
            st.subheader("1️⃣ Crop Label")
            
            # 1. Rotate
            rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
            
            # 2. Pan (Shift)
            shifted_img = shift_image(rotated_img, st.session_state.pan_x, st.session_state.pan_y)

            # --- DISPLAY CROPPER ---
            st.markdown('<div class="cropper-container">', unsafe_allow_html=True)
            # Center the cropper in a column
            c_main = st.container()
            with c_main:
                # Hint for scrolling
                st.caption("📱 **Tip:** Swipe on the white space to scroll. Use D-Pad below to move image.")
                
                cropped_img = st_cropper(
                    shifted_img, 
                    realtime_update=True, 
                    box_color='#FF0000', 
                    return_type="image",
                    key=f"crop_{st.session_state.upload_counter}_{st.session_state.rotation}_{st.session_state.pan_x}_{st.session_state.pan_y}"
                )
                st.session_state.cropped_image = cropped_img
            st.markdown('</div>', unsafe_allow_html=True)

            # --- CONTROLS (MOVED BELOW IMAGE) ---
            st.markdown("### 🎮 Controls")
            
            # Create 3 main columns for controls
            # Use columns([1,2,1]) to give the D-Pad (middle) more room
            col_left, col_mid, col_right = st.columns([1, 2, 1])
            
            # --- LEFT: Rotation ---
            with col_left:
                st.markdown("**Rotate**")
                if st.button("↺ Left", use_container_width=True):
                    st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                    st.rerun()
                if st.button("↻ Right", use_container_width=True):
                    st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                    st.rerun()

            # --- MIDDLE: D-Pad ---
            with col_mid:
                st.markdown("**Move Image**")
                step = 50
                
                # Using 3 columns for D-Pad grid
                d1, d2, d3 = st.columns(3)
                
                # Row 1
                with d2:
                    if st.button("⬆️", key="up", use_container_width=True):
                        st.session_state.pan_y -= step
                        st.rerun()
                
                # Row 2
                d4, d5, d6 = st.columns(3)
                with d4:
                    if st.button("⬅️", key="left", use_container_width=True):
                        st.session_state.pan_x -= step
                        st.rerun()
                with d5:
                    if st.button("🎯", key="reset", help="Center Image", use_container_width=True):
                        st.session_state.pan_x = 0
                        st.session_state.pan_y = 0
                        st.rerun()
                with d6:
                    if st.button("➡️", key="right", use_container_width=True):
                        st.session_state.pan_x += step
                        st.rerun()

                # Row 3
                d7, d8, d9 = st.columns(3)
                with d8:
                    if st.button("⬇️", key="down", use_container_width=True):
                        st.session_state.pan_y += step
                        st.rerun()

            # --- RIGHT: Actions ---
            with col_right:
                st.markdown("**Action**")
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.last_processed_file = None
                    st.session_state.original_image = None
                    st.rerun()
                
                st.markdown("---")
                # Confirm button is highlighted
                if st.button("✅ Crop", type="primary", use_container_width=True):
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
    st.info("👆 Upload an image to start.")