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

# IMPORTANT: For HEIC support, add these lines at the very top after imports
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
# FIXED: Handle iPhone Image Issues
# ------------------------------
def process_uploaded_image(uploaded_file):
    """
    Process uploaded image handling iPhone-specific issues:
    - HEIC format conversion
    - EXIF orientation correction
    - Color mode standardization
    """
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # FIXED: Auto-rotate based on EXIF orientation (critical for iPhone)
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # FIXED: Resize very large images to prevent memory issues on mobile
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
# Display Nutrition Data
# ------------------------------
def display_nutrition_data(data, cropped_image):
    """Display nutrition data with cropped image side by side"""
    if not isinstance(data, dict):
        st.error("Invalid data")
        return
    
    # If nested under 'nutritional_information', unwrap
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

    # Build rows HTML
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

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(cropped_image, caption="Cropped Nutrition Label")
    
    with col2:
        st.markdown(f"""
        <style>
            .nutrition-table {{
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 20px;
            }}
            .nutrition-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px;
                text-align: center;
            }}
            .nutrition-table td {{
                padding: 8px;
                border: 1px solid #ddd;
                text-align: center;
            }}
            .nutrition-table tr:nth-child(even) {{
                background: #f8f8f8;
            }}
        </style>

        <div style="margin-bottom:15px;">
            <strong>Serving Size:</strong> {serving_size}
        </div>

        <table class="nutrition-table">
            <thead>
                <tr>
                    <th>Macros</th>
                    <th>Per 100g</th>
                    <th>Per Serving</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            result_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=result_str,
                file_name="nutrition_data.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"
            )
        with col_btn2:
            # FIXED: Clear uploaded file properly
            if st.button("🔄 Process Another Image", use_container_width=True, key="reset_btn"):
                # Clear ALL session state including file uploader
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


# ------------------------------
# FIXED: Mobile-friendly cropping helper
# ------------------------------
def mobile_crop_helper(image):
    """
    Provide mobile-friendly cropping with zoom controls
    """
    st.markdown("""
    <style>
        /* Make cropper more mobile-friendly */
        .stCropper > div {
            touch-action: pan-x pan-y pinch-zoom !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add zoom controls for mobile
    st.markdown("**🔍 Zoom Controls (for mobile users):**")
    zoom_col1, zoom_col2, zoom_col3 = st.columns(3)
    
    with zoom_col1:
        if st.button("➕ Zoom In", use_container_width=True, key="zoom_in"):
            current_zoom = st.session_state.get("zoom_level", 1.0)
            st.session_state.zoom_level = min(current_zoom + 0.2, 3.0)
            st.rerun()
    
    with zoom_col2:
        if st.button("➖ Zoom Out", use_container_width=True, key="zoom_out"):
            current_zoom = st.session_state.get("zoom_level", 1.0)
            st.session_state.zoom_level = max(current_zoom - 0.2, 0.5)
            st.rerun()
    
    with zoom_col3:
        if st.button("🔄 Reset Zoom", use_container_width=True, key="reset_zoom"):
            st.session_state.zoom_level = 1.0
            st.rerun()
    
    # Apply zoom
    zoom_level = st.session_state.get("zoom_level", 1.0)
    if zoom_level != 1.0:
        new_size = tuple(int(dim * zoom_level) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        st.info(f"Current zoom: {int(zoom_level * 100)}%")
    
    return image


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Nutritional Tracker")

st.title("🍎 Nutritional Information Extractor")

# FIXED: Initialize session state with unique file ID tracking
if "file_id" not in st.session_state:
    st.session_state.file_id = None

for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data", "zoom_level"]:
    if key not in st.session_state:
        st.session_state[key] = None if "image" in key else (False if "confirmed" in key else 1.0)

# FIXED: Add accept parameter for better mobile camera support
uploaded_file = st.file_uploader(
    "📷 Upload or take a photo of the nutrition label", 
    type=["png", "jpg", "jpeg", "heic", "heif"],
    label_visibility="visible",
    help="You can upload from gallery or take a new photo",
    key="file_uploader"
)

# FIXED: Track file changes properly
if uploaded_file:
    # Generate unique file ID based on name and size
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Only process if it's a NEW file
    if st.session_state.file_id != current_file_id:
        st.session_state.file_id = current_file_id
        
        # Process the new image
        with st.spinner("📸 Processing uploaded image..."):
            processed_img = process_uploaded_image(uploaded_file)
            
            if processed_img is None:
                st.error("Failed to process image. Please try again.")
                st.stop()
            
            # Reset all state for new image
            st.session_state.original_image = processed_img
            st.session_state.crop_confirmed = False
            st.session_state.cropped_image = None
            st.session_state.results_data = None
            st.session_state.rotation = 0
            st.session_state.zoom_level = 1.0
        
        st.success("✅ Image uploaded successfully!")
        st.rerun()

    img = st.session_state.original_image

    # Step 1: Crop
    if not st.session_state.crop_confirmed:
        st.subheader("📐 Step 1: Crop the Nutrition Label")
        
        # Rotation controls
        col_rot1, col_rot2, col_rot3, col_rot4 = st.columns(4)
        with col_rot1:
            if st.button("↺ Rotate Left", use_container_width=True, key="rot_left"):
                st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                st.rerun()
        with col_rot2:
            if st.button("↻ Rotate Right", use_container_width=True, key="rot_right"):
                st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                st.rerun()
        with col_rot3:
            if st.button("⟳ Reset Rotation", use_container_width=True, key="rot_reset"):
                st.session_state.rotation = 0
                st.rerun()
        with col_rot4:
            if st.button("🗑️ Clear Crop", use_container_width=True, key="clear_crop"):
                st.session_state.cropped_image = None
                st.session_state.zoom_level = 1.0
                st.rerun()
        
        rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
        
        st.info("🔍 **Drag to select the nutrition facts area, then click Confirm Crop**")
        
        col_crop1, col_crop2 = st.columns([1, 1])
        
        with col_crop1:
            # FIXED: Apply zoom for mobile-friendly cropping
            zoomed_img = mobile_crop_helper(rotated_img)
            
            # Cropper with unique key
            cropped_img = st_cropper(
                zoomed_img, 
                realtime_update=True, 
                box_color='#FF0000', 
                aspect_ratio=None,
                return_type="image",
                key=f"cropper_{st.session_state.get('rotation', 0)}_{st.session_state.get('zoom_level', 1.0)}"
            )
            
            # Scale back if zoomed
            if st.session_state.get("zoom_level", 1.0) != 1.0:
                zoom = st.session_state.zoom_level
                original_size = tuple(int(dim / zoom) for dim in cropped_img.size)
                cropped_img = cropped_img.resize(original_size, Image.Resampling.LANCZOS)
            
            st.session_state.cropped_image = cropped_img
        
        with col_crop2:
            st.write("**Preview of Cropped Area:**")
            if cropped_img and cropped_img.size != rotated_img.size:
                st.image(cropped_img, caption="Your selection")
                st.success("✅ Area selected! Click 'Confirm Crop' below to proceed.")
            else:
                st.image(rotated_img, caption="Original image - drag on left to select")
                st.warning("👈 Drag on the left image to select nutrition facts area")
        
        if st.button("✅ Confirm Crop & Continue →", type="primary", use_container_width=True, key="confirm_crop"):
            if st.session_state.cropped_image and st.session_state.cropped_image.size != rotated_img.size:
                st.session_state.crop_confirmed = True
                st.rerun()
            else:
                st.error("Please select an area to crop first!")

    # Step 2: Process
    elif st.session_state.crop_confirmed and not st.session_state.results_data:
        st.subheader("⚙️ Step 2: Process the Image")
        
        col_process1, col_process2 = st.columns([1, 1])
        
        with col_process1:
            st.image(st.session_state.cropped_image, caption="Selected Nutrition Label")
            if st.button("✏️ Edit Crop", use_container_width=True, key="edit_crop"):
                st.session_state.crop_confirmed = False
                st.rerun()
        
        with col_process2:
            st.markdown("### Enter Product Details")
            barcode = st.text_input("**Barcode (Required)**", placeholder="Enter 13-digit barcode", key="barcode_input")
            
            st.markdown("---")
            st.markdown("**Instructions:**")
            st.markdown("1. Enter the 13-digit barcode")
            st.markdown("2. Click 'Extract Nutrition Data'")
            st.markdown("3. Wait for processing to complete")
            
            if st.button("🚀 Extract Nutrition Data", type="primary", use_container_width=True, key="extract_btn"):
                if not barcode or len(barcode) != 13 or not barcode.isdigit():
                    st.error("Please enter a valid 13-digit barcode")
                else:
                    with st.spinner("🔍 Processing image..."):
                        save_path = os.path.join(SAVE_DIR, f"{barcode}.jpg")
                        st.session_state.cropped_image.save(save_path, format="JPEG")
                        
                        try:
                            ocr_output = run_ocr(save_path)
                            input_text = json.dumps(ocr_output, separators=(",", ":"))
                            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                            with torch.no_grad():
                                outputs = model.generate(**inputs, max_new_tokens=2048, num_beams=4, early_stopping=True)
                            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            repaired = repair_output(decoded)
                            st.session_state.results_data = repaired
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {e}")
                            st.exception(e)

    # Step 3: Display Results
    elif st.session_state.results_data:
        st.subheader("✅ Analysis Complete")
        display_nutrition_data(st.session_state.results_data, st.session_state.cropped_image)

else:
    st.info("📤 Upload a nutrition label image to begin analysis.")
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; border-radius: 10px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border: 2px dashed #667eea; margin-top: 20px;">
        <h3 style="color: #2c3e50;">How to use this tool:</h3>
        <div style="text-align: left; display: inline-block; margin-top: 20px;">
            <p>📸 <strong>Step 1:</strong> Upload a clear photo of the nutrition label</p>
            <p>✂️ <strong>Step 2:</strong> Crop the nutrition facts area (use zoom controls on mobile)</p>
            <p>🔢 <strong>Step 3:</strong> Enter the product barcode</p>
            <p>📊 <strong>Step 4:</strong> View and download the extracted data</p>
        </div>
        <p style="margin-top: 20px; color: #7f8c8d;">
            Supported formats: PNG, JPG, JPEG, HEIC (iPhone)
        </p>
        <p style="margin-top: 10px; color: #95a5a6; font-size: 0.9em;">
            💡 <strong>Mobile Tip:</strong> Use the zoom controls to make cropping easier!
        </p>
    </div>
    """, unsafe_allow_html=True)