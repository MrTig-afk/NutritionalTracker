import os
import io
import json
import torch
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from dotenv import load_dotenv
from ocr import run_ocr
from repair import repair_output

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
# FIXED: Nested Data Table Display
# ------------------------------
def display_nutrition_data(data, cropped_image):
    """Display nutrition data always with all keys populated with cropped image side by side"""
    if not isinstance(data, dict):
        st.error("Invalid data")
        return
    
    # If nested under 'nutritional_information', unwrap
    if "nutritional_information" in data:
        data = data["nutritional_information"]

    per_100g = data.get("per_100g", {})
    per_serving = data.get("per_serving", {})
    serving_size = data.get("serving_size", "-")

    # FIXED: Use 'calories' key instead of 'energy' to match JSON
    rows = [
        ("Calories", ["calories"]),  # FIXED: Use 'calories' to match JSON key
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
                # Add "kCal" to calories if not present
                if label == "Calories" and "kcal" not in str(val_100g).lower():
                    val_100g = f"{val_100g}kCal"
            if k in per_serving and per_serving[k] is not None:
                val_serving = per_serving[k]
                # Add "kCal" to calories if not present
                if label == "Calories" and "kcal" not in str(val_serving).lower():
                    val_serving = f"{val_serving}kCal"
        rows_html += f"<tr><td>{label}</td><td>{val_100g}</td><td>{val_serving}</td></tr>"

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # REMOVED: use_column_width parameter
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

        # FIXED: Use columns for download button so table doesn't disappear
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            # Download JSON
            result_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=result_str,
                file_name="nutrition_data.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"  # Add key to prevent rerun
            )
        with col_btn2:
            if st.button("🔄 Process Another Image", use_container_width=True):
                for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data"]:
                    st.session_state[key] = None if "image" in key else False
                st.rerun()


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Nutritional Tracker")

st.title("🍎 Nutritional Information Extractor")

# Initialize session state
for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if "image" in key else False

# File uploader
uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"], label_visibility="collapsed")

if uploaded_file:
    if st.session_state.original_image is None:
        st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.crop_confirmed = False
        st.session_state.cropped_image = None
        st.session_state.results_data = None

    img = st.session_state.original_image

    # Step 1: Crop
    if not st.session_state.crop_confirmed:
        st.subheader("📐 Step 1: Crop the Nutrition Label")
        
        # FIXED: Add rotation controls to reduce blankness
        col_rot1, col_rot2, col_rot3, col_rot4 = st.columns(4)
        with col_rot1:
            if st.button("↺ Rotate Left", use_container_width=True):
                st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                st.rerun()
        with col_rot2:
            if st.button("↻ Rotate Right", use_container_width=True):
                st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                st.rerun()
        with col_rot3:
            if st.button("⟳ Reset Rotation", use_container_width=True):
                st.session_state.rotation = 0
                st.rerun()
        with col_rot4:
            if st.button("🗑️ Clear Crop", use_container_width=True):
                st.session_state.cropped_image = None
                st.rerun()
        
        rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
        
        # FIXED: Show instructions and preview to reduce blankness
        st.info("🔍 **Drag to select the nutrition facts area, then click Confirm Crop**")
        
        col_crop1, col_crop2 = st.columns([1, 1])
        
        with col_crop1:
            cropped_img = st_cropper(
                rotated_img, 
                realtime_update=True, 
                box_color='#FF0000', 
                aspect_ratio=None,
                return_type="image"
            )
            st.session_state.cropped_image = cropped_img
        
        with col_crop2:
            # FIXED: Show preview of cropped area
            st.write("**Preview of Cropped Area:**")
            if cropped_img and cropped_img.size != rotated_img.size:
                # REMOVED: use_column_width parameter
                st.image(cropped_img, caption="Your selection")
                st.success("✅ Area selected! Click 'Confirm Crop' below to proceed.")
            else:
                # REMOVED: use_column_width parameter
                st.image(rotated_img, caption="Original image - drag on left to select")
                st.warning("👈 Drag on the left image to select nutrition facts area")
        
        # Confirm button at bottom
        if st.button("✅ Confirm Crop & Continue →", type="primary", use_container_width=True):
            if st.session_state.cropped_image and st.session_state.cropped_image.size != rotated_img.size:
                st.session_state.crop_confirmed = True
                st.rerun()
            else:
                st.error("Please select an area to crop first!")

    # Step 2: Process
    elif st.session_state.crop_confirmed and not st.session_state.results_data:
        st.subheader("⚙️ Step 2: Process the Image")
        
        # FIXED: Show preview and processing area
        col_process1, col_process2 = st.columns([1, 1])
        
        with col_process1:
            # REMOVED: use_column_width parameter
            st.image(st.session_state.cropped_image, caption="Selected Nutrition Label")
            if st.button("✏️ Edit Crop", use_container_width=True):
                st.session_state.crop_confirmed = False
                st.rerun()
        
        with col_process2:
            st.markdown("### Enter Product Details")
            barcode = st.text_input("**Barcode (Required)**", placeholder="Enter 13-digit barcode")
            
            st.markdown("---")
            st.markdown("**Instructions:**")
            st.markdown("1. Enter the 13-digit barcode")
            st.markdown("2. Click 'Extract Nutrition Data'")
            st.markdown("3. Wait for processing to complete")
            
            if st.button("🚀 Extract Nutrition Data", type="primary", use_container_width=True):
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

    # Step 3: Display Results
    elif st.session_state.results_data:
        st.subheader("✅ Analysis Complete")
        display_nutrition_data(st.session_state.results_data, st.session_state.cropped_image)

else:
    st.info("📤 Upload a nutrition label image to begin analysis.")
    # FIXED: Add more informative empty state
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; border-radius: 10px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border: 2px dashed #667eea; margin-top: 20px;">
        <h3 style="color: #2c3e50;">How to use this tool:</h3>
        <div style="text-align: left; display: inline-block; margin-top: 20px;">
            <p>📸 <strong>Step 1:</strong> Upload a clear photo of the nutrition label</p>
            <p>✂️ <strong>Step 2:</strong> Crop the nutrition facts area</p>
            <p>🔢 <strong>Step 3:</strong> Enter the product barcode</p>
            <p>📊 <strong>Step 4:</strong> View and download the extracted data</p>
        </div>
        <p style="margin-top: 20px; color: #7f8c8d;">
            Supported formats: PNG, JPG, JPEG
        </p>
    </div>
    """, unsafe_allow_html=True)