import os
import io
import json
import torch
import streamlit as st
from PIL import Image, ExifTags
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
# FIXED: Fix iPhone image orientation
# ------------------------------
def fix_iphone_orientation(image):
    """Fix iPhone image orientation based on EXIF data"""
    try:
        # Check for EXIF orientation tag
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif:
                # Get orientation tag
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                
                orientation_value = exif.get(orientation)
                
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
    except Exception as e:
        st.warning(f"Note: Could not fix image orientation: {e}")
    
    return image


# ------------------------------
# FIXED: Nested Data Table Display with responsive design
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

    # FIXED: Use responsive columns for mobile
    st.markdown("""
    <style>
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .stButton > button {
                width: 100% !important;
                margin-bottom: 10px;
            }
            .mobile-stack {
                flex-direction: column !important;
            }
        }
        .nutrition-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .nutrition-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 14px;
        }
        .nutrition-table td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: center;
            font-size: 14px;
        }
        .nutrition-table tr:nth-child(even) {
            background: #f8f8f8;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # FIXED: Stack columns on mobile
    st.markdown('<div class="mobile-stack" style="display: flex; flex-wrap: wrap; gap: 20px;">', unsafe_allow_html=True)
    
    # Image column
    st.markdown('<div style="flex: 1; min-width: 300px;">', unsafe_allow_html=True)
    st.image(cropped_image, caption="Cropped Nutrition Label", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Table column
    st.markdown('<div style="flex: 1; min-width: 300px;">', unsafe_allow_html=True)
    st.markdown(f"""
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

    # FIXED: Stack buttons on mobile
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------
# Streamlit UI - iPhone compatible
# ------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Nutritional Tracker",
    initial_sidebar_state="collapsed"
)

# Mobile viewport meta tag
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

st.title("🍎 Nutritional Information Extractor")

# Initialize session state
for key in ["rotation", "cropped_image", "original_image", "crop_confirmed", "results_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if "image" in key else False

# File uploader - optimized for mobile
uploaded_file = st.file_uploader(
    "Choose a nutrition label image", 
    type=["png","jpg","jpeg"], 
    help="Take a clear photo of the nutrition label or select from gallery"
)

if uploaded_file:
    if st.session_state.original_image is None:
        # FIXED: Apply iPhone orientation fix
        image = Image.open(uploaded_file).convert("RGB")
        image = fix_iphone_orientation(image)
        st.session_state.original_image = image
        st.session_state.crop_confirmed = False
        st.session_state.cropped_image = None
        st.session_state.results_data = None

    img = st.session_state.original_image

    # Step 1: Crop - optimized for mobile
    if not st.session_state.crop_confirmed:
        st.subheader("📐 Step 1: Crop the Nutrition Label")
        
        # FIXED: Mobile-friendly rotation controls
        st.markdown("**Adjust orientation:**")
        rot_col1, rot_col2, rot_col3 = st.columns(3)
        with rot_col1:
            if st.button("↺ Left", use_container_width=True, help="Rotate 90° left"):
                st.session_state.rotation = (st.session_state.get("rotation", 0) - 90) % 360
                st.rerun()
        with rot_col2:
            if st.button("↻ Right", use_container_width=True, help="Rotate 90° right"):
                st.session_state.rotation = (st.session_state.get("rotation", 0) + 90) % 360
                st.rerun()
        with rot_col3:
            if st.button("Reset", use_container_width=True, help="Reset rotation"):
                st.session_state.rotation = 0
                st.rerun()
        
        rotated_img = img.rotate(st.session_state.get("rotation", 0), expand=True)
        
        # FIXED: Mobile-friendly instructions
        st.info("👆 **Drag to select nutrition facts area**")
        
        # FIXED: Stack crop area and preview vertically on mobile
        cropped_img = st_cropper(
            rotated_img, 
            realtime_update=True, 
            box_color='#FF0000', 
            aspect_ratio=None,
            return_type="image",
            key=f"cropper_{st.session_state.get('rotation', 0)}"
        )
        st.session_state.cropped_image = cropped_img
        
        # Show preview
        if cropped_img and cropped_img.size != rotated_img.size:
            st.markdown("**Preview:**")
            st.image(cropped_img, caption="Your selection", use_container_width=True)
            st.success("✅ Ready to crop!")
        
        # FIXED: Mobile-friendly action buttons
        col_action1, col_action2 = st.columns(2)
        with col_action1:
            if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
                st.session_state.cropped_image = None
                st.rerun()
        with col_action2:
            if st.button("✅ Confirm Crop", use_container_width=True, type="primary"):
                if st.session_state.cropped_image and st.session_state.cropped_image.size != rotated_img.size:
                    st.session_state.crop_confirmed = True
                    st.rerun()
                else:
                    st.error("Please select an area first!")

    # Step 2: Process - optimized for mobile
    elif st.session_state.crop_confirmed and not st.session_state.results_data:
        st.subheader("⚙️ Step 2: Process the Image")
        
        # FIXED: Stack image and form on mobile
        st.markdown("**Selected area:**")
        st.image(st.session_state.cropped_image, caption="Nutrition label to analyze", use_container_width=True)
        
        if st.button("✏️ Edit Crop", use_container_width=True, type="secondary"):
            st.session_state.crop_confirmed = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Enter product barcode:**")
        
        # FIXED: Better mobile input
        barcode = st.text_input(
            "Barcode (13 digits)", 
            placeholder="0000000000000",
            label_visibility="collapsed",
            max_chars=13,
            help="Enter the 13-digit barcode from the product"
        )
        
        # Show barcode validation
        if barcode:
            if len(barcode) == 13 and barcode.isdigit():
                st.success("✓ Valid barcode format")
            else:
                st.error("Barcode must be 13 digits")
        
        # FIXED: Mobile-friendly process button
        if st.button("🚀 Extract Nutrition Data", type="primary", use_container_width=True, disabled=not barcode):
            if not barcode or len(barcode) != 13 or not barcode.isdigit():
                st.error("Please enter a valid 13-digit barcode")
            else:
                with st.spinner("Processing image..."):
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
    # FIXED: Better mobile-first empty state
    st.info("📤 Upload a nutrition label image to begin.")
    
    st.markdown("""
    <div style="text-align: center; padding: 30px 15px; border-radius: 10px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border: 2px dashed #667eea; margin: 20px 0;">
        <div style="font-size: 48px; margin-bottom: 15px;">📸</div>
        <h3 style="color: #2c3e50; margin-bottom: 15px;">How to use:</h3>
        <div style="text-align: left; display: inline-block;">
            <div style="margin: 10px 0; font-size: 14px;">
                <strong>1.</strong> Take a clear photo of the nutrition label
            </div>
            <div style="margin: 10px 0; font-size: 14px;">
                <strong>2.</strong> Crop the nutrition facts area
            </div>
            <div style="margin: 10px 0; font-size: 14px;">
                <strong>3.</strong> Enter the product barcode
            </div>
            <div style="margin: 10px 0; font-size: 14px;">
                <strong>4.</strong> Get instant nutrition data
            </div>
        </div>
        <div style="margin-top: 20px; font-size: 12px; color: #7f8c8d;">
            📱 Optimized for iPhone & mobile devices
        </div>
    </div>
    """, unsafe_allow_html=True)