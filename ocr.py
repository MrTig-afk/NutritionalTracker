import os
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Cache the model globally to avoid reloading on every call
_MODEL_CACHE = None

def get_ocr_model():
    """
    Load and cache the OCR model.
    Only loads once, then reuses the same model instance.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("Loading OCR model... (this may take a moment)")
        # Use the custom architecture for better accuracy
        _MODEL_CACHE = ocr_predictor(
            det_arch='db_resnet50', 
            reco_arch='crnn_vgg16_bn', 
            pretrained=True
        )
        print("OCR model loaded successfully!")
    return _MODEL_CACHE


def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to handle iPhone-specific issues before OCR:
    - Fix EXIF orientation
    - Convert to RGB
    - Optimize size
    - Enhance contrast if needed
    
    Returns path to preprocessed image.
    """
    try:
        # Open and fix orientation
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # Critical for iPhone images
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (helps OCR performance)
        max_dimension = 2000
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save preprocessed image
        preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
        img.save(preprocessed_path, format='JPEG', quality=95)
        
        return preprocessed_path
    
    except Exception as e:
        print(f"Warning: Image preprocessing failed: {e}")
        return image_path  # Fall back to original


def clean_json(doctr_result):
    """
    Remove unnecessary fields from docTR output to reduce token usage
    and keep only the text content we need.
    """
    result_dict = doctr_result.export()
    
    for page in result_dict.get("pages", []):
        page.pop("confidence", None)
        page.pop("geometry", None)
        page.pop("objectness_score", None)
        page.pop("orientation", None)
        page.pop("language", None)
        
        for block in page.get("blocks", []):
            block.pop("geometry", None)
            block.pop("confidence", None)
            block.pop("objectness_score", None)
            
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


def run_ocr(image_path):
    """
    Main OCR function called by the Streamlit app.
    
    Args:
        image_path: Path to the cropped nutrition label image
        
    Returns:
        dict: Cleaned OCR results with text content
    """
    preprocessed_path = None
    
    try:
        # Step 1: Preprocess the image (handles iPhone issues)
        print(f"Preprocessing image: {image_path}")
        preprocessed_path = preprocess_image_for_ocr(image_path)
        
        # Step 2: Load the image with docTR
        print("Loading image with docTR...")
        doc = DocumentFile.from_images(preprocessed_path)
        
        # Step 3: Get the OCR model
        model = get_ocr_model()
        
        # Step 4: Run OCR
        print("Running OCR extraction...")
        result = model(doc)
        
        # Step 5: Clean and return results
        cleaned_result = clean_json(result)
        print(f"OCR completed. Extracted {len(str(cleaned_result))} characters.")
        
        return cleaned_result
    
    except Exception as e:
        print(f"ERROR in run_ocr: {e}")
        # Return minimal structure to avoid breaking the pipeline
        return {
            "pages": [{
                "blocks": [{
                    "lines": [{
                        "words": [{"value": f"OCR Error: {str(e)}"}]
                    }]
                }]
            }]
        }
    
    finally:
        # Cleanup preprocessed file
        if preprocessed_path and preprocessed_path != image_path:
            if os.path.exists(preprocessed_path):
                try:
                    os.remove(preprocessed_path)
                except:
                    pass


# --- TESTING/COMPARISON FUNCTIONS (Optional) ---

def compare_models_on_s3():
    """
    Original comparison function - useful for testing but not used by main app.
    Requires AWS credentials or anonymous access to S3.
    """
    import boto3
    import random
    from botocore import UNSIGNED
    from botocore.config import Config
    
    BUCKET_NAME = 'openfoodfacts-cropped-nutrition-label'
    PREFIX = 'crops/v1/'
    
    # Initialize models
    print("Loading models for comparison...")
    model_default = ocr_predictor(pretrained=True)
    model_custom = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    
    # Anonymous S3 access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    print(f"Listing files in s3://{BUCKET_NAME}/{PREFIX} ...")
    
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, MaxKeys=1000)
        
        all_keys = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    all_keys.append(obj['Key'])
        
        if len(all_keys) < 2:
            selected_keys = all_keys
        else:
            selected_keys = random.sample(all_keys, 2)
        
        # Run comparison
        for i, key in enumerate(selected_keys):
            print(f"\n{'='*20} PROCESSING IMAGE {i+1} {'='*20}")
            print(f"File: {key}")
            
            local_filename = f"temp_img_{i}.jpg"
            s3.download_file(BUCKET_NAME, key, local_filename)
            
            try:
                doc = DocumentFile.from_images(local_filename)
                
                print("Running Default Model...")
                res_default = model_default(doc)
                json_default = clean_json(res_default)
                
                print("Running Custom Arch Model...")
                res_custom = model_custom(doc)
                json_custom = clean_json(res_custom)
                
                str_default = json.dumps(json_default, indent=2)
                str_custom = json.dumps(json_custom, indent=2)
                
                if str_default == str_custom:
                    print("\n[Result]: Both models produced IDENTICAL output.")
                    print("--- Shared Output ---")
                    print(str_default)
                else:
                    print("\n[Result]: Outputs DIFFER.")
                    print("--- Default Output ---")
                    print(str_default)
                    print("\n--- Custom Output ---")
                    print(str_custom)
            
            except Exception as e:
                print(f"Error processing image: {e}")
            finally:
                if os.path.exists(local_filename):
                    os.remove(local_filename)
    
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Check your bucket name and network connection.")
    
    print("\nDone.")


if __name__ == "__main__":
    # If run directly, do model comparison
    print("Running in comparison mode...")
    compare_models_on_s3()