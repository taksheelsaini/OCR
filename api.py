from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# --- 1. INITIALIZATION ---

app = FastAPI(title="TrOCR Multi-Line API")

# Allow the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production (e.g., ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- 2. LOAD UNTOUCHED EXISTING MODEL ---
# We load the model exactly as it exists in the trocr_finetuned directory
# WITHOUT modifying any of the user's existing training or evaluation scripts.

MODEL_DIR = 'trocr_finetuned'

# Determine device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Loading processor and model from {MODEL_DIR} to {device}...")
try:
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model. Error: {e}")
    processor = None
    model = None


# --- 3. HELPER FUNCTIONS ---

def segment_lines(image: Image.Image) -> list[Image.Image]:
    """
    Slices a multi-line image into individual horizontal lines using OpenCV
    horizontal projection profiles with strict noise filtering.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Apply Otsu's thresholding which is often cleaner for handwriting than adaptive
    # We blur first to remove high-frequency noise (like paper texture/smudges)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate the horizontal projection profile (sum of rows)
    proj = np.sum(thresh, axis=1)
    
    # Define a STRICT threshold to identify what constitutes a "line of text" vs "noise/space"
    # Increased to 15% of the max projection to ignore smudges
    max_proj = np.max(proj)
    threshold = max_proj * 0.15 
    
    # Find the boundaries (y-coordinates) of the text lines
    in_line = False
    start_y = 0
    lines_y = [] # List of tuples: (start_y, end_y)
    
    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start_y = y
        elif val <= threshold and in_line:
            in_line = False
            # Add padding to top and bottom of the line
            padding = 15
            end_y = min(gray.shape[0], y + padding)
            start_y_padded = max(0, start_y - padding)
            
            # STRICT FILTER: Ignore anything less than 35 pixels tall
            # TrOCR expects 64px tall images, so tiny 15px strips are definitely just noise/smudges
            if end_y - start_y_padded > 35: 
                lines_y.append((start_y_padded, end_y))
                
    # If the exact bottom touches text and it's tall enough
    if in_line:
        start_y_padded = max(0, start_y - 15)
        end_y = gray.shape[0]
        if end_y - start_y_padded > 35:
            lines_y.append((start_y_padded, end_y))
        
    line_images = []
    # Extract the cropped PIL images
    for sy, ey in lines_y:
        # Crop exactly that horizontal strip across the full width
        cropped = img_cv[sy:ey, 0:img_cv.shape[1]]
        line_images.append(Image.fromarray(cropped).convert('RGB'))
        
    # Fallback: if segmentation is too strict and finds nothing, 
    # assume it's a single line image and return the whole thing
    if not line_images:
        line_images.append(image)
        
    return line_images

def predict_pil_image(image: Image.Image) -> str:
    """Runs TrOCR inference on a single horizontal line of text."""
    if model is None or processor is None:
        return "[Model not loaded. Ensure trocr_finetuned exists.]"
        
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# --- 4. API ENDPOINTS ---

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    """
    Receives an image, saves a temporary copy, segments it into lines, 
    runs TrOCR on each line, and returns the combined text.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG.")
        
    try:
        # Read the uploaded image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB for OpenCV processing
        if image.mode != "RGB":
             image = image.convert("RGB")
        
        # 1. Segment the image into individual lines
        line_images = segment_lines(image)
        
        # 2. Run OCR on each individual line
        extracted_lines = []
        for line_img in line_images:
            # The model was trained on 384x64, but the processor handles resizing automatically
            text = predict_pil_image(line_img)
            extracted_lines.append(text)
            
        # 3. Combine the results
        final_text = "\n".join(extracted_lines)
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "text": final_text,
            "segments_detected": len(line_images)
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

from fastapi.staticfiles import StaticFiles
# Serve the Vite/React frontend directly from the API port
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    # Start the server locally on port 8000
    print("Starting API Server on http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
