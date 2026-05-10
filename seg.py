# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
# import io
# import os
# import uvicorn
# from datetime import datetime
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Macular Segmentation API",
#     description="API for segmenting macular regions in fundus images",
#     version="1.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configuration
# IMG_HEIGHT = 256
# IMG_WIDTH = 256
# MODEL_PATH = "best_macular_model.h5"
# OUTPUT_DIR = "segmentation_results"

# # Create output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Global model variable
# model = None

# def extract_best_channel_and_enhance(image):
#     """
#     Extract the best channel for macular visibility and enhance it
#     Green channel is typically best for macular segmentation
#     """
#     # Extract green channel (index 1)
#     green_channel = image[:, :, 1]
    
#     # Apply CLAHE for local contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply((green_channel * 255).astype(np.uint8))
    
#     # Additional enhancement - Gaussian blur to reduce noise
#     enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
#     # Normalize back to 0-1
#     enhanced = enhanced.astype(np.float32) / 255.0
    
#     return enhanced

# def load_model():
#     """Load the trained macular segmentation model"""
#     global model
#     try:
#         # Custom objects for loading the model
#         def combined_loss(y_true, y_pred):
#             bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#             smooth = 1e-6
#             y_true_f = tf.keras.backend.flatten(y_true)
#             y_pred_f = tf.keras.backend.flatten(y_pred)
#             intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
#             dice_loss = 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
#             alpha = 0.3
#             beta = 0.7
#             tp = tf.keras.backend.sum(y_true_f * y_pred_f)
#             fp = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
#             fn = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
#             tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
#             return bce + dice_loss + tversky_loss

#         def dice_coefficient(y_true, y_pred, smooth=1e-6):
#             y_true_f = tf.keras.backend.flatten(y_true)
#             y_pred_f = tf.keras.backend.flatten(y_pred)
#             intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
#             return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

#         def iou_score(y_true, y_pred, smooth=1e-6):
#             y_true_f = tf.keras.backend.flatten(y_true)
#             y_pred_f = tf.keras.backend.flatten(y_pred)
#             intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
#             union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
#             return (intersection + smooth) / (union + smooth)

#         custom_objects = {
#             'combined_loss': combined_loss,
#             'dice_coefficient': dice_coefficient,
#             'iou_score': iou_score
#         }
        
#         model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
#         logger.info("Model loaded successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         return False

# @app.on_event("startup")
# async def startup_event():
#     """Load model on startup"""
#     if not os.path.exists(MODEL_PATH):
#         logger.error(f"Model file not found: {MODEL_PATH}")
#         raise Exception(f"Model file not found: {MODEL_PATH}")
    
#     success = load_model()
#     if not success:
#         raise Exception("Failed to load model")

# @app.post("/segment")
# async def segment_macular(file: UploadFile = File(...)):
#     """
#     Segment macular region and save result with original image overlay
    
#     Returns:
#     - result_path: Path to the saved segmentation result image
#     - statistics: Basic segmentation statistics
#     """
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     # Check file type
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     try:
#         # Read and preprocess image
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Convert to RGB if not already
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array and resize
#         image_np = np.array(image)
#         image_resized = cv2.resize(image_np, (IMG_WIDTH, IMG_HEIGHT))
#         image_normalized = image_resized.astype(np.float32) / 255.0
        
#         # Extract and enhance green channel for model input
#         enhanced_image = extract_best_channel_and_enhance(image_normalized)
#         enhanced_image = np.expand_dims(enhanced_image, axis=-1)  # Add channel dimension
#         enhanced_image = np.expand_dims(enhanced_image, axis=0)   # Add batch dimension
        
#         # Make prediction
#         prediction = model.predict(enhanced_image, verbose=0)
        
#         # Process prediction
#         prediction = prediction.squeeze()  # Remove batch dimension
#         binary_mask = (prediction > 0.5).astype(np.uint8)
        
#         # Create overlay visualization
#         # Convert original resized image back to uint8
#         original_vis = (image_resized).astype(np.uint8)
        
#         # Create colored mask (red overlay for macular region)
#         colored_mask = np.zeros_like(original_vis)
#         colored_mask[binary_mask == 1] = [255, 0, 0]  # Red color for macular region
        
#         # Blend original image with colored mask
#         alpha = 0.4  # Transparency for overlay
#         result_image = cv2.addWeighted(original_vis, 1-alpha, colored_mask, alpha, 0)
        
#         # Generate unique filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"macular_segmentation_{timestamp}.png"
#         result_path = os.path.join(OUTPUT_DIR, filename)
        
#         # Save the result image
#         result_pil = Image.fromarray(result_image)
#         result_pil.save(result_path)
        
#         # Calculate basic statistics
#         total_pixels = binary_mask.size
#         macular_pixels = np.sum(binary_mask)
#         macular_percentage = (macular_pixels / total_pixels) * 100
#         max_confidence = float(np.max(prediction))
        
#         return {
#             "success": True,
#             "result_path": result_path,
#             "filename": filename,
#             "statistics": {
#                 "total_pixels": int(total_pixels),
#                 "macular_pixels": int(macular_pixels),
#                 "macular_percentage": round(macular_percentage, 2),
#                 "max_confidence": round(max_confidence, 4)
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Error during segmentation: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Macular Segmentation API",
#         "version": "1.0.0",
#         "status": "active"
#     }

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Macular Segmentation API",
    description="API for segmenting macular regions in fundus images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_PATH = "best_macular_model.h5"
OUTPUT_DIR = "segmentation_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model variable
model = None

def get_largest_component(binary_mask):
    """
    Find and keep only the largest connected component (circular region)
    This helps remove small artifacts and keep only the main macular region
    """
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # No components found (only background)
        return binary_mask
    
    # Find the largest component (excluding background which is label 0)
    # stats contains [x, y, width, height, area] for each component
    largest_component_idx = 1  # Start from 1 to skip background
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_component_idx = i
    
    # Create mask with only the largest component
    largest_mask = (labels == largest_component_idx).astype(np.uint8)
    
    # Optional: Apply morphological operations to make the region more circular
    # Create circular kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small gaps and smooth the boundary
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel)
    
    return largest_mask

def extract_best_channel_and_enhance(image):
    """
    Extract the best channel for macular visibility and enhance it
    Green channel is typically best for macular segmentation
    """
    # Extract green channel (index 1)
    green_channel = image[:, :, 1]
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((green_channel * 255).astype(np.uint8))
    
    # Additional enhancement - Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Normalize back to 0-1
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced

def load_model():
    """Load the trained macular segmentation model"""
    global model
    try:
        # Custom objects for loading the model
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            smooth = 1e-6
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            dice_loss = 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
            alpha = 0.3
            beta = 0.7
            tp = tf.keras.backend.sum(y_true_f * y_pred_f)
            fp = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
            fn = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
            tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            return bce + dice_loss + tversky_loss

        def dice_coefficient(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

        def iou_score(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
            return (intersection + smooth) / (union + smooth)

        custom_objects = {
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient,
            'iou_score': iou_score
        }
        
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise Exception(f"Model file not found: {MODEL_PATH}")
    
    success = load_model()
    if not success:
        raise Exception("Failed to load model")

@app.post("/segment")
async def segment_macular(file: UploadFile = File(...)):
    """
    Segment macular region and save result with original image overlay
    
    Returns:
    - result_path: Path to the saved segmentation result image
    - statistics: Basic segmentation statistics
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and resize
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (IMG_WIDTH, IMG_HEIGHT))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Extract and enhance green channel for model input
        enhanced_image = extract_best_channel_and_enhance(image_normalized)
        enhanced_image = np.expand_dims(enhanced_image, axis=-1)  # Add channel dimension
        enhanced_image = np.expand_dims(enhanced_image, axis=0)   # Add batch dimension
        
        # Make prediction
        prediction = model.predict(enhanced_image, verbose=0)
        
        # Process prediction
        prediction = prediction.squeeze()  # Remove batch dimension
        binary_mask = (prediction > 0.5).astype(np.uint8)
        
        # Keep only the largest connected component (main macular region)
        binary_mask = get_largest_component(binary_mask)
        
        # Create overlay visualization
        # Convert original resized image back to uint8
        original_vis = (image_resized).astype(np.uint8)
        
        # Create colored mask (red overlay for macular region)
        colored_mask = np.zeros_like(original_vis)
        colored_mask[binary_mask == 1] = [255, 0, 0]  # Red color for macular region
        
        # Blend original image with colored mask
        alpha = 0.4  # Transparency for overlay
        result_image = cv2.addWeighted(original_vis, 1-alpha, colored_mask, alpha, 0)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"macular_segmentation_{timestamp}.png"
        result_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the result image
        result_pil = Image.fromarray(result_image)
        result_pil.save(result_path)
        
        # Calculate basic statistics
        total_pixels = binary_mask.size
        macular_pixels = np.sum(binary_mask)
        macular_percentage = (macular_pixels / total_pixels) * 100
        max_confidence = float(np.max(prediction))
        
        return {
            "success": True,
            "result_path": result_path,
            "filename": filename,
            "statistics": {
                "total_pixels": int(total_pixels),
                "macular_pixels": int(macular_pixels),
                "macular_percentage": round(macular_percentage, 2),
                "max_confidence": round(max_confidence, 4)
            }
        }
        
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Macular Segmentation API",
        "version": "1.0.0",
        "status": "active"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )