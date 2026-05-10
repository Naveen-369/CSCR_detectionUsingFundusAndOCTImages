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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pennylane as qml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCT Classification API with Grad-CAM",
    description="API for classifying OCT images with quantum neural network and Grad-CAM visualization",
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
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_PATH = "models/quantum_neural_network_oct.h5"
SCALER_PARAMS_PATH = "models/scaler_params.npy"
QUANTUM_WEIGHTS_PATH = "models/quantum_weights.npy"
OUTPUT_DIR = "oct_results"
N_QUBITS = 8
N_LAYERS = 2
N_FEATURES = 64
# Your selected features from training
SELECTED_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
model = None
scaler = None
quantum_weights = None
dev = None

# Class names (adjust according to your dataset)
CLASS_NAMES = {0: "Normal", 1: "CSR"}

def extract_features_from_images(images, n_features=64):
    """Extract features from images using various statistical measures"""
    features = []

    for img in images:
        # Flatten and get statistical features
        flat_img = img.flatten()

        # Statistical features
        mean_val = np.mean(flat_img)
        std_val = np.std(flat_img)
        min_val = np.min(flat_img)
        max_val = np.max(flat_img)
        median_val = np.median(flat_img)

        # Histogram features
        hist, _ = np.histogram(flat_img, bins=10)
        hist = hist / np.sum(hist)  # Normalize

        # Gradient features
        grad_x = np.gradient(img, axis=0)
        grad_y = np.gradient(img, axis=1)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(grad_magnitude)
        grad_std = np.std(grad_magnitude)

        # Local Binary Pattern-like features
        center_pixels = img[1:-1, 1:-1]
        neighbor_pixels = [
            img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
            img[1:-1, :-2],               img[1:-1, 2:],
            img[2:, :-2], img[2:, 1:-1], img[2:, 2:]
        ]

        lbp_features = []
        for neighbor in neighbor_pixels:
            lbp_features.append(np.mean(neighbor > center_pixels))

        # Combine all features
        feature_vector = [mean_val, std_val, min_val, max_val, median_val, grad_mean, grad_std] + \
                        list(hist) + lbp_features

        # Add texture features (variance in local windows)
        h, w = img.shape
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                window = img[i:i+8, j:j+8]
                feature_vector.append(np.var(window))

        features.append(feature_vector[:n_features])  # Limit to n_features

    return np.array(features)

def setup_quantum_circuit():
    """Setup quantum circuit for feature processing"""
    global dev
    
    dev = qml.device("default.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev)
    def quantum_node(weights, x):
        """Quantum node for processing features"""
        qml.templates.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        qml.templates.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    
    return quantum_node

def quanv(features, weights, quantum_node):
    """Apply the quantum circuit to the features."""
    processed_features = []
    for row in features:
        result = quantum_node(weights, row)
        processed_features.append(result)
    return np.array(processed_features)

def generate_gradcam_heatmap(model, img_array, class_idx, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for the prediction
    Since we're working with feature-based model, we'll create a feature importance heatmap
    """
    try:
        # For feature-based models, we'll create a simple importance visualization
        # Get the model's weights from the first dense layer
        first_layer_weights = model.layers[0].get_weights()[0]  # Shape: (n_features, 128)
        
        # Calculate feature importance based on weights
        feature_importance = np.abs(first_layer_weights).mean(axis=1)
        
        # Normalize importance scores
        if feature_importance.max() > feature_importance.min():
            feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min())
        
        # Create a simple heatmap visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(range(len(feature_importance)), feature_importance)
        
        # Color bars based on importance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.hot(feature_importance[i]))
        
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance Visualization (Grad-CAM style)')
        
        # Save the plot
        importance_path = os.path.join(OUTPUT_DIR, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return importance_path, feature_importance
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return None, None

def load_model_and_preprocessors():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_selector, quantum_weights
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Load scaler parameters
        if os.path.exists(SCALER_PARAMS_PATH):
            scaler_params = np.load(SCALER_PARAMS_PATH)
            scaler_mean = scaler_params[0]  # scaler.mean_
            scaler_scale = scaler_params[1]  # scaler.scale_
            
            # Create scaler with loaded parameters
            scaler = StandardScaler()
            scaler.mean_ = scaler_mean
            scaler.scale_ = scaler_scale
            scaler.var_ = scaler_scale ** 2  # variance = scale^2
            scaler.n_features_in_ = len(scaler_mean)
            scaler.n_samples_seen_ = 1000  # dummy value
            
            logger.info(f"Scaler loaded successfully with {len(scaler_mean)} features")
            logger.info(f"Scaler mean range: {scaler_mean.min():.4f} to {scaler_mean.max():.4f}")
            logger.info(f"Scaler scale range: {scaler_scale.min():.4f} to {scaler_scale.max():.4f}")
        else:
            logger.error(f"Scaler parameters file not found at {SCALER_PARAMS_PATH}")
            return False
            
        # Load quantum weights
        if os.path.exists(QUANTUM_WEIGHTS_PATH):
            quantum_weights = np.load(QUANTUM_WEIGHTS_PATH)
            logger.info(f"Quantum weights loaded successfully, shape: {quantum_weights.shape}")
        else:
            logger.error(f"Quantum weights file not found at {QUANTUM_WEIGHTS_PATH}")
            return False
        
        # Setup quantum circuit
        setup_quantum_circuit()
        
        logger.info("All components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model and preprocessors: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessors on startup"""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise Exception(f"Model file not found: {MODEL_PATH}")
    
    success = load_model_and_preprocessors()
    if not success:
        raise Exception("Failed to load model and preprocessors")

@app.post("/classify")
async def classify_oct(file: UploadFile = File(...)):
    """
    Classify OCT image and generate Grad-CAM visualization
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
        
        # Convert to grayscale and resize
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array and resize
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (IMG_WIDTH, IMG_HEIGHT))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Extract features
        features = extract_features_from_images([image_normalized], n_features=N_FEATURES)
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Feature range: {features.min():.4f} to {features.max():.4f}")
        
        # Select the EXACT features that were selected during training [0,1,2,3,4,5,6,7]
        features_selected = features[:, SELECTED_FEATURES]
        
        logger.info(f"Selected features shape: {features_selected.shape}")
        logger.info(f"Selected features: {SELECTED_FEATURES}")
        logger.info(f"Selected feature range: {features_selected.min():.4f} to {features_selected.max():.4f}")
        
        # Scale features using the EXACT scaler from training
        features_scaled = scaler.transform(features_selected)
        
        logger.info(f"Scaled features shape: {features_scaled.shape}")
        logger.info(f"Scaled feature range: {features_scaled.min():.4f} to {features_scaled.max():.4f}")
        
        # Apply quantum processing
        quantum_node = setup_quantum_circuit()
        quantum_features = quanv(features_scaled, quantum_weights, quantum_node)
        
        logger.info(f"Quantum features shape: {quantum_features.shape}")
        logger.info(f"Quantum feature range: {quantum_features.min():.4f} to {quantum_features.max():.4f}")
        
        # Make prediction
        prediction = model.predict(quantum_features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Predicted class: {predicted_class}, confidence: {confidence}")
        
        # Generate Grad-CAM visualization
        gradcam_path, feature_importance = generate_gradcam_heatmap(model, quantum_features, predicted_class)
        
        # Create result visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original image
        axes[0].imshow(image_resized, cmap='gray')
        axes[0].set_title('Original OCT Image')
        axes[0].axis('off')
        
        # Prediction result
        class_names_list = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        axes[1].bar(class_names_list, prediction[0])
        axes[1].set_title(f'Prediction: {CLASS_NAMES[predicted_class]} ({confidence:.2%})')
        axes[1].set_ylabel('Confidence')
        axes[1].set_ylim(0, 1)
        
        # Feature visualization
        axes[2].plot(features[0][:20])  # Plot first 20 features
        axes[2].set_title('First 20 Extracted Features')
        axes[2].set_xlabel('Feature Index')
        axes[2].set_ylabel('Feature Value')
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"oct_classification_{timestamp}.png"
        result_path = os.path.join(OUTPUT_DIR, result_filename)
        
        plt.tight_layout()
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "prediction": {
                "class": CLASS_NAMES[predicted_class],
                "class_index": int(predicted_class),
                "confidence": round(confidence, 4),
                "all_probabilities": {
                    CLASS_NAMES[i]: round(float(prob), 4) 
                    for i, prob in enumerate(prediction[0])
                }
            },
            "result_path": result_path,
            "gradcam_path": gradcam_path,
            "filename": result_filename,
            "feature_stats": {
                "num_features_extracted": int(features.shape[1]),
                "num_features_selected": int(features_selected.shape[1]),
                "num_features_used": int(quantum_features.shape[1]),
                "feature_importance_available": feature_importance is not None,
                "raw_prediction_values": prediction[0].tolist()
            },
            "debug_info": {
                "feature_range": f"{features.min():.4f} to {features.max():.4f}",
                "scaled_range": f"{features_scaled.min():.4f} to {features_scaled.max():.4f}",
                "quantum_range": f"{quantum_features.min():.4f} to {quantum_features.max():.4f}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OCT Classification API with Quantum Neural Network",
        "version": "1.0.0",
        "status": "active",
        "model_type": "Quantum Neural Network",
        "classes": list(CLASS_NAMES.values()),
        "selected_features": SELECTED_FEATURES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "quantum_circuit_ready": dev is not None,
        "scaler_loaded": scaler is not None,
        "quantum_weights_loaded": quantum_weights is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )