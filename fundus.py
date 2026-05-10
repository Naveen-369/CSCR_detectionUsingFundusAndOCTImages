from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import io
import os
import uvicorn
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pennylane as qml
from pennylane import numpy as pnp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum EfficientNet Classification API with Grad-CAM",
    description="API for classifying eye disease images using quantum-enhanced EfficientNet with Grad-CAM visualization",
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
MODEL_PATH = "quantum_efficientnet_cscr_healthy.pth"
OUTPUT_DIR = "classification_results"
N_QUBITS = 4

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
dev = None

# Class names
CLASS_NAMES = {0: "CSCR", 1: "Healthy"}

# Data preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Quantum circuit definition (same as training)
def quantum_circuit(inputs, weights):
    """Quantum circuit for processing features"""
    for i in range(N_QUBITS):
        qml.RY(inputs[i].float(), wires=i)

    for layer in range(2):  # 2 layers
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0].float(), wires=i)
            qml.RZ(weights[layer, i, 1].float(), wires=i)

        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def setup_quantum_circuit():
    """Setup quantum circuit"""
    global dev
    dev = qml.device("default.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_layer(inputs, weights):
        return quantum_circuit(inputs, weights)
    
    return quantum_layer

# Quantum EfficientNet model definition (same as training)
class QuantumEfficientNet(nn.Module):
    def __init__(self, num_classes=2, n_qubits=4):
        super(QuantumEfficientNet, self).__init__()

        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        for param in list(self.efficientnet.parameters())[:-10]:
            param.requires_grad = False

        self.features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        feature_dim = self.efficientnet.classifier[1].in_features

        self.feature_reducer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_qubits)
        )

        self.n_qubits = n_qubits
        self.quantum_weights = nn.Parameter(torch.randn(2, n_qubits, 2, dtype=torch.float32) * 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits + n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        classical_features = self.features(x)
        classical_features = self.feature_reducer(classical_features)
        classical_features = classical_features.float()

        quantum_outputs = []
        quantum_layer = setup_quantum_circuit()
        
        for i in range(x.shape[0]):
            quantum_input = classical_features[i].float()
            quantum_output = quantum_layer(quantum_input, self.quantum_weights.float())
            quantum_tensor = torch.stack([torch.tensor(q, dtype=torch.float32, device=x.device)
                                        for q in quantum_output])
            quantum_outputs.append(quantum_tensor)

        quantum_features = torch.stack(quantum_outputs).float()
        combined_features = torch.cat([classical_features, quantum_features], dim=1)
        output = self.classifier(combined_features.float())
        return output

    def get_conv_features(self, x):
        """Extract convolutional features for Grad-CAM"""
        # Get features from EfficientNet backbone before the classifier
        features = self.efficientnet.features(x)
        return features

class GradCAM:
    """Grad-CAM implementation for the quantum-enhanced EfficientNet"""
    def __init__(self, model, target_layer_name='features'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []

    def save_gradient(self, grad):
        self.gradients = grad

    def save_activation(self, module, input, output):
        self.activations = output

    def register_hooks(self):
        # Register hooks on EfficientNet features
        target_layer = self.model.efficientnet.features
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_backward_hook(lambda m, gi, go: self.save_gradient(go[0])))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()
        self.register_hooks()

        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU to keep positive influences
        cam = torch.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        self.remove_hooks()
        return cam.detach().cpu().numpy()

def create_gradcam_overlay(original_img, cam, alpha=0.4):
    """Create Grad-CAM overlay on original image"""
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Convert to heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Convert original image tensor to numpy
    if isinstance(original_img, torch.Tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_img = original_img * std + mean
        original_img = torch.clamp(original_img, 0, 1)
        
        # Convert to numpy
        original_np = original_img.permute(1, 2, 0).cpu().numpy()
        original_np = (original_np * 255).astype(np.uint8)
    else:
        original_np = original_img

    # Create overlay
    overlay = cv2.addWeighted(original_np, 1-alpha, heatmap, alpha, 0)
    return overlay, heatmap

def load_model():
    """Load the trained quantum EfficientNet model"""
    global model
    
    try:
        model = QuantumEfficientNet(num_classes=2, n_qubits=N_QUBITS).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
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
    
    # Setup quantum circuit
    setup_quantum_circuit()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify eye disease image and generate Grad-CAM visualization
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Store original for visualization
        original_img = np.array(image.resize((224, 224)))
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        logger.info(f"Prediction: {CLASS_NAMES[predicted_class]} with confidence {confidence:.4f}")
        
        # Generate Grad-CAM
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(input_tensor, predicted_class)
        
        # Create visualizations
        overlay, heatmap = create_gradcam_overlay(input_tensor[0], cam)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        axes[0, 1].imshow(cam, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(f'Grad-CAM Overlay\nPrediction: {CLASS_NAMES[predicted_class]}')
        axes[1, 0].axis('off')
        
        # Prediction probabilities
        class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        probs = probabilities[0].cpu().numpy()
        bars = axes[1, 1].bar(class_names, probs)
        axes[1, 1].set_title(f'Classification Results\nConfidence: {confidence:.2%}')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_ylim(0, 1)
        
        # Color bars based on prediction
        for i, bar in enumerate(bars):
            if i == predicted_class:
                bar.set_color('green')
            else:
                bar.set_color('lightcoral')
        
        # Add probability text on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"eye_disease_classification_{timestamp}.png"
        result_path = os.path.join(OUTPUT_DIR, result_filename)
        
        plt.tight_layout()
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "prediction": {
                "class": CLASS_NAMES[predicted_class],
                "class_index": predicted_class,
                "confidence": round(confidence, 4),
                "all_probabilities": {
                    CLASS_NAMES[i]: round(float(prob), 4) 
                    for i, prob in enumerate(probs)
                }
            },
            "result_path": result_path,
            "filename": result_filename,
            "gradcam_stats": {
                "cam_shape": list(cam.shape),
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
                "cam_mean": float(cam.mean())
            },
            "model_info": {
                "architecture": "Quantum-Enhanced EfficientNet-B0",
                "quantum_qubits": N_QUBITS,
                "device": str(device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum EfficientNet Eye Disease Classification API",
        "version": "1.0.0",
        "status": "active",
        "model_architecture": "Quantum-Enhanced EfficientNet-B0",
        "classes": list(CLASS_NAMES.values()),
        "quantum_qubits": N_QUBITS,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "quantum_circuit_ready": dev is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )