from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
OCT_MODEL_PATH = "models/quantum_neural_network_oct.h5"
OCT_SCALER_PARAMS_PATH = "models/scaler_params.npy"
OCT_QUANTUM_WEIGHTS_PATH = "models/quantum_weights.npy"
FUNDUS_MODEL_PATH = "quantum_efficientnet_cscr_healthy.pth"
SEGMENTATION_MODEL_PATH = "best_macular_model.h5"

OUTPUT_DIR = "static/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fundus_model = None
oct_model = None
segmentation_model = None
oct_scaler = None
oct_quantum_weights = None

# Class names
OCT_CLASS_NAMES = {0: "Normal", 1: "CSR"}
FUNDUS_CLASS_NAMES = {0: "CSCR", 1: "Healthy"}
OCT_N_QUBITS = 8
OCT_N_LAYERS = 2
OCT_N_FEATURES = 64
OCT_SELECTED_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]
FUNDUS_N_QUBITS = 4

# ============ OCT FUNCTIONS ============

def oct_extract_features_from_images(images, n_features=64):
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

def oct_setup_quantum_circuit():
    """Setup quantum circuit for feature processing"""
    dev = qml.device("default.qubit", wires=OCT_N_QUBITS)
    
    @qml.qnode(dev)
    def quantum_node(weights, x):
        """Quantum node for processing features"""
        qml.templates.AngleEmbedding(x, wires=range(OCT_N_QUBITS), rotation="Y")
        qml.templates.BasicEntanglerLayers(weights, wires=range(OCT_N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(OCT_N_QUBITS)]
    
    return quantum_node

def oct_quanv(features, weights, quantum_node):
    """Apply the quantum circuit to the features."""
    processed_features = []
    for row in features:
        result = quantum_node(weights, row)
        processed_features.append(result)
    return np.array(processed_features)

def oct_generate_gradcam_heatmap(model, img_array, class_idx):
    """
    Generate Grad-CAM heatmap for the prediction
    """
    try:
        # For feature-based models, create a simple importance visualization
        first_layer_weights = model.layers[0].get_weights()[0]
        
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_path = os.path.join(OUTPUT_DIR, f"oct_feature_importance_{timestamp}.png")
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return importance_path, feature_importance
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return None, None

def load_oct_model():
    """Load the OCT model and preprocessing components"""
    global oct_model, oct_scaler, oct_quantum_weights
    
    try:
        # Custom objects for OCT model
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            smooth = 1e-6
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            dice_loss = 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
            return bce + dice_loss

        custom_objects = {'combined_loss': combined_loss}
        
        oct_model = tf.keras.models.load_model(OCT_MODEL_PATH, custom_objects=custom_objects)
        logger.info("OCT model loaded successfully")
        
        # Load scaler parameters
        if os.path.exists(OCT_SCALER_PARAMS_PATH):
            scaler_params = np.load(OCT_SCALER_PARAMS_PATH)
            scaler_mean = scaler_params[0]
            scaler_scale = scaler_params[1]
            
            oct_scaler = StandardScaler()
            oct_scaler.mean_ = scaler_mean
            oct_scaler.scale_ = scaler_scale
            oct_scaler.var_ = scaler_scale ** 2
            oct_scaler.n_features_in_ = len(scaler_mean)
            logger.info("OCT scaler loaded successfully")
            
        # Load quantum weights
        if os.path.exists(OCT_QUANTUM_WEIGHTS_PATH):
            oct_quantum_weights = np.load(OCT_QUANTUM_WEIGHTS_PATH)
            logger.info("OCT quantum weights loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading OCT model: {str(e)}")
        return False

# ============ FUNDUS FUNCTIONS ============

# Fundus data preprocessing
fundus_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fundus_quantum_circuit(inputs, weights):
    """Quantum circuit for processing features"""
    for i in range(FUNDUS_N_QUBITS):
        qml.RY(inputs[i].float(), wires=i)

    for layer in range(2):  # 2 layers
        for i in range(FUNDUS_N_QUBITS):
            qml.RY(weights[layer, i, 0].float(), wires=i)
            qml.RZ(weights[layer, i, 1].float(), wires=i)

        for i in range(FUNDUS_N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[FUNDUS_N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(FUNDUS_N_QUBITS)]

def fundus_setup_quantum_circuit():
    """Setup quantum circuit"""
    dev = qml.device("default.qubit", wires=FUNDUS_N_QUBITS)
    
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_layer(inputs, weights):
        return fundus_quantum_circuit(inputs, weights)
    
    return quantum_layer

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
        quantum_layer = fundus_setup_quantum_circuit()
        
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
        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        self.remove_hooks()
        return cam.detach().cpu().numpy()

def fundus_create_gradcam_overlay(original_img, cam, alpha=0.4):
    """Create Grad-CAM overlay on original image"""
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    if isinstance(original_img, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_img = original_img * std + mean
        original_img = torch.clamp(original_img, 0, 1)
        original_np = original_img.permute(1, 2, 0).cpu().numpy()
        original_np = (original_np * 255).astype(np.uint8)
    else:
        original_np = original_img

    overlay = cv2.addWeighted(original_np, 1-alpha, heatmap, alpha, 0)
    return overlay, heatmap

def load_fundus_model():
    """Load the trained quantum EfficientNet model"""
    global fundus_model
    try:
        fundus_model = QuantumEfficientNet(num_classes=2, n_qubits=FUNDUS_N_QUBITS).to(device)
        fundus_model.load_state_dict(torch.load(FUNDUS_MODEL_PATH, map_location=device))
        fundus_model.eval()
        logger.info("Fundus model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading fundus model: {str(e)}")
        return False

# ============ SEGMENTATION FUNCTIONS ============

def segmentation_get_largest_component(binary_mask):
    """Find and keep only the largest connected component"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:
        return binary_mask
    
    largest_component_idx = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_component_idx = i
    
    largest_mask = (labels == largest_component_idx).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel)
    
    return largest_mask

def segmentation_extract_best_channel_and_enhance(image):
    """Extract the best channel for macular visibility and enhance it"""
    green_channel = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((green_channel * 255).astype(np.uint8))
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    enhanced = enhanced.astype(np.float32) / 255.0
    return enhanced

def load_segmentation_model():
    """Load the trained macular segmentation model"""
    global segmentation_model
    try:
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            smooth = 1e-6
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            dice_loss = 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
            return bce + dice_loss

        custom_objects = {'combined_loss': combined_loss}
        
        segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, custom_objects=custom_objects)
        logger.info("Segmentation model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading segmentation model: {str(e)}")
        return False

# ============ LOAD ALL MODELS ============

def load_all_models():
    """Load all models at startup"""
    success = True
    if os.path.exists(OCT_MODEL_PATH):
        success = success and load_oct_model()
    if os.path.exists(FUNDUS_MODEL_PATH):
        success = success and load_fundus_model()
    if os.path.exists(SEGMENTATION_MODEL_PATH):
        success = success and load_segmentation_model()
    return success

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_type = request.form.get('image_type', 'fundus')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image_type == 'oct':
            return process_oct_image(image, file.filename)
        else:  # fundus
            return process_fundus_image(image, file.filename)
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

def process_oct_image(image, filename):
    """Process OCT image"""
    if oct_model is None:
        return jsonify({'error': 'OCT model not loaded'}), 500
    
    # Convert to grayscale and resize
    if image.mode != 'L':
        image = image.convert('L')
    
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Extract features
    features = oct_extract_features_from_images([image_normalized], n_features=OCT_N_FEATURES)
    features_selected = features[:, OCT_SELECTED_FEATURES]
    features_scaled = oct_scaler.transform(features_selected)
    
    # Apply quantum processing
    quantum_node = oct_setup_quantum_circuit()
    quantum_features = oct_quanv(features_scaled, oct_quantum_weights, quantum_node)
    
    # Make prediction
    prediction = oct_model.predict(quantum_features, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    
    # Generate Grad-CAM visualization
    gradcam_path, feature_importance = oct_generate_gradcam_heatmap(oct_model, quantum_features, predicted_class)
    
    # Create result visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    axes[0].imshow(image_resized, cmap='gray')
    axes[0].set_title('Original OCT Image')
    axes[0].axis('off')
    
    # Prediction result
    class_names_list = [OCT_CLASS_NAMES[i] for i in range(len(OCT_CLASS_NAMES))]
    axes[1].bar(class_names_list, prediction[0])
    axes[1].set_title(f'Prediction: {OCT_CLASS_NAMES[predicted_class]} ({confidence:.2%})')
    axes[1].set_ylabel('Confidence')
    axes[1].set_ylim(0, 1)
    
    # Feature visualization
    axes[2].plot(features[0][:20])
    axes[2].set_title('First 20 Extracted Features')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Feature Value')
    
    # Save result image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"oct_result_{timestamp}.png"
    result_path = os.path.join(OUTPUT_DIR, result_filename)
    plt.tight_layout()
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return jsonify({
        'success': True,
        'image_type': 'oct',
        'prediction': OCT_CLASS_NAMES[predicted_class],
        'confidence': round(confidence, 4),
        'result_image': result_filename,
        'all_probabilities': {
            OCT_CLASS_NAMES[i]: round(float(prob), 4) 
            for i, prob in enumerate(prediction[0])
        }
    })

def process_fundus_image(image, filename):
    """Process fundus image"""
    if fundus_model is None:
        return jsonify({'error': 'Fundus model not loaded'}), 500
    
    # Convert to RGB
    image = image.convert('RGB')
    original_img = np.array(image.resize((224, 224)))
    
    # Apply preprocessing
    input_tensor = fundus_transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = fundus_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    result_data = {
        'success': True,
        'image_type': 'fundus',
        'prediction': FUNDUS_CLASS_NAMES[predicted_class],
        'confidence': round(confidence, 4),
        'all_probabilities': {
            FUNDUS_CLASS_NAMES[i]: round(float(prob), 4) 
            for i, prob in enumerate(probabilities[0])
        }
    }
    
    # Generate Grad-CAM
    gradcam = GradCAM(fundus_model)
    cam = gradcam.generate_cam(input_tensor, predicted_class)
    overlay, heatmap = fundus_create_gradcam_overlay(input_tensor[0], cam)
    
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
    axes[1, 0].set_title(f'Grad-CAM Overlay\nPrediction: {FUNDUS_CLASS_NAMES[predicted_class]}')
    axes[1, 0].axis('off')
    
    # Prediction probabilities
    class_names = [FUNDUS_CLASS_NAMES[i] for i in range(len(FUNDUS_CLASS_NAMES))]
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
    
    # Save fundus classification result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fundus_result_filename = f"fundus_classification_{timestamp}.png"
    fundus_result_path = os.path.join(OUTPUT_DIR, fundus_result_filename)
    plt.tight_layout()
    plt.savefig(fundus_result_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    result_data['result_image'] = fundus_result_filename
    
    # If prediction is not "Healthy", run segmentation
    if FUNDUS_CLASS_NAMES[predicted_class] != "Healthy" and segmentation_model is not None:
        segmentation_result = run_segmentation(image)
        if segmentation_result:
            result_data['segmentation'] = segmentation_result
    
    return jsonify(result_data)

def run_segmentation(image):
    """Run macular segmentation on fundus image"""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and resize
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Extract and enhance green channel for model input
        enhanced_image = segmentation_extract_best_channel_and_enhance(image_normalized)
        enhanced_image = np.expand_dims(enhanced_image, axis=-1)
        enhanced_image = np.expand_dims(enhanced_image, axis=0)
        
        # Make prediction
        prediction = segmentation_model.predict(enhanced_image, verbose=0)
        prediction = prediction.squeeze()
        binary_mask = (prediction > 0.5).astype(np.uint8)
        binary_mask = segmentation_get_largest_component(binary_mask)
        
        # Create overlay visualization
        original_vis = (image_resized).astype(np.uint8)
        colored_mask = np.zeros_like(original_vis)
        colored_mask[binary_mask == 1] = [255, 0, 0]
        
        alpha = 0.4
        result_image = cv2.addWeighted(original_vis, 1-alpha, colored_mask, alpha, 0)
        
        # Save segmentation result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_filename = f"segmentation_{timestamp}.png"
        seg_path = os.path.join(OUTPUT_DIR, seg_filename)
        result_pil = Image.fromarray(result_image)
        result_pil.save(seg_path)
        
        # Calculate statistics
        total_pixels = binary_mask.size
        macular_pixels = np.sum(binary_mask)
        macular_percentage = (macular_pixels / total_pixels) * 100
        
        return {
            'segmentation_image': seg_filename,
            'statistics': {
                'total_pixels': int(total_pixels),
                'macular_pixels': int(macular_pixels),
                'macular_percentage': round(macular_percentage, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        return None

@app.route('/results/<filename>')
def get_result_image(filename):
    """Serve result images"""
    try:
        return send_file(os.path.join(OUTPUT_DIR, filename))
    except Exception as e:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/download/<filename>')
def download_result(filename):
    """Download result images"""
    try:
        return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'oct': oct_model is not None,
            'fundus': fundus_model is not None,
            'segmentation': segmentation_model is not None
        },
        'device': str(device)
    })

# Initialize models when app starts
with app.app_context():
    logger.info("Loading models...")
    if load_all_models():
        logger.info("All models loaded successfully")
    else:
        logger.warning("Some models failed to load")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)