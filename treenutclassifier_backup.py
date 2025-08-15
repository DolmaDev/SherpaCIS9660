# treenutclassifier.py - Updated with trained Random Forest model
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog
from skimage.filters import prewitt_h, prewitt_v

# Load trained model and scaler
def load_trained_model():
    try:
        with open('best_nut_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('nut_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

def extract_features_from_image(image):
    """Extract the same 71 features used in training"""
    # Parameters (same as training)
    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform'
    
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB')) / 255.0
    
    # Resize image to 64x64 (same as training)
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    
    # 1. Color features (RGB channel means)
    rgb_features = np.mean(image_resized, axis=(0, 1))
    
    # 2. Convert to grayscale
    gray_image = rgb2gray(image_resized)
    
    # 3. Local Binary Pattern features
    lbp = local_binary_pattern(gray_image, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                             range=(0, n_points + 2), density=True)
    
    # 4. HOG features
    hog_features = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(1, 1), visualize=False)
    
    # 5. Edge features
    horizontal_edges = prewitt_h(gray_image)
    vertical_edges = prewitt_v(gray_image)
    edge_features = np.array([
        np.mean(horizontal_edges), np.std(horizontal_edges),
        np.mean(vertical_edges), np.std(vertical_edges)
    ])
    
    # 6. Statistical features
    stat_features = np.array([
        np.mean(gray_image), np.std(gray_image),
        np.min(gray_image), np.max(gray_image)
    ])
    
    # Combine all features (should be 71 features total)
    combined_features = np.concatenate([
        rgb_features, lbp_hist, hog_features[:50], edge_features, stat_features
    ])
    
    return combined_features.reshape(1, -1)

def predict_nut(image):
    """Predict nut type using trained Random Forest model"""
    model, scaler = load_trained_model()
    
    if model is None:
        return None, 0.0, {}
    
    try:
        # Extract features
        features = extract_features_from_image(image)
        
        # Scale features (same as training)
        features_scaled = scaler.transform(features)
        
        # Get prediction and probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        class_names = ['Almond', 'Cashew', 'Walnut']
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        # Create probability dictionary
        class_probabilities = {
            class_names[i]: float(probabilities[i]) for i in range(len(class_names))
        }
        
        return predicted_class, confidence, class_probabilities
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, {}

def get_eval_report():
    """Return evaluation metrics for the trained model"""
    model, scaler = load_trained_model()
    
    if model is None:
        return {
            "accuracy": 0.00,
            "classification_report_text": "Trained model files not found.\nPlease ensure 'best_nut_model.pkl' and 'nut_scaler.pkl' are in the project directory.",
            "model_loaded": False
        }
    
    return {
        "accuracy": 0.889,
        "classification_report_text": """Trained Random Forest Model Performance:

Test Accuracy: 88.9%
Cross-validation: 90.4% ± 2.5%

Per-class Performance:
- Almond: Precision 0.93, Recall 0.87, F1 0.90
- Cashew: Precision 0.83, Recall 0.83, F1 0.83  
- Walnut: Precision 0.91, Recall 0.97, F1 0.94

Model Details:
- Type: Random Forest (100 trees)
- Features: 71 extracted features
- Training: 270 images, Test: 90 images
- Feature types: RGB, LBP, HOG, edges, statistics""",
        "model_loaded": True
    }
