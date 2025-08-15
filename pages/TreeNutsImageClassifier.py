import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_LITE_DISABLE_XNNPACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import treenutclassifier          
import numpy as np
from PIL import Image

# =============== Page configuration ===============
st.set_page_config(
    page_title="Tree Nuts Classifier",
    page_icon="🌰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============== Styling ===============
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #8B4513; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center; }
    .confident-prediction { background: linear-gradient(135deg,#d4edda,#c3e6cb); border: 2px solid #28a745; }
    .uncertain-prediction { background: linear-gradient(135deg,#fff3cd,#ffeaa7); border: 2px solid #ffc107; }
    .unknown-prediction { background: linear-gradient(135deg,#f8d7da,#f5c6cb); border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# =============== Title & intro ===============
st.markdown('<h1 class="main-header">Tree Nuts Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Upload an image and let the AI classify the type of tree nut using trained Random Forest model<br>
    <small>Supports: Almond, Cashew, and Walnut</small>
</div>
""", unsafe_allow_html=True)

# =============== Sidebar ===============
with st.sidebar:
    st.header("About This App")
    st.markdown("""
**What it does**
- Identifies nuts from uploaded images using trained Random Forest
- Built with 88.9% accuracy on test data
- Uses 71 extracted features per image

**Supported Nuts**
- Almond
- Cashew  
- Walnut

**How to use**
1. Upload an image (JPG/PNG)
2. Click Analyze
3. View results with confidence scores
""")

    st.header("Model Details")
    st.markdown("""
- **Model Type:** Random Forest (100 trees)
- **Test Accuracy:** 88.9%
- **Cross-validation:** 90.4% ± 2.5%
- **Features:** 71 extracted features
- **Training Data:** 270 images
- **Test Data:** 90 images
- **Feature Types:** RGB, LBP, HOG, edges, statistics
""")

    # Show model status
    st.header("Model Status")
    model, scaler = treenutclassifier.load_trained_model()
    if model is not None:
        st.success("Trained model loaded successfully")
        st.info("Ready for predictions")
    else:
        st.error("Model files not found")
        st.warning("Please ensure model files are in project directory")

# =============== Nut info (no emojis) ===============
nut_info = {
    'Almond': {
        'description': 'Almonds are nutrient-dense tree nuts rich in vitamin E, healthy fats, and protein.',
        'benefits': ['High in vitamin E', 'Good source of protein', 'Rich in monounsaturated fats', 'May help lower cholesterol'],
    },
    'Cashew': {
        'description': 'Cashews are creamy, kidney-shaped nuts with a mild, sweet flavor.',
        'benefits': ['Rich in copper and zinc', 'Good source of magnesium', 'Contains heart-healthy fats', 'May support bone health'],
    },
    'Walnut': {
        'description': 'Walnuts have a brain-like appearance and are rich in omega-3 fatty acids.',
        'benefits': ['High in omega-3s', 'May support brain health', 'Rich in antioxidants', 'May help reduce inflammation'],
    }
}

# =============== Main Layout ===============
# Create tabs first (always visible)
tab_ui, tab_eval = st.tabs(["Classifier", "Model Evaluation"])

with tab_ui:
    # File uploader ALWAYS visible - prevents page jumping
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a nut for identification",
        key="page_nut_uploader" 
    )
    
    # Create consistent layout columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Show uploaded image immediately when available
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Could not read image: {e}")
                image = None
        else:
            st.info("Upload an image above to get started")
            image = None

    with col2:
        st.header("Analysis Results")
        
        # Analyze button - always in same position
        if uploaded_file is not None and image is not None:
            analyze = st.button("Analyze Image", type="primary", key="analyze_btn")
        else:
            analyze = st.button("Analyze Image", type="primary", disabled=True, key="analyze_btn_disabled")
            analyze = False
        
        # Analysis results using trained Random Forest model
        if analyze and uploaded_file is not None and image is not None:
            # Use trained Random Forest model
            with st.spinner("Analyzing image with trained Random Forest model..."):
                try:
                    predicted_class, confidence, class_probabilities = treenutclassifier.predict_nut(image)
                    
                    if predicted_class is not None:
                        # Confidence band
                        if confidence >= 0.8:
                            box_class = "confident-prediction"
                            certainty_text = "High Confidence"
                        elif confidence >= 0.6:
                            box_class = "uncertain-prediction"
                            certainty_text = "Moderate Confidence"
                        else:
                            box_class = "unknown-prediction"
                            certainty_text = "Low Confidence"

                        nut_data = nut_info.get(predicted_class, {'description': '', 'benefits': []})
                        
                        # Results display (no emojis)
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h2>{predicted_class}</h2>
                            <h4>{certainty_text}</h4>
                            <h3>Confidence: {confidence:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        st.progress(confidence)

                        if confidence < 0.6:
                            st.warning("Low confidence - image may not match trained nut types or quality is unclear.")
                            st.info("Tips: Use a well-lit, close-up image where the nut is the main subject.")

                        # Detailed probabilities
                        st.subheader("Detailed Probabilities")
                        for nut_name, probability in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
                            c1, c2, c3 = st.columns([2, 4, 1])
                            with c1:
                                st.write(f"{nut_name}")
                            with c2:
                                st.progress(max(0.0, min(1.0, probability)))
                            with c3:
                                st.write(f"{probability:.1%}")

                        # Nut information
                        if confidence >= 0.6:
                            st.subheader(f"About {predicted_class}")
                            st.write(nut_data['description'])
                            st.subheader("Health Benefits")
                            for benefit in nut_data['benefits']:
                                st.write(f"• {benefit}")
                            
                            # Show feature extraction info
                            with st.expander("Feature Extraction Details"):
                                st.write("""
                                This prediction is based on 71 extracted features:
                                - **Color features (3):** RGB channel means
                                - **Texture features (10):** Local Binary Pattern histogram  
                                - **Shape features (50):** Histogram of Oriented Gradients
                                - **Edge features (4):** Horizontal and vertical edge statistics
                                - **Statistical features (4):** Grayscale mean, std, min, max
                                """)
                    else:
                        st.error("Prediction failed. Please try again with a different image.")
                        
                except Exception as e:
                    st.error("Error during prediction.")
                    st.write(f"Error details: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image above to begin analysis")
        else:
            st.info("Click 'Analyze Image' to start classification")

with tab_eval:
    st.subheader("Model Performance Evaluation")
    try:
        rep = treenutclassifier.get_eval_report()
        
        if rep.get("model_loaded", False):
            # Show accuracy
            if "accuracy" in rep:
                st.metric("Test Accuracy", f"{float(rep['accuracy']):.1%}")
            
            # Show detailed report
            if rep.get("classification_report_text"):
                st.subheader("Detailed Performance Report")
                st.code(rep["classification_report_text"], language="text")
            
            # Show model info
            st.subheader("Model Information")
            st.write("""
            **Training Process:**
            - Feature extraction from 360 nut images
            - 75/25 train/test split (270 train, 90 test)
            - 5-fold cross-validation
            - Standardized feature scaling
            
            **Model Comparison:**
            - Random Forest: 88.9% (selected)
            - Logistic Regression: 88.9%
            - SVM (RBF): 87.8%
            - K-Nearest Neighbors: 77.8%
            - Gaussian Naive Bayes: 77.8%
            """)
            
        else:
            st.error("Model evaluation unavailable")
            st.write(rep.get("classification_report_text", "No details available"))
            
    except Exception as e:
        st.error(f"Could not load evaluation from backend: {e}")

# =============== Footer ===============
st.markdown("---")
st.caption("Powered by Random Forest • 88.9% test accuracy • 71 extracted features • Trained on 360 nut images")
