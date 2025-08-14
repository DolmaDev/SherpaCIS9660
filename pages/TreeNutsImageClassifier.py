import numpy as np
from PIL import Image
import streamlit as st
import os, threading
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quiet TF logs
os.environ.setdefault("TF_LITE_DISABLE_XNNPACK", "1")  # avoid macOS deadlocks
os.environ.setdefault("OMP_NUM_THREADS", "1")        # be conservative with threads


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
st.markdown('<h1 class="main-header"> Tree Nuts Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Upload an image and let the image classifier identify the type of tree nut!<br>
    <small>Supports: Almonds, Cashews, and Walnuts</small>
</div>
""", unsafe_allow_html=True)

# =============== Sidebar ===============
with st.sidebar:
    st.header("About This App")
    st.markdown("""
**What it does**
- Identifies nuts from uploaded images
- Built for academic purposes

**Supported Nuts**
- Almonds
- Cashews
- Walnuts

**How to use**
1. Upload an image (JPG/PNG)
2. Click **Analyze**
3. View results
""")

    st.header("Model Info")
    st.markdown("""
- **Backbone:** MobileNetV2 (converted to TFLite)
- **Input Size:** 224×224
- **Classes:** 3
""")

# =============== TFLite loader (via TensorFlow Lite) ===============
@st.cache_resource(show_spinner=True)
def load_tflite(model_path: str = "treenuts_classifier_final.tflite"):
    try:
        from tensorflow.lite import Interpreter  # uses TensorFlow's bundled TFLite
    except Exception as e:
        st.error("TensorFlow Lite is not available. Install TensorFlow (prefer Python 3.12) "
                 "or deploy on Streamlit Cloud with Python 3.12.")
        st.caption(f"Import error: {e}")
        return None, None, None

    try:
        interpreter = Interpreter(model_path=model_path, num_thread=1)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error("Failed to load TFLite model.")
        st.exception(e)
        return None, None, None


# =============== Utilities ===============
def preprocess_image(image: Image.Image, target_size=(224, 224), dtype=np.float32) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = (np.asarray(image, dtype=dtype) / 255.0)[None, ...]  # (1,H,W,C)
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def tflite_predict(interpreter, in_details, out_details, image: Image.Image, class_names):
    """
    Runs a single TFLite inference with a session-level lock to prevent macOS/XNNPACK deadlocks.
    Assumes you've already:
      - disabled XNNPACK via TF_LITE_DISABLE_XNNPACK=1
      - created the interpreter with num_threads=1
    """
    # Determine expected input size and dtype from the model
    in_info = in_details[0]
    _, in_h, in_w, in_c = in_info["shape"]  # shape like [1, H, W, C]
    target_size = (in_w, in_h)  # PIL expects (width, height)

    # Preprocess -> float32 in [0,1]
    x = preprocess_image(image, target_size=target_size)

    # Match dtype (handle quantized uint8 properly)
    if in_info["dtype"] == np.uint8:
        x = (x * 255.0).astype(np.uint8)  # scale to [0..255] before casting
    elif x.dtype != in_info["dtype"]:
        x = x.astype(in_info["dtype"])

    # ---- Session-level lock to avoid native thread deadlocks ----
    lock = st.session_state.get("_tflite_lock")
    if lock is None:
        import threading
        lock = threading.Lock()
        st.session_state["_tflite_lock"] = lock

    with lock:
        interpreter.set_tensor(in_info["index"], x)
        interpreter.invoke()
        out_info = out_details[0]
        out_raw = interpreter.get_tensor(out_info["index"])[0]

    # Dequantize output if needed
    qscale, qzero = (0.0, 0)
    if "quantization" in out_info and out_info["quantization"] != (0.0, 0):
        qscale, qzero = out_info["quantization"]

    if qscale and qscale != 0.0:
        out = (out_raw.astype(np.float32) - qzero) * qscale
    else:
        out = out_raw.astype(np.float32)

    # Normalize to probabilities if logits
    if not np.isclose(out.sum(), 1.0, atol=1e-3):
        out = softmax(out)

    pred_i = int(np.argmax(out))
    probs = {class_names[i]: float(out[i]) for i in range(len(class_names))}
    return class_names[pred_i], float(out[pred_i]), probs


# =============== Nut info ===============
nut_info = {
    'Almonds': {
        'description': 'Almonds are nutrient-dense tree nuts rich in vitamin E, healthy fats, and protein.',
        'benefits': ['High in vitamin E', 'Good source of protein', 'Rich in monounsaturated fats', 'May help lower cholesterol'],
    },
    'Cashews': {
        'description': 'Cashews are creamy, kidney-shaped nuts with a mild, sweet flavor.',
        'benefits': ['Rich in copper and zinc', 'Good source of magnesium', 'Contains heart-healthy fats', 'May support bone health'],
    },
    'Walnuts': {
        'description': 'Walnuts have a brain-like appearance and are rich in omega-3 fatty acids.',
        'benefits': ['High in omega-3s', 'May support brain health', 'Rich in antioxidants', 'May help reduce inflammation'],
    }
}

# =============== Layout ===============
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload an image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a nut for identification"
    )

with col2:
    st.header("Analysis Results")

    analyze = st.button("Analyze", type="primary", disabled=(uploaded_file is None))

    if uploaded_file is None:
        st.info("Upload an image, then click **Analyze** to begin.")
    elif analyze:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.stop()

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Load TFLite model
        with st.spinner("Loading model…"):
            interp, in_det, out_det = load_tflite("treenuts_classifier_final.tflite")
        if interp is None:
            st.stop()

        # Predict
        with st.spinner("Analyzing image…"):
            class_names = ['Almonds', 'Cashews', 'Walnuts']
            try:
                predicted_class, confidence, class_probabilities = tflite_predict(
                    interp, in_det, out_det, image, class_names
                )
            except Exception as e:
                st.error("Error during prediction.")
                st.exception(e)
                st.stop()

        # Confidence band
        if confidence >= 0.8:
            box_class = "confident-prediction"
            certainty_text = "High Confidence"
        elif confidence >= 0.6:
            box_class = "uncertain-prediction"
            certainty_text = "Moderate Confidence"
        else:
            box_class = "unknown-prediction"
            certainty_text = "Low Confidence — May not be a trained nut"

        nut_data = nut_info.get(predicted_class, {'emoji': '🌰', 'description': '', 'benefits': []})
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2>{nut_data['emoji']} {predicted_class}</h2>
            <h4>{certainty_text}</h4>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

        if confidence < 0.6:
            st.warning("⚠️ Low confidence — image may not match (Almonds/Cashews/Walnuts) or quality is unclear.")
            st.info("💡 Tips: Use a well-lit, close-up image where the nut is the main subject.")

        st.subheader("Detailed Probabilities")
        for nut_name, probability in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
            c1, c2, c3 = st.columns([2, 4, 1])
            with c1:
                st.write(f"{nut_info.get(nut_name, {}).get('emoji','🌰')} {nut_name}")
            with c2:
                st.progress(max(0.0, min(1.0, probability)))
            with c3:
                st.write(f"{probability:.1%}")

        if confidence >= 0.6:
            st.subheader(f"About {predicted_class}")
            st.write(nut_data['description'])
            st.subheader("Health Benefits")
            for b in nut_data['benefits']:
                st.write(f"• {b}")

# =============== Footer ===============
st.markdown("---")
st.caption("Runs on TFLite (no TensorFlow) • MobileNetV2 transfer learning (converted)")
