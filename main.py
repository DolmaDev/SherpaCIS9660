import os
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")  # <- use vendorized distutils
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")       # quieter logs
os.environ.setdefault("TF_LITE_DISABLE_XNNPACK", "1")    # avoid some macOS deadlocks
os.environ.setdefault("OMP_NUM_THREADS", "1")            # be conservative
os.environ.setdefault("TF_METAL_ENABLE", "0")            # TEMP: disable Metal if installed
os.environ.setdefault("KERAS_BACKEND", "tensorflow")     # Ensure Keras uses TensorFlow backend
#import setuptools  #Commenting out for new version teest

import streamlit as st

# st.set_option("server.runOnSave", False)    
@st.cache_resource(show_spinner=True)
def load_model_safely():
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        # This line often prevents the C++ mutex crash with mismatched protobuf:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        import tensorflow as tf
        # TODO: load/return your model, for example:
        # model = tf.keras.models.load_model("models/nut_classifier.keras")
        # return model
        return tf  # placeholder if you’re not loading a saved model yet
    except Exception as e:
        st.error("TensorFlow failed to load in this environment.")
        st.exception(e)
        return None
# ---- Page Config ----
st.set_page_config(page_title="SherpaCIS9660", page_icon="🗺️", layout="centered")

# ---- Custom Styles ----
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2rem;
}
/* Make ALL Streamlit buttons big & pretty */
div.stButton > button {
    width: 100%;
    height: 120px;
    font-size: 1.25rem;
    font-weight: 700;
    border-radius: 14px;
    color: #fff;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: 0;
    transition: transform .15s ease-in-out, filter .15s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.02);
    filter: brightness(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---- Title & Subtitle ----
st.markdown('<div class="title">SherpaCIS9660</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your hub for interactive AI-powered tools</div>', unsafe_allow_html=True)

# ---- Buttons ----
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🍽️ MunchMap Agent", key="munchmap", help="Restaurant Recommender Tool"):
        try:
            st.switch_page("pages/MunchMap.py")  # Streamlit ≥ 1.26
        except AttributeError:
            st.session_state["_nav"] = "munchmap"
            st.experimental_rerun()

with col2:
    if st.button("📊 Project 1 - Data Explorer", key="proj1", help="Interactive data analysis"):
        try:
            st.switch_page("pages/Proj1.py")
        except AttributeError:
            st.session_state["_nav"] = "proj1"
            st.experimental_rerun()

with col3:
    if st.button("🌰Tree Nut Image CLassifier", key="coming", help="Interactive Image Classifier"):
        try:
            st.switch_page("pages/TreeNutsImageClassifier.py")
        except AttributeError:
            st.session_state["_nav"] = "treenutsimageclassifier"
            st.experimental_rerun()



# ---- Fallback router for older Streamlit ----
_nav = st.session_state.get("_nav")
if _nav:
    st.session_state["_nav"] = None
    if _nav == "munchmap":
        st.write("Opening MunchMap… use the sidebar to select the page if it doesn’t switch automatically.")
    elif _nav == "proj1":
        st.write("Opening Project 1… use the sidebar to select the page if it doesn’t switch automatically.")
    elif _nav == "about":
        st.write("Opening About page… use the sidebar to select the page if it doesn’t switch automatically.")
