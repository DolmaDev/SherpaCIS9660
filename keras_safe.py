# keras_safe.py
from __future__ import annotations
import os
import traceback

# Ensure a default backend BEFORE importing keras anywhere
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

KERAS_AVAILABLE = True
KERAS_IMPORT_ERROR = None
TF_AVAILABLE = True
TF_IMPORT_ERROR = None

try:
    import keras  # Keras 3
except Exception as e:
    KERAS_AVAILABLE = False
    KERAS_IMPORT_ERROR = e

# TensorFlow is optional for Keras 3 (other backends exist), but we default to TF here
try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e

def backend_ok() -> bool:
    """Return True if Keras + TF are importable."""
    return KERAS_AVAILABLE and TF_AVAILABLE

def backend_status() -> dict:
    """Diagnostics you can show in the UI."""
    return {
        "keras_available": KERAS_AVAILABLE,
        "tf_available": TF_AVAILABLE,
        "keras_backend_env": os.environ.get("KERAS_BACKEND", ""),
        "keras_error": str(KERAS_IMPORT_ERROR) if KERAS_IMPORT_ERROR else "",
        "tf_error": str(TF_IMPORT_ERROR) if TF_IMPORT_ERROR else "",
    }

def format_exception(e: BaseException) -> str:
    """Compact traceback for user-visible error boxes."""
    return "".join(traceback.format_exception_only(type(e), e)).strip()

def load_model_safe(path: str):
    """Load a Keras model or raise a friendly RuntimeError with diagnostics."""
    if not backend_ok():
        raise RuntimeError(f"Keras backend not ready: {backend_status()}")
    try:
        from keras import models
        return models.load_model(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{path}': {format_exception(e)}")

def predict_safe(model, X):
    """Run model.predict with friendly error messages."""
    if not backend_ok():
        raise RuntimeError(f"Keras backend not ready: {backend_status()}")
    try:
        return model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {format_exception(e)}")

# ---- Optional: decorator to wrap any function with a clean error ----
def safe_call(fn):
    """Decorator that turns exceptions into RuntimeError with short messages."""
    def _wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"{fn.__name__} failed: {format_exception(e)}")
    return _wrap

# ---- Optional: Streamlit helper (call from your page to show status) ----
def show_backend_status(st):
    """Render backend diagnostics in the UI (pass the Streamlit module as 'st')."""
    st.caption("Keras/TensorFlow backend status")
    st.json(backend_status())
