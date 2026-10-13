import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import joblib
import pickle
import streamlit as st
import torch

# Suppress scikit-learn unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path to import model_architecture
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from model_architecture import MiniXception
except ImportError:
    # Fallback if imported from another location
    import model_architecture
    MiniXception = model_architecture.MiniXception


# -------------------------------------------------------------
# Emotion Classes & Label Mappings
# -------------------------------------------------------------
EMOTIONS_ALPHA = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTIONS_SVM = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

CANONICAL_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_EMOJIS = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}


def find_file(filename: str) -> Path:
    """
    Search for a file in root directory, current working directory,
    or script directory.
    """
    candidates = [
        ROOT_DIR / filename,
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate '{filename}'. Searched: {[str(p) for p in candidates]}")


@st.cache_resource(show_spinner="Loading models into memory...")
def load_all_models() -> Dict[str, Any]:
    """
    Loads and caches all three models and the Haar cascade detector:
    1. MiniXception (CNN)
    2. HOG + Linear SVM
    3. LBP + KNN
    4. Haar Cascade Frontal Face Detector
    """
    device = torch.device("cpu")

    # 1. Mini-Xception CNN
    cnn_path = find_file("best_mini_xception.pth")
    model_xc = MiniXception(in_channels=1, num_classes=7)
    model_xc.load_state_dict(torch.load(str(cnn_path), map_location=device))
    model_xc.to(device)
    model_xc.eval()

    # 2. HOG + Linear SVM (explicitly linear, not RBF)
    svm_path = find_file("hog_svm_fer2013_linear.pkl")
    bundle_svm = joblib.load(str(svm_path))
    svm_classifiers = bundle_svm["classifiers"]
    svm_scaler = bundle_svm["scaler"]

    # 3. LBP + KNN (trained model saved as knn_trained from lbp_knn.ipynb)
    try:
        knn_path = find_file("knn_trained")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)
        knn_source_name = "knn_trained"
        print(f"[Model Loader] KNN model loaded: knn_trained ({knn_path})", flush=True)
    except FileNotFoundError:
        knn_path = find_file("knn_lbp_fer2013.pkl")
        bundle_knn = joblib.load(str(knn_path))
        knn_model = bundle_knn["model"] if isinstance(bundle_knn, dict) else bundle_knn
        knn_source_name = "knn_lbp_fer2013.pkl"
        print(f"[Model Loader] KNN model loaded: knn_lbp_fer2013.pkl ({knn_path})", flush=True)

    knn_classes = EMOTIONS_ALPHA

    # 4. Haar Cascade Face Detector
    try:
        xml_path = find_file("haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(str(xml_path))
    except FileNotFoundError:
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            face_cascade = cv2.CascadeClassifier()

    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade XML detector.")

    return {
        "model_xc": model_xc,
        "svm_classifiers": svm_classifiers,
        "svm_scaler": svm_scaler,
        "svm_rbf_model": None,
        "svm_rbf_scaler": None,
        "knn_model": knn_model,
        "knn_classes": knn_classes,
        "knn_source": knn_source_name,
        "face_cascade": face_cascade,
        "device": device
    }


@st.cache_resource(show_spinner="Loading HOG + RBF SVM model (~817 MB)...")
def load_rbf_svm() -> Dict[str, Any]:
    """
    Loads and caches the HOG + RBF SVM model bundle from hog_svm_fer2013_rbf.pkl.
    Loaded lazily on demand when enabled by the user to avoid overhead at startup.
    """
    rbf_path = find_file("hog_svm_fer2013_rbf.pkl")
    print(f"[Model Loader] Loading RBF SVM model from: {rbf_path}...", flush=True)
    bundle_rbf = joblib.load(str(rbf_path))
    model_rbf = bundle_rbf["model"]
    scaler_rbf = bundle_rbf["scaler"]
    print(f"[Model Loader] RBF SVM model loaded: hog_svm_fer2013_rbf.pkl ({rbf_path})", flush=True)
    return {
        "model": model_rbf,
        "scaler": scaler_rbf,
    }
