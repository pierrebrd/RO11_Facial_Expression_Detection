import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import hog, local_binary_pattern
import gradio as gr
from model_architecture import MiniXception

# -------------------------------------------------------------
# 1. Définition des classes et correspondances d'émotions
# -------------------------------------------------------------
# Ordre alphabétique standard FER-2013 (utilisé par ImageFolder / Xception et LBP/KNN)
EMOTIONS_ALPHA = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Mapping du modèle HOG+SVM (issu de hog_svm.ipynb)
EMOTIONS_SVM = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

device = torch.device("cpu")

# -------------------------------------------------------------
# 2. Chargement des modèles entraînés
# -------------------------------------------------------------
print("Chargement de MiniXception...")
model_xc = MiniXception(in_channels=1, num_classes=7)
model_xc.load_state_dict(torch.load("best_mini_xception.pth", map_location=device))
model_xc.eval()

print("Chargement de HOG + SVM linéaire...")
bundle_svm = joblib.load("hog_svm_fer2013_linear.pkl")
svm_classifiers = bundle_svm["classifiers"]
svm_scaler = bundle_svm["scaler"]

print("Chargement de LBP + KNN...")
bundle_knn = joblib.load("knn_lbp_fer2013.pkl")
knn_model = bundle_knn["model"]

# Détecteur de visages OpenCV (robuste : local d'abord, puis cv2.data)
xml_name = "haarcascade_frontalface_default.xml"
if os.path.exists(xml_name):
    face_cascade = cv2.CascadeClassifier(xml_name)
elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + xml_name)
else:
    face_cascade = cv2.CascadeClassifier()

if face_cascade.empty():
    print("Avertissement: Impossible de charger le fichier XML Haar Cascade frontal.")


# -------------------------------------------------------------
# 3. Pipeline d'inférence temps réel
# -------------------------------------------------------------
def process_frame(frame):
    """
    Traite une image capturée par la webcam :
    1. Détecte le visage
    2. Redimensionne en 48x48 niveaux de gris
    3. Exécute les 3 modèles en parallèle
    4. Retourne la frame annotée et les prédictions
    """
    if frame is None or face_cascade.empty():
        return frame, {}, {}, {}

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return frame, {}, {}, {}

    # Si plusieurs visages sont détectés, on prend le plus grand
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_48 = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    # --- A. Inférence Mini-Xception (CNN) ---
    face_norm = (face_48.astype(np.float32) / 127.5) - 1.0  # Normalisation [-1, 1]
    tensor = torch.tensor(face_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model_xc(tensor)
        probs_xc = F.softmax(logits, dim=1).squeeze().numpy()
    xc_scores = {EMOTIONS_ALPHA[i]: float(probs_xc[i]) for i in range(len(EMOTIONS_ALPHA))}

    # --- B. Inférence HOG + SVM ---
    hog_vec = hog(
        face_48,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    hog_scaled = svm_scaler.transform([hog_vec])
    raw_svm_scores = [float(svm_classifiers[i].decision_function(hog_scaled)[0]) for i in range(7)]
    exp_scores = np.exp(np.array(raw_svm_scores) - np.max(raw_svm_scores))
    svm_probs = exp_scores / np.sum(exp_scores)
    svm_scores = {EMOTIONS_SVM[i]: float(svm_probs[i]) for i in range(7)}

    # --- C. Inférence LBP + KNN ---
    lbp = local_binary_pattern(face_48, P=8, R=1, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    knn_probs = knn_model.predict_proba([hist])[0]
    knn_scores = {EMOTIONS_ALPHA[i]: float(knn_probs[i]) for i in range(len(EMOTIONS_ALPHA))}

    # Dessin de la boîte englobante sur la frame
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return annotated, xc_scores, svm_scores, knn_scores


# -------------------------------------------------------------
# 4. Interface Gradio
# -------------------------------------------------------------
with gr.Blocks(title="Benchmark Émotions en Direct") as demo:
    gr.Markdown("# Détection d'émotions en direct : Comparatif Side-by-Side")
    gr.Markdown("Comparaison en temps réel entre **Mini-Xception (CNN)**, **HOG + SVM** et **LBP + KNN** sur FER-2013.")

    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True, label="Flux Caméra")
        with gr.Column():
            out_xc = gr.Label(num_top_classes=3, label="1. Mini-Xception (Deep Learning)")
            out_svm = gr.Label(num_top_classes=3, label="2. HOG + SVM Linéaire")
            out_knn = gr.Label(num_top_classes=3, label="3. LBP + KNN")

    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[webcam, out_xc, out_svm, out_knn]
    )

if __name__ == "__main__":
    # Récupération du port dynamique Render (ou 7860 par défaut en local)
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
