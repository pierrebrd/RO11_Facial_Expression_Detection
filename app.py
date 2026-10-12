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

# Détecteur Haar Cascade (local en priorité)
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
# 3. Pipeline d'inférence temps réel optimisé
# -------------------------------------------------------------
def process_frame(frame):
    """
    Traite une frame issue de la webcam :
    - Détection tolérante et rapide
    - Envoi aux 3 modèles
    - Renvoi de l'image avec rectangle et des 3 classements
    """
    if frame is None or face_cascade.empty():
        msg = {"Aucun signal caméra": 1.0}
        return frame, msg, msg, msg

    # Réduction de la résolution pour fluidifier le traitement
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        proc_frame = cv2.resize(frame, (640, int(h * scale)))
    else:
        proc_frame = frame.copy()

    gray = cv2.cvtColor(proc_frame, cv2.COLOR_RGB2GRAY)
    # Détection plus sensible et tolérante (scaleFactor=1.1, minNeighbors=3)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        waiting_msg = {"En attente de visage...": 1.0}
        return proc_frame, waiting_msg, waiting_msg, waiting_msg

    # On sélectionne le plus grand visage
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w_box, h_box = faces[0]
    face_roi = gray[y:y+h_box, x:x+w_box]
    face_48 = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    # --- A. Mini-Xception (CNN) ---
    face_norm = (face_48.astype(np.float32) / 127.5) - 1.0
    tensor = torch.tensor(face_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model_xc(tensor)
        probs_xc = F.softmax(logits, dim=1).squeeze().numpy()
    xc_scores = {EMOTIONS_ALPHA[i]: float(probs_xc[i]) for i in range(len(EMOTIONS_ALPHA))}

    # --- B. HOG + SVM ---
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

    # --- C. LBP + KNN ---
    lbp = local_binary_pattern(face_48, P=8, R=1, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    knn_probs = knn_model.predict_proba([hist])[0]
    knn_scores = {EMOTIONS_ALPHA[i]: float(knn_probs[i]) for i in range(len(EMOTIONS_ALPHA))}

    # Dessin du rectangle vert sur le visage détecté
    annotated = proc_frame.copy()
    cv2.rectangle(annotated, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)

    return annotated, xc_scores, svm_scores, knn_scores


# -------------------------------------------------------------
# 4. Interface Gradio fluide
# -------------------------------------------------------------
with gr.Blocks(title="Benchmark Émotions en Direct") as demo:
    gr.Markdown("# Détection d'émotions en direct : Comparatif Side-by-Side")
    gr.Markdown("Comparaison en temps réel entre **Mini-Xception (CNN)**, **HOG + SVM** et **LBP + KNN** sur FER-2013.")

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(sources=["webcam"], streaming=True, label="1. Votre Webcam (Flux direct)")
            annotated_output = gr.Image(label="2. Détection du visage en direct")
        with gr.Column(scale=1):
            out_xc = gr.Label(num_top_classes=3, label="Modèle 1 : Mini-Xception (CNN)")
            out_svm = gr.Label(num_top_classes=3, label="Modèle 2 : HOG + SVM Linéaire")
            out_knn = gr.Label(num_top_classes=3, label="Modèle 3 : LBP + KNN")

    # stream_every=0.3 évite la saturation réseau et du CPU
    # Séparation stricte : la webcam reste fluide à 30 FPS en local, annotated_output reçoit le retour serveur
    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[annotated_output, out_xc, out_svm, out_knn],
        stream_every=0.3
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
