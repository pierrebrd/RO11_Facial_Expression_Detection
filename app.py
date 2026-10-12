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
# 1. Emotion Classes & Label Mappings
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
# 2. Loading Trained Models
# -------------------------------------------------------------
print("Loading MiniXception...")
model_xc = MiniXception(in_channels=1, num_classes=7)
model_xc.load_state_dict(torch.load("best_mini_xception.pth", map_location=device))
model_xc.eval()

print("Loading HOG + Linear SVM...")
bundle_svm = joblib.load("hog_svm_fer2013_linear.pkl")
svm_classifiers = bundle_svm["classifiers"]
svm_scaler = bundle_svm["scaler"]

print("Loading LBP + KNN...")
bundle_knn = joblib.load("knn_lbp_fer2013.pkl")
knn_model = bundle_knn["model"]

# Haar Cascade Face Detector (local XML prioritized)
xml_name = "haarcascade_frontalface_default.xml"
if os.path.exists(xml_name):
    face_cascade = cv2.CascadeClassifier(xml_name)
elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + xml_name)
else:
    face_cascade = cv2.CascadeClassifier()

if face_cascade.empty():
    print("Warning: Failed to load frontal Haar Cascade XML file.")


# -------------------------------------------------------------
# 3. Real-Time Inference Pipeline
# -------------------------------------------------------------
def process_frame(frame):
    """
    Processes a webcam frame:
    - Detects face with Haar Cascade
    - Preprocesses ROI to 48x48 grayscale
    - Runs Mini-Xception, HOG+SVM, and LBP+KNN in parallel
    - Returns annotated frame and predictions for each model
    """
    if frame is None or face_cascade.empty():
        msg = {"No camera feed": 1.0}
        return frame, msg, msg, msg

    # Resize input frame to ensure fluid processing
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        proc_frame = cv2.resize(frame, (640, int(h * scale)))
    else:
        proc_frame = frame.copy()

    gray = cv2.cvtColor(proc_frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        waiting_msg = {"Waiting for face...": 1.0}
        return proc_frame, waiting_msg, waiting_msg, waiting_msg

    # Select the largest face detected
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
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2",
        transform_sqrt=True,
        feature_vector=True
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

    # Draw bounding box around detected face
    annotated = proc_frame.copy()
    cv2.rectangle(annotated, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)

    return annotated, xc_scores, svm_scores, knn_scores


# -------------------------------------------------------------
# 4. Gradio User Interface
# -------------------------------------------------------------
with gr.Blocks(title="Live Facial Emotion Benchmark") as demo:
    gr.Markdown("# Live Facial Emotion Recognition: Side-by-Side Benchmark")
    gr.Markdown("Real-time comparative benchmark between **Mini-Xception (CNN)**, **HOG + SVM**, and **LBP + KNN** trained on FER-2013.")

    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(sources=["webcam"], streaming=True, label="1. Live Webcam Feed")
            annotated_output = gr.Image(label="2. Real-Time Face Detection")
        with gr.Column(scale=1):
            out_xc = gr.Label(num_top_classes=3, label="Model 1: Mini-Xception (CNN)")
            out_svm = gr.Label(num_top_classes=3, label="Model 2: HOG + Linear SVM")
            out_knn = gr.Label(num_top_classes=3, label="Model 3: LBP + KNN")

    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[annotated_output, out_xc, out_svm, out_knn]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
