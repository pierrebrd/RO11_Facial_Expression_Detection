import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import hog, local_binary_pattern

from models import EMOTIONS_ALPHA, EMOTIONS_SVM, CANONICAL_EMOTIONS, EMOTION_EMOJIS


@dataclass
class ModelResult:
    name: str
    top_emotion: str
    top_prob: float
    emoji: str
    probabilities: Dict[str, float]
    # Latencies in milliseconds (ms)
    haar_latency_ms: float
    feature_prep_latency_ms: float
    image_proc_latency_ms: float   # haar + feature_prep
    network_latency_ms: float      # neural net / classifier inference
    total_latency_ms: float        # image_proc + network


@dataclass
class FrameAnalysisResult:
    has_face: bool
    num_faces: int
    annotated_frame: np.ndarray    # RGB image
    face_48: Optional[np.ndarray]   # 48x48 grayscale face ROI
    face_box: Optional[Tuple[int, int, int, int]]
    haar_latency_ms: float
    cnn: Optional[ModelResult] = None
    svm: Optional[ModelResult] = None
    knn: Optional[ModelResult] = None


def run_cnn_inference(
    face_48: np.ndarray,
    model_xc: torch.nn.Module,
    device: torch.device,
    haar_latency_ms: float
) -> ModelResult:
    """Runs MiniXception inference and records prep and network latency."""
    t0_prep = time.perf_counter()
    face_norm = (face_48.astype(np.float32) / 127.5) - 1.0
    tensor = torch.tensor(face_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    prep_ms = (time.perf_counter() - t0_prep) * 1000.0

    t0_net = time.perf_counter()
    with torch.no_grad():
        logits = model_xc(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    net_ms = (time.perf_counter() - t0_net) * 1000.0

    scores = {EMOTIONS_ALPHA[i]: float(probs[i]) for i in range(len(EMOTIONS_ALPHA))}
    top_idx = int(np.argmax(probs))
    top_emotion = EMOTIONS_ALPHA[top_idx]
    top_prob = float(probs[top_idx])

    return ModelResult(
        name="Mini-Xception (CNN)",
        top_emotion=top_emotion,
        top_prob=top_prob,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        probabilities=scores,
        haar_latency_ms=haar_latency_ms,
        feature_prep_latency_ms=prep_ms,
        image_proc_latency_ms=haar_latency_ms + prep_ms,
        network_latency_ms=net_ms,
        total_latency_ms=haar_latency_ms + prep_ms + net_ms
    )


def run_svm_inference(
    face_48: np.ndarray,
    svm_classifiers: Any,
    svm_scaler: Any,
    haar_latency_ms: float
) -> ModelResult:
    """Runs HOG + Linear SVM inference and records feature extraction and classifier latency."""
    t0_prep = time.perf_counter()
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
    prep_ms = (time.perf_counter() - t0_prep) * 1000.0

    t0_net = time.perf_counter()
    raw_scores = [float(svm_classifiers[i].decision_function(hog_scaled)[0]) for i in range(7)]
    exp_scores = np.exp(np.array(raw_scores) - np.max(raw_scores))
    svm_probs = exp_scores / np.sum(exp_scores)
    net_ms = (time.perf_counter() - t0_net) * 1000.0

    scores = {EMOTIONS_SVM[i]: float(svm_probs[i]) for i in range(7)}
    top_idx = int(np.argmax(svm_probs))
    top_emotion = EMOTIONS_SVM[top_idx]
    top_prob = float(svm_probs[top_idx])

    return ModelResult(
        name="HOG + Linear SVM",
        top_emotion=top_emotion,
        top_prob=top_prob,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        probabilities=scores,
        haar_latency_ms=haar_latency_ms,
        feature_prep_latency_ms=prep_ms,
        image_proc_latency_ms=haar_latency_ms + prep_ms,
        network_latency_ms=net_ms,
        total_latency_ms=haar_latency_ms + prep_ms + net_ms
    )


def run_knn_inference(
    face_48: np.ndarray,
    knn_model: Any,
    knn_classes: List[str],
    haar_latency_ms: float
) -> ModelResult:
    """Runs LBP + KNN inference and records texture extraction and neighbor search latency."""
    t0_prep = time.perf_counter()
    lbp = local_binary_pattern(face_48, P=8, R=1, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    prep_ms = (time.perf_counter() - t0_prep) * 1000.0

    t0_net = time.perf_counter()
    knn_probs = knn_model.predict_proba([hist])[0]
    net_ms = (time.perf_counter() - t0_net) * 1000.0

    scores = {knn_classes[i]: float(knn_probs[i]) for i in range(len(knn_classes))}
    top_idx = int(np.argmax(knn_probs))
    top_emotion = knn_classes[top_idx]
    top_prob = float(knn_probs[top_idx])

    return ModelResult(
        name="LBP + KNN",
        top_emotion=top_emotion,
        top_prob=top_prob,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        probabilities=scores,
        haar_latency_ms=haar_latency_ms,
        feature_prep_latency_ms=prep_ms,
        image_proc_latency_ms=haar_latency_ms + prep_ms,
        network_latency_ms=net_ms,
        total_latency_ms=haar_latency_ms + prep_ms + net_ms
    )


def process_frame_benchmark(
    frame: np.ndarray,
    models: Dict[str, Any],
    enable_cnn: bool = True,
    enable_svm: bool = True,
    enable_knn: bool = True,
    scale_factor: float = 1.1,
    min_neighbors: int = 3,
    min_size: int = 40,
    max_frame_width: int = 640
) -> FrameAnalysisResult:
    """
    Complete analysis pipeline:
    1. Downscales frame if larger than max_frame_width
    2. Runs Haar Cascade face detection (timing measured)
    3. Crops and resizes primary face ROI to 48x48
    4. Runs inference for enabled models (CNN, SVM, KNN)
    5. Returns annotated RGB frame and timing breakdowns
    """
    if frame is None:
        raise ValueError("Received empty or None frame.")

    # Downscale frame for real-time responsiveness if needed
    h, w = frame.shape[:2]
    if w > max_frame_width:
        scale = float(max_frame_width) / float(w)
        proc_frame = cv2.resize(frame, (max_frame_width, int(h * scale)))
    else:
        proc_frame = frame.copy()

    # Convert to grayscale for Haar detection
    if len(proc_frame.shape) == 3:
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = proc_frame.copy()
        proc_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    face_cascade: cv2.CascadeClassifier = models["face_cascade"]

    # 1. Haar Cascade Face Detection timing
    t0_haar = time.perf_counter()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )
    t_haar_done = time.perf_counter()
    haar_only_ms = (t_haar_done - t0_haar) * 1000.0

    # Prepare RGB frame for Streamlit display
    annotated_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

    if len(faces) == 0:
        return FrameAnalysisResult(
            has_face=False,
            num_faces=0,
            annotated_frame=annotated_rgb,
            face_48=None,
            face_box=None,
            haar_latency_ms=haar_only_ms
        )

    # 2. Select the largest detected face as primary
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    primary_box = faces_sorted[0]
    x, y, w_box, h_box = primary_box

    # Crop and resize to 48x48
    t0_crop = time.perf_counter()
    face_roi = gray[y : y + h_box, x : x + w_box]
    face_48 = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    t_crop_done = time.perf_counter()

    haar_total_prep_ms = haar_only_ms + (t_crop_done - t0_crop) * 1000.0

    # 3. Model Inferences (only for enabled models)
    cnn_res = None
    if enable_cnn:
        cnn_res = run_cnn_inference(
            face_48=face_48,
            model_xc=models["model_xc"],
            device=models["device"],
            haar_latency_ms=haar_total_prep_ms
        )

    svm_res = None
    if enable_svm:
        svm_res = run_svm_inference(
            face_48=face_48,
            svm_classifiers=models["svm_classifiers"],
            svm_scaler=models["svm_scaler"],
            haar_latency_ms=haar_total_prep_ms
        )

    knn_res = None
    if enable_knn:
        knn_res = run_knn_inference(
            face_48=face_48,
            knn_model=models["knn_model"],
            knn_classes=models["knn_classes"],
            haar_latency_ms=haar_total_prep_ms
        )

    # 4. Draw annotations on frame
    cv2.rectangle(annotated_rgb, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    for (ox, oy, ow, oh) in faces_sorted[1:]:
        cv2.rectangle(annotated_rgb, (ox, oy), (ox + ow, oy + oh), (0, 200, 255), 1)

    labels = []
    if cnn_res is not None:
        labels.append(f"CNN: {cnn_res.top_emotion}")
    if svm_res is not None:
        labels.append(f"SVM: {svm_res.top_emotion}")
    if knn_res is not None:
        labels.append(f"KNN: {knn_res.top_emotion}")

    label_text = " | ".join(labels) if labels else "Face detected"
    text_y = max(y - 10, 20)
    cv2.putText(
        annotated_rgb,
        label_text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return FrameAnalysisResult(
        has_face=True,
        num_faces=len(faces),
        annotated_frame=annotated_rgb,
        face_48=face_48,
        face_box=(x, y, w_box, h_box),
        haar_latency_ms=haar_total_prep_ms,
        cnn=cnn_res,
        svm=svm_res,
        knn=knn_res
    )
