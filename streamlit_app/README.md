# Facial Emotion Recognition: Streamlit Benchmark

This Streamlit application provides a comparative benchmark between machine learning and deep learning approaches trained on FER-2013:

1. **Mini-Xception (CNN)**: Deep convolutional neural network with Depthwise Separable Convolutions and Residual Connections.
2. **HOG + Linear SVM**: Histogram of Oriented Gradients (9 orientations, 8x8 cells, 2x2 blocks) feature extraction classified by 7 One-vs-Rest Linear SVMs (~1-2 ms inference).
3. **HOG + RBF SVM (Optional)**: One-vs-Rest Support Vector Machine with Radial Basis Function kernel from `hog_svm_fer2013_rbf.pkl` (~817 MB, optional and unchecked by default).
4. **LBP + KNN**: Local Binary Patterns texture extraction (P=8, R=1, 256-bin histogram) using `knn_trained` from `lbp_knn.ipynb`.

---

## Architecture

- **`app.py`**: Streamlit application UI, navigation, and live loop.
- **`model_loader.py`**: Caches and loads Mini-Xception, Linear SVM, `knn_trained`, Haar Cascade, and lazily loads RBF SVM on demand.
- **`emotion_pipeline.py`**: Face detection, ROI 48px crop, feature extraction, model inference, multi-line video overlays, and `LatencyTracker`.
- **`ui_components.py`**: Model prediction cards, spaced emotion confidence bars, and running mean latency summary table.

---

## Cloud Deployment & Camera Architecture

When running on the internet (Streamlit Cloud, Render, etc.), the server has no local physical webcam attached (`/dev/video0`). Instead, the app accesses the client's webcam through the browser:

- **Live Webcam (WebRTC)**: Real-time browser webcam streaming via `streamlit-webrtc` using Google public STUN servers. Multi-line face bounding box overlays and latencies are rendered directly on the stream.
- **Webcam Snapshot**: Uses `st.camera_input` to capture and benchmark individual photos directly from the client's browser.
- **Sample / Upload Image**: Test on bundled images in `test_images/` or upload custom files.
- **Localhost Device**: Fallback to local OpenCV camera when running locally on desktop.

---

## Features & Controls

- **Navigation (Left Sidebar)**: Select between Live Webcam, Webcam Snapshot, and Sample / Upload Image.
- **Model Toggles (Left Sidebar)**: Enable or disable any model independently (`Mini-Xception (CNN)`, `HOG + Linear SVM`, `HOG + RBF SVM`, `LBP + KNN`). RBF SVM is unchecked by default to save memory and CPU cycles.
- **Mean Latency Benchmark**: Running average table tracking mean image processing time, prediction latency, total latency, and FPS.
- **Detection Tuning**: Adjust Haar Cascade scale factor, min neighbors, min size, and frame width.
