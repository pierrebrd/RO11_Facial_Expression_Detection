# Facial Emotion Recognition: Streamlit Benchmark


## Cloud Deployment & Camera Architecture

When running on the internet (e.g. Streamlit Cloud, Render, etc.), the server has no local physical webcam attached (`/dev/video0`). Instead, the app accesses the client's webcam through the browser:

- **Live Webcam (WebRTC)**: When deployed with `streamlit-webrtc` and `av` (in `requirements.txt`), live video frames stream from the client's browser directly to the Python backend over WebRTC using public STUN servers. Real-time bounding boxes, emotions, and latencies are rendered directly on the stream.
- **Webcam Snapshot**: Uses `st.camera_input` to capture and benchmark individual photos directly from the client's browser without requiring WebRTC.
- **Sample / Upload Image**: Test on bundled images in `test_images/` or upload custom files.
- **Localhost Device**: When running locally on your own desktop with a physical camera, local OpenCV device access is available as a fallback.

---

## Features & Controls

- **Navigation (Left Sidebar)**: Select between Live Webcam, Webcam Snapshot, and Sample / Upload Image.
- **Model Toggles (Left Sidebar)**: Enable or disable any of the three models independently (`Mini-Xception (CNN)`, `HOG + Linear SVM`, `LBP + KNN`).
- **Detection Tuning**: Adjust Haar Cascade scale factor, min neighbors, min size, and frame width.
