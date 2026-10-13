# Facial Emotion Recognition: Streamlit Benchmark

## Navigation & Architecture

Modes are selected from the **left sidebar** to guarantee exclusive camera resource access and prevent hardware lock conflicts:

- **Live Webcam**: Real-time video processing directly from your connected camera device (device index 0, 1, etc.) via OpenCV.
- **Webcam Snapshot**: Capture an instantaneous photo using your web browser's camera input.
- **Sample / Upload Image**: Test bundled samples from `test_images/` (`happy.jpg`, `sad.avif`, `suprised_boy.jpeg`) or upload custom images.

---

## Latency Benchmarking & Controls

- **Model Selection (Sidebar)**: Enable or disable any of the three models independently (`Mini-Xception (CNN)`, `HOG + Linear SVM`, `LBP + KNN`) to focus analysis and optimize frame rates.
- **Image Processing Latency**: Multi-scale Haar Cascade face detection + 48x48 ROI cropping + feature extraction (normalization, HOG descriptor, or LBP texture).
- **Network / Inference Latency**: Forward pass, SVM decision functions, or KNN neighbor search.
- **Total Latency & FPS**: End-to-end processing time and inferred throughput.
