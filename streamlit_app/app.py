import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Setup sys.path to find local modules and project root
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for p in [str(CURRENT_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import load_all_models
from pipeline import process_frame_benchmark, FrameAnalysisResult
from ui import render_active_models

# Optional WebRTC support for live browser webcam streaming in cloud deployments
try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase
    )
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


if HAS_WEBRTC:
    class EmotionVideoProcessor(VideoProcessorBase):
        """
        Processes live video frames from the browser via WebRTC in real time.
        """
        def __init__(self):
            self.models = None
            self.enabled_models = ["cnn", "svm", "knn"]
            self.scale_factor = 1.10
            self.min_neighbors = 3
            self.min_size = 40
            self.max_frame_width = 640
            self.latest_analysis = None
            self.lock = threading.Lock()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            if self.models is None:
                return frame

            analysis: FrameAnalysisResult = process_frame_benchmark(
                frame=img_bgr,
                models=self.models,
                enable_cnn="cnn" in self.enabled_models,
                enable_svm="svm" in self.enabled_models,
                enable_knn="knn" in self.enabled_models,
                scale_factor=self.scale_factor,
                min_neighbors=self.min_neighbors,
                min_size=self.min_size,
                max_frame_width=self.max_frame_width
            )

            with self.lock:
                self.latest_analysis = analysis

            # Return the annotated RGB frame to the WebRTC video stream
            return av.VideoFrame.from_ndarray(analysis.annotated_frame, format="rgb24")


def main():
    st.set_page_config(
        page_title="Facial Emotion Benchmark",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -------------------------------------------------------------
    # Sidebar: Navigation (Tabs on the Left)
    # -------------------------------------------------------------
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Mode Selection",
        [
            "Live Webcam",
            "Webcam Snapshot",
            "Sample / Upload Image"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Selection")
    enable_cnn = st.sidebar.checkbox("Mini-Xception (CNN)", value=True, key="enable_cnn")
    enable_svm = st.sidebar.checkbox("HOG + Linear SVM", value=True, key="enable_svm")
    enable_knn = st.sidebar.checkbox("LBP + KNN", value=True, key="enable_knn")

    enabled_models = [
        k for k, v in [("cnn", enable_cnn), ("svm", enable_svm), ("knn", enable_knn)] if v
    ]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Face Detection Tuning")
    scale_factor = st.sidebar.slider(
        "Haar Scale Factor",
        min_value=1.05,
        max_value=1.40,
        value=1.10,
        step=0.01,
        help="Factor by which image size is reduced at each scale."
    )
    min_neighbors = st.sidebar.slider(
        "Haar Min Neighbors",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of neighbor rectangles required to retain a face candidate."
    )
    min_size = st.sidebar.slider(
        "Haar Min Face Size (px)",
        min_value=30,
        max_value=150,
        value=40,
        step=5,
        help="Minimum size in pixels of detected face."
    )
    max_frame_width = st.sidebar.slider(
        "Max Frame Width (px)",
        min_value=320,
        max_value=1280,
        value=640,
        step=80,
        help="Width to which input frames are scaled for fast processing."
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("About the Models"):
        st.markdown(
            """
            - **Mini-Xception (CNN)**:
              Deep convolutional neural network with Depthwise Separable Convolutions and Residual Connections, trained directly on 48x48 FER-2013 images.
            - **HOG + Linear SVM**:
              Histogram of Oriented Gradients (9 orientations, 8x8 cells, 2x2 blocks) scaled with StandardScaler and classified by 7 binary Linear SVMs (One-vs-Rest).
            - **LBP + KNN**:
              Local Binary Patterns texture features (P=8, R=1, 256-bin normalized histogram) classified by K-Nearest Neighbors.
            """
        )

    # -------------------------------------------------------------
    # Main Header
    # -------------------------------------------------------------
    st.title("Facial Emotion Recognition: Benchmark")
    st.markdown(
        "Comparative benchmark between **Mini-Xception (CNN)**, "
        "**HOG + Linear SVM**, and **LBP + KNN** on FER-2013."
    )

    # Load models
    try:
        models = load_all_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # -------------------------------------------------------------
    # Mode 1: Live Webcam
    # -------------------------------------------------------------
    if mode == "Live Webcam":
        st.subheader("Live Webcam")

        if not enabled_models:
            st.warning("Please enable at least one model in Model Selection on the left.")
            return

        if HAS_WEBRTC:
            st.caption(
                "Streaming from your browser webcam via WebRTC. "
                "Click **START** below to begin real-time detection and latency benchmarking."
            )

            rtc_config = RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                    ]
                }
            )

            ctx = webrtc_streamer(
                key="live-webcam-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=EmotionVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.models = models
                ctx.video_processor.enabled_models = enabled_models
                ctx.video_processor.scale_factor = scale_factor
                ctx.video_processor.min_neighbors = min_neighbors
                ctx.video_processor.min_size = min_size
                ctx.video_processor.max_frame_width = max_frame_width

                if ctx.state.playing and ctx.video_processor.latest_analysis is not None:
                    st.markdown("---")
                    with ctx.video_processor.lock:
                        analysis_snapshot = ctx.video_processor.latest_analysis
                    render_active_models(analysis_snapshot, enabled_models)

        else:
            st.warning(
                "When deployed on the internet, your web browser cannot access a server's local camera. "
                "Browser streaming requires `streamlit-webrtc` (added to `requirements.txt`). "
                "You can also use **Webcam Snapshot** on the left to capture frames directly."
            )

            st.markdown("#### Local OpenCV Camera (Localhost Only)")
            c1, c2 = st.columns([1, 1])
            cam_idx = c1.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1, key="live_idx")
            start_stream = c2.toggle("Start Live Stream", value=False, key="live_toggle")

            v_col, roi_col = st.columns([3, 1])
            video_ph = v_col.empty()
            roi_ph = roi_col.empty()
            results_ph = st.empty()

            if not start_stream:
                video_ph.info("Toggle 'Start Live Stream' above to launch local camera device.")
            else:
                cap = cv2.VideoCapture(int(cam_idx))
                if not cap.isOpened():
                    st.error(
                        f"Could not open camera device at index {cam_idx}. "
                        "On cloud servers, no physical camera is attached. "
                        "Please use 'Webcam Snapshot' on the left."
                    )
                else:
                    try:
                        while start_stream:
                            ret, frame = cap.read()
                            if not ret:
                                video_ph.warning("Unable to read frame from webcam.")
                                break

                            analysis: FrameAnalysisResult = process_frame_benchmark(
                                frame=frame,
                                models=models,
                                enable_cnn=enable_cnn,
                                enable_svm=enable_svm,
                                enable_knn=enable_knn,
                                scale_factor=scale_factor,
                                min_neighbors=min_neighbors,
                                min_size=min_size,
                                max_frame_width=max_frame_width
                            )

                            video_ph.image(
                                analysis.annotated_frame,
                                caption="Live Camera Feed",
                                use_container_width=True
                            )

                            if analysis.has_face and analysis.face_48 is not None:
                                roi_ph.image(analysis.face_48, caption="48x48 Face ROI", width=120)
                            else:
                                roi_ph.info("No face detected")

                            with results_ph.container():
                                render_active_models(analysis, enabled_models)

                            time.sleep(0.01)
                    finally:
                        cap.release()

    # -------------------------------------------------------------
    # Mode 2: Webcam Snapshot (Browser)
    # -------------------------------------------------------------
    elif mode == "Webcam Snapshot":
        st.subheader("Webcam Snapshot")
        st.caption("Capture a photo directly from your browser webcam.")

        camera_image = st.camera_input("Take a photo")

        if camera_image is not None:
            if not enabled_models:
                st.warning("Please enable at least one model in Model Selection on the left.")
            else:
                file_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    analysis = process_frame_benchmark(
                        frame=frame,
                        models=models,
                        enable_cnn=enable_cnn,
                        enable_svm=enable_svm,
                        enable_knn=enable_knn,
                        scale_factor=scale_factor,
                        min_neighbors=min_neighbors,
                        min_size=min_size,
                        max_frame_width=max_frame_width
                    )

                    v_col, roi_col = st.columns([3, 1])
                    with v_col:
                        st.image(
                            analysis.annotated_frame,
                            caption="Captured Frame with Detection",
                            use_container_width=True
                        )
                    with roi_col:
                        if analysis.has_face and analysis.face_48 is not None:
                            st.image(analysis.face_48, caption="48x48 Face ROI", width=120)
                        else:
                            st.warning("No face detected.")

                    render_active_models(analysis, enabled_models)

    # -------------------------------------------------------------
    # Mode 3: Sample / Upload Image
    # -------------------------------------------------------------
    elif mode == "Sample / Upload Image":
        st.subheader("Sample / Upload Image")

        source_type = st.radio(
            "Choose image source:",
            ["Bundled Sample Images", "Upload Custom Image"],
            horizontal=True
        )

        img_to_process = None

        if source_type == "Bundled Sample Images":
            test_dir = ROOT_DIR / "test_images"
            if test_dir.exists():
                sample_files = [f.name for f in sorted(test_dir.iterdir()) if f.is_file()]
                selected_sample = st.selectbox("Select a sample image:", sample_files)
                if selected_sample:
                    sample_path = test_dir / selected_sample
                    try:
                        pil_img = Image.open(sample_path)
                        img_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        st.error(f"Error reading image: {e}")
            else:
                st.warning("test_images/ directory not found.")
        else:
            uploaded_file = st.file_uploader(
                "Upload an image (JPG, PNG, JPEG, AVIF)",
                type=["jpg", "jpeg", "png", "webp", "avif"]
            )
            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file)
                    img_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")

        if img_to_process is not None:
            if not enabled_models:
                st.warning("Please enable at least one model in Model Selection on the left.")
            else:
                analysis = process_frame_benchmark(
                    frame=img_to_process,
                    models=models,
                    enable_cnn=enable_cnn,
                    enable_svm=enable_svm,
                    enable_knn=enable_knn,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=min_size,
                    max_frame_width=max_frame_width
                )

                v_col, roi_col = st.columns([3, 1])
                with v_col:
                    st.image(
                        analysis.annotated_frame,
                        caption="Detection Result",
                        use_container_width=True
                    )
                with roi_col:
                    if analysis.has_face and analysis.face_48 is not None:
                        st.image(analysis.face_48, caption="48x48 Face ROI", width=120)
                    else:
                        st.warning("No face detected.")

                render_active_models(analysis, enabled_models)


if __name__ == "__main__":
    main()
