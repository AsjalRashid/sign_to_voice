from __future__ import annotations

import io
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from signvoice.vision.capture import ThreadedCapture
from signvoice.vision.landmark import (
    FEATURE_DIM,
    LandmarkExtractor,
    draw_landmarks,
    ensure_model,
)


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "mediapipe"

st.set_page_config(
    page_title="SignVoice · Landmark Studio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


class LiveStats:
    """Small thread-safe bridge from the WebRTC worker to the Streamlit UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames = 0
        self._latest: dict | None = None

    def update(self, result) -> None:
        with self._lock:
            self._frames += 1
            self._latest = {
                "frames": self._frames,
                "left": bool(result.left_found),
                "right": bool(result.right_found),
                "pose": bool(result.pose_found),
                "inference_ms": float(result.inference_ms),
                "features": result.features.copy(),
            }

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._latest is None:
                return None
            return {**self._latest, "features": self._latest["features"].copy()}


class BrowserLandmarkProcessor(VideoProcessorBase):
    """Processes browser webcam frames with the same LandmarkExtractor API."""

    def __init__(self, stats: LiveStats, model_dir: str, pose_model: str,
                 frame_interval: int, min_confidence: float, swap_handedness: bool):
        self.stats = stats
        self.extractor = LandmarkExtractor(
            model_dir=model_dir,
            pose_model=pose_model,
            min_detect_conf=min_confidence,
            min_presence_conf=min_confidence,
            min_tracking_conf=min_confidence,
            swap_handedness=swap_handedness,
            mode="video",
            frame_interval=frame_interval,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        result = self.extractor.process(image)
        draw_landmarks(image, result)
        self.stats.update(result)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def on_ended(self) -> None:
        self.extractor.close()

    def __del__(self):  # best-effort cleanup when WebRTC tears down a processor
        extractor = getattr(self, "extractor", None)
        if extractor is not None:
            try:
                extractor.close()
            except Exception:
                pass


@st.cache_resource(show_spinner="Loading MediaPipe models…")
def get_image_extractor(model_dir: str, pose_model: str, frame_interval: int,
                        min_confidence: float, swap_handedness: bool):
    return LandmarkExtractor(
        model_dir=model_dir,
        pose_model=pose_model,
        min_detect_conf=min_confidence,
        min_presence_conf=min_confidence,
        min_tracking_conf=min_confidence,
        swap_handedness=swap_handedness,
        mode="image",
        frame_interval=frame_interval,
    )


def show_feature_details(features: np.ndarray, left: bool, right: bool, pose: bool,
                         inference_ms: float | None = None) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Feature dimensions", f"{features.size}")
    b.metric("Left hand", "Detected" if left else "Not detected")
    c.metric("Right hand", "Detected" if right else "Not detected")
    d.metric("Upper-body pose", "Detected" if pose else "Not detected")
    if inference_ms is not None:
        st.caption(f"Landmark inference: {inference_ms:.1f} ms per frame")

    with st.expander("Inspect the 147-value feature vector", expanded=False):
        chart_df = pd.DataFrame({"feature index": np.arange(features.size), "value": features})
        st.line_chart(chart_df, x="feature index", y="value")
        st.download_button(
            "Download feature vector (.npy)",
            data=numpy_bytes(features.astype(np.float32)),
            file_name="landmark_features.npy",
            mime="application/octet-stream",
        )
        st.caption("Layout: left hand 0–62 · right hand 63–125 · selected pose points 126–146.")


def numpy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


@st.fragment(run_every=1.0)
def live_metrics(stats: LiveStats) -> None:
    """Refresh metrics independently while the WebRTC video is running."""
    snapshot = stats.snapshot()
    if snapshot is None:
        st.info("Start the camera and allow browser camera access to see landmark statistics.")
        return
    st.metric("Frames processed", snapshot["frames"])
    show_feature_details(
        snapshot["features"], snapshot["left"], snapshot["right"],
        snapshot["pose"], snapshot["inference_ms"],
    )


def browser_camera(model_dir: str, pose_model: str, frame_interval: int,
                   confidence: float, swap_handedness: bool) -> None:
    st.subheader("Browser webcam")
    st.write("Uses WebRTC to send your browser camera frames to the MediaPipe pipeline. This is the recommended mode for a hosted Streamlit deployment.")
    if "browser_stats" not in st.session_state:
        st.session_state.browser_stats = LiveStats()
    stats = st.session_state.browser_stats

    ctx = webrtc_streamer(
        key="signvoice-landmark-demo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: BrowserLandmarkProcessor(
            stats, model_dir, pose_model, frame_interval, confidence, swap_handedness
        ),
        async_processing=True,
    )
    if ctx.state.playing:
        st.success("Camera stream is running. Landmark overlays are drawn on the outgoing video.")
    live_metrics(stats)


def local_camera(model_dir: str, pose_model: str, frame_interval: int,
                 confidence: float, swap_handedness: bool, width: int, height: int) -> None:
    st.subheader("Local camera · ThreadedCapture")
    st.warning("This mode opens a camera attached to the machine running Streamlit. It is intended for local/self-hosted use; a hosted cloud server usually cannot access your laptop webcam. Use Browser webcam for deployment.")

    if "local_cam" not in st.session_state:
        st.session_state.local_cam = None
    if "local_extractor" not in st.session_state:
        st.session_state.local_extractor = None
    if "local_last_id" not in st.session_state:
        st.session_state.local_last_id = 0

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start local camera", type="primary", disabled=st.session_state.local_cam is not None):
            try:
                extractor = LandmarkExtractor(
                    model_dir=model_dir, pose_model=pose_model,
                    min_detect_conf=confidence, min_presence_conf=confidence,
                    min_tracking_conf=confidence, swap_handedness=swap_handedness,
                    mode="video", frame_interval=frame_interval,
                )
                cam = ThreadedCapture(0, width=width, height=height, mirror=True).start()
                st.session_state.local_extractor = extractor
                st.session_state.local_cam = cam
                st.session_state.local_last_id = cam.frames_captured
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start local camera: {exc}")
    with c2:
        if st.button("Stop local camera", disabled=st.session_state.local_cam is None):
            stop_local_camera()
            st.rerun()

    if st.session_state.local_cam is None:
        st.info("Press **Start local camera** to run capture.py → landmarks.py on a locally attached webcam.")
        return

    local_frame_view()


@st.fragment(run_every=0.12)
def local_frame_view() -> None:
    cam = st.session_state.get("local_cam")
    extractor = st.session_state.get("local_extractor")
    if cam is None or extractor is None:
        return
    frame = cam.read(after_id=st.session_state.local_last_id, timeout=0.05)
    if frame is None:
        st.caption("Waiting for the next frame…")
        return
    st.session_state.local_last_id = frame.id
    image = frame.image.copy()
    result = extractor.process(frame.image)
    draw_landmarks(image, result)
    st.image(image, channels="BGR", use_container_width=True)
    st.caption(f"Capture frame ID: {frame.id} · Capture frame age: {frame.age_ms:.1f} ms")
    show_feature_details(
        result.features, result.left_found, result.right_found,
        result.pose_found, result.inference_ms,
    )


def stop_local_camera() -> None:
    cam = st.session_state.get("local_cam")
    extractor = st.session_state.get("local_extractor")
    if cam is not None:
        try:
            cam.stop()
        except Exception:
            pass
    if extractor is not None:
        try:
            extractor.close()
        except Exception:
            pass
    st.session_state.local_cam = None
    st.session_state.local_extractor = None
    st.session_state.local_last_id = 0


def image_demo(model_dir: str, pose_model: str, frame_interval: int,
               confidence: float, swap_handedness: bool) -> None:
    st.subheader("Upload an image")
    st.write("Run the extractor in IMAGE mode on a still image. This is useful for testing model setup without granting camera access.")
    uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The uploaded file could not be decoded as an image.")
        return

    try:
        extractor = get_image_extractor(
            model_dir, pose_model, 1, confidence, swap_handedness
        )
        result = extractor.process(image)
        annotated = image.copy()
        draw_landmarks(annotated, result)
        left, right = st.columns(2)
        with left:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with right:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Landmark overlay", use_container_width=True)
        show_feature_details(
            result.features, result.left_found, result.right_found,
            result.pose_found, result.inference_ms,
        )
    except Exception as exc:
        st.exception(exc)


def pipeline_explainer() -> None:
    st.subheader("How the two modules work together")
    st.markdown(
        """
        ```text
        Camera source
          ├─ Browser webcam: WebRTC frames (hosted deployment)
          └─ Local webcam: ThreadedCapture → latest Frame (self-hosted)
                                      │
                                      ▼
                              LandmarkExtractor
                              ├─ HandLandmarker (up to 2 hands)
                              ├─ PoseLandmarker (sampled every N frames)
                              └─ 147-value feature vector
                                      │
                                      ▼
                              draw_landmarks()
                              ├─ Annotated video/image
                              └─ Detection + inference diagnostics
        ```
        """
    )
    st.markdown("**Feature layout**")
    st.dataframe(
        pd.DataFrame([
            {"Feature range": "0–62", "Component": "Left hand", "Values": "21 landmarks × (x, y, z)"},
            {"Feature range": "63–125", "Component": "Right hand", "Values": "21 landmarks × (x, y, z)"},
            {"Feature range": "126–146", "Component": "Selected pose", "Values": "7 landmarks × (x, y, z)"},
        ]),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        "The pose model is evaluated every `frame_interval` processed frames. Between pose evaluations, the last pose coordinates are reused. Missing components remain zero-filled in the feature vector."
    )
    with st.expander("Model files and source modules"):
        st.code("signvoice/vision/capture.py\nsignvoice/vision/landmarks.py\nmodels/mediapipe/", language="text")
        st.write("The model assets are downloaded on first use by `ensure_model()` and cached under `models/mediapipe/`.")


# Sidebar configuration
st.sidebar.title("Landmark settings")
st.sidebar.caption("These settings apply when a new processor/extractor is created.")
pose_choice = st.sidebar.selectbox(
    "Pose model", ["pose_landmarker_lite.task", "pose_landmarker_full.task"], index=0
)
frame_interval = st.sidebar.slider("Run pose every N frames", 1, 10, 3)
confidence = st.sidebar.slider("Detection / presence / tracking confidence", 0.1, 0.9, 0.5, 0.05)
swap_handedness = st.sidebar.checkbox("Swap left/right labels", value=False)
width = st.sidebar.selectbox("Local camera width", [640, 960, 1280], index=0)
height = st.sidebar.selectbox("Local camera height", [480, 540, 720], index=0)

st.title("🤟 SignVoice · Landmark Studio")
st.write("A live showcase of the SignVoice camera capture and MediaPipe landmark extraction pipeline.")
st.caption(f"Feature vector size: {FEATURE_DIM} · Hands: up to 2 · Pose points: 7")

# Ensure model assets exist before starting a mode, and surface download errors clearly.
with st.expander("Model status", expanded=False):
    if st.button("Download / verify selected model assets"):
        try:
            hand_path = ensure_model("hand_landmarker.task", MODEL_DIR)
            pose_path = ensure_model(pose_choice, MODEL_DIR)
            st.success(f"Models ready:\n- {hand_path}\n- {pose_path}")
        except Exception as exc:
            st.error(str(exc))

mode = st.radio(
    "Choose a demo", ["Browser webcam", "Local webcam", "Upload image", "Pipeline overview"],
    horizontal=True,
)

# Do not leave a server-side webcam thread running after the user switches demos.
if mode != "Local webcam" and st.session_state.get("local_cam") is not None:
    stop_local_camera()

if mode == "Browser webcam":
    browser_camera(str(MODEL_DIR), pose_choice, frame_interval, confidence, swap_handedness)
elif mode == "Local webcam":
    local_camera(str(MODEL_DIR), pose_choice, frame_interval, confidence,
                 swap_handedness, width, height)
elif mode == "Upload image":
    image_demo(str(MODEL_DIR), pose_choice, frame_interval, confidence, swap_handedness)
else:
    pipeline_explainer()

st.divider()
st.caption("SignVoice landmark demo · Raw, un-normalised MediaPipe coordinates · Missing landmarks are represented by zeros.")
