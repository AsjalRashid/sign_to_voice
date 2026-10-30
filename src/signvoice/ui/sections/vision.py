from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from signvoice.ui import runtime
from signvoice.ui.compat import stretch
from signvoice.ui.registry import Section, register_section
from signvoice.ui.stages import FrameContext, build_stages, run_stages
from signvoice.vision.capture import CaptureError
from signvoice.vision.landmark import FEATURE_DIM, LEFT_HAND, POSE, RIGHT_HAND, draw_landmarks

TEXT_REFRESH_S = 0.25          # how often the text panels redraw
CHART_REFRESH_S = 1.0          # how often the history chart redraws
RUNNING = "vision_running"



def _sidebar() -> runtime.VisionSettings:
    with st.sidebar:
        st.header("Vision settings")
        st.caption("Changing a setting while the camera runs restarts it with the new value.")

        with st.expander("Source", expanded=True):
            kind = st.radio("Input", ["webcam", "video"], key="v_kind", horizontal=True,
                            format_func={"webcam": "Webcam", "video": "Video file"}.get)
            camera_index = st.number_input("Camera index", 0, 10, 0, key="v_cam", disabled=kind != "webcam")
            resolution = st.selectbox("Resolution", list(runtime.RESOLUTIONS), index=1, key="v_res",
                                      disabled=kind != "webcam")
            video_path = st.text_input("Video file path", key="v_path", disabled=kind != "video",
                                       placeholder="e.g. samples/hello.mp4")
            mirror = st.checkbox("Mirror (selfie view)", True, key="v_mirror")

        with st.expander("Capture backend", expanded=True):
            backend = st.radio("Backend", ["threaded", "plain"], key="v_backend",
                               format_func={"threaded": "ThreadedCapture (latest frame)",
                                            "plain": "Plain cv2.VideoCapture"}.get)
            delay = st.slider("Simulated downstream work (ms per frame)", 0, 200, 0, 10, key="v_delay",
                              help="Stands in for the classifier, LLM calls and so on. Raise it and compare "
                                   "the two backends: the plain one falls behind and shows old frames.")

        with st.expander("Detection"):
            use_pose = st.checkbox("Pose landmarks", True, key="v_pose")
            swap = st.checkbox("Swap left / right", False, key="v_swap",
                               help="Use if your left hand is reported as the right one.")
            detect = st.slider("Min detection confidence", 0.1, 0.9, 0.5, 0.05, key="v_det")
            presence = st.slider("Min presence confidence", 0.1, 0.9, 0.5, 0.05, key="v_pres")
            tracking = st.slider("Min tracking confidence", 0.1, 0.9, 0.5, 0.05, key="v_track")

        with st.expander("Display"):
            draw = st.checkbox("Draw landmarks", True, key="v_draw")
            overlay = st.checkbox("Text overlay on the video", True, key="v_overlay")

    width, height = runtime.RESOLUTIONS[resolution]
    return runtime.VisionSettings(
        source_kind=kind, camera_index=int(camera_index), video_path=video_path.strip(),
        width=width, height=height, mirror=mirror, backend=backend, extra_delay_ms=int(delay),
        use_pose=use_pose, swap_handedness=swap, min_detect=detect, min_presence=presence,
        min_tracking=tracking, draw=draw, overlay=overlay,
    )



def _flag(label: str, on: bool) -> str:
    return f":green[●] {label}" if on else f":gray[○] {label}"


def _render_signals(ph, result) -> None:
    if result is None:
        ph.caption("Waiting for the camera...")
        return
    ph.markdown(
        f"{_flag('Left hand', result.left_found)}  \n"
        f"{_flag('Right hand', result.right_found)}  \n"
        f"{_flag('Pose', result.pose_found)}  \n\n"
        f"Feature vector: {FEATURE_DIM} values, {int(np.count_nonzero(result.features))} non-zero"
    )


def _fmt(value: float, unit: str = "") -> str:
    return f"{value:.1f}{unit}"


def _render_timing(ph, source, stats, cfg, skipped: int) -> None:
    infer = list(stats.inference_ms)
    ages = list(stats.age_ms)
    with ph.container():
        a, b = st.columns(2)
        a.metric("Capture FPS", _fmt(source.fps))
        b.metric("Processing FPS", _fmt(stats.fps))
        a.metric("Inference, median", _fmt(runtime.percentile(infer, 50), " ms"))
        b.metric("Inference, p90", _fmt(runtime.percentile(infer, 90), " ms"))
        if cfg.threaded:
            a.metric("Frame age on screen", _fmt(runtime.percentile(ages, 50), " ms"),
                     help="Time from the camera capturing the frame to it being drawn (median).")
        else:
            a.metric("Frame age on screen", "n/a", help="Plain cv2 gives no capture timestamps.")
        b.metric("Frames skipped", skipped,
                 help="Camera frames dropped because newer ones arrived while processing.")


def _render_features(ph, result) -> None:
    f = result.features
    left, right, pose = (f[s].reshape(-1, 3) for s in (LEFT_HAND, RIGHT_HAND, POSE))
    rows = [("Left wrist", left[0], result.left_found), ("Left index tip", left[8], result.left_found),
            ("Right wrist", right[0], result.right_found), ("Right index tip", right[8], result.right_found),
            ("Nose", pose[0], result.pose_found)]
    table = pd.DataFrame([{"point": name, "x": round(float(v[0]), 3), "y": round(float(v[1]), 3),
                           "z": round(float(v[2]), 3), "detected": ok} for name, v, ok in rows])
    with ph.container():
        st.dataframe(table, hide_index=True, **stretch(st.dataframe))
        st.caption("Raw MediaPipe values: x and y are fractions of the image width and height. "
                   "All zeros means that part was not detected.")


def _render_stages(ph, stage_out: dict) -> None:
    rows = [{"stage": stage, "metric": k, "value": str(v)} for stage, vals in stage_out.items()
            for k, v in vals.items()]
    if not rows:
        ph.info("No pipeline stages registered. Add a module to signvoice/ui/stage_plugins/ "
                "(see ui/README.md).")
        return
    ph.dataframe(pd.DataFrame(rows), hide_index=True, **stretch(st.dataframe))


def _render_chart(ph, stats, cfg) -> None:
    data = {"inference (ms)": list(stats.inference_ms)}
    if cfg.threaded and len(stats.age_ms) == len(stats.inference_ms):
        data["frame age on screen (ms)"] = list(stats.age_ms)
    ph.line_chart(pd.DataFrame(data), height=220)


def _how_it_works() -> None:
    st.markdown(
        """
**Two loops, two threads.** A background thread reads the camera and keeps overwriting a single
"latest frame" slot. The loop you see here asks for the newest frame, runs MediaPipe on it, and
draws the result. If processing is slower than the camera, frames are skipped but never queued,
so what you see is never old.

**Try it.** Raise *Simulated downstream work* in the sidebar, then switch the backend between
*ThreadedCapture* and *Plain cv2.VideoCapture* and wave your hand. The plain backend lags behind
because the driver buffers frames while the loop is busy.

**What the numbers mean**
- *Capture FPS*: how fast the camera delivers frames.
- *Processing FPS*: how fast this loop handles them (includes drawing in the browser).
- *Inference*: time inside the landmark extractor only (hand + pose models).
- *Frame age on screen*: capture to display, measured exactly with the threaded backend.
- *Frames skipped*: camera frames never processed because a newer one replaced them.

Processing FPS here is a little lower than in a bare Python loop, because every frame is also
encoded and sent to the browser.
        """
    )

def _describe(cfg: runtime.VisionSettings) -> str:
    where = f"webcam {cfg.camera_index}, {cfg.width}x{cfg.height}" if cfg.source_kind == "webcam" else cfg.video_path
    backend = "ThreadedCapture" if cfg.threaded else "plain cv2.VideoCapture"
    return f"{backend} · {where} · pose {'on' if cfg.use_pose else 'off'}"


def _run_loop(cfg, ui) -> None:
    try:
        with st.spinner("Loading MediaPipe models and opening the source (the first run downloads the models)..."):
            extractor = runtime.build_extractor(cfg)
            try:
                source = runtime.open_source(cfg)
            except Exception:
                extractor.close()
                raise
    except (CaptureError, RuntimeError, OSError) as exc:
        st.session_state[RUNNING] = False
        ui["status"].error(f"Could not start: {exc}")
        return

    ui["status"].caption(f"Running · {_describe(cfg)}")
    stats = runtime.RollingStats()
    stages = build_stages()
    last_id, last_text, last_chart, end_message = 0, 0.0, 0.0, None
    stage_out: dict = {}
    try:
        while True:
            frame = source.read(after_id=last_id, timeout=1.0)
            if frame is None:
                end_message = getattr(source, "error", None) or "The video source ended."
                break
            last_id = frame.id

            result = extractor.process(frame.image)
            if cfg.extra_delay_ms:
                time.sleep(cfg.extra_delay_ms / 1000)
            stage_out = run_stages(stages, FrameContext(frame, result, cfg))

            view = frame.image.copy()                 # frames are shared so cant draw on the original
            if cfg.draw:
                draw_landmarks(view, result)
            age = frame.age_ms if cfg.threaded else None
            stats.add(result.inference_ms, age)
            if cfg.overlay:
                runtime.put_overlay(view, [f"FPS {stats.fps:.0f}   inference {result.inference_ms:.0f} ms"])
            ui["screen"].image(view, channels="BGR", output_format="JPEG", **stretch(st.image))

            now = time.perf_counter()
            if now - last_text >= TEXT_REFRESH_S:
                skipped = max(0, source.frames_captured - stats.processed)
                _render_signals(ui["signals"], result)
                _render_timing(ui["timing"], source, stats, cfg, skipped)
                _render_features(ui["features"], result)
                _render_stages(ui["stages"], stage_out)
                last_text = now
            if now - last_chart >= CHART_REFRESH_S:
                _render_chart(ui["chart"], stats, cfg)
                last_chart = now
    finally:                                         
        source.stop()
        extractor.close()

    st.session_state[RUNNING] = False
    ui["status"].info(f"Stopped: {end_message}")



def render() -> None:
    cfg = _sidebar()
    st.caption(SECTION.blurb)

    start_col, stop_col, _ = st.columns([1, 1, 6])
    if start_col.button("▶ Start", key="vision_start", type="primary", **stretch(st.button)):
        st.session_state[RUNNING] = True
    if stop_col.button("■ Stop", key="vision_stop", **stretch(st.button)):
        st.session_state[RUNNING] = False

    screen_col, info_col = st.columns([3, 2])
    ui = {}
    with screen_col:
        ui["status"] = st.empty()
        ui["screen"] = st.empty()
    with info_col:
        st.markdown("**Detection signals**")
        ui["signals"] = st.empty()
        st.markdown("**Timing**")
        ui["timing"] = st.empty()
    tab_chart, tab_features, tab_stages, tab_how = st.tabs(
        ["Timing history", "Feature vector", "Pipeline stages", "How it works"])
    with tab_chart:
        ui["chart"] = st.empty()
    with tab_features:
        ui["features"] = st.empty()
    with tab_stages:
        ui["stages"] = st.empty()
    with tab_how:
        _how_it_works()

    ui["screen"].image(runtime.idle_image(cfg.width, cfg.height, "Camera off"),
                       channels="BGR", output_format="JPEG", **stretch(st.image))
    _render_signals(ui["signals"], None)
    _render_stages(ui["stages"], {})

    if st.session_state.get(RUNNING, False):
        _run_loop(cfg, ui)
    else:
        ui["status"].caption("Press Start to open the camera.")


SECTION = register_section(Section(
    key="vision", label="Vision", icon="👁️", order=10,
    blurb="Threaded OpenCV capture and MediaPipe hand and pose landmarks, live.",
    render=render,
))