from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from signvoice.vision.capture import CaptureError, Frame, ThreadedCapture
from signvoice.vision.landmark import LandmarkExtractor

RESOLUTIONS = {"320 x 240": (320, 240), "640 x 480": (640, 480), "1280 x 720": (1280, 720)}
POSE_MODEL = "pose_landmarker_lite.task"


@dataclass(frozen=True)
class VisionSettings:
    source_kind: str = "webcam"     
    camera_index: int = 0
    video_path: str = ""
    width: int = 640
    height: int = 480
    mirror: bool = True
    backend: str = "threaded"             
    extra_delay_ms: int = 0               
    use_pose: bool = True
    swap_handedness: bool = False
    min_detect: float = 0.5
    min_presence: float = 0.5
    min_tracking: float = 0.5
    draw: bool = True
    overlay: bool = True
    model_dir: str = "models/mediapipe"

    @property
    def source(self) -> Union[int, str]:
        return self.video_path if self.source_kind == "video" else self.camera_index

    @property
    def threaded(self) -> bool:
        return self.backend == "threaded"


class PlainCapture:
    """
    A plain cv2.VideoCapture read on the caller's thread, behind the same interface as
    ThreadedCapture. While the caller is busy the camera driver queues frames, so a slow
    consumer ends up looking at old ones. Only here so the UI can show the difference.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480,
                 mirror: bool = True) -> None:
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(source, str)
        self._period = 0.0
        self._next_due = 0.0
        self._count = 0
        self._fps = 0.0
        self._last_t: Optional[float] = None
        self.error: Optional[str] = None

    def start(self) -> "PlainCapture":
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"Cannot open video source {self.source!r}.")
        if self._is_file:
            self._period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)   # play files at normal speed
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return self

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, after_id: int = 0, timeout: float = 1.0) -> Optional[Frame]:
        if self._cap is None:
            return None
        if self._period:
            wait = self._next_due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            self._next_due = max(self._next_due, time.perf_counter()) + self._period
        ok, image = self._cap.read()
        if not ok:
            return None
        if self.mirror:
            image = cv2.flip(image, 1)
        now = time.perf_counter()
        if self._last_t is not None:
            inst = 1.0 / max(now - self._last_t, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = now
        self._count += 1
        return Frame(self._count, now, image)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frames_captured(self) -> int:
        return self._count

    def __enter__(self) -> "PlainCapture":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def build_extractor(cfg: VisionSettings) -> LandmarkExtractor:
    return LandmarkExtractor(
        model_dir=cfg.model_dir,
        pose_model=POSE_MODEL if cfg.use_pose else None,
        min_detect_conf=cfg.min_detect,
        min_presence_conf=cfg.min_presence,
        min_tracking_conf=cfg.min_tracking,
        swap_handedness=cfg.swap_handedness,
    )


def open_source(cfg: VisionSettings):
    """Open the configured camera or video file with the configured backend."""
    if cfg.source_kind == "video" and not Path(cfg.video_path).is_file():
        raise CaptureError(f"Video file not found: {cfg.video_path!r}")
    if cfg.threaded:
        return ThreadedCapture(cfg.source, width=cfg.width, height=cfg.height, mirror=cfg.mirror).start()
    return PlainCapture(cfg.source, cfg.width, cfg.height, cfg.mirror).start()


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


class RollingStats:
    """Rolling windows of inference time and frame age, plus loop FPS."""

    def __init__(self, window: int = 150) -> None:
        self.inference_ms: deque[float] = deque(maxlen=window)
        self.age_ms: deque[float] = deque(maxlen=window)
        self._stamps: deque[float] = deque(maxlen=31)
        self.processed = 0

    def add(self, inference_ms: float, age_ms: Optional[float] = None, now: Optional[float] = None) -> None:
        self.inference_ms.append(inference_ms)
        if age_ms is not None:
            self.age_ms.append(age_ms)
        self._stamps.append(time.perf_counter() if now is None else now)
        self.processed += 1

    @property
    def fps(self) -> float:
        """Frames processed per second over the last ~30 frames."""
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


def put_overlay(image: np.ndarray, lines: Sequence[str]) -> None:
    """Write lines of text in the top-left corner of a BGR image, in place."""
    y = 24
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def idle_image(width: int, height: int, text: str) -> np.ndarray:
    """A neutral BGR placeholder so the layout does not jump when the camera is off."""
    image = np.full((height, width, 3), 38, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(image, text, ((width - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
    return image