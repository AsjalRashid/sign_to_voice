"""
Tests for signvoice.vision.capture.ThreadedCapture.

No webcam needed: file sources use a generated video, and "camera" behaviour is
checked with a fake cv2.VideoCapture. Run with:  pytest tests/test_capture.py -v
"""
import time

import cv2
import numpy as np
import pytest

from signvoice.vision.capture import CaptureError, ThreadedCapture


@pytest.fixture
def video(tmp_path):
    """2-second, 30 FPS clip. Left half white, right half black (to check mirroring)."""
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, :80] = 255
    for _ in range(60):
        writer.write(frame)
    writer.release()
    return path


def drain(cam, delay=0.0):
    """Read until the stream ends. Returns the list of frames received."""
    frames, last = [], 0
    while (f := cam.read(after_id=last, timeout=1.0)) is not None:
        frames.append(f)
        last = f.id
        time.sleep(delay)
    return frames


def test_ids_strictly_increase_and_stream_ends_cleanly(video):
    with ThreadedCapture(video) as cam:
        frames = drain(cam)
        assert len(frames) > 30
        assert all(b.id > a.id for a, b in zip(frames, frames[1:]))
        assert cam.ended and cam.error is None      # end of file is not an error


def test_slow_consumer_skips_frames_but_never_sees_stale_ones(video):
    with ThreadedCapture(video) as cam:
        last, got, ages = 0, 0, []
        while (f := cam.read(after_id=last, timeout=1.0)) is not None:
            ages.append(f.age_ms)                    # age right when we receive it
            last, got = f.id, got + 1
            time.sleep(0.06)                         # pretend inference takes 60 ms
        assert got < cam.frames_captured             # frames were skipped...
        assert max(ages) < 150                       # ...but each one was fresh


def test_timestamp_is_capture_time(video):
    with ThreadedCapture(video) as cam:
        f = cam.read(after_id=0)
        time.sleep(0.1)
        assert 100 <= f.age_ms < 1000                # age grows from capture time


def test_mirror_flips_horizontally(video):
    with ThreadedCapture(video, mirror=False) as cam:
        plain = cam.read(after_id=0).image
    with ThreadedCapture(video, mirror=True) as cam:
        flipped = cam.read(after_id=0).image
    assert plain[:, :80].mean() > plain[:, 80:].mean()       # white on the left
    assert flipped[:, 80:].mean() > flipped[:, :80].mean()   # white on the right


def test_missing_source_raises(tmp_path):
    with pytest.raises(CaptureError):
        ThreadedCapture(str(tmp_path / "nope.mp4")).start()


def test_stop_is_idempotent_and_start_twice_is_safe(video):
    cam = ThreadedCapture(video).start()
    assert cam.start() is cam                        # no second thread
    cam.stop()
    cam.stop()                                       # must not raise


# ---- live-camera code path, using a fake cv2.VideoCapture ------------------ #
class FakeCam:
    frames_before_dying = None                       # None = never dies

    def __init__(self, source):
        self.n = 0

    def isOpened(self): return True
    def set(self, *a): return True
    def get(self, *a): return 0.0
    def release(self): pass

    def read(self):
        time.sleep(1 / 30)
        if self.frames_before_dying is not None and self.n >= self.frames_before_dying:
            return False, None
        self.n += 1
        return True, np.zeros((120, 160, 3), np.uint8)


def test_live_camera_path_delivers_frames(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCam)
    with ThreadedCapture(0, start_timeout=1.0) as cam:
        f1 = cam.read(after_id=0)
        f2 = cam.read(after_id=f1.id)
        assert f2.id > f1.id
        assert 15 < cam.fps < 40


def test_camera_loss_is_reported_as_error(monkeypatch):
    class Dying(FakeCam):
        frames_before_dying = 5
    monkeypatch.setattr(cv2, "VideoCapture", Dying)
    with ThreadedCapture(0, max_read_failures=3, start_timeout=1.0) as cam:
        drain(cam)
        assert cam.ended
        assert cam.error is not None