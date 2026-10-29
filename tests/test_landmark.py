"""
Tests for signvoice.vision.landmarks.LandmarkExtractor, using fake MediaPipe models
(no .task downloads, no webcam). Run with:  pytest tests/test_landmarks.py -v
"""
import types

import numpy as np
import pytest
from mediapipe.tasks.python import vision

from signvoice.vision import landmark as L


class P:                                    # fake landmark
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Cat:                                  # fake handedness category
    def __init__(self, name):
        self.category_name = name


def hand_pts(x0):
    return [P(x0 + 0.001 * i, 0.5) for i in range(21)]


class FakeModel:
    """Stands in for HandLandmarker / PoseLandmarker and records how it was called."""

    def __init__(self, options, result):
        self.options, self.result = options, result
        self.video_calls, self.image_calls, self.closed = [], 0, False

    def detect_for_video(self, image, ts):
        self.video_calls.append(ts)
        return self.result

    def detect(self, image):
        self.image_calls += 1
        return self.result

    def close(self):
        self.closed = True


HAND_RESULT = types.SimpleNamespace(hand_landmarks=[hand_pts(0.3)], handedness=[[Cat("Left")]])
POSE_RESULT = types.SimpleNamespace(pose_landmarks=[[P(i / 100, 0.5) for i in range(33)]])
FRAME = np.zeros((120, 160, 3), np.uint8)


@pytest.fixture
def make(monkeypatch):
    """Factory: build an extractor whose MediaPipe models are fakes."""
    monkeypatch.setattr(L, "ensure_model", lambda name, d="": f"/fake/{name}")
    made = {}

    def build(**kwargs):
        hand_result = kwargs.pop("hand_result", HAND_RESULT)
        monkeypatch.setattr(vision.HandLandmarker, "create_from_options",
                            staticmethod(lambda o: made.setdefault("hand", FakeModel(o, hand_result))))
        monkeypatch.setattr(vision.PoseLandmarker, "create_from_options",
                            staticmethod(lambda o: made.setdefault("pose", FakeModel(o, POSE_RESULT))))
        return L.LandmarkExtractor(**kwargs), made

    return build


# ---- modes ----------------------------------------------------------------- #
def test_invalid_mode_is_rejected(make):
    with pytest.raises(ValueError):
        make(mode="live_stream")


def test_video_mode_is_default_and_passed_to_both_models(make):
    ex, models = make()
    assert models["hand"].options.running_mode == vision.RunningMode.VIDEO
    assert models["pose"].options.running_mode == vision.RunningMode.VIDEO


def test_image_mode_uses_detect_not_detect_for_video(make):
    ex, models = make(mode="image")
    assert models["hand"].options.running_mode == vision.RunningMode.IMAGE
    ex.process(FRAME)
    assert models["hand"].image_calls == 1 and models["hand"].video_calls == []
    assert models["pose"].image_calls == 1 and models["pose"].video_calls == []


def test_video_timestamps_strictly_increase(make):
    ex, models = make()
    for _ in range(50):
        ex.process(FRAME)                           # many calls inside the same millisecond
    ts = models["hand"].video_calls
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_explicit_timestamp_is_used_and_kept_increasing(make):
    ex, models = make()
    ex.process(FRAME, ts_ms=1000)
    ex.process(FRAME, ts_ms=1033)
    ex.process(FRAME, ts_ms=1033)            # repeated: must be bumped, not rejected
    assert models["hand"].video_calls == [1000, 1033, 1034]


# ---- feature layout ---------------------------------------------------------- #
def test_features_land_in_the_right_slots(make):
    ex, _ = make()
    r = ex.process(FRAME)
    assert r.features.shape == (L.FEATURE_DIM,) and r.features.dtype == np.float32
    assert (r.left_found, r.right_found, r.pose_found) == (True, False, True)
    blocks = L.split_feature_vector(r.features)
    assert np.allclose(blocks["left"][:, 0], [0.3 + 0.001 * i for i in range(21)])
    assert not blocks["right"].any()                          # undetected hand stays zero
    assert np.allclose(blocks["pose"][:, 0], [i / 100 for i in L.POSE_IDX])   # picked the right joints


def test_pose_model_can_be_disabled(make):
    ex, models = make(pose_model=None)
    r = ex.process(FRAME)
    assert "pose" not in models
    assert not r.pose_found and not r.features[L.POSE].any()
    assert r.left_found


def test_no_hands_gives_zero_features(make):
    empty = types.SimpleNamespace(hand_landmarks=[], handedness=[])
    ex, _ = make(hand_result=empty, pose_model=None)
    r = ex.process(FRAME)
    assert not r.hands_found and not r.features.any()


def test_close_closes_models(make):
    ex, models = make()
    ex.close()
    assert models["hand"].closed and models["pose"].closed


# ---- left/right assignment ---------------------------------------------------- #
def test_hands_assigned_by_label(make):
    ex, _ = make()
    res = types.SimpleNamespace(hand_landmarks=[hand_pts(0.7), hand_pts(0.2)],
                                handedness=[[Cat("Right")], [Cat("Left")]])
    left, right = ex._assign_hands(res)
    assert left[0, 0] == pytest.approx(0.2) and right[0, 0] == pytest.approx(0.7)


def test_duplicate_labels_fall_back_to_screen_position(make):
    ex, _ = make()
    res = types.SimpleNamespace(hand_landmarks=[hand_pts(0.7), hand_pts(0.2)],
                                handedness=[[Cat("Right")], [Cat("Right")]])
    left, right = ex._assign_hands(res)
    assert left[0, 0] == pytest.approx(0.2) and right[0, 0] == pytest.approx(0.7)


def test_swap_handedness_inverts_labels(make):
    ex, _ = make(swap_handedness=True)
    res = types.SimpleNamespace(hand_landmarks=[hand_pts(0.3)], handedness=[[Cat("Left")]])
    left, right = ex._assign_hands(res)
    assert left is None and right is not None


# ---- drawing --------------------------------------------------------------------- #
def test_draw_does_nothing_when_nothing_found():
    img = np.zeros((120, 160, 3), np.uint8)
    L.draw_landmarks(img, L.LandmarkResults(L.empty_features(), False, False, False, 0.0))
    assert not img.any()


def test_draw_marks_a_found_hand():
    feats = L.empty_features()
    feats[L.LEFT_HAND] = np.array([[0.3 + 0.01 * i, 0.5, 0] for i in range(21)], np.float32).ravel()
    img = np.zeros((120, 160, 3), np.uint8)
    L.draw_landmarks(img, L.LandmarkResults(feats, True, False, False, 0.0))
    assert img.any()