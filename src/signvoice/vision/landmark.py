from __future__ import annotations
 
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
 
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


POSE_IDX = (0, 11, 12, 13, 14, 15, 16)      # nose, shoulders, elbows, wrists only instead of all 33 body landmark locations
ONE_HAND_PTS = 21
HAND_DIM = ONE_HAND_PTS * 3                 # 63
POSE_DIM = len(POSE_IDX) * 3                # 21
FEATURE_DIM = 2 * HAND_DIM + POSE_DIM       # 147
 
LEFT_HAND = slice(0, HAND_DIM)              # 0 TO 62
RIGHT_HAND = slice(HAND_DIM, 2 * HAND_DIM)  # 63 TO 125
POSE = slice(2 * HAND_DIM, FEATURE_DIM)     # 126 TO 146
 
POSE_POS = {"nose": 0, "l_shoulder": 1, "r_shoulder": 2,
            "l_elbow": 3, "r_elbow": 4, "l_wrist": 5, "r_wrist": 6}

# Dictates how to draw lines between points to construct a visual skeleton graph.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_CONNECTIONS = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
 
 
def ensure_model(filename: str, model_dir: str | Path = "models/mediapipe") -> Path:
    """Returns the path to a model file, downloading it first if it is missing.

    Args:
        filename(str): Name of the model file to find or download.
        model_dir(str|Path): Target directory where models are stored.

    Returns:
        The Path instance pointing to the local validated model file.

    Raises:
        FileNotFoundError: If the filename is not recognized in MODEL_URLS.
        RuntimeError: If downloading from the remote URL fails.
    """
    path = Path(model_dir) / filename
    if path.exists():
        return path
    url = MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(f"{path} not found and no download URL is known for it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as exc:                      
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {filename} from {url} ({exc}). "
            f"Download it manually and place it at {path}."
        ) from exc
    return path

#HELPER FUNCTIONS
@dataclass
class LandmarkResults:
    """A container holding structured tracking outputs from an image pipeline.

    Attributes:
        features(np.ndarray): Flattened array containing (X, Y, Z) metrics for left hand,
            right hand, and body pose structures.
        left_found(bool): True if the left hand is detected in the current frame.
        right_found(bool): True if the right hand is detected in the current frame.
        pose_found(bool): True if the body pose points are detected in the frame.
        inference_ms(float): Duration of the tracking processing block in milliseconds.
    """
    
    features: np.ndarray
    left_found: bool
    right_found: bool
    pose_found: bool
    inference_ms: float
    
    @property
    def hands_found(self) -> bool:
        """Evaluates if at least one hand has been tracked in the current frame."""
        return self.left_found or self.right_found
    
def empty_features() -> np.ndarray :
    """
        Generates a clean, zero-initialized numpy feature array.

    Returns:
        A zeroed 1D array of size FEATURE_DIM with float32 precision.
    """
    return np.zeros(FEATURE_DIM, dtype=np.float32)

def split_feature_vector(vec:np.ndarray) -> dict[str, np.ndarray]:
    """
    Slices and reshapes a combined feature vector back into component parts.

    Args:
        vec(np.ndarray): An array whose trailing dimension matches FEATURE_DIM.

    Returns:
        A dictionary containing keys 'left', 'right', and 'pose' mapped to arrays 
        reshaped into spatial coordinates of shape (..., N, 3).
    """
    leading_dims = vec.shape[:-1]
    
    return{
        'left' : vec[..., LEFT_HAND].reshape(*leading_dims, ONE_HAND_PTS, 3),
        'right' : vec[..., RIGHT_HAND].reshape(*leading_dims, ONE_HAND_PTS, 3),
        'pose' : vec[..., POSE].reshape(*leading_dims, len(POSE_IDX), 3),
    }

def _to_xyz(landmarks) -> np.ndarray:
    """
    Returns the x, y and z coordianted of a landmark detection
    
    Args:
            landmarks: An iterable collection of MediaPipe landmark instances.
    
    Returns:
        A float32 matrix of shape (num_points, 3) containing normalized X, Y, Z 
        coordinates.
    
    """
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

#Extractor 

RUNNING_MODES = {
    "video": vision.RunningMode.VIDEO,
    "image": vision.RunningMode.IMAGE,
}

class LandmarkExtractor: 
    """
    A wrapper for managing MediaPipe Hand and Pose landmarker models.

    Handles structural setup, operational runtime states, model orchestration,
    and concurrent model inference processing loops.
    
    """
        
    def __init__(self,
                 model_dir: str|Path = "models\mediapipe",
                 hand_model: str = 'hand_landmarker.task',
                 pose_model: str = 'pose_landmarker_lite.task',
                 min_detect_conf: float = 0.5,
                 min_presence_conf: float = 0.5,
                 min_tracking_conf: float = 0.5,
                 swap_handedness: bool = False,
                 mode: str = "video",
                 frame_interval: int = 3,
                 ):
        """
        Initializes the hand and pose landmarkers with specific metrics.

        Args:
            model_dir(str|Path): Local path where model assets reside or will be saved.
            hand_model(str): Filename string of the target hand landmarker asset.
            pose_model(str): Filename string of the target pose landmarker asset. If 
                evaluates to empty or None, pose tracking is bypassed.
            min_detect_conf(float): Minimum confidence score for detection validation.
            min_presence_conf(float): Minimum confidence score for presence signaling.
            min_tracking_conf(float): Minimum confidence score for tracking stability.
            swap_handedness(bool): Explicitly mirrors left and right landmarker labels.
            mode(str): Operational environment framework. Must be "video" or "image".
            frame_interval(int): Specifies after how many frames to run the pose detection model.

        Raises:
            ValueError: If an unsupported mode argument is provided.
        """
        if mode not in RUNNING_MODES:
            raise ValueError(f"mode must be one of {sorted(RUNNING_MODES)}, got {mode!r}")
        self.mode = mode
        running_mode = RUNNING_MODES[mode]
        self.swap_handedness = swap_handedness
        self.frame_interval = frame_interval
        
        self._processed_frames = 0
        self._last_pose_xyz = None
        
        self.last_ts_ms = 0 
        
        hand_opts = vision.HandLandmarkerOptions(
            base_options = mp_python.BaseOptions(model_asset_path=str(ensure_model(hand_model, model_dir))),
            running_mode = running_mode,
            num_hands = 2,
            min_hand_detection_confidence = min_detect_conf,
            min_hand_presence_confidence = min_presence_conf,
            min_tracking_confidence =  min_tracking_conf,
        )
        self._hands = vision.HandLandmarker.create_from_options(hand_opts)        #Initializes the Landmark detector 
        
        self._pose = None
        if pose_model:
            pose_opts = vision.PoseLandmarkerOptions(
                base_options = mp_python.BaseOptions(model_asset_path=str(ensure_model(pose_model, model_dir))),
                running_mode = running_mode,
                num_poses = 1,
                min_pose_detection_confidence = min_detect_conf,
                min_pose_presence_confidence = min_presence_conf,
                min_tracking_confidence =  min_tracking_conf,
            )
            self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
    
    def close(self) -> None:
        """Releases allocated resources and terminates active landmarker handles."""        
        for model in (self._hands, self._pose):
            if model is not None:
                model.close()
        self._hands = self._pose = None 
    
    def __enter__(self) -> 'LandmarkExtractor':
        """Enters context manager scope."""
        return self 
    
    def __exit__(self, *exc) -> None:
        """Cleans up active instances when exiting context manager scope."""
        return self.close

    def _next_timestamp_ms(self, requested: Optional[int] = None) -> int:
        """Generates a strictly increasing millisecond timestamp for video streams.

        Args:
            requested(int): Explicit timestamp override value if supplied.

        Returns:
            A monotonically increasing millisecond integer value.
        """
        ts = int(time.perf_counter() *1000) if requested is None else int(requested)
        
        if ts <= self.last_ts_ms:
            ts = self.last_ts_ms + 1
        
        self.last_ts_ms = ts 
        return ts 
    
    def _detect(self, landmarker, frame: np.ndarray, ts: int):
        """Executes landmarker inference using the active landmarker for detection.

        Args:
            landmarker: Active MediaPipe landmarker runner client instance.
            frame: Prepared MediaPipe Image structural wrapper.
            ts: Calculated incremental video micro-timestamp window reference.

        Returns:
            MediaPipe detection results object matching the selected running mode.
        """
        if self.mode == 'video':
            return landmarker.detect_for_video(frame, ts)
        return landmarker.detect(frame)
    
    def process(self, bgr_frame: np.ndarray, ts_ms: Optional[int] = None) -> LandmarkResults:
        """Processes an incoming image frame to perform raw tracking extraction.
        Args:
            bgr_frame(np.ndarray): OpenCV default BGR source image array.
            ts_ms(int|None): Optional explicit millisecond timestamp tracking marker window.
        Returns:
            A LandmarkResults instance holding feature vectors and diagnostic data.
        """
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
        ts = self._next_timestamp_ms(ts_ms) if self.mode == "video" else 0

        left, right = self._assign_hands(self._detect(self._hands, frame, ts ))
        
        pose_xyz = None
        if self._pose is not None:
            if (self._processed_frames % self.frame_interval) == 0: # Gates pose detection to run every N frame    
                pose_res = self._detect(self._pose, frame, ts)
                self._last_pose_xyz = (
                    _to_xyz(pose_res.pose_landmarks[0])[list(POSE_IDX)]
                    if pose_res.pose_landmarks
                    else None
                )
            pose_xyz = self._last_pose_xyz
        
        self._processed_frames += 1 
        
        features = empty_features()
        if left is not None:
            features[LEFT_HAND] = left.ravel()
        if right is not None:
            features[RIGHT_HAND] = right.ravel()
        if pose_xyz is not None:
            features[POSE] = pose_xyz.ravel()
            
        return LandmarkResults(
            features = features,
            left_found= left is not None,
            right_found= right is not None,
            pose_found= pose_xyz is not None,
            inference_ms= (time.perf_counter() - t0) *1000
        )
    
    def _assign_hands(self, res) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Map detected hands to (left, right) using MediaPipe's handedness labels.
        Args:
            res: Direct return output container from a HandLandmarker execution logic block.
        Returns:
            A tuple of (left_hand, right_hand) where each element is either a numpy float32 coordinate matrix or 
            None if missing.
        """
        hands = []
        for landmarks, handedness in zip(res.hand_landmarks, res.handedness):
            label = handedness[0].category_name          # "Left" or "Right"
            if self.swap_handedness:
                label = "Right" if label == "Left" else "Left"
            hands.append((label, _to_xyz(landmarks)))
 
        if not hands:
            return None, None
        if len(hands) == 2 and hands[0][0] == hands[1][0]:
            hands.sort(key=lambda h: h[1][0, 0])
            return hands[0][1], hands[1][1]
 
        left = right = None
        for label, xyz in hands:
            if label == "Left":
                left = xyz
            else:
                right = xyz
        return left, right
    
def draw_landmarks(frame: np.ndarray, res: LandmarkResults) -> None:
    """Draws tracking connections and landmark nodes directly onto a frame canvas.
    Args:
        frame(np.ndarray): The BGR numpy frame matrix canvas where spatial lines are written.
        res(LandmarkerResult): Completed structural LandmarkResults packet container.
    """
    h,w = frame.shape[:2]
    blocks = split_feature_vector(res.features)

    def px(xyz: np.ndarray) -> list[tuple[int, int]]:
        return [(int(x * w), int(y * h)) for x, y, _ in xyz]
 
    def draw(points, connections, line_color, dot_color):
        for a, b in connections:
            cv2.line(frame, points[a], points[b], line_color, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(frame, p, 3, dot_color, -1, cv2.LINE_AA)
 
    if res.left_found:
        draw(px(blocks["left"]), HAND_CONNECTIONS, (255, 140, 0), (255, 255, 255))
    if res.right_found:
        draw(px(blocks["right"]), HAND_CONNECTIONS, (0, 200, 120), (255, 255, 255))
    if res.pose_found:
        draw(px(blocks["pose"]), POSE_CONNECTIONS, (200, 200, 200), (0, 0, 255))    
    