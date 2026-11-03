"""
collect.py - capture and curate sign-language landmark training sequences.

This script collects labelled sign-language samples from a webcam. It coordinates
camera capture through ThreadedCapture and hand/pose landmark extraction through
LandmarkExtractor, then records a fixed number of consecutive feature vectors for
each requested sign.

A take is stored as a float32 NumPy array with shape (frames, FEATURE_DIM). The
feature layout is defined by signvoice.vision.landmark: left-hand coordinates,
right-hand coordinates, and selected body-pose coordinates are flattened into
one vector per frame. Coordinates are retained in the extractor's raw,
unnormalised form; undetected landmark groups are represented by zeros. The camera
feed is mirrored before landmark extraction to match the live application's input.

Collection proceeds sign by sign. For each take, the script waits for a start
keypress (or starts automatically when --auto is enabled), displays a countdown,
records the requested number of frames, and replays the resulting landmark
skeleton so the operator can keep or discard the sample. The replay is rendered
from the saved feature vectors on a dark canvas; raw camera frames are not saved.
A minimum hand-coverage threshold is used to flag takes with insufficient hand
visibility for review.

Accepted takes are written to numbered .npy files. A language-level metadata.csv
file is appended with one row per accepted take, recording its label, signer,
capture conditions, file path, frame count, feature dimension, hand coverage,
effective processing rate, and camera source. Existing numbered samples are
left in place, and numbering resumes after the highest existing numeric filename.

Output layout:
    <out>/<lang>/<signer>/<SIGN>/0001.npy ...
    <out>/<lang>/metadata.csv

Examples:
    python scripts/collect.py --signer p01 --lighting daylight --background wall 
        --signs SEND,CLEAR,STOP --reps 30

    python scripts/collect.py --signer p02 --lighting lamp --background shelf 
        --vocab languages/sign/asl/vocab.yaml --reps 30

    python scripts/collect.py --signer p01 --lighting daylight --background wall 
        --signs ABCDEFGHIKLMNOPQRSTUVWXY --split-chars --reps 20

Keys:
    SPACE  start the next take (countdown follows)
    D      discard the take being reviewed
    K      keep the take immediately and stop replaying it
    S      skip the current sign
    Q      quit collection; previously saved takes are retained
"""

from __future__ import annotations
 
import argparse
import csv
import re
import sys
import time
import yaml
from datetime import datetime
import itertools
from pathlib import Path
import cv2
from types import SimpleNamespace
import numpy as np
from signvoice.vision.capture import ThreadedCapture
from signvoice.vision.landmark import (FEATURE_DIM, LEFT_HAND, POSE, RIGHT_HAND, LandmarkExtractor, draw_landmarks)


WINDOW = "collect.py"
WHITE, RED, GREEN, AMBER = (255, 255, 255), (60, 60, 255), (80, 220, 80), (0, 190, 255)

class QuitAll(Exception):
    """Signal that the operator requested the collection session to stop."""
 
class SkipSign(Exception):
    """Signal that the operator wants to abandon the current sign."""

class Feed:
    """
    Provides sequential camera frames and their corresponding landmark results.

    The class tracks the most recently consumed frame ID so that each call to
    step() requests a newer frame from ThreadedCapture. The retrieved frame is
    passed through the configured LandmarkExtractor.

    Attributes:
        cam(ThreadedCapture): Camera capture object supplying frames.
        extractor(LandmarkExtractor): Processor used to detect and encode landmarks.
        last_id(int): ID of the most recently consumed camera frame.
    """
        
    def __init__(self, cam : ThreadedCapture, extractor: LandmarkExtractor) -> None:
        """
        Initialise the frame feed and its frame-ID cursor.

        Args:
            cam(ThreadedCapture): Active threaded camera capture instance.
            extractor(LandmarkExtractor): Landmark extractor used for each frame.
        """
        
        self.cam = cam
        self.extractor = extractor 
        self.last_id = 0 
        
    def step(self):
        """
        Retrieve the next frame and run landmark extraction on it.

        Raises:
            RuntimeError: If the capture thread reports an error or no frame is received within the read timeout.

        Returns:
            frame(np.ndarray): Copy of the retrieved camera image.
            results(object): Landmark extraction result for the retrieved frame.
        """
        
        frame = self.cam.read(self.last_id, 2.0)
        if frame is None:
            raise RuntimeError(self.cam.error or "No frames reseved from the camera")
        self.last_id = frame.id
        
        return frame.image.copy(), self.extractor.process(frame.image)
        
def put_text(frame, lines: list[str], color=WHITE, scale=0.7) -> None:
    """
    Draw a stack of outlined text lines near the upper-left corner of a frame.

    Args:
        frame(np.ndarray): Image canvas on which the text is drawn in place.
        lines(list[str]): Text lines to render, in display order.
        color(tuple[int, int, int]): BGR colour used for the foreground text.
        scale(float): OpenCV font scale; also controls vertical line spacing.
    """
        
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(36 * scale / 0.7)
 
 
def handle_key(key: int, allow_skip: bool = True) -> int:
    """
    Draw a stack of outlined text lines near the upper-left corner of a frame.

    Args:
        frame(np.ndarray): Image canvas on which the text is drawn in place.
        lines(list[str]): Text lines to render, in display order.
        color(tuple[int, int, int]): BGR colour used for the foreground text.
        scale(float): OpenCV font scale; also controls vertical line spacing.
    """
        
    key &= 0xFF
    if key == ord("q"):
        raise QuitAll
    if key == ord("s") and allow_skip:
        raise SkipSign
    return key

def show_live(feed: Feed, lines: list[str], color = WHITE):
    """
    Display the latest camera frame with its detected landmarks and status text.

    Args:
        feed(Feed): Source of camera frames and landmark results.
        lines(list[str]): Status or instruction lines to overlay.
        color(tuple[int, int, int]): BGR colour used for the overlay text.

    Returns:
        int: Normalized keyboard code captured while the window is displayed.

    Raises:
        QuitAll: If the operator presses Q.
        SkipSign: If the operator presses S.
    """
        
    frame, result = feed.step()
    draw_landmarks(frame, result)
    put_text(frame, lines, color)
    cv2.imshow(WINDOW, frame)
    
    return handle_key(cv2.waitKey(1))

# RECORDING STEPS

def wait_for(feed: Feed , header: str, auto: bool = False) -> None:
    """
    Wait for the operator to start the next recording.

    In manual mode, the live preview remains active until SPACE is pressed. In
    automatic mode, the function returns after the short preparation delay.

    Args:
        feed(Feed): Camera and landmark feed displayed while waiting.
        header(str): Status text identifying the sign and take.
        auto(bool): Start automatically after the preparation delay if True.

    Raises:
        QuitAll: If the operator presses Q.
        SkipSign: If the operator presses S.
    """
    
    t0 = time.perf_counter()
    while True:
        if auto and time.time() - t0 > 0.8:
            return
        key = show_live(feed, [header, 'Get ready' if auto else 'Press SPACE to start'])
        if key == ord(' '):
            return
    

def countdown(feed: Feed, header: str, seconds: int) -> None:
    """
    Display a live-preview countdown before recording begins.

    Args:
        feed(Feed): Camera and landmark feed displayed during the countdown.
        header(str): Status text identifying the sign and take.
        seconds(int): Countdown duration in seconds.

    Raises:
        QuitAll: If the operator presses Q during the countdown.
        SkipSign: If the operator presses S during the countdown.
    """
    
    t0 = time.perf_counter()
    while(sec_left := seconds - (time.time() - t0)) > 0:
        show_live(feed, [header, f"Starting in {int(sec_left)}"], AMBER)

def record_take(feed: Feed, header: str, n_frames: int) -> tuple:        
    """
    Capture a take as a sequence of consecutive landmark feature vectors.

    Each frame contributes one flattened feature vector. Hand presence is tracked
    per frame to calculate the proportion of frames in which at least one hand
    was detected. The returned frame rate is the effective capture and
    processing rate, rather than the camera hardware's nominal FPS.

    Args:
        feed(Feed): Camera and landmark feed used to collect the sequence.
        header(str): Status text identifying the sign and take.
        n_frames(int): Number of frames to record.

    Returns:
        features(np.ndarray): Float32 feature array with shape
            (n_frames, FEATURE_DIM).
        coverage(float): Fraction of recorded frames containing at least one
            detected hand, in the range [0.0, 1.0].
        fps(float): Effective number of processed frames per second.

    Raises:
        QuitAll: If the operator presses Q during recording.
    """
    features, presence = [], []
    t0 = time.perf_counter()
    while len_features := len(features) < n_frames:
        frame, results = feed.step()
        features.append(results.features) # (n_frame, 147)
        presence.append(results.hands_found)
        draw_landmarks(frame, results)
        put_text(frame, [header, f"Recording {len_features}/{n_frames}"])
        cv2.imshow(WINDOW, frame)
        handle_key(cv2.waitKey(1), allow_skip= False)
    fps = n_frames / max(time.perf_counter() - t0, 1e-6) # Effective recording/processing rate not hardware fps
    
    return np.stack(features).astype(np.float32), float(np.mean(presence)), fps


def result_from_features(vec: np.ndarray) -> SimpleNamespace:
    """
    Reconstruct the result attributes required by draw_landmarks().

    A missing landmark group is encoded as an all-zero slice in the stored
    feature vector. This function therefore infers each group's presence by
    checking whether its corresponding slice contains any non-zero values.

    Args:
        vec(np.ndarray): One flattened feature vector with shape
            (FEATURE_DIM,).

    Returns:
        SimpleNamespace: Lightweight result object containing the original
            feature vector and the left_found, right_found, pose_found, and
            hands_found flags expected by draw_landmarks().
    """
    left, right, pose = (bool(np.any(vec[part])) for part in (LEFT_HAND, RIGHT_HAND, POSE))
    return SimpleNamespace(features=vec, left_found=left, right_found=right, pose_found=pose,
                           hands_found=left or right)    
    
def review(data: np.ndarray,
           header: str,
           message: str,
           fps: float,
           color,
           loops: int = 2,
           size: tuple[int, int] = (640, 480)
           ) -> bool:
    """
    Replay a recorded take as a landmark skeleton for operator approval.

    Each feature vector is drawn on a dark canvas at an interval derived from
    the take's effective recording FPS. The replay contains only the extracted
    skeleton representation, not the original camera image. The operator can
    keep the take immediately, discard it, or allow the configured replay
    passes to finish.

    Args:
        data(np.ndarray): Recorded feature sequence with shape
            (frames, FEATURE_DIM).
        header(str): Status text identifying the sign and take.
        message(str): Additional review status, such as hand-coverage details.
        fps(float): Effective recording FPS used to determine playback timing.
        color(tuple[int, int, int]): BGR colour used for review text.
        loops(int): Maximum number of replay passes. A value less than or equal
            to zero repeats indefinitely until the operator chooses an action.
        size(tuple[int, int]): Replay canvas dimensions as (width, height).

    Returns:
        bool: True if the take is kept; False if the operator discards it.
    """
    
    width, height = size
    speed = 1.0 / min(max(fps, default= 5.0), 60.0)  # Gives recorded speed  for the original video with lower and upper bound of 5 and 60 respectively
    n_passes = itertools.count() if loops <=0 else range(loops) # Infinate looping (until user exits by keystroke) if loops is 0. If a number is given it will loop only within  that given range
    
    for loop_num in n_passes:
        for i,vec in enumerate(data):
            t0 = time.perf_counter()
            canvas = np.full((height, width, 3), 28, np.uint8)
            draw_landmarks(canvas, result_from_features(vec))
            progress = f"Replay {i + 1}/{len(data)}  pass {loop_num + 1}" + (f"/{loops}" if loops > 0 else "")
            put_text(canvas, [header, message, "K keep   D discard", progress], color)
            cv2.imshow(WINDOW, canvas)
            wait_ms = max(1, int((speed - (time.perf_counter() - t0)) * 1000))
            key = handle_key(cv2.waitKey(wait_ms), allow_skip=False)
            if key == ord("k"):
                return True
            if key == ord("d"):
                return False
    return True

# DATA KEEPING                        
CSV_FIELDS = [
    "timestamp", "lang", "signer", "sign", "file", "frames", "feature_dim",
    "hand_coverage", "fps", "lighting", "background", "camera",
]

def cleaned_name(text: str) -> str:
    """
    Convert a label or identifier into a filesystem-safe name.

    Leading and trailing whitespace is removed, and characters outside the
    letters, digits, underscore, and hyphen set are replaced with underscores.

    Args:
        text(str): Original label or identifier.

    Returns:
        str: Sanitized name suitable for use as a directory or filename
            component.
    """
    return re.sub(r"[^A-Za-z0-9_\-]","_",text.strip())

def next_index(sign_dir: Path) -> int:
    """
    Find the next numeric sample index for a sign directory.

    Only .npy files whose stems consist entirely of digits are considered.

    Args:
        sign_dir(Path): Directory containing previously saved take files.

    Returns:
        int: One greater than the largest existing numeric stem, or 1 if no
            numbered .npy files are present.
    """
    last_index = [int(p.stem) for p in sign_dir.glob("*.npy") if p.stem.isdigit()]
    return max(last_index, default=0) + 1

def append_metadata(csv_path: Path, row: dict) -> None:
    """
    Append one accepted take's metadata to the language-level CSV file.

    The CSV header is written only when the destination file does not already
    exist. Fields are emitted in the order defined by CSV_FIELDS.

    Args:
        csv_path(Path): Destination metadata CSV path.
        row(dict): Metadata values for the take, keyed by CSV_FIELDS names.
    """
    create_file = not csv_path.exists()
    with csv_path.open('a',newline="", encoding= "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames= CSV_FIELDS)
        if create_file:
            writer.writeheader()
        writer.writerow(row)
        
def load_signs(args) -> list[str]:
    """
    Build the deduplicated list of sign labels requested for collection.

    Labels can be loaded from a vocabulary YAML file, supplied as a
    comma-separated --signs value, or supplied as individual characters when
    --split-chars is enabled. All labels are sanitized and converted to
    uppercase before duplicates are removed while preserving input order.

    Args:
        args(argparse.Namespace): Parsed command-line arguments. Uses the
            vocab, signs, and split_chars attributes.

    Returns:
        list[str]: Ordered, unique sign labels to collect.

    Raises:
        SystemExit: If neither a vocabulary nor any sign labels are supplied.
    """
        
    signs: list[str] = []
    if args.vocab:
        import yaml  # only needed for --vocab
 
        entries = yaml.safe_load(Path(args.vocab).read_text(encoding="utf-8"))
        signs += [str(e["label"]) for e in entries]
    if args.signs:
        signs += list(args.signs) if args.split_chars else [s for s in args.signs.split(",") if s.strip()]
    signs = [cleaned_name(s).upper() for s in signs]
    if not signs:
        sys.exit("No signs given. Use --signs or --vocab.")
    return list(dict.fromkeys(signs))

def parse_args() -> argparse.Namespace:
    """
    Parse command-line options controlling the collection session.

    The arguments configure the signer and capture conditions, sign vocabulary,
    number and length of takes, review behaviour, camera settings, landmark
    models, and output directory.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
        
    p = argparse.ArgumentParser(description="Record and store landmark sequences for sign language translation training")
    
    p.add_argument("--signer", required= True, help="An ID to differentiate signers")
    p.add_argument("--lighting", required=True, help="The lighting condition in which the video was taken e.g daylight, brigt, lamp, white light etc")
    p.add_argument("--background", required=True, help="The type of background in the video like plain wall, shelf, trees etc")
    p.add_argument("--lang", default="asl", help="Sign language type or name like asl(Americam Sign Language), psl(Pakistan Sign Language)")
    p.add_argument("--signs", help="comma-separated signs, e.g. SEND,CLEAR,STOP")    
    p.add_argument("--split-chars", action="store_true",help="treat --signs as individual characters, e.g. ABC... for the alphabet")
    p.add_argument("--vocab", help="The vocab.yaml for the language in use")
    p.add_argument("--reps", type=int, default=30, help="Number of takes for this sign")
    p.add_argument("--frames", type=int, default=30, help="frames per take")
    p.add_argument("--countdown", type=int, default=3, help="seconds before each take starts")
    p.add_argument("--review-seconds", type=float, default=1.5, help="window to press D after a take")
    p.add_argument("--playback-loops", type=int, default=2,help="times the replay repeats (0 = until you press K or D)")
    p.add_argument("--min-coverage", type=float, default=0.9, help="reject a take if hands are visible in fewer than this fraction of frames")
    p.add_argument("--auto", action="store_true", help="start takes automatically")
    p.add_argument("--camera", type=int, default=0, help="The source value representing which camera to utilise")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model-dir", default="models/mediapipe", help="where MediaPipe .task files live")
    p.add_argument("--swap-handedness", action="store_true", help="use if left/right hands come out inverted on your setup")
    p.add_argument("--out", default="data/raw", help="root output folder")    
    
    return p.parse_args()
    
def main() -> None:
    """
    Run the sign-language data collection workflow.

    The function parses options, prepares the language output directory, opens
    the landmark extractor and mirrored camera stream, and collects the
    requested number of takes for each sign. Accepted sequences are saved as
    numbered NumPy files and described in metadata.csv. A Q keypress stops the
    session while preserving samples already written to disk. OpenCV windows
    are closed during cleanup, and a per-sign collection summary is printed.

    Raises:
        RuntimeError: If camera frame acquisition fails.
        SystemExit: If no sign labels are provided.
    """
    
    args = parse_args()
    args.signer = cleaned_name(args.signer)
    signs = load_signs(args)
    
    lang_dir = Path(args.out)/args.lang
    csv_path = lang_dir/"metadata.csv"
    lang_dir.mkdir(parents = True, exist_ok = True)
    counts: dict[str, int] = {}
    
    try:
        with (LandmarkExtractor(model_dir= args.model_dir, swap_handedness= args.swap_handedness) as extractor ,
              ThreadedCapture(source= args.camera, width= args.width, height= args.height, mirror= True) as cam):
            
            feed = Feed(cam, extractor)
            
            for sign in signs:
                sign_dir = lang_dir/args.signer/sign
                sign_dir.mkdir(parents = True, exist_ok = True)
                saved = len(list(sign_dir.glob("*.npy")))
                next = next_index(sign_dir)
                counts[sign] = saved
                
                while saved < args.reps:
                    header = f"{sign}  |  take {saved + 1}/{args.reps}  |  {args.signer}"
                    
                    try:
                        wait_for(feed, header, auto = args.auto)
                        countdown(feed, header, seconds= args.countdown)
                        data, coverage, fps = record_take(feed, header, n_frames = args.frames)
                    
                    except SkipSign:
                        break
                    
                    if coverage < args.min_coverage :
                        msg = f"Keep this take? ({coverage:.0%} hands found, {fps:.0f} FPS)"
                        keep = review(data, header, msg, GREEN, fps, args.playback_loops, (args.width, args.height))
                    
                    else:
                        msg = f"Saved ({coverage:.0%} hand coverage, {fps:.0f} FPS)"
                        keep = review(data, header, msg, GREEN, args.review_seconds)

                    if not keep:
                        continue  # discarded
                    
                    file_path = sign_dir / f"{next:04d}.npy" 
                    np.save(file_path, data)
                    
                    append_metadata(csv_path, {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "lang": args.lang, "signer": args.signer, "sign": sign,
                        "file": str(file_path.relative_to(lang_dir)), "frames": data.shape[0],
                        "feature_dim": data.shape[1], "hand_coverage": f"{coverage:.3f}",
                        "fps": f"{fps:.1f}", "lighting": args.lighting,
                        "background": args.background, "camera": args.camera,
                    })
                    saved += 1
                    next += 1
                    counts[sign] = saved
                    
    except QuitAll:
        print("Quit requested.")
    finally:
        cv2.destroyAllWindows()
 
    print(f"\nTakes on disk for signer '{args.signer}' (feature dim {FEATURE_DIM}):")
    for sign, n in counts.items():
        print(f"  {sign:<12} {n}/{args.reps}")
    print(f"Metadata: {csv_path}")

if __name__ == "__main__":
    main()