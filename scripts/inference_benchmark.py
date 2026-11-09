#!/usr/bin/env python3
"""
benchmark_inference.py - how fast is the capture + landmark loop WITHOUT Streamlit?

Runs the same pieces the app uses (ThreadedCapture + LandmarkExtractor) in a bare Python
loop: no browser, no JPEG encoding, no widgets. Compare its numbers with the Vision
section's "Inference" and "Processing FPS" to see whether the models/CPU or the Streamlit
layer is the bottleneck.

It runs a guided 2 x 2 test, so that two suspects are checked in one go:
    pose model on / off      (what does the pose model cost?)
    hands in view / not      (is the hand detector slower when it has nothing to track?)

Run (from the repo root, with your virtual environment active):
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --seconds 15 --draw --csv bench.csv
    python scripts/benchmark_inference.py --video samples/demo.mp4 --unguided   # no webcam

For comparable numbers: plug the laptop in, use the best-performance power mode, close the
browser and other heavy apps, keep the lighting and your distance the same each time, and
use the same resolution as the app (640 x 480 by default).

What the columns mean:
    inference   time inside LandmarkExtractor.process() only (the same number the app shows)
    draw        copy of the frame + drawing the landmarks (only with --draw)
    age at use  how old the frame was when the loop picked it up
    age at end  how old it was after processing (+ drawing); the app's "frame age" is like this
    FPS         frames processed per second of wall-clock time
    skipped     camera frames replaced by newer ones before the loop got to them
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    from signvoice.vision.capture import CaptureError, ThreadedCapture
except ModuleNotFoundError:   # allow running without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from signvoice.vision.capture import CaptureError, ThreadedCapture
from signvoice.vision.landmark import LandmarkExtractor, draw_landmarks

POSE_MODEL = "pose_landmarker_lite.task"
CONFIGS = {"pose": POSE_MODEL, "nopose": None}
POSE_INTERVALS = (1, 2, 3)


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def run_phase(cam, extractor, seconds: float, draw: bool, state: dict) -> dict:
    """Process frames for `seconds`. Returns per-frame rows plus phase totals."""
    rows = []
    captured_before = cam.frames_captured
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        frame = cam.read(after_id=state["last_id"], timeout=1.0)
        if frame is None:
            print("  (no new frame: the source ended or stalled; ending this phase early)")
            break
        state["last_id"] = frame.id
        age_use = frame.age_ms
        result = extractor.process(frame.image)
        draw_ms = 0.0
        if draw:
            t0 = time.perf_counter()
            view = frame.image.copy()
            draw_landmarks(view, result)
            draw_ms = (time.perf_counter() - t0) * 1000
        rows.append({"inference_ms": result.inference_ms, "draw_ms": draw_ms, "age_use_ms": age_use,
                     "age_end_ms": frame.age_ms, "hands": bool(result.hands_found)})
    elapsed = time.perf_counter() - t_start
    captured = cam.frames_captured - captured_before
    return {"rows": rows, "elapsed": elapsed, "skipped": max(0, captured - len(rows))}


def summarise(phase: dict) -> dict:
    rows = phase["rows"]
    col = lambda k: [r[k] for r in rows]
    return {
        "frames": len(rows),
        "fps": len(rows) / phase["elapsed"] if phase["elapsed"] else 0.0,
        "inf_med": pct(col("inference_ms"), 50), "inf_p90": pct(col("inference_ms"), 90),
        "inf_max": max(col("inference_ms"), default=float("nan")),
        "draw_med": pct(col("draw_ms"), 50),
        "age_use": pct(col("age_use_ms"), 50),
        "age_end_med": pct(col("age_end_ms"), 50), "age_end_p90": pct(col("age_end_ms"), 90),
        "hands_share": float(np.mean(col("hands"))) if rows else float("nan"),
        "skipped": phase["skipped"],
    }


def countdown(label: str, seconds: int) -> None:
    for s in range(seconds, 0, -1):
        print(f"  {label} ... starting in {s}", end="\r", flush=True)
        time.sleep(1)
    print(" " * 60, end="\r")


def print_table(results: list[tuple[str, dict]], draw: bool) -> None:
    lines = [
        ("frames processed", "frames", "{:d}"), ("FPS", "fps", "{:.1f}"),
        ("inference median (ms)", "inf_med", "{:.1f}"), ("inference p90 (ms)", "inf_p90", "{:.1f}"),
        ("inference max (ms)", "inf_max", "{:.1f}"),
        ("draw median (ms)", "draw_med", "{:.1f}"),
        ("frame age at use, median (ms)", "age_use", "{:.1f}"),
        ("frame age at end, median (ms)", "age_end_med", "{:.1f}"),
        ("frame age at end, p90 (ms)", "age_end_p90", "{:.1f}"),
        ("frames with hands found", "hands_share", "{:.0%}"), ("frames skipped", "skipped", "{:d}"),
    ]
    if not draw:
        lines = [l for l in lines if l[1] != "draw_med"]
    width = max(len(l[0]) for l in lines)
    names = [name for name, _ in results]
    colw = max(14, max(len(n) for n in names) + 2)
    print("\n" + " " * width + "".join(f"{n:>{colw}}" for n in names))
    for label, key, fmt in lines:
        print(f"{label:<{width}}" + "".join(f"{fmt.format(s[key]):>{colw}}" for _, s in results))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bare-loop capture + landmark benchmark (no Streamlit)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="use a video file instead of a webcam")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seconds", type=float, default=10.0, help="length of each phase")
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=["pose", "nopose"])
    p.add_argument(
        "--pose-intervals", nargs="+", type=int, choices=POSE_INTERVALS,
        default=[2, 3],
        help="pose inference interval(s): 1=every frame, 2=every 2nd, 3=every 3rd "
             "(used for the pose config; default: 2 3)",
    )
    p.add_argument("--unguided", action="store_true",
                   help="one phase per config, no prompts (use with --video, or to move freely)")
    p.add_argument("--draw", action="store_true", help="also time copying and drawing the landmarks")
    p.add_argument("--warmup", type=int, default=30, help="frames discarded before measuring")
    p.add_argument("--countdown", type=int, default=3, help="seconds to get ready before each phase")
    p.add_argument("--model-dir", default="models/mediapipe")
    p.add_argument("--csv", help="write every measured frame to this CSV file")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    source = args.video if args.video else args.camera
    try:
        cam = ThreadedCapture(source, width=args.width, height=args.height).start()
    except CaptureError as exc:
        sys.exit(f"Could not open the source: {exc}")

    first = cam.read(after_id=0)
    import cv2
    import mediapipe as mp
    count0, t0 = cam.frames_captured, time.perf_counter()
    time.sleep(1.0)                                     # measure the delivery rate over a full second
    camera_fps = (cam.frames_captured - count0) / (time.perf_counter() - t0)
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}, "
          f"{os.cpu_count()} logical CPUs | opencv {cv2.__version__}, mediapipe {mp.__version__}")
    print(f"Frames are {first.image.shape[1]}x{first.image.shape[0]}, camera delivers ~{camera_fps:.0f} FPS")

    phases = [("", None)] if args.unguided else [("hands in view", True), ("hands out of view", False)]
    results, csv_rows = [], []
    state = {"last_id": first.id}
    try:
        for config in args.configs:
            intervals = args.pose_intervals if config == "pose" else [1]
            for frame_interval in intervals:
                run_name = config if config == "nopose" else f"{config} (every {frame_interval} frame{'s' if frame_interval != 1 else ''})"
                print(f"\n=== {run_name} ===")
                with LandmarkExtractor(
                    model_dir=args.model_dir,
                    pose_model=CONFIGS[config],
                    frame_interval=frame_interval,
                ) as extractor:
                    for _ in range(args.warmup):  # models and tracker settle, not measured
                        frame = cam.read(after_id=state["last_id"], timeout=1.0)
                        if frame is None:
                            break
                        state["last_id"] = frame.id
                        extractor.process(frame.image)
                    for label, hands_in_view in phases:
                        if hands_in_view is not None:
                            what = "keep BOTH hands in view, signing or moving slowly" if hands_in_view \
                                else "take your hands out of the camera view"
                            print(f"Phase: {label}: {what}")
                            countdown(label, args.countdown)
                        phase = run_phase(cam, extractor, args.seconds, args.draw, state)
                        name = run_name + (f", {label}" if label else "")
                        results.append((name, summarise(phase)))
                        csv_rows += [
                            {"config": config, "pose_frame_interval": frame_interval,
                             "phase": label or "all", **r}
                            for r in phase["rows"]
                        ]
    finally:
        cam.stop()

    print_table(results, args.draw)
    if args.csv:
        if csv_rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nPer-frame data written to {args.csv}")
        else:
            print("\nNo benchmark rows were collected; CSV was not written.")


if __name__ == "__main__":
    main()