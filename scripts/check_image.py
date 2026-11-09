
#!/usr/bin/env python3
"""
check_image.py - run the landmark extractor on images and visualise the result.

Examples:
    python scripts/check_image.py path/to/image.jpg

    python scripts/check_image.py path/to/folder --limit 10 --show

    python scripts/check_image.py --list failures.txt

    python scripts/check_image.py --list failures.txt --padding 50

    python scripts/check_image.py --list failures.txt --padding 75 --show

    python scripts/check_image.py --list failures.txt --padding 0

By default:
    - Images are mirrored before detection (selfie view).
    - 50 pixels of padding are added to all four sides.
    - Padding uses replicated edge pixels.
    - Annotated images are saved to check_out/.

Use --no-mirror to disable mirroring.
Use --padding 0 to disable padding.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from signvoice.vision.landmark import LandmarkExtractor, draw_landmarks
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from signvoice.vision.landmark import LandmarkExtractor, draw_landmarks


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_SIZE = 640


def find_images(folder: Path, limit: int) -> list[Path]:
    """Find image files directly inside a folder."""
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return files[:limit] if limit > 0 else files


def gather_images(
    entries: list[str],
    list_file: str | None,
    limit: int,
) -> list[Path]:
    """Collect image paths from arguments, folders, wildcards, and list files."""

    entries = list(entries)

    if list_file:
        lines = Path(list_file).read_text(encoding="utf-8").splitlines()
        entries += [
            line.strip().strip('"')
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]

    images: list[Path] = []

    for entry in entries:
        exists = Path(entry).exists()
        wildcard = not exists and any(c in entry for c in "*?[")

        matches = (
            [Path(m) for m in sorted(glob.glob(entry))]
            if wildcard else [Path(entry)]
        )

        if not matches or not any(m.exists() for m in matches):
            print(f"Nothing found for: {entry}")
            continue

        for match in matches:
            if match.is_dir():
                images += find_images(match, limit)
            elif match.is_file():
                images.append(match)

    return list(dict.fromkeys(images))


def unique_name(stem: str, used: set[str]) -> str:
    """Generate an output filename that cannot overwrite an earlier one."""

    name, n = f"{stem}_landmarks.png", 2

    while name in used:
        name, n = f"{stem}_{n}_landmarks.png", n + 1

    used.add(name)
    return name


def read_image(path: Path) -> np.ndarray | None:
    """Read an image, including paths containing non-ASCII characters."""

    data = np.fromfile(str(path), dtype=np.uint8)

    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def add_padding(
    image: np.ndarray,
    pixels: int,
    mode: str = "replicate",
) -> np.ndarray:
    """
    Add padding to all four sides of an image.

    Args:
        image (np.ndarray): Input BGR image.
        pixels (int): Padding width in pixels.
        mode (str): 'replicate' extends edge pixels;
                    'constant' adds a black border.

    Returns:
        np.ndarray: Padded image.
    """

    if pixels <= 0:
        return image

    border_type = (
        cv2.BORDER_REPLICATE
        if mode == "replicate"
        else cv2.BORDER_CONSTANT
    )

    return cv2.copyMakeBorder(
        image,
        top=pixels,
        bottom=pixels,
        left=pixels,
        right=pixels,
        borderType=border_type,
        value=(0, 0, 0),
    )


def describe(result) -> str:
    """Format the landmark detection results."""

    def yes(flag):
        return "yes" if flag else "no"

    return (
        f"left hand {yes(result.left_found)}, "
        f"right hand {yes(result.right_found)}, "
        f"pose {yes(result.pose_found)} "
        f"({result.inference_ms:.0f} ms)"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the landmark extractor on images and visualise results."
    )

    p.add_argument(
        "paths",
        nargs="*",
        help="Image files, folders, and/or wildcard patterns.",
    )

    p.add_argument(
        "--list",
        dest="list_file",
        help="Text file containing one image or folder path per line.",
    )

    p.add_argument("--out", default="check_out")
    p.add_argument("--show", action="store_true")

    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum images per folder (0 = all).",
    )

    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--swap-handedness", action="store_true")
    p.add_argument("--model-dir", default="models/mediapipe")

    p.add_argument(
        "--padding",
        type=int,
        default=45,
        help="Padding in pixels on each side (default: 45; use 0 to disable).",
    )

    p.add_argument(
        "--padding-mode",
        choices=("replicate", "constant"),
        default="replicate",
        help="Padding style: replicate edge pixels or add a black border.",
    )

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.padding < 0:
        sys.exit("Padding cannot be negative.")

    if not args.paths and not args.list_file:
        sys.exit("Give at least one image, folder, wildcard, or --list file.txt")

    images = gather_images(args.paths, args.list_file, args.limit)

    if not images:
        sys.exit("No images found.")

    kwargs = {
        "model_dir": args.model_dir,
        "swap_handedness": args.swap_handedness,
        "mode": "image",
    }

    if args.no_pose:
        kwargs["pose_model"] = None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hands_seen = 0
    processed = 0
    used_names: set[str] = set()

    print(f"Images:       {len(images)}")
    print(f"Padding:      {args.padding}px")
    print(f"Padding mode: {args.padding_mode}")
    print(f"Mirror:       {not args.no_mirror}")
    print()

    with LandmarkExtractor(**kwargs) as extractor:

        for number, image_path in enumerate(images, 1):

            image = read_image(image_path)

            if image is None:
                print(f"[{number}/{len(images)}] {image_path.name}: could not read image")
                continue

            original_height, original_width = image.shape[:2]

            # First attempt: detect using the original image.
            result = extractor.process(image)

            # Preserve the original image for annotation if detection succeeds.
            original_image = image.copy()

            if not args.no_mirror:
                original_image = cv2.flip(original_image, 1)

            result = extractor.process(original_image)
            display_image = original_image.copy()

            if not result.hands_found and args.padding > 0:
                padded_image = add_padding(
                    original_image,
                    pixels=args.padding,
                    mode=args.padding_mode,
                )

                result = extractor.process(padded_image)
                display_image = padded_image

            # Mirror the image for display and detection consistency.
            if not args.no_mirror:
                display_image = cv2.flip(display_image, 1)
                
            processed += 1
            hands_seen += bool(result.hands_found)

            status = "DETECTED" if result.hands_found else "NOT DETECTED"

            print(
                f"[{number}/{len(images)}] {image_path.name} "
                f"({original_width}x{original_height} -> "
                f"{display_image.shape[1]}x{display_image.shape[0]}): "
                f"{status} | {describe(result)}"
                
            )

            scale = max(1.0, DISPLAY_SIZE / max(display_image.shape[:2]))

            view = (
                cv2.resize(
                    display_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                if scale > 1
                else display_image.copy()
            )

            draw_landmarks(view, result)

            label = "HANDS FOUND" if result.hands_found else "NO HANDS FOUND"

            cv2.putText(
                view,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                view,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 220, 80) if result.hands_found else (60, 60, 255),
                2,
                cv2.LINE_AA,
            )

            saved = out_dir / unique_name(image_path.stem, used_names)

            ok, encoded = cv2.imencode(".png", view)

            if ok:
                encoded.tofile(str(saved))

            if args.show:
                cv2.imshow("check_image", view)

                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break

    if args.show:
        cv2.destroyAllWindows()

    print("\n========== SUMMARY ==========")
    print(f"Processed:     {processed}")
    print(f"Hands found:   {hands_seen}")
    print(f"Not detected:  {processed - hands_seen}")

    if processed:
        print(f"Detection rate: {hands_seen / processed * 100:.2f}%")

    print(f"Annotated images saved to: {out_dir}/")


if __name__ == "__main__":
    main()
