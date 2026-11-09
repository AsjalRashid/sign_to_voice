

"""Process a static sign-language image dataset into normalized landmarks.

Expected input:
    data/raw/static/asl/<split>/<letter>/<image>

Output:
    data/processed/static/asl/<split>/<letter>/<image>.npy

Each output file contains one normalized 147-dimensional landmark vector.

Detection strategy:
    1. Attempt detection on the original image.
    2. If detection fails, add configurable padding and retry.
    3. Normalize and save the extracted landmarks.
"""

from __future__ import annotations

import argparse
import csv
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from signvoice.vision.landmark import FEATURE_DIM, LandmarkExtractor
from signvoice.vision.normalise import (
    body_reference,
    hand_reference,
    normalise,
)


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

SPLITS = ("train", "test", "valid")


def parse_args() -> argparse.Namespace:
    """Parse dataset processing options."""

    parser = argparse.ArgumentParser(
        description="Extract and normalize landmarks from a static image dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/static/asl"),
        help="Root directory containing split and letter folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/static/asl"),
        help="Output directory for normalized landmark arrays.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/mediapipe",
        help="Directory containing MediaPipe model files.",
    )
    parser.add_argument(
        "--reference",
        choices=("body", "hand"),
        default="hand",
        help="Normalization reference. Hand is the default for static handshapes.",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror images before extraction (default: true).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=45,
        help="Fallback padding in pixels on each side (default: 45). Use 0 to disable.",
    )
    parser.add_argument(
        "--padding-mode",
        choices=("replicate", "constant"),
        default="replicate",
        help="Padding method: replicate edge pixels or add a constant border.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )

    args = parser.parse_args()

    if args.padding < 0:
        parser.error("--padding cannot be negative.")

    return args


def discover_images(split_dir: Path) -> list[Path]:
    """Find supported images recursively inside a dataset split."""

    return sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def add_padding(
    image: np.ndarray,
    pixels: int,
    mode: str = "replicate",
) -> np.ndarray:
    """Add padding around an image without modifying the original.

    Args:
        image (np.ndarray): Input image.
        pixels (int): Number of pixels to add on each side.
        mode (str): Padding method, either 'replicate' or 'constant'.

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
        pixels,
        pixels,
        pixels,
        pixels,
        border_type,
        value=(0, 0, 0),
    )


def normalize_image(
    image: np.ndarray,
    extractor: LandmarkExtractor,
    reference: str,
    padding: int = 45,
    padding_mode: str = "replicate",
) -> tuple[np.ndarray, bool]:
    """Extract and normalize one image using padding as a detection fallback.

    Detection is first attempted on the original image. If no hand is
    detected, padding is added and detection is attempted again.

    Args:
        image (np.ndarray): Input image, already mirrored if required.
        extractor (LandmarkExtractor): Initialized MediaPipe extractor.
        reference (str): Normalization reference, either 'hand' or 'body'.
        padding (int): Number of fallback padding pixels per side.
        padding_mode (str): Padding method.

    Returns:
        tuple[np.ndarray, bool]:
            - Normalized 147-dimensional float32 vector.
            - Whether fallback padding was used.

    Raises:
        ValueError: If neither detection attempt succeeds or normalization
            produces an invalid vector.
    """

    # First attempt: original image without padding.
    detection_image = image
    result = extractor.process(detection_image)
    used_padding = False

    # Second attempt: add padding only if the first attempt fails.
    if not result.hands_found and padding > 0:
        detection_image = add_padding(
            image,
            pixels=padding,
            mode=padding_mode,
        )

        result = extractor.process(detection_image)
        used_padding = True

    if not result.hands_found:
        raise ValueError(
            f"MediaPipe did not detect either hand "
            f"(original and {padding}px padding attempts failed)."
        )

    # Extract features.
    raw = np.asarray(
        result.features,
        dtype=np.float32,
    ).reshape(1, FEATURE_DIM)

    # Select normalization reference.
    reference_fn = (
        hand_reference if reference == "hand" else body_reference
    )

    # Use the dimensions of the image that produced the landmarks.
    height, width = detection_image.shape[:2]

    normalized = normalise(
        raw,
        reference=reference_fn,
        aspect=width / height if height > 0 else 1.0,
    )

    normalized = np.asarray(normalized, dtype=np.float32)

    if normalized.shape != (1, FEATURE_DIM):
        raise ValueError(
            f"Unexpected normalized shape: {normalized.shape}"
        )

    if not np.isfinite(normalized).all():
        raise ValueError("Normalized vector contains NaN or infinity.")

    return normalized[0], used_padding


def save_vector(vector: np.ndarray, output_path: Path) -> None:
    """Save one vector atomically as a NumPy .npy file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".npy.tmp")

    with temporary_path.open("wb") as file:
        np.save(file, vector)

    temporary_path.replace(output_path)


def process_dataset(args: argparse.Namespace) -> None:
    """Traverse all splits and classes, extracting normalized landmarks."""

    input_root = args.input
    output_root = args.out

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {input_root}"
        )

    processed = 0
    skipped = 0
    failed = 0
    fallback_attempts = 0
    fallback_recovered = 0

    failure_records = []

    print(f"Input:         {input_root}")
    print(f"Output:        {output_root}")
    print(f"Reference:     {args.reference}")
    print(f"Mirror:        {args.mirror}")
    print(f"Fallback:      {args.padding}px")
    print(f"Padding mode:  {args.padding_mode}")

    # Reuse one MediaPipe extractor for the entire dataset.
    with LandmarkExtractor(
        model_dir=args.model_dir,
        swap_handedness=False,
        mode="image",
    ) as extractor:

        for split in SPLITS:
            split_dir = input_root / split

            if not split_dir.is_dir():
                print(f"\nMissing split: {split_dir}")
                continue

            for letter in string.ascii_uppercase:
                letter_dir = split_dir / letter

                if not letter_dir.is_dir():
                    continue

                images = discover_images(letter_dir)

                print(
                    f"\n[{split}/{letter}] "
                    f"Found {len(images)} images."
                )

                for image_path in images:

                    # Preserve the original directory structure.
                    relative_path = image_path.relative_to(input_root)

                    output_path = (
                        output_root / relative_path
                    ).with_suffix(".npy")

                    if output_path.exists() and not args.overwrite:
                        skipped += 1
                        continue

                    try:
                        image = cv2.imread(
                            str(image_path),
                            cv2.IMREAD_COLOR,
                        )

                        if image is None:
                            raise ValueError("Could not decode image.")

                        # Mirror before detection so both attempts use
                        # exactly the same image orientation.
                        if args.mirror:
                            image = cv2.flip(image, 1)

                        vector, used_padding = normalize_image(
                            image=image,
                            extractor=extractor,
                            reference=args.reference,
                            padding=args.padding,
                            padding_mode=args.padding_mode,
                        )

                        save_vector(vector, output_path)

                        processed += 1

                        if used_padding:
                            fallback_attempts += 1
                            fallback_recovered += 1

                    except Exception as exc:
                        failed += 1

                        failure_records.append({
                            "image": str(image_path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })

                        print(f"FAILED: {image_path}")
                        print(f"        {type(exc).__name__}: {exc}")

    # Write a fresh failure report, including a header if there are
    # no failures. This prevents stale failure records from remaining.
    failure_log = output_root / "failures.csv"
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    with failure_log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "error_type", "error"],
        )
        writer.writeheader()
        writer.writerows(failure_records)

    print("\n========== DATASET PROCESSING SUMMARY ==========")
    print(f"Processed:          {processed}")
    print(f"Skipped:            {skipped}")
    print(f"Failed:             {failed}")
    print(f"Fallback recovered: {fallback_recovered}")
    print(f"Failure log:        {failure_log}")

    if failure_records:
        error_counts = Counter(
            record["error_type"] for record in failure_records
        )

        print("\nFailure breakdown:")

        for error_type, count in error_counts.most_common():
            print(f"{error_type}: {count}")

    print("================================================")


def main() -> None:
    """Run the static dataset preprocessing pipeline."""

    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()

