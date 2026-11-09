
#!/usr/bin/env python3
"""
Create a balanced list of images that were successfully detected
during the original, unpadded dataset processing.

Selects up to 300 images per class, excluding previously failed images.
"""

import argparse
import random
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = list("abcdefghijklmnopqrstuvwxyz")


def load_failed_images(path: Path) -> set[str]:
    """Load failed image paths from a text file."""

    return {
        str(Path(line.strip())).replace("\\", "/").lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def create_successful_list(
    dataset_dir: Path,
    failure_file: Path,
    output_file: Path,
    samples_per_class: int = 300,
    seed: int = 42,
) -> None:

    failed = load_failed_images(failure_file)
    rng = random.Random(seed)

    selected = []
    print("\n========== CLASS SUMMARY ==========")

    for letter in CLASSES:
        class_dir = dataset_dir / "train" / letter

        if not class_dir.is_dir():
            print(f"{letter.upper()}: directory not found")
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        successful = [
            image for image in images
            if str(image).replace("\\", "/").lower() not in failed
        ]

        rng.shuffle(successful)

        chosen = successful[:samples_per_class]
        selected.extend(chosen)

        print(
            f"{letter.upper()}: "
            f"{len(images)} total | "
            f"{len(images) - len(successful)} failed | "
            f"{len(successful)} successful | "
            f"{len(chosen)} selected"
        )

        if len(chosen) < samples_per_class:
            print(
                f"    WARNING: Only {len(chosen)} successful "
                f"images available; wanted {samples_per_class}."
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        "\n".join(str(path) for path in selected) + "\n",
        encoding="utf-8",
    )

    print("\n========== FINAL SUMMARY ==========")
    print(f"Classes:  {len(CLASSES)}")
    print(f"Selected: {len(selected)}")
    print(f"Output:   {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/static/asl"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("failures.txt"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("successful_images.txt"),
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_successful_list(
        dataset_dir=args.input,
        failure_file=args.failures,
        output_file=args.out,
        samples_per_class=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
