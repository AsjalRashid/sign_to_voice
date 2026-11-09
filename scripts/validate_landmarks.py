
"""Validate extracted MediaPipe landmark vectors stored as .npy files.

Checks:
    - Expected vector shape: (147,)
    - Expected data type: float32
    - No NaN or infinite values
    - Flags all-zero vectors as suspicious

Generates:
    - A CSV report containing the validation result for every file.
    - A terminal summary grouped by dataset split and class.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_SHAPE = (147,)
EXPECTED_DTYPE = np.dtype(np.float32)

SPLITS = {"train", "test", "valid"}


def validate_file(path: Path) -> dict:
    """Load and validate a single NumPy landmark vector.

    Args:
        path (Path): Path to the .npy file.

    Returns:
        dict: Validation results, including issues and vector statistics.
    """
    result = {
        "path": str(path),
        "status": "VALID",
        "issues": [],
        "shape": "",
        "dtype": "",
        "min": "",
        "max": "",
    }

    try:
        vector = np.load(path, allow_pickle=False)

    except Exception as exc:
        result["status"] = "INVALID"
        result["issues"].append(
            f"Load error: {type(exc).__name__}: {exc}"
        )
        return result

    result["shape"] = str(vector.shape)
    result["dtype"] = str(vector.dtype)

    # Check vector shape.
    if vector.shape != EXPECTED_SHAPE:
        result["issues"].append(
            f"Incorrect shape: expected {EXPECTED_SHAPE}, "
            f"got {vector.shape}"
        )

    # Check data type.
    if vector.dtype != EXPECTED_DTYPE:
        result["issues"].append(
            f"Incorrect dtype: expected {EXPECTED_DTYPE}, "
            f"got {vector.dtype}"
        )

    # Check whether the array contains numeric values.
    if not np.issubdtype(vector.dtype, np.number):
        result["issues"].append("Array contains non-numeric values")

    else:
        # Check for NaN and infinite values.
        finite_mask = np.isfinite(vector)

        if not np.all(finite_mask):
            nan_count = int(np.isnan(vector).sum())
            inf_count = int(np.isinf(vector).sum())

            result["issues"].append(
                f"Non-finite values: {nan_count} NaN, "
                f"{inf_count} Inf"
            )

        # Record statistics using finite values only.
        finite_values = vector[finite_mask]

        if finite_values.size:
            result["min"] = float(finite_values.min())
            result["max"] = float(finite_values.max())

        # An all-zero vector is suspicious but not automatically invalid.
        if vector.size > 0 and np.all(vector == 0):
            result["issues"].append("WARNING: vector contains only zeros")

    # Separate warnings from actual validation failures.
    has_error = any(
        not issue.startswith("WARNING:")
        for issue in result["issues"]
    )

    if has_error:
        result["status"] = "INVALID"
    elif result["issues"]:
        result["status"] = "WARNING"

    return result


def get_split_and_class(path: Path, root: Path) -> tuple[str, str]:
    """Extract dataset split and class label from a file path."""
    relative = path.relative_to(root)
    parts = relative.parts

    split = parts[0] if len(parts) > 0 and parts[0] in SPLITS else "unknown"
    label = parts[1] if len(parts) > 1 else "unknown"

    return split, label


def print_summary(results: list[dict], root: Path) -> None:
    """Print overall and per-split/class validation statistics."""
    total = len(results)

    counts = Counter(result["status"] for result in results)

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Dataset:  {root}")
    print(f"Total:    {total}")
    print(f"Valid:    {counts['VALID']}")
    print(f"Warnings: {counts['WARNING']}")
    print(f"Invalid:  {counts['INVALID']}")

    print("\n========== SPLIT / CLASS SUMMARY ==========")

    grouped = defaultdict(Counter)

    for result in results:
        path = Path(result["path"])
        split, label = get_split_and_class(path, root)
        grouped[(split, label)][result["status"]] += 1

    current_split = None

    for (split, label), counts in sorted(grouped.items()):
        if split != current_split:
            current_split = split
            print(f"\n[{split.upper()}]")

        total_class = sum(counts.values())

        print(
            f"  {label:>3}: "
            f"{total_class:4} files | "
            f"Valid: {counts['VALID']:4} | "
            f"Warnings: {counts['WARNING']:3} | "
            f"Invalid: {counts['INVALID']:3}"
        )

    print("========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate extracted landmark .npy files."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/static/asl"),
        help="Root directory containing processed landmark vectors.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default= Path('reports') / 'landmark_validation.csv',
        help="CSV report output path.",
    )

    args = parser.parse_args()

    root = args.input

    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")

    files = sorted(root.rglob("*.npy"))

    if not files:
        print(f"No .npy files found in: {root}")
        return

    print(f"Validating {len(files)} files...\n")

    results = [
        validate_file(path)
        for path in files
    ]

    print_summary(results, root)

    report_path = args.report or root / "validation_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "path",
                "status",
                "shape",
                "dtype",
                "min",
                "max",
                "issues",
            ],
        )

        writer.writeheader()

        for result in results:
            row = result.copy()
            row["issues"] = "; ".join(row["issues"])
            writer.writerow(row)

    print(f"CSV report saved to: {report_path}")


if __name__ == "__main__":
    main()
