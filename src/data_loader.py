"""
Data loader module for Brain Tumor Detection and Classification.

Responsible for discovering image file paths and integer class labels
from the dataset directory tree.
"""

import logging
import os

# Ordered class names; index in this list == class_index
CLASS_NAMES: list[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Supported image file extensions (lowercase)
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
# The Kaggle archive names this class ``notumor``; keep ``no_tumor`` as the
# canonical model label while accepting both spellings on disk.
_CLASS_DIRECTORY_ALIASES: dict[str, tuple[str, ...]] = {
    "glioma": ("glioma",),
    "meningioma": ("meningioma",),
    "no_tumor": ("no_tumor", "notumor", "no-tumor"),
    "pituitary": ("pituitary",),
}

logger = logging.getLogger(__name__)


def discover_dataset(data_dir: str) -> list[tuple[str, int]]:
    """
    Walk *data_dir* expecting subdirectories named after CLASS_NAMES.

    Returns a list of (absolute_file_path, class_index) tuples where
    class_index matches the position of the class name in CLASS_NAMES.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist, or if any of the expected class
        subdirectories is missing inside *data_dir*.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{data_dir}'"
        )

    samples: list[tuple[str, int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        existing_dirs = [
            os.path.join(data_dir, dirname)
            for dirname in _CLASS_DIRECTORY_ALIASES[class_name]
            if os.path.isdir(os.path.join(data_dir, dirname))
        ]
        if not existing_dirs:
            raise FileNotFoundError(
                f"Expected class subdirectory for '{class_name}' not found in "
                f"'{data_dir}'. Accepted folder names: "
                f"{', '.join(_CLASS_DIRECTORY_ALIASES[class_name])}."
            )
        if len(existing_dirs) > 1:
            raise ValueError(
                f"Multiple folder aliases found for class '{class_name}' in "
                f"'{data_dir}': {', '.join(os.path.basename(p) for p in existing_dirs)}. "
                "Keep only one folder for this class to avoid duplicate samples."
            )
        class_dir = existing_dirs[0]

        class_files: list[str] = []
        for entry in os.scandir(class_dir):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in _IMAGE_EXTENSIONS:
                    class_files.append(os.path.abspath(entry.path))

        logger.info(
            "Class '%s' (index %d): found %d image(s) in '%s'",
            class_name,
            class_index,
            len(class_files),
            class_dir,
        )

        for file_path in sorted(class_files):
            samples.append((file_path, class_index))

    return samples


def get_class_distribution(samples: list[tuple[str, int]]) -> dict[str, int]:
    """
    Return a dict mapping class name to sample count.

    Parameters
    ----------
    samples:
        List of (file_path, class_index) tuples as returned by
        :func:`discover_dataset`.

    Returns
    -------
    dict[str, int]
        Keys are class names from CLASS_NAMES; values are the number of
        samples for each class.  Classes with zero samples are included
        with a count of 0.
    """
    distribution: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for _, class_index in samples:
        distribution[CLASS_NAMES[class_index]] += 1
    return distribution


def print_dataset_info(data_dir: str) -> None:
    """
    Convenience function that discovers the dataset, computes class
    distribution, and prints a summary table to stdout.

    Parameters
    ----------
    data_dir:
        Root directory of the dataset (e.g. ``"data/Training"``).
    """
    samples = discover_dataset(data_dir)
    distribution = get_class_distribution(samples)

    total = len(samples)
    print(f"\nDataset: {os.path.abspath(data_dir)}")
    print(f"Total samples: {total}")
    print(f"{'Class':<20} {'Count':>8} {'Percentage':>12}")
    print("-" * 42)
    for class_name in CLASS_NAMES:
        count = distribution[class_name]
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"{class_name:<20} {count:>8} {pct:>11.1f}%")
    print("-" * 42)
    print(f"{'TOTAL':<20} {total:>8} {'100.0%':>12}")
    print()
