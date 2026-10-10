"""
Splitter module for Brain Tumor Detection and Classification.

Partitions path/label pairs into non-overlapping train/val/test sets
using stratified splitting via scikit-learn.
"""

import logging
from typing import Optional

from sklearn.model_selection import train_test_split

from src.data_loader import CLASS_NAMES

logger = logging.getLogger(__name__)

# Type alias for readability
Sample = tuple[str, int]


def stratified_train_val_split(
    samples: list[Sample], val_ratio: float = 0.15, random_seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    """Split the supplied Training partition into train and validation sets.

    The dataset's separately supplied Testing partition should be passed
    directly to final evaluation and must not be included here.
    """
    if not samples:
        raise ValueError("Dataset is empty; cannot split")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    labels = [label for _, label in samples]
    train, val = train_test_split(
        samples, test_size=val_ratio, stratify=labels, random_state=random_seed
    )
    return train, val


def stratified_split(
    samples: list[Sample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """
    Partition *samples* into non-overlapping train, validation, and test subsets
    using stratified splitting.

    The split is performed in two stages:

    * **Stage 1** — Split off the test subset (``test_ratio`` fraction) from
      the full dataset, stratified by class label.
    * **Stage 2** — Split the remaining samples into train and val subsets
      (stratified), where val takes ``val_ratio / (train_ratio + val_ratio)``
      of the remaining pool.

    After splitting, total and per-class counts for each subset are logged via
    ``logging.INFO`` and also printed to stdout.

    Parameters
    ----------
    samples:
        List of ``(absolute_file_path, class_index)`` tuples as returned by
        :func:`src.data_loader.discover_dataset`.
    train_ratio:
        Fraction of the full dataset reserved for training (default 0.70).
    val_ratio:
        Fraction of the full dataset reserved for validation (default 0.15).
    test_ratio:
        Fraction of the full dataset reserved for testing (default 0.15).
    random_seed:
        Seed for :func:`sklearn.model_selection.train_test_split` to ensure
        reproducible splits.

    Returns
    -------
    tuple[list[Sample], list[Sample], list[Sample]]
        A ``(train_samples, val_samples, test_samples)`` tuple where each
        element is a list of ``(path, class_index)`` tuples.

    Raises
    ------
    ValueError
        If *samples* is empty.
    ValueError
        If any class has fewer than 2 samples (insufficient for stratification).
    """
    if not samples:
        raise ValueError("Dataset is empty; cannot split")

    # Validate that every class has at least 2 samples (required for stratification)
    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for _, class_index in samples:
        if 0 <= class_index < len(CLASS_NAMES):
            class_counts[CLASS_NAMES[class_index]] += 1

    for cls_name, count in class_counts.items():
        # Only check classes that appear in the dataset
        if count > 0 and count < 2:
            raise ValueError(
                f"Too few samples in class '{cls_name}' for stratified split "
                f"(found {count}, need at least 2)"
            )

    labels = [label for _, label in samples]

    # Stage 1: split off the test set
    minimum_partition = len(class_counts)
    test_size = max(int(len(samples) * test_ratio + 0.999999), minimum_partition)
    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed,
    )

    # Stage 2: split remaining into train and val
    # val_ratio relative to the remaining pool
    remaining_total = train_ratio + val_ratio
    val_size_relative = val_ratio / remaining_total
    val_size = max(
        int(len(train_val_samples) * val_size_relative + 0.999999), minimum_partition
    )

    train_val_labels = [label for _, label in train_val_samples]
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_seed,
    )

    # Log and print per-subset statistics
    _log_split_info(train_samples, val_samples, test_samples)

    return train_samples, val_samples, test_samples


def assert_no_leakage(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
) -> None:
    """
    Assert that no file path appears in more than one split.

    Checks all three pairwise intersections (train∩val, train∩test, val∩test).
    Raises :class:`AssertionError` listing the total count of overlapping paths
    if any overlap is found.

    Parameters
    ----------
    train:
        Training split as a list of ``(path, class_index)`` tuples.
    val:
        Validation split as a list of ``(path, class_index)`` tuples.
    test:
        Test split as a list of ``(path, class_index)`` tuples.

    Raises
    ------
    AssertionError
        If any file path appears in more than one subset.  The message
        reports the count of overlapping paths.
    """
    train_paths = {path for path, _ in train}
    val_paths = {path for path, _ in val}
    test_paths = {path for path, _ in test}

    overlapping: set[str] = set()
    overlapping |= train_paths & val_paths
    overlapping |= train_paths & test_paths
    overlapping |= val_paths & test_paths

    if overlapping:
        raise AssertionError(
            f"{len(overlapping)} file paths appear in multiple splits"
        )


def get_split_info(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
) -> dict:
    """
    Return a summary dictionary with total and per-class counts for each split.

    Parameters
    ----------
    train:
        Training split as a list of ``(path, class_index)`` tuples.
    val:
        Validation split as a list of ``(path, class_index)`` tuples.
    test:
        Test split as a list of ``(path, class_index)`` tuples.

    Returns
    -------
    dict
        A nested dictionary with the structure::

            {
                "train": {"total": int, "per_class": {class_name: int, ...}},
                "val":   {"total": int, "per_class": {class_name: int, ...}},
                "test":  {"total": int, "per_class": {class_name: int, ...}},
            }
    """
    return {
        "train": _subset_info(train),
        "val": _subset_info(val),
        "test": _subset_info(test),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _subset_info(samples: list[Sample]) -> dict:
    """Return total count and per-class count dict for a single split."""
    per_class: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for _, class_index in samples:
        if 0 <= class_index < len(CLASS_NAMES):
            per_class[CLASS_NAMES[class_index]] += 1
    return {"total": len(samples), "per_class": per_class}


def _log_split_info(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
) -> None:
    """Log and print total and per-class sample counts for each split."""
    info = get_split_info(train, val, test)

    for split_name in ("train", "val", "test"):
        subset = info[split_name]
        total = subset["total"]
        per_class = subset["per_class"]

        msg_lines = [f"[Split: {split_name}] total={total}"]
        for class_name in CLASS_NAMES:
            count = per_class[class_name]
            msg_lines.append(f"  {class_name}: {count}")

        full_msg = "\n".join(msg_lines)
        logger.info(full_msg)
        print(full_msg)
