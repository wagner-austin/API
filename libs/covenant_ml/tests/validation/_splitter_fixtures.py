"""Shared fixtures and helpers for test_splitter splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation import (
    CVSplitInfo,
)


def _make_intp_array(values: tuple[int, ...]) -> NDArray[np.intp]:
    """Create intp array from tuple of ints.

    Args:
        values: Tuple of integer values.

    Returns:
        Array of intp dtype.
    """
    result: NDArray[np.intp] = np.zeros(len(values), dtype=np.intp)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _make_labels(n_pos: int, n_neg: int) -> NDArray[np.int64]:
    """Create binary label array with specified class counts.

    Args:
        n_pos: Number of positive samples (label=1).
        n_neg: Number of negative samples (label=0).

    Returns:
        Label array with n_pos ones followed by n_neg zeros.
    """
    pos: NDArray[np.int64] = np.ones(n_pos, dtype=np.int64)
    neg: NDArray[np.int64] = np.zeros(n_neg, dtype=np.int64)
    result: NDArray[np.int64] = np.concatenate([pos, neg])
    return result


def _make_features(n_samples: int, n_features: int, seed: int = 42) -> NDArray[np.float64]:
    """Create feature matrix with reproducible random values.

    Args:
        n_samples: Number of samples (rows).
        n_features: Number of features (columns).
        seed: Random seed for reproducibility.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def _count_class(y: NDArray[np.int64], class_value: int) -> int:
    """Count occurrences of a class value.

    Args:
        y: Label array.
        class_value: Class to count.

    Returns:
        Number of samples with given class value.
    """
    mask: NDArray[np.bool_] = y == class_value
    return int(np.sum(mask))


def _get_unique_indices(split_info: CVSplitInfo) -> set[int]:
    """Get all unique indices across all validation folds.

    Args:
        split_info: Complete split information.

    Returns:
        Set of all unique indices appearing in validation folds.
    """
    all_indices: set[int] = set()
    for fold in split_info["folds"]:
        val_indices = fold["val_indices"]
        for i in range(len(val_indices)):
            all_indices.add(int(val_indices.item(i)))
    return all_indices


def _indices_to_set(indices: NDArray[np.intp]) -> set[int]:
    """Convert index array to set of ints."""
    result: set[int] = set()
    for i in range(len(indices)):
        result.add(int(indices.item(i)))
    return result


def _check_all_ones(counts: NDArray[np.int64]) -> bool:
    """Check if all counts equal 1."""
    return all(int(counts.item(i)) == 1 for i in range(len(counts)))


def _make_groups(samples_per_group: tuple[int, ...]) -> NDArray[np.int64]:
    """Create group ID array with specified samples per group.

    Args:
        samples_per_group: Tuple of sample counts per group.
            E.g., (3, 2, 4) creates groups [0,0,0,1,1,2,2,2,2].

    Returns:
        Array of group IDs.
    """
    groups: list[int] = []
    for group_id, count in enumerate(samples_per_group):
        groups.extend([group_id] * count)
    result: NDArray[np.int64] = np.array(groups, dtype=np.int64)
    return result


def _make_labels_for_groups(
    samples_per_group: tuple[int, ...],
    positive_groups: set[int],
) -> NDArray[np.int64]:
    """Create label array where specified groups have at least one positive.

    Args:
        samples_per_group: Tuple of sample counts per group.
        positive_groups: Set of group IDs that should have positive samples.
            First sample in each positive group is set to 1.

    Returns:
        Binary label array.
    """
    n_samples = sum(samples_per_group)
    labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    idx = 0
    for group_id, count in enumerate(samples_per_group):
        if group_id in positive_groups:
            # Set first sample in group to positive
            labels[idx] = 1
        idx += count

    return labels


def _get_groups_for_indices(
    groups: NDArray[np.int64],
    indices: NDArray[np.intp],
) -> set[int]:
    """Get unique group IDs for the given sample indices.

    Args:
        groups: Full group ID array.
        indices: Sample indices to look up.

    Returns:
        Set of unique group IDs.
    """
    result: set[int] = set()
    for i in range(len(indices)):
        idx = int(indices.item(i))
        group_id = int(groups.item(idx))
        result.add(group_id)
    return result


def _make_int64_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create int64 array from tuple of ints.

    Args:
        values: Tuple of integer values.

    Returns:
        Array of int64 dtype.
    """
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result
