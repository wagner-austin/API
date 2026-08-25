"""Stratified k-fold cross-validation splitter.

Provides stratified splitting that maintains class proportions across folds.
Each sample appears in exactly one validation fold.

Includes GroupKFold support for time-series data where multiple observations
per entity (e.g., customer) must stay together in the same fold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.validation.types import CVSplit, CVSplitInfo

_log = get_logger(__name__)


def _get_class_indices(y: NDArray[np.int64], class_value: int) -> NDArray[np.intp]:
    """Get indices of samples belonging to a specific class.

    Args:
        y: Label array of shape (n_samples,).
        class_value: Class value to find (0 or 1 for binary).

    Returns:
        Array of indices where y equals class_value.
    """
    mask: NDArray[np.bool_] = y == class_value
    indices: NDArray[np.intp] = np.flatnonzero(mask)
    return indices


def _shuffle_indices(
    indices: NDArray[np.intp],
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Shuffle indices in place and return.

    Args:
        indices: Array of indices to shuffle.
        rng: Random number generator for reproducibility.

    Returns:
        Shuffled indices (same array, modified in place).
    """
    rng.shuffle(indices)
    return indices


def _split_array_into_folds(
    indices: NDArray[np.intp],
    n_folds: int,
) -> tuple[NDArray[np.intp], ...]:
    """Split array of indices into approximately equal folds.

    Uses numpy.array_split which handles uneven divisions by distributing
    extra elements to the first folds.

    Args:
        indices: Array of indices to split.
        n_folds: Number of folds to create.

    Returns:
        Tuple of n_folds arrays, each containing a portion of indices.
    """
    splits = np.array_split(indices, n_folds)
    result: list[NDArray[np.intp]] = []
    for split in splits:
        typed_split: NDArray[np.intp] = np.asarray(split, dtype=np.intp)
        result.append(typed_split)
    return tuple(result)


def _concat_indices(*arrays: NDArray[np.intp]) -> NDArray[np.intp]:
    """Concatenate multiple index arrays.

    Args:
        *arrays: Variable number of index arrays.

    Returns:
        Single concatenated array of indices.
    """
    if len(arrays) == 0:
        empty: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        return empty
    result: NDArray[np.intp] = np.concatenate(arrays)
    return result


def stratified_kfold_split(
    y: NDArray[np.int64],
    n_folds: int,
    random_state: int,
) -> CVSplitInfo:
    """Create stratified k-fold cross-validation splits.

    Maintains class proportions across all folds. Each sample appears in
    exactly one validation fold and (n_folds - 1) training folds.

    Args:
        y: Binary labels of shape (n_samples,).
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.

    Returns:
        CVSplitInfo containing all fold splits and metadata.

    Raises:
        ValueError: If n_folds < 2 or not enough samples per class.
    """
    n_samples = len(y)

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    # Get indices for each class
    pos_indices = _get_class_indices(y, 1)
    neg_indices = _get_class_indices(y, 0)

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos < n_folds:
        raise ValueError(
            f"Not enough positive samples ({n_pos}) for {n_folds} folds. "
            f"Need at least {n_folds} positive samples."
        )
    if n_neg < n_folds:
        raise ValueError(
            f"Not enough negative samples ({n_neg}) for {n_folds} folds. "
            f"Need at least {n_folds} negative samples."
        )

    # Shuffle indices
    rng = np.random.default_rng(random_state)
    pos_indices = _shuffle_indices(pos_indices.copy(), rng)
    neg_indices = _shuffle_indices(neg_indices.copy(), rng)

    # Split each class into folds
    pos_folds = _split_array_into_folds(pos_indices, n_folds)
    neg_folds = _split_array_into_folds(neg_indices, n_folds)

    # Combine class folds to create stratified folds
    folds: list[CVSplit] = []

    for fold_num in range(n_folds):
        # Validation set: samples from this fold for both classes
        val_indices = _concat_indices(pos_folds[fold_num], neg_folds[fold_num])

        # Training set: samples from all other folds
        train_pos_parts: list[NDArray[np.intp]] = []
        train_neg_parts: list[NDArray[np.intp]] = []

        for other_fold in range(n_folds):
            if other_fold != fold_num:
                train_pos_parts.append(pos_folds[other_fold])
                train_neg_parts.append(neg_folds[other_fold])

        train_indices = _concat_indices(
            *train_pos_parts,
            *train_neg_parts,
        )

        # Shuffle train and val indices for randomness
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    _log.info(
        "Created stratified k-fold splits",
        extra={
            "n_folds": n_folds,
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_ratio": n_pos / n_samples,
        },
    )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


def get_fold_data(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    split: CVSplit,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Extract train and validation data for a single fold.

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Labels of shape (n_samples,).
        split: A single fold split with train and val indices.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val).
    """
    x_train: NDArray[np.float64] = x[split["train_indices"]]
    y_train: NDArray[np.int64] = y[split["train_indices"]]
    x_val: NDArray[np.float64] = x[split["val_indices"]]
    y_val: NDArray[np.int64] = y[split["val_indices"]]

    return x_train, y_train, x_val, y_val


def _get_group_labels(
    groups: NDArray[np.int64],
    y: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Compute a label for each unique group.

    A group is labeled positive (1) if any sample in the group is positive.
    This allows stratification by group while preserving class balance.

    Args:
        groups: Group IDs for each sample, shape (n_samples,).
        y: Binary labels for each sample, shape (n_samples,).

    Returns:
        Tuple of (unique_groups, group_labels) where:
            - unique_groups: Sorted array of unique group IDs.
            - group_labels: Binary label (0 or 1) for each unique group.
    """
    unique_groups: NDArray[np.int64] = np.unique(groups)
    n_groups = len(unique_groups)

    group_labels: NDArray[np.int64] = np.zeros(n_groups, dtype=np.int64)

    for i in range(n_groups):
        group_id = int(unique_groups.item(i))
        mask: NDArray[np.bool_] = groups == group_id
        group_y: NDArray[np.int64] = y[mask]
        # Group is positive if any sample is positive
        if int(np.sum(group_y)) > 0:
            group_labels[i] = 1

    return unique_groups, group_labels


def _get_sample_indices_for_groups(
    groups: NDArray[np.int64],
    selected_groups: NDArray[np.int64],
) -> NDArray[np.intp]:
    """Get all sample indices that belong to the selected groups.

    Args:
        groups: Group IDs for each sample, shape (n_samples,).
        selected_groups: Array of group IDs to select.

    Returns:
        Array of sample indices belonging to the selected groups.
    """
    # Create mask for all samples belonging to selected groups
    mask: NDArray[np.bool_] = np.zeros(len(groups), dtype=np.bool_)
    for i in range(len(selected_groups)):
        group_id = int(selected_groups.item(i))
        group_mask: NDArray[np.bool_] = groups == group_id
        mask = mask | group_mask

    indices: NDArray[np.intp] = np.flatnonzero(mask)
    return indices


def group_stratified_kfold_split(
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
    n_folds: int,
    random_state: int,
) -> CVSplitInfo:
    """Create group-stratified k-fold cross-validation splits.

    Ensures that all samples from the same group appear in the same fold.
    This is critical for time-series data where multiple observations per
    entity (e.g., customer statements over time) must not leak between
    train and validation sets.

    Groups are stratified by their aggregate label: a group is positive if
    any sample in the group is positive. This maintains approximate class
    balance across folds while respecting group boundaries.

    Args:
        y: Binary labels of shape (n_samples,).
        groups: Group IDs of shape (n_samples,). All samples with the same
            group ID will be assigned to the same fold.
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.

    Returns:
        CVSplitInfo containing all fold splits and metadata.

    Raises:
        ValueError: If n_folds < 2, not enough groups, or groups/y length mismatch.
    """
    n_samples = len(y)

    if len(groups) != n_samples:
        raise ValueError(f"groups length ({len(groups)}) must match y length ({n_samples})")

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    # Get unique groups and their aggregate labels
    unique_groups, group_labels = _get_group_labels(groups, y)
    n_groups = len(unique_groups)

    if n_groups < n_folds:
        raise ValueError(
            f"Not enough groups ({n_groups}) for {n_folds} folds. Need at least {n_folds} groups."
        )

    # Get indices for positive and negative groups
    pos_group_mask: NDArray[np.bool_] = group_labels == 1
    neg_group_mask: NDArray[np.bool_] = group_labels == 0
    pos_group_indices: NDArray[np.intp] = np.flatnonzero(pos_group_mask)
    neg_group_indices: NDArray[np.intp] = np.flatnonzero(neg_group_mask)

    n_pos_groups = len(pos_group_indices)
    n_neg_groups = len(neg_group_indices)

    if n_pos_groups < n_folds:
        raise ValueError(
            f"Not enough positive groups ({n_pos_groups}) for {n_folds} folds. "
            f"Need at least {n_folds} positive groups."
        )
    if n_neg_groups < n_folds:
        raise ValueError(
            f"Not enough negative groups ({n_neg_groups}) for {n_folds} folds. "
            f"Need at least {n_folds} negative groups."
        )

    # Shuffle group indices
    rng = np.random.default_rng(random_state)
    rng.shuffle(pos_group_indices)
    rng.shuffle(neg_group_indices)

    # Split groups into folds
    pos_group_folds = _split_array_into_folds(pos_group_indices, n_folds)
    neg_group_folds = _split_array_into_folds(neg_group_indices, n_folds)

    # Create folds by expanding group indices to sample indices
    folds: list[CVSplit] = []

    for fold_num in range(n_folds):
        # Get group indices for validation fold
        val_pos_group_idx = pos_group_folds[fold_num]
        val_neg_group_idx = neg_group_folds[fold_num]

        # Convert group indices to actual group IDs
        val_pos_groups: NDArray[np.int64] = unique_groups[val_pos_group_idx]
        val_neg_groups: NDArray[np.int64] = unique_groups[val_neg_group_idx]
        val_groups: NDArray[np.int64] = np.concatenate([val_pos_groups, val_neg_groups])

        # Get sample indices for validation groups
        val_indices = _get_sample_indices_for_groups(groups, val_groups)

        # Get group indices for training folds (all other folds)
        train_pos_parts: list[NDArray[np.intp]] = []
        train_neg_parts: list[NDArray[np.intp]] = []

        for other_fold in range(n_folds):
            if other_fold != fold_num:
                train_pos_parts.append(pos_group_folds[other_fold])
                train_neg_parts.append(neg_group_folds[other_fold])

        train_pos_group_idx = _concat_indices(*train_pos_parts)
        train_neg_group_idx = _concat_indices(*train_neg_parts)

        # Convert to group IDs
        train_pos_groups: NDArray[np.int64] = unique_groups[train_pos_group_idx]
        train_neg_groups: NDArray[np.int64] = unique_groups[train_neg_group_idx]
        train_groups: NDArray[np.int64] = np.concatenate([train_pos_groups, train_neg_groups])

        # Get sample indices for training groups
        train_indices = _get_sample_indices_for_groups(groups, train_groups)

        # Shuffle train and val indices
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    # Count positive samples for logging
    n_pos = int(np.sum(y))

    _log.info(
        "Created group-stratified k-fold splits",
        extra={
            "n_folds": n_folds,
            "n_samples": n_samples,
            "n_groups": n_groups,
            "n_positive_groups": n_pos_groups,
            "n_negative_groups": n_neg_groups,
            "n_positive_samples": n_pos,
            "positive_sample_ratio": n_pos / n_samples,
        },
    )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


def group_kfold_split(
    y: NDArray[np.int64],
    groups: NDArray[np.int64],
    n_folds: int,
    random_state: int,
) -> CVSplitInfo:
    """Plain grouped k-fold: whole groups per fold, no label stratification.

    The instrument for grouped data whose groups carry MIXED labels —
    co-elution windows holding both real and blank peaks, for example —
    where :func:`group_stratified_kfold_split`'s any-positive group label
    is undefined (every mixed group would count as positive and the
    negative stratum would be empty). Groups shuffle once and split into
    folds directly; with many mixed groups, class balance follows from
    the mixing rather than from stratification.

    Args:
        y: Binary labels of shape (n_samples,), used for balance logging.
        groups: Group IDs of shape (n_samples,). All samples with the
            same group ID land in the same fold.
        n_folds: Number of folds (must be >= 2).
        random_state: Random seed for reproducibility.

    Returns:
        CVSplitInfo containing all fold splits and metadata.

    Raises:
        ValueError: If n_folds < 2, not enough groups, or groups/y
            length mismatch.
    """
    n_samples = len(y)

    if len(groups) != n_samples:
        raise ValueError(f"groups length ({len(groups)}) must match y length ({n_samples})")

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    unique_groups: NDArray[np.int64] = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < n_folds:
        raise ValueError(
            f"Not enough groups ({n_groups}) for {n_folds} folds. Need at least {n_folds} groups."
        )

    group_indices: NDArray[np.intp] = np.arange(n_groups, dtype=np.intp)
    rng = np.random.default_rng(random_state)
    rng.shuffle(group_indices)
    group_folds = _split_array_into_folds(group_indices, n_folds)

    folds: list[CVSplit] = []
    for fold_num in range(n_folds):
        val_group_ids: NDArray[np.int64] = unique_groups[group_folds[fold_num]]
        val_indices = _get_sample_indices_for_groups(groups, val_group_ids)

        train_parts: list[NDArray[np.intp]] = [
            group_folds[other_fold] for other_fold in range(n_folds) if other_fold != fold_num
        ]
        train_group_idx = _concat_indices(*train_parts)
        train_group_ids: NDArray[np.int64] = unique_groups[train_group_idx]
        train_indices = _get_sample_indices_for_groups(groups, train_group_ids)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    n_pos = int(np.sum(y))
    _log.info(
        "Created plain grouped k-fold splits",
        extra={
            "n_folds": n_folds,
            "n_samples": n_samples,
            "n_groups": n_groups,
            "n_positive_samples": n_pos,
            "positive_sample_ratio": n_pos / n_samples,
        },
    )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


__all__ = [
    "get_fold_data",
    "group_kfold_split",
    "group_stratified_kfold_split",
    "stratified_kfold_split",
]
