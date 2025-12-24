"""Stratified shuffle split cross-validation splitter strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements repeated random stratified splits for cross-validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import CVStrategyCapabilities, CVStrategyName
from ..types import CVSplit, CVSplitInfo

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


def _stratified_shuffle_split(
    y: NDArray[np.int64],
    n_splits: int,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[CVSplit, ...]:
    """Generate stratified shuffle splits.

    Creates n_splits random train/test splits where each split maintains
    the class proportions of the original data.

    Args:
        y: Binary labels of shape (n_samples,).
        n_splits: Number of splits to generate.
        test_fraction: Fraction of samples for test/validation set (0 < f < 1).
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of CVSplit objects, one per split.

    Raises:
        ValueError: If test_fraction not in (0, 1) or not enough samples.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    n_samples = len(y)
    pos_indices = _get_class_indices(y, 1)
    neg_indices = _get_class_indices(y, 0)

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    # Calculate samples for test set from each class
    n_pos_test = max(1, round(n_pos * test_fraction))
    n_neg_test = max(1, round(n_neg * test_fraction))

    if n_pos_test >= n_pos:
        raise ValueError(
            f"Not enough positive samples ({n_pos}) for test_fraction={test_fraction}. "
            f"Need at least {n_pos_test + 1} positive samples."
        )
    if n_neg_test >= n_neg:
        raise ValueError(
            f"Not enough negative samples ({n_neg}) for test_fraction={test_fraction}. "
            f"Need at least {n_neg_test + 1} negative samples."
        )

    splits: list[CVSplit] = []

    for split_num in range(n_splits):
        # Shuffle indices
        pos_shuffled = pos_indices.copy()
        neg_shuffled = neg_indices.copy()
        rng.shuffle(pos_shuffled)
        rng.shuffle(neg_shuffled)

        # Split each class
        pos_test = pos_shuffled[:n_pos_test]
        pos_train = pos_shuffled[n_pos_test:]
        neg_test = neg_shuffled[:n_neg_test]
        neg_train = neg_shuffled[n_neg_test:]

        # Combine and shuffle
        val_indices: NDArray[np.intp] = np.concatenate([pos_test, neg_test])
        train_indices: NDArray[np.intp] = np.concatenate([pos_train, neg_train])
        rng.shuffle(val_indices)
        rng.shuffle(train_indices)

        splits.append(
            CVSplit(
                fold_number=split_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    _log.info(
        "Created stratified shuffle splits",
        extra={
            "n_splits": n_splits,
            "n_samples": n_samples,
            "test_fraction": test_fraction,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_pos_test": n_pos_test,
            "n_neg_test": n_neg_test,
        },
    )

    return tuple(splits)


class ShuffleSplitSplitter:
    """Stratified shuffle split cross-validation splitter.

    Generates random stratified train/validation splits. Unlike k-fold,
    each split is independent and samples may appear in multiple
    validation sets across different splits.

    This is useful when:
    - You want more control over train/test sizes
    - You need more splits than samples would allow for k-fold
    - Independence between folds is desired

    Attributes:
        test_fraction: Fraction of samples for validation set (default 0.2).
    """

    def __init__(self, test_fraction: float = 0.2) -> None:
        """Initialize the shuffle split splitter.

        Args:
            test_fraction: Fraction of samples for validation set.
                Must be in (0, 1). Default is 0.2 (20% validation).

        Raises:
            ValueError: If test_fraction not in (0, 1).
        """
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        self._test_fraction = test_fraction

    @property
    def test_fraction(self) -> float:
        """Get the test/validation fraction.

        Returns:
            The fraction of samples reserved for validation.
        """
        return self._test_fraction

    def strategy_name(self) -> CVStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'shuffle_split'.
        """
        return "shuffle_split"

    def capabilities(self) -> CVStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating this strategy preserves class ratios
            and supports shuffling, but not groups or temporal ordering.
        """
        return CVStrategyCapabilities(
            preserves_class_ratio=True,
            supports_groups=False,
            supports_temporal=False,
            supports_shuffle=True,
        )

    def split(
        self,
        y: NDArray[np.int64],
        n_folds: int,
        random_state: int,
        *,
        groups: NDArray[np.int64] | None = None,
    ) -> CVSplitInfo:
        """Generate stratified shuffle splits.

        Args:
            y: Binary labels of shape (n_samples,).
            n_folds: Number of splits to generate (repurposed from k-fold interface).
            random_state: Random seed for reproducibility.
            groups: Not used by this strategy. Ignored if provided.

        Returns:
            CVSplitInfo containing all splits and metadata.

        Raises:
            ValueError: If n_folds < 1 or not enough samples per class.
        """
        # Groups parameter is ignored for non-group-aware strategies
        del groups

        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")

        rng = np.random.default_rng(random_state)
        folds = _stratified_shuffle_split(y, n_folds, self._test_fraction, rng)

        return CVSplitInfo(
            n_folds=n_folds,
            n_samples=len(y),
            folds=folds,
        )


def create_shuffle_split_splitter() -> ShuffleSplitSplitter:
    """Factory function to create a ShuffleSplitSplitter with default settings.

    Returns:
        A new ShuffleSplitSplitter instance with 20% validation fraction.
    """
    return ShuffleSplitSplitter(test_fraction=0.2)


__all__ = [
    "ShuffleSplitSplitter",
    "create_shuffle_split_splitter",
]
