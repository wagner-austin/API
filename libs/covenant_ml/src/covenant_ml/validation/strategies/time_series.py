"""Time series cross-validation splitter strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements temporal splitting that respects time ordering.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import CVStrategyCapabilities, CVStrategyName
from ..types import CVSplit, CVSplitInfo

_log = get_logger(__name__)


def _validate_time_series_params(
    n_samples: int,
    n_splits: int,
    min_train_size: int | None,
) -> None:
    """Validate time series split parameters.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of splits to generate.
        min_train_size: Minimum samples in first training set.

    Raises:
        ValueError: If parameters are invalid.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")

    if n_samples < n_splits + 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_splits} splits. "
            f"Need at least {n_splits + 1} samples."
        )

    if min_train_size is not None:
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
        if min_train_size > n_samples - n_splits:
            raise ValueError(
                f"min_train_size ({min_train_size}) too large for "
                f"{n_samples} samples and {n_splits} splits"
            )


def _compute_fold_size(
    n_samples: int,
    n_splits: int,
    min_train_size: int | None,
) -> int:
    """Compute the fold size for time series splits.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of splits to generate.
        min_train_size: Minimum samples in first training set.

    Returns:
        The fold size to use.
    """
    if min_train_size is not None:
        remaining = n_samples - min_train_size
        fold_size = remaining // n_splits
        return max(1, fold_size)
    return n_samples // (n_splits + 1)


def _time_series_split(
    n_samples: int,
    n_splits: int,
    min_train_size: int | None = None,
) -> tuple[CVSplit, ...]:
    """Generate time series splits with expanding window.

    Each split uses all prior data for training and the next chunk for
    validation. The training set grows with each split.

    Example with n_samples=100, n_splits=4:
    - Split 0: train [0:20], val [20:40]
    - Split 1: train [0:40], val [40:60]
    - Split 2: train [0:60], val [60:80]
    - Split 3: train [0:80], val [80:100]

    Args:
        n_samples: Total number of samples in the dataset.
        n_splits: Number of splits to generate.
        min_train_size: Minimum number of samples in first training set.
            If None, defaults to n_samples / (n_splits + 1).

    Returns:
        Tuple of CVSplit objects, one per split.

    Raises:
        ValueError: If n_splits < 1 or not enough samples.
    """
    _validate_time_series_params(n_samples, n_splits, min_train_size)

    fold_size = _compute_fold_size(n_samples, n_splits, min_train_size)

    splits: list[CVSplit] = []
    all_indices: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)

    for split_num in range(n_splits):
        if min_train_size is not None:
            train_end = min_train_size + (split_num * fold_size)
        else:
            train_end = (split_num + 1) * fold_size

        val_start = train_end
        val_end = min(train_end + fold_size, n_samples)

        # For the last split, use all remaining samples
        if split_num == n_splits - 1:
            val_end = n_samples

        train_indices: NDArray[np.intp] = all_indices[:train_end]
        val_indices: NDArray[np.intp] = all_indices[val_start:val_end]

        splits.append(
            CVSplit(
                fold_number=split_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    _log.info(
        "Created time series splits",
        extra={
            "n_splits": n_splits,
            "n_samples": n_samples,
            "fold_size": fold_size,
            "min_train_size": min_train_size,
        },
    )

    return tuple(splits)


class TimeSeriesSplitter:
    """Time series cross-validation splitter.

    Generates temporal splits that respect time ordering. Training data
    always precedes validation data chronologically, preventing data
    leakage from future to past.

    Uses an expanding window approach where each subsequent fold has
    a larger training set (all prior data) and the next temporal chunk
    for validation.

    This is the appropriate strategy when:
    - Data has a natural temporal ordering
    - Future information should not leak into training
    - You want to simulate real-world prediction scenarios

    Attributes:
        min_train_size: Minimum samples in initial training set.
    """

    def __init__(self, min_train_size: int | None = None) -> None:
        """Initialize the time series splitter.

        Args:
            min_train_size: Minimum number of samples in the first
                training set. If None, defaults to n_samples / (n_splits + 1).
        """
        self._min_train_size = min_train_size

    @property
    def min_train_size(self) -> int | None:
        """Get the minimum training set size.

        Returns:
            The minimum training size, or None if using default calculation.
        """
        return self._min_train_size

    def strategy_name(self) -> CVStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'time_series'.
        """
        return "time_series"

    def capabilities(self) -> CVStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating this strategy supports temporal ordering
            but not class ratio preservation, groups, or shuffling.
        """
        return CVStrategyCapabilities(
            preserves_class_ratio=False,
            supports_groups=False,
            supports_temporal=True,
            supports_shuffle=False,
        )

    def split(
        self,
        y: NDArray[np.int64],
        n_folds: int,
        random_state: int,
        *,
        groups: NDArray[np.int64] | None = None,
    ) -> CVSplitInfo:
        """Generate time series splits.

        Note: random_state is accepted for interface compatibility but
        has no effect since time series splitting is deterministic.

        Args:
            y: Binary labels of shape (n_samples,). Used only for length.
            n_folds: Number of splits to generate.
            random_state: Ignored for time series splitting (deterministic).
            groups: Not used by this strategy. Ignored if provided.

        Returns:
            CVSplitInfo containing all splits and metadata.

        Raises:
            ValueError: If n_folds < 1 or not enough samples.
        """
        # Groups and random_state are ignored for time series
        del groups
        del random_state

        n_samples = len(y)
        folds = _time_series_split(n_samples, n_folds, self._min_train_size)

        return CVSplitInfo(
            n_folds=n_folds,
            n_samples=n_samples,
            folds=folds,
        )


def create_time_series_splitter() -> TimeSeriesSplitter:
    """Factory function to create a TimeSeriesSplitter with default settings.

    Returns:
        A new TimeSeriesSplitter instance with automatic min_train_size.
    """
    return TimeSeriesSplitter(min_train_size=None)


__all__ = [
    "TimeSeriesSplitter",
    "create_time_series_splitter",
]
