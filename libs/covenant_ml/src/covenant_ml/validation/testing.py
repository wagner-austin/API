"""Testing utilities for validation module.

Provides factory functions and test data generators for cross-validation tests.
This module is exported for consumers to use in their test suites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocol import CVSplitterProtocol, CVStrategyCapabilities, CVStrategyName
from .registry import CVSplitterRegistration, CVSplitterRegistry
from .types import CVSplit, CVSplitInfo


class FakeCVSplitter:
    """Fake CV splitter for testing.

    Returns predetermined splits for predictable test behavior.
    Useful for testing code that depends on CV splits without
    needing to generate actual stratified splits.
    """

    def __init__(
        self,
        name: CVStrategyName = "stratified_kfold",
        capabilities: CVStrategyCapabilities | None = None,
        splits: tuple[CVSplit, ...] | None = None,
    ) -> None:
        """Initialize fake splitter.

        Args:
            name: Strategy name to return.
            capabilities: Capabilities to return. If None, uses defaults.
            splits: Predetermined splits to return. If None, generates simple splits.
        """
        self._name = name
        self._capabilities = capabilities or CVStrategyCapabilities(
            preserves_class_ratio=True,
            supports_groups=False,
            supports_temporal=False,
            supports_shuffle=True,
        )
        self._splits = splits
        self._split_call_count = 0

    @property
    def split_call_count(self) -> int:
        """Get the number of times split was called.

        Returns:
            The count of split() invocations.
        """
        return self._split_call_count

    def strategy_name(self) -> CVStrategyName:
        """Return the configured strategy name.

        Returns:
            The strategy name set during initialization.
        """
        return self._name

    def capabilities(self) -> CVStrategyCapabilities:
        """Return the configured capabilities.

        Returns:
            The capabilities set during initialization.
        """
        return self._capabilities

    def split(
        self,
        y: NDArray[np.int64],
        n_folds: int,
        random_state: int,
        *,
        groups: NDArray[np.int64] | None = None,
    ) -> CVSplitInfo:
        """Return predetermined or generated splits.

        Args:
            y: Binary labels of shape (n_samples,).
            n_folds: Number of folds requested.
            random_state: Random seed (ignored if using predetermined splits).
            groups: Group IDs (ignored by fake).

        Returns:
            CVSplitInfo with predetermined or simple generated splits.
        """
        del groups
        del random_state
        self._split_call_count += 1

        if self._splits is not None:
            return CVSplitInfo(
                n_folds=len(self._splits),
                n_samples=len(y),
                folds=self._splits,
            )

        # Generate simple splits if none provided
        n_samples = len(y)
        fold_size = max(1, n_samples // n_folds)
        folds: list[CVSplit] = []

        indices: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)

        for fold_num in range(n_folds):
            val_start = fold_num * fold_size
            val_end = min(val_start + fold_size, n_samples)

            val_indices = indices[val_start:val_end]
            train_mask: NDArray[np.bool_] = np.ones(n_samples, dtype=np.bool_)
            train_mask[val_start:val_end] = False
            train_indices: NDArray[np.intp] = indices[train_mask]

            folds.append(
                CVSplit(
                    fold_number=fold_num,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )
            )

        return CVSplitInfo(
            n_folds=n_folds,
            n_samples=n_samples,
            folds=tuple(folds),
        )


def make_fake_cv_splitter(
    name: CVStrategyName = "stratified_kfold",
    preserves_class_ratio: bool = True,
    supports_groups: bool = False,
) -> FakeCVSplitter:
    """Create a FakeCVSplitter with specified capabilities.

    Args:
        name: Strategy name to use.
        preserves_class_ratio: Whether strategy claims to preserve class ratios.
        supports_groups: Whether strategy claims to support groups.

    Returns:
        A configured FakeCVSplitter instance.
    """
    capabilities = CVStrategyCapabilities(
        preserves_class_ratio=preserves_class_ratio,
        supports_groups=supports_groups,
        supports_temporal=False,
        supports_shuffle=True,
    )
    return FakeCVSplitter(name=name, capabilities=capabilities)


def make_test_cv_split_info(
    n_samples: int = 100,
    n_folds: int = 3,
) -> CVSplitInfo:
    """Create a CVSplitInfo for testing.

    Generates simple non-overlapping splits for test verification.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds to create.

    Returns:
        CVSplitInfo with simple non-overlapping splits.
    """
    fold_size = n_samples // n_folds
    indices: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)
    folds: list[CVSplit] = []

    for fold_num in range(n_folds):
        val_start = fold_num * fold_size
        val_end = val_start + fold_size if fold_num < n_folds - 1 else n_samples

        val_indices = indices[val_start:val_end]
        train_mask: NDArray[np.bool_] = np.ones(n_samples, dtype=np.bool_)
        train_mask[val_start:val_end] = False
        train_indices: NDArray[np.intp] = indices[train_mask]

        folds.append(
            CVSplit(
                fold_number=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    return CVSplitInfo(
        n_folds=n_folds,
        n_samples=n_samples,
        folds=tuple(folds),
    )


def make_binary_labels(
    n_samples: int = 100,
    positive_ratio: float = 0.3,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Create binary labels for testing.

    Args:
        n_samples: Total number of samples.
        positive_ratio: Fraction of positive samples (0 < ratio < 1).
        random_state: Random seed for reproducibility.

    Returns:
        Array of binary labels (0 and 1).
    """
    rng = np.random.default_rng(random_state)
    n_positive = int(n_samples * positive_ratio)
    labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    positive_indices = rng.choice(n_samples, size=n_positive, replace=False)
    labels[positive_indices] = 1
    return labels


def make_group_ids(
    n_samples: int = 100,
    n_groups: int = 10,
) -> NDArray[np.int64]:
    """Create group IDs for testing group-aware splitting.

    Assigns samples to groups in a round-robin fashion.

    Args:
        n_samples: Total number of samples.
        n_groups: Number of unique groups.

    Returns:
        Array of group IDs.
    """
    groups: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        groups[i] = i % n_groups
    return groups


def make_test_cv_registry() -> CVSplitterRegistry:
    """Create a test CV registry with fake splitters.

    Returns:
        CVSplitterRegistry populated with FakeCVSplitter instances.
    """
    registry = CVSplitterRegistry()

    # Register fake stratified_kfold
    def create_fake_stratified() -> CVSplitterProtocol:
        return make_fake_cv_splitter("stratified_kfold")

    registry.register(
        "stratified_kfold",
        CVSplitterRegistration(create_fake_stratified),
    )

    # Register fake group_stratified_kfold
    def create_fake_group_stratified() -> CVSplitterProtocol:
        return make_fake_cv_splitter("group_stratified_kfold", supports_groups=True)

    registry.register(
        "group_stratified_kfold",
        CVSplitterRegistration(create_fake_group_stratified),
    )

    # Register fake shuffle_split
    def create_fake_shuffle() -> CVSplitterProtocol:
        return make_fake_cv_splitter("shuffle_split")

    registry.register(
        "shuffle_split",
        CVSplitterRegistration(create_fake_shuffle),
    )

    # Register fake time_series
    def create_fake_time_series() -> CVSplitterProtocol:
        caps = CVStrategyCapabilities(
            preserves_class_ratio=False,
            supports_groups=False,
            supports_temporal=True,
            supports_shuffle=False,
        )
        return FakeCVSplitter(name="time_series", capabilities=caps)

    registry.register(
        "time_series",
        CVSplitterRegistration(create_fake_time_series),
    )

    return registry


__all__ = [
    "FakeCVSplitter",
    "make_binary_labels",
    "make_fake_cv_splitter",
    "make_group_ids",
    "make_test_cv_registry",
    "make_test_cv_split_info",
]
