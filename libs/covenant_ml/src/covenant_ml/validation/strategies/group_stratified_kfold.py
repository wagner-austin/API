"""Group-stratified k-fold cross-validation splitter strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Wraps the existing group_stratified_kfold_split function as a Protocol-compliant class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..protocol import CVStrategyCapabilities, CVStrategyName
from ..splitter import group_stratified_kfold_split
from ..types import CVSplitInfo


class GroupStratifiedKFoldSplitter:
    """Group-stratified k-fold cross-validation splitter.

    Ensures that all samples from the same group appear in the same fold.
    This is critical for time-series data where multiple observations per
    entity (e.g., customer statements over time) must not leak between
    train and validation sets.

    Groups are stratified by their aggregate label: a group is positive if
    any sample in the group is positive. This maintains approximate class
    balance across folds while respecting group boundaries.
    """

    def strategy_name(self) -> CVStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'group_stratified_kfold'.
        """
        return "group_stratified_kfold"

    def capabilities(self) -> CVStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating this strategy preserves class ratios,
            supports groups and shuffling, but not temporal ordering.
        """
        return CVStrategyCapabilities(
            preserves_class_ratio=True,
            supports_groups=True,
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
        """Generate group-stratified k-fold splits.

        Args:
            y: Binary labels of shape (n_samples,).
            n_folds: Number of folds (must be >= 2).
            random_state: Random seed for reproducibility.
            groups: Group IDs of shape (n_samples,). All samples with the same
                group ID will be assigned to the same fold. Required.

        Returns:
            CVSplitInfo containing all fold splits and metadata.

        Raises:
            ValueError: If n_folds < 2, not enough groups, groups not provided,
                or groups/y length mismatch.
        """
        if groups is None:
            raise ValueError("groups parameter is required for group_stratified_kfold strategy")
        return group_stratified_kfold_split(y, groups, n_folds, random_state)


def create_group_stratified_kfold_splitter() -> GroupStratifiedKFoldSplitter:
    """Factory function to create a GroupStratifiedKFoldSplitter.

    Returns:
        A new GroupStratifiedKFoldSplitter instance.
    """
    return GroupStratifiedKFoldSplitter()


__all__ = [
    "GroupStratifiedKFoldSplitter",
    "create_group_stratified_kfold_splitter",
]
