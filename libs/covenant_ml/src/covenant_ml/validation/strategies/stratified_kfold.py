"""Stratified k-fold cross-validation splitter strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Wraps the existing stratified_kfold_split function as a Protocol-compliant class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..protocol import CVStrategyCapabilities, CVStrategyName
from ..splitter import stratified_kfold_split
from ..types import CVSplitInfo


class StratifiedKFoldSplitter:
    """Stratified k-fold cross-validation splitter.

    Maintains class proportions across all folds. Each sample appears in
    exactly one validation fold and (n_folds - 1) training folds.

    This is the standard CV strategy for classification tasks where you
    want balanced class representation in each fold.
    """

    def strategy_name(self) -> CVStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'stratified_kfold'.
        """
        return "stratified_kfold"

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
        """Generate stratified k-fold splits.

        Args:
            y: Binary labels of shape (n_samples,).
            n_folds: Number of folds (must be >= 2).
            random_state: Random seed for reproducibility.
            groups: Not used by this strategy. Ignored if provided.

        Returns:
            CVSplitInfo containing all fold splits and metadata.

        Raises:
            ValueError: If n_folds < 2 or not enough samples per class.
        """
        # Groups parameter is ignored for non-group-aware strategies
        del groups
        return stratified_kfold_split(y, n_folds, random_state)


def create_stratified_kfold_splitter() -> StratifiedKFoldSplitter:
    """Factory function to create a StratifiedKFoldSplitter.

    Returns:
        A new StratifiedKFoldSplitter instance.
    """
    return StratifiedKFoldSplitter()


__all__ = [
    "StratifiedKFoldSplitter",
    "create_stratified_kfold_splitter",
]
