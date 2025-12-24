"""Protocols for cross-validation strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Defines the CVSplitter interface that implementations must satisfy.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from .types import CVSplitInfo

# =============================================================================
# Strategy Names and Capabilities
# =============================================================================

CVStrategyName = Literal[
    "stratified_kfold",
    "group_stratified_kfold",
    "shuffle_split",
    "time_series",
]


class CVStrategyCapabilities(TypedDict, total=True):
    """Describes supported features of a CV strategy implementation.

    Attributes:
        preserves_class_ratio: Whether strategy maintains class proportions.
        supports_groups: Whether strategy supports group-based splitting.
        supports_temporal: Whether strategy preserves temporal ordering.
        supports_shuffle: Whether strategy can randomize sample order.
    """

    preserves_class_ratio: bool
    supports_groups: bool
    supports_temporal: bool
    supports_shuffle: bool


# =============================================================================
# CV Splitter Protocol
# =============================================================================


class CVSplitterProtocol(Protocol):
    """Protocol for cross-validation splitter strategies.

    Implementations must provide methods to:
    - Identify the strategy by name
    - Describe its capabilities
    - Generate train/validation splits from data
    """

    def strategy_name(self) -> CVStrategyName:
        """Return the name of this CV strategy.

        Returns:
            The strategy name as a literal string.
        """
        ...

    def capabilities(self) -> CVStrategyCapabilities:
        """Return the capabilities of this CV strategy.

        Returns:
            TypedDict describing what this strategy supports.
        """
        ...

    def split(
        self,
        y: NDArray[np.int64],
        n_folds: int,
        random_state: int,
        *,
        groups: NDArray[np.int64] | None = None,
    ) -> CVSplitInfo:
        """Generate cross-validation splits.

        Args:
            y: Binary labels of shape (n_samples,).
            n_folds: Number of folds to create.
            random_state: Random seed for reproducibility.
            groups: Optional group IDs for group-aware splitting.
                Required if strategy supports_groups is True.

        Returns:
            CVSplitInfo containing all fold splits and metadata.

        Raises:
            ValueError: If n_folds < 2, not enough samples, or groups
                required but not provided.
        """
        ...


# =============================================================================
# CV Splitter Factory Protocol
# =============================================================================


class CVSplitterFactory(Protocol):
    """Factory protocol to construct a CV splitter implementation."""

    def __call__(self) -> CVSplitterProtocol:
        """Create a new CV splitter instance.

        Returns:
            A configured CV splitter ready for use.
        """
        ...


__all__ = [
    "CVSplitterFactory",
    "CVSplitterProtocol",
    "CVStrategyCapabilities",
    "CVStrategyName",
]
