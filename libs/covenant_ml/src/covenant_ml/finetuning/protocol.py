"""Protocols for fine-tuning strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Defines the FineTuningStrategy interface that implementations must satisfy.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..optimizer.types import SearchSpace
from .types import FineTuningConfig, FineTuningResult, WarmStartConfig

# =============================================================================
# Strategy Names and Capabilities
# =============================================================================

FineTuningStrategyName = Literal[
    "staged",
    "warm_start",
    "iterative_refinement",
]


class FineTuningCapabilities(TypedDict, total=True):
    """Describes supported features of a fine-tuning strategy.

    Attributes:
        supports_warm_start: Whether strategy can initialize from prior results.
        supports_staged: Whether strategy supports multi-stage optimization.
        supports_early_stop: Whether strategy can stop early on convergence.
        preserves_prior_params: Whether strategy keeps prior string params fixed.
    """

    supports_warm_start: bool
    supports_staged: bool
    supports_early_stop: bool
    preserves_prior_params: bool


# =============================================================================
# Fine-Tuning Strategy Protocol
# =============================================================================


class FineTuningStrategyProtocol(Protocol):
    """Protocol for fine-tuning strategy implementations.

    Implementations must provide methods to:
    - Identify the strategy by name
    - Describe its capabilities
    - Execute fine-tuning with optional warm-start
    """

    def strategy_name(self) -> FineTuningStrategyName:
        """Return the name of this fine-tuning strategy.

        Returns:
            The strategy name as a literal string.
        """
        ...

    def capabilities(self) -> FineTuningCapabilities:
        """Return the capabilities of this fine-tuning strategy.

        Returns:
            TypedDict describing what this strategy supports.
        """
        ...

    def fine_tune(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: FineTuningConfig,
        objective: ObjectiveProtocol,
        warm_start: WarmStartConfig | None = None,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> FineTuningResult:
        """Execute fine-tuning optimization.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Base parameter ranges to search.
            config: Fine-tuning configuration with stages.
            objective: Function to evaluate hyperparameters.
            warm_start: Optional warm-start from prior optimization.
            trial_callback: Optional callback after each trial.

        Returns:
            Complete fine-tuning result with best parameters.
        """
        ...


# =============================================================================
# Fine-Tuning Factory Protocol
# =============================================================================


class FineTuningStrategyFactory(Protocol):
    """Factory protocol to construct a fine-tuning strategy implementation."""

    def __call__(self) -> FineTuningStrategyProtocol:
        """Create a new fine-tuning strategy instance.

        Returns:
            A configured fine-tuning strategy ready for use.
        """
        ...


__all__ = [
    "FineTuningCapabilities",
    "FineTuningStrategyFactory",
    "FineTuningStrategyName",
    "FineTuningStrategyProtocol",
]
