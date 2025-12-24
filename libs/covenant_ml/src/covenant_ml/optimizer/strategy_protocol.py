"""Unified optimizer strategy protocol.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Defines a unified HyperparameterOptimizer interface that implementations must satisfy.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from .protocol import ObjectiveProtocol, TrialCallbackProtocol
from .types import (
    OptimizationConfig,
    OptimizationSummary,
    SearchSpace,
)

# =============================================================================
# Strategy Names and Capabilities
# =============================================================================

OptimizerStrategyName = Literal[
    "optuna_tpe",
    "random_search",
    "grid_search",
]


class OptimizerStrategyCapabilities(TypedDict, total=True):
    """Describes supported features of an optimizer strategy.

    Attributes:
        supports_pruning: Whether strategy supports early trial pruning.
        supports_parallel: Whether strategy can run trials in parallel.
        is_deterministic: Whether strategy produces same results given same seed.
        requires_bounds: Whether strategy requires bounded search spaces.
    """

    supports_pruning: bool
    supports_parallel: bool
    is_deterministic: bool
    requires_bounds: bool


# =============================================================================
# Unified Optimizer Protocol
# =============================================================================


class HyperparameterOptimizerProtocol(Protocol):
    """Unified protocol for hyperparameter optimization strategies.

    This protocol abstracts over different optimization algorithms
    (TPE, random search, grid search, etc.) with a common interface.
    """

    def strategy_name(self) -> OptimizerStrategyName:
        """Return the name of this optimization strategy.

        Returns:
            The strategy name as a literal string.
        """
        ...

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return the capabilities of this optimization strategy.

        Returns:
            TypedDict describing what this strategy supports.
        """
        ...

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Parameter ranges to search.
            config: Optimization settings (n_trials, timeout, etc.).
            objective: Function to evaluate hyperparameters.
            trial_callback: Optional callback after each trial.

        Returns:
            Summary with best hyperparameters and trial statistics.
        """
        ...


# =============================================================================
# Optimizer Factory Protocol
# =============================================================================


class OptimizerStrategyFactory(Protocol):
    """Factory protocol to construct an optimizer strategy implementation."""

    def __call__(self) -> HyperparameterOptimizerProtocol:
        """Create a new optimizer instance.

        Returns:
            A configured optimizer ready for use.
        """
        ...


__all__ = [
    "HyperparameterOptimizerProtocol",
    "OptimizerStrategyCapabilities",
    "OptimizerStrategyFactory",
    "OptimizerStrategyName",
]
