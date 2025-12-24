"""Registry for pluggable hyperparameter optimization strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Strategies are registered explicitly and accessed by name.
"""

from __future__ import annotations

from .strategy_protocol import (
    HyperparameterOptimizerProtocol,
    OptimizerStrategyCapabilities,
    OptimizerStrategyFactory,
    OptimizerStrategyName,
)


class OptimizerStrategyRegistration:
    """Registration record holding a factory and cached capabilities."""

    def __init__(self, factory: OptimizerStrategyFactory) -> None:
        """Initialize registration with a factory.

        Args:
            factory: Callable that creates HyperparameterOptimizerProtocol instances.
        """
        self._factory = factory
        self._capabilities_cache: OptimizerStrategyCapabilities | None = None

    def factory(self) -> OptimizerStrategyFactory:
        """Get the factory callable.

        Returns:
            The factory that creates optimizer instances.
        """
        return self._factory

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Get capabilities, caching on first access.

        Returns:
            The capabilities of this optimization strategy.
        """
        if self._capabilities_cache is None:
            optimizer = self._factory()
            self._capabilities_cache = optimizer.capabilities()
        return self._capabilities_cache


class OptimizerStrategyRegistry:
    """Registry of optimization strategies keyed by name."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._map: dict[OptimizerStrategyName, OptimizerStrategyRegistration] = {}

    def register(
        self,
        name: OptimizerStrategyName,
        registration: OptimizerStrategyRegistration,
    ) -> None:
        """Register an optimization strategy.

        Args:
            name: The strategy name.
            registration: The registration containing factory and capabilities.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._map:
            raise ValueError(f"Strategy '{name}' is already registered")
        self._map[name] = registration

    def list_strategies(self) -> list[OptimizerStrategyName]:
        """List all registered strategy names.

        Returns:
            Sorted list of registered strategy names.
        """
        return sorted(self._map.keys())

    def get(self, name: OptimizerStrategyName) -> HyperparameterOptimizerProtocol:
        """Get an optimizer instance by name.

        Args:
            name: The strategy name to retrieve.

        Returns:
            A new instance of the requested optimizer.

        Raises:
            KeyError: If the strategy is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: OptimizerStrategyName) -> OptimizerStrategyCapabilities:
        """Get capabilities for a strategy by name.

        Args:
            name: The strategy name to query.

        Returns:
            The capabilities of the requested strategy.

        Raises:
            KeyError: If the strategy is not registered.
        """
        return self._map[name].capabilities()

    def has_strategy(self, name: OptimizerStrategyName) -> bool:
        """Check if a strategy is registered.

        Args:
            name: The strategy name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._map


def default_optimizer_registry() -> OptimizerStrategyRegistry:
    """Build the default registry with supported optimization strategies.

    Includes:
    - optuna_tpe: Optuna TPE (Tree-structured Parzen Estimator) Bayesian optimization
    - random_search: Random sampling from search space
    - grid_search: Exhaustive grid search (for small spaces)

    Returns:
        A configured OptimizerStrategyRegistry with all built-in strategies.
    """
    reg = OptimizerStrategyRegistry()

    # Import strategies module
    strategies_mod = __import__(
        "covenant_ml.optimizer.strategies",
        fromlist=[
            "create_optuna_tpe_optimizer",
            "create_random_search_optimizer",
            "create_grid_search_optimizer",
        ],
    )

    # Optuna TPE
    create_optuna_tpe: OptimizerStrategyFactory = strategies_mod.create_optuna_tpe_optimizer
    reg.register("optuna_tpe", OptimizerStrategyRegistration(create_optuna_tpe))

    # Random Search
    create_random_search: OptimizerStrategyFactory = strategies_mod.create_random_search_optimizer
    reg.register("random_search", OptimizerStrategyRegistration(create_random_search))

    # Grid Search
    create_grid_search: OptimizerStrategyFactory = strategies_mod.create_grid_search_optimizer
    reg.register("grid_search", OptimizerStrategyRegistration(create_grid_search))

    return reg


__all__ = [
    "OptimizerStrategyRegistration",
    "OptimizerStrategyRegistry",
    "default_optimizer_registry",
]
