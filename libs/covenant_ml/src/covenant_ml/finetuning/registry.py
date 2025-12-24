"""Registry for pluggable fine-tuning strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Strategies are registered explicitly and accessed by name.
"""

from __future__ import annotations

from .protocol import (
    FineTuningCapabilities,
    FineTuningStrategyFactory,
    FineTuningStrategyName,
    FineTuningStrategyProtocol,
)


class FineTuningRegistration:
    """Registration record holding a factory and cached capabilities."""

    def __init__(self, factory: FineTuningStrategyFactory) -> None:
        """Initialize registration with a factory.

        Args:
            factory: Callable that creates FineTuningStrategyProtocol instances.
        """
        self._factory = factory
        self._capabilities_cache: FineTuningCapabilities | None = None

    def factory(self) -> FineTuningStrategyFactory:
        """Get the factory callable.

        Returns:
            The factory that creates fine-tuning strategy instances.
        """
        return self._factory

    def capabilities(self) -> FineTuningCapabilities:
        """Get capabilities, caching on first access.

        Returns:
            The capabilities of this fine-tuning strategy.
        """
        if self._capabilities_cache is None:
            strategy = self._factory()
            self._capabilities_cache = strategy.capabilities()
        return self._capabilities_cache


class FineTuningRegistry:
    """Registry of fine-tuning strategies keyed by name."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._map: dict[FineTuningStrategyName, FineTuningRegistration] = {}

    def register(
        self,
        name: FineTuningStrategyName,
        registration: FineTuningRegistration,
    ) -> None:
        """Register a fine-tuning strategy.

        Args:
            name: The strategy name.
            registration: The registration containing factory and capabilities.
        """
        self._map[name] = registration

    def list_strategies(self) -> list[FineTuningStrategyName]:
        """List all registered strategy names.

        Returns:
            Sorted list of registered strategy names.
        """
        return sorted(self._map.keys())

    def get(self, name: FineTuningStrategyName) -> FineTuningStrategyProtocol:
        """Get a fine-tuning strategy instance by name.

        Args:
            name: The strategy name to retrieve.

        Returns:
            A new instance of the requested fine-tuning strategy.

        Raises:
            KeyError: If the strategy is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: FineTuningStrategyName) -> FineTuningCapabilities:
        """Get capabilities for a strategy by name.

        Args:
            name: The strategy name to query.

        Returns:
            The capabilities of the requested strategy.

        Raises:
            KeyError: If the strategy is not registered.
        """
        return self._map[name].capabilities()

    def has_strategy(self, name: FineTuningStrategyName) -> bool:
        """Check if a strategy is registered.

        Args:
            name: The strategy name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._map


def default_finetuning_registry() -> FineTuningRegistry:
    """Build the default registry with supported fine-tuning strategies.

    Includes:
    - staged: Multi-stage optimization with narrowing search spaces
    - warm_start: Single-stage optimization initialized from prior results
    - iterative_refinement: Repeated refinement until convergence

    Returns:
        A configured FineTuningRegistry with all built-in strategies.
    """
    reg = FineTuningRegistry()

    # Import strategies module
    strategies_mod = __import__(
        "covenant_ml.finetuning.strategies",
        fromlist=[
            "create_staged_finetuning",
            "create_warm_start_finetuning",
            "create_iterative_refinement_finetuning",
        ],
    )

    # Staged fine-tuning
    create_staged: FineTuningStrategyFactory = strategies_mod.create_staged_finetuning
    reg.register("staged", FineTuningRegistration(create_staged))

    # Warm-start fine-tuning
    create_warm_start: FineTuningStrategyFactory = strategies_mod.create_warm_start_finetuning
    reg.register("warm_start", FineTuningRegistration(create_warm_start))

    # Iterative refinement
    create_iterative: FineTuningStrategyFactory = (
        strategies_mod.create_iterative_refinement_finetuning
    )
    reg.register("iterative_refinement", FineTuningRegistration(create_iterative))

    return reg


__all__ = [
    "FineTuningRegistration",
    "FineTuningRegistry",
    "default_finetuning_registry",
]
