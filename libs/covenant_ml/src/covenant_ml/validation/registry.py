"""Registry for pluggable cross-validation strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Strategies are registered explicitly and accessed by name.
"""

from __future__ import annotations

from .protocol import (
    CVSplitterFactory,
    CVSplitterProtocol,
    CVStrategyCapabilities,
    CVStrategyName,
)


class CVSplitterRegistration:
    """Registration record holding a factory and cached capabilities."""

    def __init__(self, factory: CVSplitterFactory) -> None:
        """Initialize registration with a factory.

        Args:
            factory: Callable that creates CVSplitterProtocol instances.
        """
        self._factory = factory
        self._capabilities_cache: CVStrategyCapabilities | None = None

    def factory(self) -> CVSplitterFactory:
        """Get the factory callable.

        Returns:
            The factory that creates splitter instances.
        """
        return self._factory

    def capabilities(self) -> CVStrategyCapabilities:
        """Get capabilities, caching on first access.

        Returns:
            The capabilities of this CV strategy.
        """
        if self._capabilities_cache is None:
            splitter = self._factory()
            self._capabilities_cache = splitter.capabilities()
        return self._capabilities_cache


class CVSplitterRegistry:
    """Registry of CV splitter strategies keyed by name."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._map: dict[CVStrategyName, CVSplitterRegistration] = {}

    def register(
        self,
        name: CVStrategyName,
        registration: CVSplitterRegistration,
    ) -> None:
        """Register a CV splitter strategy.

        Args:
            name: The strategy name.
            registration: The registration containing factory and capabilities.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._map:
            raise ValueError(f"Strategy '{name}' is already registered")
        self._map[name] = registration

    def list_strategies(self) -> list[CVStrategyName]:
        """List all registered strategy names.

        Returns:
            Sorted list of registered strategy names.
        """
        return sorted(self._map.keys())

    def get(self, name: CVStrategyName) -> CVSplitterProtocol:
        """Get a CV splitter instance by name.

        Args:
            name: The strategy name to retrieve.

        Returns:
            A new instance of the requested CV splitter.

        Raises:
            KeyError: If the strategy is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: CVStrategyName) -> CVStrategyCapabilities:
        """Get capabilities for a strategy by name.

        Args:
            name: The strategy name to query.

        Returns:
            The capabilities of the requested strategy.

        Raises:
            KeyError: If the strategy is not registered.
        """
        return self._map[name].capabilities()

    def has_strategy(self, name: CVStrategyName) -> bool:
        """Check if a strategy is registered.

        Args:
            name: The strategy name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._map


def default_cv_registry() -> CVSplitterRegistry:
    """Build the default registry with supported CV strategies.

    Includes:
    - stratified_kfold: Stratified k-fold maintaining class proportions
    - group_stratified_kfold: Group-aware stratified k-fold
    - shuffle_split: Stratified shuffle split with configurable test size
    - time_series: Time series split preserving temporal order

    Returns:
        A configured CVSplitterRegistry with all built-in strategies.
    """
    reg = CVSplitterRegistry()

    # Import strategies module
    strategies_mod = __import__(
        "covenant_ml.validation.strategies",
        fromlist=[
            "create_stratified_kfold_splitter",
            "create_group_stratified_kfold_splitter",
            "create_shuffle_split_splitter",
            "create_time_series_splitter",
        ],
    )

    # Stratified K-Fold
    create_stratified_kfold: CVSplitterFactory = strategies_mod.create_stratified_kfold_splitter
    reg.register("stratified_kfold", CVSplitterRegistration(create_stratified_kfold))

    # Group Stratified K-Fold
    create_group_stratified: CVSplitterFactory = (
        strategies_mod.create_group_stratified_kfold_splitter
    )
    reg.register("group_stratified_kfold", CVSplitterRegistration(create_group_stratified))

    # Shuffle Split
    create_shuffle_split: CVSplitterFactory = strategies_mod.create_shuffle_split_splitter
    reg.register("shuffle_split", CVSplitterRegistration(create_shuffle_split))

    # Time Series Split
    create_time_series: CVSplitterFactory = strategies_mod.create_time_series_splitter
    reg.register("time_series", CVSplitterRegistration(create_time_series))

    return reg


__all__ = [
    "CVSplitterRegistration",
    "CVSplitterRegistry",
    "default_cv_registry",
]
