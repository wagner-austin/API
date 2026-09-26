"""Tests for optimizer strategy registry.

Tests cover:
- Registry creation
- Strategy registration
- Strategy retrieval
- Default registry population
- Error handling
"""

from __future__ import annotations

import pytest

from covenant_ml.optimizer import (
    OptimizerStrategyName,
    OptimizerStrategyRegistration,
    OptimizerStrategyRegistry,
    default_optimizer_registry,
)
from covenant_ml.optimizer.strategies import RandomSearchOptimizer

# =============================================================================
# OptimizerStrategyRegistry Tests
# =============================================================================


class TestOptimizerStrategyRegistry:
    """Tests for OptimizerStrategyRegistry."""

    def test_empty_registry(self) -> None:
        """New registry has no registered strategies."""
        registry = OptimizerStrategyRegistry()
        assert registry.list_strategies() == []

    def test_register_and_get(self) -> None:
        """Can register and retrieve a strategy."""
        registry = OptimizerStrategyRegistry()

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)

        optimizer = registry.get(OptimizerStrategyName.RANDOM_SEARCH)
        assert optimizer.strategy_name() is OptimizerStrategyName.RANDOM_SEARCH

    def test_list_strategies(self) -> None:
        """List returns all registered strategy names."""
        registry = OptimizerStrategyRegistry()

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)
        registry.register(OptimizerStrategyName.GRID_SEARCH, registration)

        strategies = registry.list_strategies()
        assert OptimizerStrategyName.RANDOM_SEARCH in strategies
        assert OptimizerStrategyName.GRID_SEARCH in strategies
        assert len(strategies) == 2

    def test_duplicate_registration_raises(self) -> None:
        """Registering same name twice raises ValueError."""
        registry = OptimizerStrategyRegistry()

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)

    def test_has_strategy_returns_true_when_registered(self) -> None:
        """has_strategy returns True for registered strategies."""
        registry = OptimizerStrategyRegistry()

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)

        assert registry.has_strategy(OptimizerStrategyName.RANDOM_SEARCH) is True

    def test_has_strategy_returns_false_when_not_registered(self) -> None:
        """has_strategy returns False for unregistered strategies."""
        registry = OptimizerStrategyRegistry()
        assert registry.has_strategy(OptimizerStrategyName.RANDOM_SEARCH) is False

    def test_get_capabilities_returns_strategy_capabilities(self) -> None:
        """get_capabilities returns the strategy's capabilities."""
        registry = OptimizerStrategyRegistry()

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        registry.register(OptimizerStrategyName.RANDOM_SEARCH, registration)

        caps = registry.get_capabilities(OptimizerStrategyName.RANDOM_SEARCH)
        assert caps["supports_pruning"] is False
        assert caps["supports_parallel"] is True
        assert caps["is_deterministic"] is True
        assert caps["requires_bounds"] is True


# =============================================================================
# OptimizerStrategyRegistration Tests
# =============================================================================


class TestOptimizerStrategyRegistration:
    """Tests for OptimizerStrategyRegistration."""

    def test_capabilities_caches_result(self) -> None:
        """capabilities() caches the result after first call."""
        call_count = 0

        def counting_factory() -> RandomSearchOptimizer:
            nonlocal call_count
            call_count += 1
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(counting_factory)

        # First call should invoke factory
        caps1 = registration.capabilities()
        assert call_count == 1
        assert caps1["is_deterministic"] is True

        # Second call should use cache
        caps2 = registration.capabilities()
        assert call_count == 1  # Still 1, not 2
        assert caps2["is_deterministic"] is True

    def test_capabilities_returns_correct_values(self) -> None:
        """capabilities() returns the optimizer's capabilities."""

        def factory() -> RandomSearchOptimizer:
            return RandomSearchOptimizer()

        registration = OptimizerStrategyRegistration(factory)
        caps = registration.capabilities()

        assert caps["supports_pruning"] is False
        assert caps["supports_parallel"] is True


# =============================================================================
# Default Registry Tests
# =============================================================================


class TestDefaultOptimizerRegistry:
    """Tests for the default optimizer registry."""

    def test_default_registry_has_strategies(self) -> None:
        """Default registry has expected strategies."""
        registry = default_optimizer_registry()
        strategies = registry.list_strategies()

        assert OptimizerStrategyName.RANDOM_SEARCH in strategies
        assert OptimizerStrategyName.GRID_SEARCH in strategies
        assert OptimizerStrategyName.OPTUNA_TPE in strategies

    def test_random_search_works(self) -> None:
        """Random search from registry works correctly."""
        registry = default_optimizer_registry()
        optimizer = registry.get(OptimizerStrategyName.RANDOM_SEARCH)

        assert optimizer.strategy_name() is OptimizerStrategyName.RANDOM_SEARCH

    def test_grid_search_works(self) -> None:
        """Grid search from registry works correctly."""
        registry = default_optimizer_registry()
        optimizer = registry.get(OptimizerStrategyName.GRID_SEARCH)

        assert optimizer.strategy_name() is OptimizerStrategyName.GRID_SEARCH

    def test_optuna_tpe_works(self) -> None:
        """Optuna TPE from registry works correctly."""
        registry = default_optimizer_registry()
        optimizer = registry.get(OptimizerStrategyName.OPTUNA_TPE)

        assert optimizer.strategy_name() is OptimizerStrategyName.OPTUNA_TPE

    def test_each_call_returns_fresh_registry(self) -> None:
        """Each call to default_optimizer_registry returns a new instance."""
        registry1 = default_optimizer_registry()
        registry2 = default_optimizer_registry()

        # They should be different instances
        assert registry1 is not registry2

    def test_each_get_returns_fresh_instance(self) -> None:
        """Each get call returns a new optimizer instance."""
        registry = default_optimizer_registry()

        optimizer1 = registry.get(OptimizerStrategyName.RANDOM_SEARCH)
        optimizer2 = registry.get(OptimizerStrategyName.RANDOM_SEARCH)

        assert optimizer1 is not optimizer2
