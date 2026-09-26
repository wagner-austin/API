"""Tests for fine-tuning strategy registry.

Tests cover:
- FineTuningRegistration
- FineTuningRegistry
- default_finetuning_registry
"""

from __future__ import annotations

import pytest

from covenant_ml.finetuning import (
    FineTuningRegistration,
    FineTuningRegistry,
    FineTuningStrategyName,
    default_finetuning_registry,
)
from covenant_ml.finetuning.testing import FakeFineTuningStrategy

# =============================================================================
# FineTuningRegistration Tests
# =============================================================================


class TestFineTuningRegistration:
    """Tests for FineTuningRegistration."""

    def test_factory_returns_callable(self) -> None:
        """Factory method returns the factory callable."""

        def factory() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy()

        registration = FineTuningRegistration(factory)
        assert registration.factory() is factory

    def test_capabilities_caches_result(self) -> None:
        """Capabilities are cached after first access."""
        call_count = 0

        def factory() -> FakeFineTuningStrategy:
            nonlocal call_count
            call_count += 1
            return FakeFineTuningStrategy()

        registration = FineTuningRegistration(factory)

        # First call creates instance
        caps1 = registration.capabilities()
        assert call_count == 1

        # Second call uses cache
        caps2 = registration.capabilities()
        assert call_count == 1

        assert caps1 == caps2


# =============================================================================
# FineTuningRegistry Tests
# =============================================================================


class TestFineTuningRegistry:
    """Tests for FineTuningRegistry."""

    def test_empty_registry(self) -> None:
        """New registry has no strategies."""
        registry = FineTuningRegistry()
        assert registry.list_strategies() == []

    def test_register_and_get(self) -> None:
        """Can register and retrieve a strategy."""
        registry = FineTuningRegistry()

        def factory() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy(name=FineTuningStrategyName.STAGED)

        registration = FineTuningRegistration(factory)
        registry.register(FineTuningStrategyName.STAGED, registration)

        strategy = registry.get(FineTuningStrategyName.STAGED)
        assert strategy.strategy_name() is FineTuningStrategyName.STAGED

    def test_list_strategies(self) -> None:
        """List returns all registered strategy names."""
        registry = FineTuningRegistry()

        def factory_staged() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy(name=FineTuningStrategyName.STAGED)

        def factory_warm() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy(name=FineTuningStrategyName.WARM_START)

        registry.register(FineTuningStrategyName.STAGED, FineTuningRegistration(factory_staged))
        registry.register(FineTuningStrategyName.WARM_START, FineTuningRegistration(factory_warm))

        strategies = registry.list_strategies()
        assert FineTuningStrategyName.STAGED in strategies
        assert FineTuningStrategyName.WARM_START in strategies
        assert len(strategies) == 2

    def test_get_unknown_strategy_raises(self) -> None:
        """Getting unknown strategy raises KeyError."""
        registry = FineTuningRegistry()

        with pytest.raises(KeyError):
            registry.get(FineTuningStrategyName.STAGED)

    def test_get_capabilities(self) -> None:
        """Can retrieve capabilities for a strategy."""
        registry = FineTuningRegistry()

        def factory() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy()

        registry.register(FineTuningStrategyName.STAGED, FineTuningRegistration(factory))

        caps = registry.get_capabilities(FineTuningStrategyName.STAGED)
        assert caps["supports_warm_start"] is True
        assert caps["supports_staged"] is True

    def test_has_strategy(self) -> None:
        """has_strategy returns correct boolean."""
        registry = FineTuningRegistry()

        def factory() -> FakeFineTuningStrategy:
            return FakeFineTuningStrategy()

        registry.register(FineTuningStrategyName.STAGED, FineTuningRegistration(factory))

        assert registry.has_strategy(FineTuningStrategyName.STAGED) is True
        # Check for a valid name that wasn't registered
        assert registry.has_strategy(FineTuningStrategyName.WARM_START) is False


# =============================================================================
# Default Registry Tests
# =============================================================================


class TestDefaultFineTuningRegistry:
    """Tests for default_finetuning_registry."""

    def test_has_all_strategies(self) -> None:
        """Default registry has expected strategies."""
        registry = default_finetuning_registry()
        strategies = registry.list_strategies()

        assert FineTuningStrategyName.STAGED in strategies
        assert FineTuningStrategyName.WARM_START in strategies
        assert FineTuningStrategyName.ITERATIVE_REFINEMENT in strategies

    def test_staged_works(self) -> None:
        """Staged strategy from registry works."""
        registry = default_finetuning_registry()
        strategy = registry.get(FineTuningStrategyName.STAGED)

        assert strategy.strategy_name() is FineTuningStrategyName.STAGED
        caps = strategy.capabilities()
        assert caps["supports_staged"] is True

    def test_warm_start_works(self) -> None:
        """Warm start strategy from registry works."""
        registry = default_finetuning_registry()
        strategy = registry.get(FineTuningStrategyName.WARM_START)

        assert strategy.strategy_name() is FineTuningStrategyName.WARM_START
        caps = strategy.capabilities()
        assert caps["supports_warm_start"] is True

    def test_iterative_refinement_works(self) -> None:
        """Iterative refinement strategy from registry works."""
        registry = default_finetuning_registry()
        strategy = registry.get(FineTuningStrategyName.ITERATIVE_REFINEMENT)

        assert strategy.strategy_name() is FineTuningStrategyName.ITERATIVE_REFINEMENT
        caps = strategy.capabilities()
        assert caps["supports_early_stop"] is True

    def test_each_call_returns_fresh_registry(self) -> None:
        """Each call to default_finetuning_registry returns new instance."""
        registry1 = default_finetuning_registry()
        registry2 = default_finetuning_registry()

        assert registry1 is not registry2

    def test_each_get_returns_fresh_instance(self) -> None:
        """Each get call returns a new strategy instance."""
        registry = default_finetuning_registry()

        strategy1 = registry.get(FineTuningStrategyName.STAGED)
        strategy2 = registry.get(FineTuningStrategyName.STAGED)

        assert strategy1 is not strategy2
