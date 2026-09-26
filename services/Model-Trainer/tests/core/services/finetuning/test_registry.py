"""Tests for the FineTuningRegistry."""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
    StrategyName,
)
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.finetuning.registry import (
    FineTuningRegistry,
    StrategyRegistration,
    default_registry,
)
from model_trainer.core.types import LMModelProto


class FakeStrategy:
    """Fake strategy for testing registry."""

    def __init__(self, strategy_name: str = "fake") -> None:
        """Initialize fake strategy with a name."""
        self._name = strategy_name

    def name(self) -> StrategyName:
        """Return the strategy name."""
        return StrategyName.FULL

    def capabilities(self) -> StrategyCapabilities:
        """Return fake capabilities."""
        return {
            "supports_quantization": False,
            "supports_gradient_checkpointing": True,
            "requires_peft": False,
            "trainable_param_fraction": 1.0,
        }

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Fake adapt method."""
        return AdaptedModel(
            model=model,
            base_model_id=model_id,
            strategy_name=StrategyName.FULL,
            is_peft_model=False,
            lora_config=None,
        )

    def save_adapted(self, adapted: AdaptedModel, out_dir: str) -> None:
        """Fake save method."""
        pass

    def load_adapted(
        self,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Fake load method."""
        return AdaptedModel(
            model=base_model,
            base_model_id=model_id,
            strategy_name=StrategyName.FULL,
            is_peft_model=False,
            lora_config=None,
        )


def create_fake_strategy() -> FakeStrategy:
    """Factory function for fake strategy."""
    return FakeStrategy()


class TestStrategyRegistration:
    """Tests for StrategyRegistration class."""

    def test_factory_returns_provided_factory(self) -> None:
        """Test that factory() returns the factory function."""
        reg = StrategyRegistration(create_fake_strategy)
        assert reg.factory() is create_fake_strategy

    def test_capabilities_creates_strategy_once(self) -> None:
        """Test that capabilities() caches the result."""
        call_count = 0

        def counting_factory() -> FakeStrategy:
            nonlocal call_count
            call_count += 1
            return FakeStrategy()

        reg = StrategyRegistration(counting_factory)

        # First call should create strategy
        caps1 = reg.capabilities()
        assert call_count == 1

        # Second call should use cache
        caps2 = reg.capabilities()
        assert call_count == 1  # Still 1

        # Results should be same object
        assert caps1 is caps2

    def test_capabilities_returns_correct_values(self) -> None:
        """Test that capabilities() returns correct values."""
        reg = StrategyRegistration(create_fake_strategy)
        caps = reg.capabilities()

        assert caps["supports_quantization"] is False
        assert caps["supports_gradient_checkpointing"] is True
        assert caps["requires_peft"] is False
        assert caps["trainable_param_fraction"] == 1.0


class TestFineTuningRegistry:
    """Tests for FineTuningRegistry class."""

    def test_empty_registry_lists_no_strategies(self) -> None:
        """Test that empty registry returns empty list."""
        reg = FineTuningRegistry()
        assert reg.list_strategies() == []

    def test_register_adds_strategy(self) -> None:
        """Test that register() adds a strategy."""
        reg = FineTuningRegistry()
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        assert reg.is_registered(StrategyName.FULL)
        assert StrategyName.FULL in reg.list_strategies()

    def test_list_strategies_returns_sorted_names(self) -> None:
        """Test that list_strategies() returns sorted names."""
        reg = FineTuningRegistry()
        # Register in non-alphabetical order
        reg.register(StrategyName.LORA, StrategyRegistration(create_fake_strategy))
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))
        reg.register(StrategyName.QLORA, StrategyRegistration(create_fake_strategy))

        names = reg.list_strategies()
        assert names == [StrategyName.FULL, StrategyName.LORA, StrategyName.QLORA]

    def test_get_returns_strategy_instance(self) -> None:
        """Test that get() returns a strategy instance."""
        reg = FineTuningRegistry()
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        strategy = reg.get(StrategyName.FULL)
        expected = FakeStrategy()
        assert type(strategy) is type(expected)

    def test_get_creates_new_instance_each_time(self) -> None:
        """Test that get() creates new instances each time."""
        reg = FineTuningRegistry()
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        s1 = reg.get(StrategyName.FULL)
        s2 = reg.get(StrategyName.FULL)
        assert s1 is not s2

    def test_get_raises_for_unregistered_strategy(self) -> None:
        """Test that get() raises KeyError for unregistered strategy."""
        reg = FineTuningRegistry()
        # Only register "full", then try to get "lora" which is not registered
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        with pytest.raises(KeyError):
            reg.get(StrategyName.LORA)

    def test_get_capabilities_returns_cached_capabilities(self) -> None:
        """Test that get_capabilities() returns cached capabilities."""
        call_count = 0

        def counting_factory() -> FakeStrategy:
            nonlocal call_count
            call_count += 1
            return FakeStrategy()

        reg = FineTuningRegistry()
        reg.register(StrategyName.FULL, StrategyRegistration(counting_factory))

        caps1 = reg.get_capabilities(StrategyName.FULL)
        caps2 = reg.get_capabilities(StrategyName.FULL)

        # Should only create strategy once
        assert call_count == 1
        assert caps1 is caps2

    def test_get_capabilities_raises_for_unregistered(self) -> None:
        """Test that get_capabilities() raises KeyError for unregistered."""
        reg = FineTuningRegistry()
        # Only register "full", then try to get capabilities for "lora"
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        with pytest.raises(KeyError):
            reg.get_capabilities(StrategyName.LORA)

    def test_is_registered_true_for_registered(self) -> None:
        """Test that is_registered() returns True for registered strategies."""
        reg = FineTuningRegistry()
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        assert reg.is_registered(StrategyName.FULL) is True

    def test_is_registered_false_for_unregistered(self) -> None:
        """Test that is_registered() returns False for unregistered strategies."""
        reg = FineTuningRegistry()
        # Only register "full", check that "lora" is not registered
        reg.register(StrategyName.FULL, StrategyRegistration(create_fake_strategy))

        assert reg.is_registered(StrategyName.LORA) is False


class TestDefaultRegistry:
    """Tests for the default_registry() function."""

    def test_default_registry_contains_all_strategies(self) -> None:
        """Test that the default registry has every declared strategy.

        Asserted against ``StrategyName`` rather than a hand-written list,
        so a strategy declared but never registered fails here -- which is the
        gap a restated list would hide, and the reason the names were collapsed
        onto one declaration in the first place.
        """
        assert default_registry().list_strategies() == sorted(StrategyName)

    def test_default_registry_full_strategy(self) -> None:
        """Test that full strategy is correctly registered."""
        reg = default_registry()
        strategy = reg.get(StrategyName.FULL)

        assert strategy.name() is StrategyName.FULL
        caps = strategy.capabilities()
        assert caps["requires_peft"] is False
        assert caps["trainable_param_fraction"] == 1.0

    def test_default_registry_lora_strategy(self) -> None:
        """Test that lora strategy is correctly registered."""
        reg = default_registry()
        strategy = reg.get(StrategyName.LORA)

        assert strategy.name() is StrategyName.LORA
        caps = strategy.capabilities()
        assert caps["requires_peft"] is True

    def test_default_registry_qlora_strategy(self) -> None:
        """Test that qlora strategy is correctly registered."""
        reg = default_registry()
        strategy = reg.get(StrategyName.QLORA)

        assert strategy.name() is StrategyName.QLORA
        caps = strategy.capabilities()
        assert caps["requires_peft"] is True
        assert caps["supports_quantization"] is True

    def test_default_registry_returns_new_instance(self) -> None:
        """Test that default_registry() returns new registry each time."""
        reg1 = default_registry()
        reg2 = default_registry()
        assert reg1 is not reg2

    def test_default_registry_get_capabilities_cached(self) -> None:
        """Test that capabilities are cached within a registry instance."""
        reg = default_registry()

        # Get capabilities twice
        caps1 = reg.get_capabilities(StrategyName.FULL)
        caps2 = reg.get_capabilities(StrategyName.FULL)

        # Should be same object (cached)
        assert caps1 is caps2
