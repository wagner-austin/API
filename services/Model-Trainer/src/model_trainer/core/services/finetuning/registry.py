"""Registry for pluggable fine-tuning strategies.

Follows the covenant_ml pattern: Protocol + Registration + Registry.
Strict typing; no optional fallbacks. Strategies are registered explicitly.
"""

from __future__ import annotations

from model_trainer.core.contracts.finetuning import (
    FineTuningStrategy,
    StrategyCapabilities,
    StrategyFactory,
    StrategyName,
)


class StrategyRegistration:
    """Registration record holding a factory and cached capabilities.

    Caches capabilities after first access to avoid repeated instantiation.
    """

    def __init__(self, factory: StrategyFactory) -> None:
        """Initialize registration with a factory function.

        Args:
            factory: Callable that creates strategy instances.
        """
        self._factory = factory
        self._capabilities_cache: StrategyCapabilities | None = None

    def factory(self) -> StrategyFactory:
        """Return the factory function.

        Returns:
            Strategy factory callable.
        """
        return self._factory

    def capabilities(self) -> StrategyCapabilities:
        """Return cached capabilities, creating strategy once if needed.

        Returns:
            Strategy capabilities.
        """
        if self._capabilities_cache is None:
            strategy = self._factory()
            self._capabilities_cache = strategy.capabilities()
        return self._capabilities_cache


class FineTuningRegistry:
    """Registry of fine-tuning strategies keyed by name.

    Provides lookup, enumeration, and capability querying for strategies.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._map: dict[StrategyName, StrategyRegistration] = {}

    def register(self, name: StrategyName, registration: StrategyRegistration) -> None:
        """Register a strategy by name.

        Args:
            name: Strategy identifier.
            registration: Registration containing factory and capabilities.
        """
        self._map[name] = registration

    def list_strategies(self) -> list[StrategyName]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy names.
        """
        return sorted(self._map.keys())

    def get(self, name: StrategyName) -> FineTuningStrategy:
        """Get a strategy instance by name.

        Args:
            name: Strategy identifier.

        Returns:
            Strategy instance.

        Raises:
            KeyError: If strategy is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: StrategyName) -> StrategyCapabilities:
        """Get capabilities for a strategy without full instantiation.

        Args:
            name: Strategy identifier.

        Returns:
            Strategy capabilities.

        Raises:
            KeyError: If strategy is not registered.
        """
        return self._map[name].capabilities()

    def is_registered(self, name: StrategyName) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy identifier.

        Returns:
            True if strategy is registered.
        """
        return name in self._map


def default_registry() -> FineTuningRegistry:
    """Build the default registry with all supported strategies.

    Includes:
        - full: Train all parameters (no adapters)
        - lora: LoRA via PEFT library
        - qlora: Quantized LoRA (4-bit + LoRA)

    Returns:
        Registry with all strategies registered.
    """
    reg = FineTuningRegistry()

    # Full fine-tuning strategy
    full_mod = __import__(
        "model_trainer.core.services.finetuning.strategies.full",
        fromlist=["create_full_strategy"],
    )
    create_full_strategy: StrategyFactory = full_mod.create_full_strategy
    reg.register("full", StrategyRegistration(create_full_strategy))

    # LoRA strategy
    lora_mod = __import__(
        "model_trainer.core.services.finetuning.strategies.lora",
        fromlist=["create_lora_strategy"],
    )
    create_lora_strategy: StrategyFactory = lora_mod.create_lora_strategy
    reg.register("lora", StrategyRegistration(create_lora_strategy))

    # QLoRA strategy
    qlora_mod = __import__(
        "model_trainer.core.services.finetuning.strategies.qlora",
        fromlist=["create_qlora_strategy"],
    )
    create_qlora_strategy: StrategyFactory = qlora_mod.create_qlora_strategy
    reg.register("qlora", StrategyRegistration(create_qlora_strategy))

    return reg


__all__ = [
    "FineTuningRegistry",
    "StrategyRegistration",
    "default_registry",
]
