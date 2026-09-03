"""Protocols and types for pluggable fine-tuning strategies.

Follows the covenant_ml pattern of Protocol + Registry for extensibility.
Strategies adapt models for efficient fine-tuning (LoRA, QLoRA, Full).

Strict typing: no Any, cast, type: ignore, .pyi, or stubs.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from model_trainer.core.contracts.model import (
    LoraConfig,
    ModelTrainConfig,
)
from model_trainer.core.contracts.strategy_names import StrategyName
from model_trainer.core.types import LMModelProto


class StrategyCapabilities(TypedDict):
    """Declares what features a fine-tuning strategy supports.

    Used for capability discovery and validation before applying strategy.
    """

    supports_quantization: bool
    supports_gradient_checkpointing: bool
    requires_peft: bool
    trainable_param_fraction: float  # Approximate fraction of params trained (1.0 = all)


class AdaptedModel:
    """Model adapted for fine-tuning with strategy-specific metadata.

    Wraps the base model with information about how it was adapted,
    enabling correct save/load behavior for LoRA adapters vs full weights.

    Attributes:
        model: The adapted model (may have LoRA adapters attached).
        base_model_id: HuggingFace model ID of the base model.
        strategy_name: Name of the strategy that adapted this model.
        is_peft_model: Whether model has PEFT adapters (affects save/load).
        lora_config: LoRA configuration if applicable.
    """

    model: LMModelProto
    base_model_id: str
    strategy_name: StrategyName
    is_peft_model: bool
    lora_config: LoraConfig | None

    def __init__(
        self: AdaptedModel,
        *,
        model: LMModelProto,
        base_model_id: str,
        strategy_name: StrategyName,
        is_peft_model: bool,
        lora_config: LoraConfig | None,
    ) -> None:
        """Initialize adapted model wrapper.

        Args:
            model: The adapted model instance.
            base_model_id: HuggingFace model ID of the base model.
            strategy_name: Name of the fine-tuning strategy used.
            is_peft_model: Whether the model has PEFT/LoRA adapters.
            lora_config: LoRA configuration if adapters were applied.
        """
        self.model = model
        self.base_model_id = base_model_id
        self.strategy_name = strategy_name
        self.is_peft_model = is_peft_model
        self.lora_config = lora_config


class FineTuningStrategy(Protocol):
    """Protocol for pluggable fine-tuning strategy implementations.

    Each strategy defines how to adapt a base model for training.
    Strategies are registered in FineTuningRegistry and selected by name.

    Implementations:
        - FullFineTuneStrategy: Train all parameters (no adapters)
        - LoRAStrategy: Low-rank adaptation via PEFT
        - QLoRAStrategy: Quantized LoRA (4-bit base + LoRA)
    """

    def name(self: FineTuningStrategy) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal type.
        """
        ...

    def capabilities(self: FineTuningStrategy) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities describing what this strategy supports.
        """
        ...

    def adapt(
        self: FineTuningStrategy,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Adapt a model for fine-tuning using this strategy.

        Args:
            model: Base model to adapt.
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration with strategy-specific settings.

        Returns:
            AdaptedModel wrapping the modified model.

        Raises:
            ValueError: If required config is missing for this strategy.
            RuntimeError: If required libraries are not available.
        """
        ...

    def save_adapted(
        self: FineTuningStrategy,
        adapted: AdaptedModel,
        out_dir: str,
    ) -> None:
        """Save an adapted model to disk.

        For PEFT models, saves only adapter weights.
        For full models, saves complete weights.

        Args:
            adapted: The adapted model to save.
            out_dir: Output directory path.
        """
        ...

    def load_adapted(
        self: FineTuningStrategy,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Load adapter weights and apply to base model.

        Args:
            base_model: Base model to apply adapters to.
            model_id: HuggingFace model ID.
            adapter_path: Path to saved adapter weights.

        Returns:
            AdaptedModel with adapters loaded.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
        """
        ...


class StrategyFactory(Protocol):
    """Factory protocol to construct a strategy implementation."""

    def __call__(self) -> FineTuningStrategy:
        """Create a new strategy instance.

        Returns:
            Strategy implementation.
        """
        ...


__all__ = [
    "AdaptedModel",
    "FineTuningStrategy",
    "StrategyCapabilities",
    "StrategyFactory",
    "StrategyName",
]
