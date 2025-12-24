"""Full fine-tuning strategy - trains all model parameters.

No adapters, no quantization. Simple baseline strategy for comparison.
"""

from __future__ import annotations

from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
    StrategyName,
)
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.types import LMModelProto


class FullFineTuneStrategy:
    """Full parameter fine-tuning strategy.

    Trains all model parameters without any parameter-efficient methods.
    Use for small models or when maximum quality is required.

    Attributes:
        _name: Strategy identifier "full".
    """

    def __init__(self) -> None:
        """Initialize full fine-tuning strategy."""
        self._name: StrategyName = "full"

    def name(self) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "full".
        """
        return self._name

    def capabilities(self) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities showing full parameter training with no special requirements.
        """
        return StrategyCapabilities(
            supports_quantization=False,
            supports_gradient_checkpointing=True,
            requires_peft=False,
            requires_unsloth=False,
            trainable_param_fraction=1.0,
        )

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Adapt a model for full fine-tuning.

        Enables gradient checkpointing if configured, but otherwise
        leaves the model unchanged since all parameters will be trained.

        Args:
            model: Base model to adapt.
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration.

        Returns:
            AdaptedModel wrapping the original model.
        """
        # Enable gradient checkpointing for memory efficiency if hook is set
        if Hooks.enable_gradient_checkpointing is not None:
            Hooks.enable_gradient_checkpointing(model)

        return AdaptedModel(
            model=model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=False,
            lora_config=None,
        )

    def save_adapted(
        self,
        adapted: AdaptedModel,
        out_dir: str,
    ) -> None:
        """Save full model weights to disk.

        Args:
            adapted: The adapted model to save.
            out_dir: Output directory path.
        """
        # Full models save all weights via save_pretrained
        adapted.model.save_pretrained(out_dir)

    def load_adapted(
        self,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Load model from saved checkpoint.

        For full fine-tuning, the adapter_path contains complete model weights,
        not adapter deltas. The base_model parameter is ignored.

        Args:
            base_model: Base model (unused for full strategy).
            model_id: HuggingFace model ID.
            adapter_path: Path to saved model weights.

        Returns:
            AdaptedModel with loaded weights.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
        """
        from pathlib import Path

        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Model path not found: {adapter_path}")

        # For full fine-tuning, we load the entire model from the checkpoint
        # The base_model is not used - we load fresh from adapter_path
        if Hooks.load_full_model is None:
            raise RuntimeError("Full model loader hook not configured. Set Hooks.load_full_model.")
        loaded_model = Hooks.load_full_model(adapter_path)

        return AdaptedModel(
            model=loaded_model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=False,
            lora_config=None,
        )


def create_full_strategy() -> FullFineTuneStrategy:
    """Factory function to create a FullFineTuneStrategy.

    Returns:
        New FullFineTuneStrategy instance.
    """
    return FullFineTuneStrategy()


__all__ = [
    "FullFineTuneStrategy",
    "create_full_strategy",
]
