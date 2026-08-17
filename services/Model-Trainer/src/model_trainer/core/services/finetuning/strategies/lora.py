"""LoRA fine-tuning strategy via PEFT library.

Low-Rank Adaptation reduces trainable parameters by decomposing weight updates
into low-rank matrices, enabling efficient fine-tuning of large models.
"""

from __future__ import annotations

from pathlib import Path

from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
    StrategyName,
)
from model_trainer.core.contracts.model import LoraConfig, ModelTrainConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.types import LMModelProto


def _require_lora_config(cfg: ModelTrainConfig) -> LoraConfig:
    """Extract and validate LoRA config from training config.

    Args:
        cfg: Training configuration.

    Returns:
        LoraConfig from the training config.

    Raises:
        ValueError: If LoRA config is missing or disabled.
    """
    lora_cfg = cfg.get("lora")
    if lora_cfg is None:
        raise ValueError("LoRA strategy requires lora config in ModelTrainConfig")
    if not lora_cfg["enabled"]:
        raise ValueError("LoRA strategy requires lora.enabled=True")
    return lora_cfg


class LoRAStrategy:
    """LoRA fine-tuning strategy via PEFT.

    Trains only low-rank adapter matrices while keeping base model frozen.
    Typically reduces trainable parameters to ~0.1-1% of original.

    Attributes:
        _name: Strategy identifier "lora".
    """

    def __init__(self) -> None:
        """Initialize LoRA strategy."""
        self._name: StrategyName = "lora"

    def name(self) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "lora".
        """
        return self._name

    def capabilities(self) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities showing LoRA is parameter-efficient and requires PEFT.
        """
        return StrategyCapabilities(
            supports_quantization=False,
            supports_gradient_checkpointing=True,
            requires_peft=True,
            trainable_param_fraction=0.01,  # Approximate: varies by rank
        )

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Adapt a model with LoRA adapters via PEFT.

        Args:
            model: Base model to wrap with adapters.
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration with LoRA settings.

        Returns:
            AdaptedModel with LoRA adapters attached.

        Raises:
            ValueError: If LoRA config is missing.
        """
        lora_cfg = _require_lora_config(cfg)

        # Enable gradient checkpointing for memory efficiency if available
        Hooks.enable_gradient_checkpointing(model)

        peft_model = Hooks.create_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        )

        return AdaptedModel(
            model=peft_model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=True,
            lora_config=lora_cfg,
        )

    def save_adapted(
        self,
        adapted: AdaptedModel,
        out_dir: str,
    ) -> None:
        """Save LoRA adapter weights to disk.

        Only saves the adapter weights, not the full base model.

        Args:
            adapted: The adapted model with LoRA adapters.
            out_dir: Output directory path.

        Raises:
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Hooks.save_peft_model(adapted.model, out_dir)

    def load_adapted(
        self,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Load LoRA adapter weights and apply to base model.

        Args:
            base_model: Base model to apply adapters to.
            model_id: HuggingFace model ID.
            adapter_path: Path to saved adapter weights.

        Returns:
            AdaptedModel with adapters loaded.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
        """
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        loaded_model = Hooks.load_peft_model(base_model, adapter_path)

        # Try to load adapter config to reconstruct LoraConfig
        # For now, return None as we don't have the original config stored
        return AdaptedModel(
            model=loaded_model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=True,
            lora_config=None,  # Config not preserved on load
        )


def create_lora_strategy() -> LoRAStrategy:
    """Factory function to create a LoRAStrategy.

    Returns:
        New LoRAStrategy instance.
    """
    return LoRAStrategy()


__all__ = [
    "LoRAStrategy",
    "create_lora_strategy",
]
