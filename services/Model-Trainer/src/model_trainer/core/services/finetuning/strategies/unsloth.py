"""Unsloth-optimized LoRA fine-tuning strategy.

Unsloth provides 2x-5x faster fine-tuning through optimized kernels
and memory-efficient implementations for supported model architectures.
"""

from __future__ import annotations

from pathlib import Path

from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
    StrategyName,
)
from model_trainer.core.contracts.model import (
    LoraConfig,
    ModelTrainConfig,
    UnslothConfig,
)
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
        raise ValueError("Unsloth strategy requires lora config in ModelTrainConfig")
    if not lora_cfg["enabled"]:
        raise ValueError("Unsloth strategy requires lora.enabled=True")
    return lora_cfg


def _require_unsloth_config(cfg: ModelTrainConfig) -> UnslothConfig:
    """Extract and validate Unsloth config from training config.

    Args:
        cfg: Training configuration.

    Returns:
        UnslothConfig from the training config.

    Raises:
        ValueError: If Unsloth config is missing or disabled.
    """
    unsloth_cfg = cfg.get("unsloth")
    if unsloth_cfg is None:
        raise ValueError("Unsloth strategy requires unsloth config in ModelTrainConfig")
    if not unsloth_cfg["enabled"]:
        raise ValueError("Unsloth strategy requires unsloth.enabled=True")
    return unsloth_cfg


class UnslothStrategy:
    """Unsloth-optimized LoRA fine-tuning strategy.

    Uses Unsloth's optimized kernels for 2x-5x faster training.
    Automatically handles 4-bit quantization and LoRA application.

    Attributes:
        _name: Strategy identifier "unsloth".
    """

    def __init__(self) -> None:
        """Initialize Unsloth strategy."""
        self._name: StrategyName = "unsloth"

    def name(self) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "unsloth".
        """
        return self._name

    def capabilities(self) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities showing Unsloth requires the unsloth library.
        """
        return StrategyCapabilities(
            supports_quantization=True,
            supports_gradient_checkpointing=True,
            requires_peft=False,  # Unsloth has its own PEFT implementation
            requires_unsloth=True,
            trainable_param_fraction=0.01,  # LoRA adapter params only
        )

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Adapt a model using Unsloth's optimized LoRA.

        Note: For Unsloth, the model should be loaded via Unsloth's loader.
        This method expects an Unsloth-loaded model and applies LoRA.

        Args:
            model: Model loaded via Unsloth (or compatible).
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration with LoRA and Unsloth settings.

        Returns:
            AdaptedModel with Unsloth-optimized LoRA adapters.

        Raises:
            ValueError: If required configs are missing.
            RuntimeError: If Unsloth hook is not set.
        """
        lora_cfg = _require_lora_config(cfg)
        # Validate unsloth config exists
        _require_unsloth_config(cfg)

        if Hooks.apply_unsloth_peft is None:
            raise RuntimeError("Unsloth hook not configured. Set Hooks.apply_unsloth_peft.")

        peft_model = Hooks.apply_unsloth_peft(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
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
        """Save Unsloth LoRA adapter weights to disk.

        Unsloth models save via the standard PEFT save_pretrained method.

        Args:
            adapted: The adapted model with Unsloth LoRA adapters.
            out_dir: Output directory path.

        Raises:
            RuntimeError: If PEFT save hook is not set.
        """
        if Hooks.save_peft_model is None:
            raise RuntimeError("PEFT save hook not configured. Set Hooks.save_peft_model.")

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Hooks.save_peft_model(adapted.model, out_dir)

    def load_adapted(
        self,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Load LoRA adapter weights onto an Unsloth-loaded base model.

        The base_model should be loaded via Unsloth for optimal performance.

        Args:
            base_model: Unsloth-loaded base model to apply adapters to.
            model_id: HuggingFace model ID.
            adapter_path: Path to saved adapter weights.

        Returns:
            AdaptedModel with adapters loaded.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
            RuntimeError: If PEFT load hook is not set.
        """
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        if Hooks.load_peft_model is None:
            raise RuntimeError("PEFT load hook not configured. Set Hooks.load_peft_model.")

        loaded_model = Hooks.load_peft_model(base_model, adapter_path)

        return AdaptedModel(
            model=loaded_model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=True,
            lora_config=None,  # Config not preserved on load
        )


def create_unsloth_strategy() -> UnslothStrategy:
    """Factory function to create an UnslothStrategy.

    Returns:
        New UnslothStrategy instance.
    """
    return UnslothStrategy()


__all__ = [
    "UnslothStrategy",
    "create_unsloth_strategy",
]
