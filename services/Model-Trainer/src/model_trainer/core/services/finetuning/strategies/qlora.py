"""QLoRA fine-tuning strategy - Quantized LoRA.

Combines 4-bit quantization of base model with LoRA adapters for
memory-efficient fine-tuning of large models on consumer hardware.
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
    QuantizationConfig,
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
        raise ValueError("QLoRA strategy requires lora config in ModelTrainConfig")
    if not lora_cfg["enabled"]:
        raise ValueError("QLoRA strategy requires lora.enabled=True")
    return lora_cfg


def _require_quantization_config(cfg: ModelTrainConfig) -> QuantizationConfig:
    """Extract and validate quantization config from training config.

    Args:
        cfg: Training configuration.

    Returns:
        QuantizationConfig from the training config.

    Raises:
        ValueError: If quantization config is missing or invalid.
    """
    quant_cfg = cfg.get("quantization")
    if quant_cfg is None:
        raise ValueError("QLoRA strategy requires quantization config in ModelTrainConfig")
    if not quant_cfg["load_in_4bit"] and not quant_cfg["load_in_8bit"]:
        raise ValueError("QLoRA strategy requires either load_in_4bit=True or load_in_8bit=True")
    return quant_cfg


class QLoRAStrategy:
    """Quantized LoRA fine-tuning strategy.

    Loads base model in 4-bit or 8-bit precision, then applies LoRA adapters
    for training. Dramatically reduces memory usage while maintaining quality.

    Attributes:
        _name: Strategy identifier "qlora".
    """

    def __init__(self) -> None:
        """Initialize QLoRA strategy."""
        self._name: StrategyName = "qlora"

    def name(self) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "qlora".
        """
        return self._name

    def capabilities(self) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities showing QLoRA supports quantization and requires PEFT.
        """
        return StrategyCapabilities(
            supports_quantization=True,
            supports_gradient_checkpointing=True,
            requires_peft=True,
            trainable_param_fraction=0.01,  # Only LoRA params are trainable
        )

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Adapt a model with quantization and LoRA.

        Note: For QLoRA, the model should already be loaded with quantization.
        This method expects a pre-quantized model and adds LoRA adapters.

        Args:
            model: Base model (should be quantized already).
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration with LoRA and quantization settings.

        Returns:
            AdaptedModel with LoRA adapters on quantized base.

        Raises:
            ValueError: If required configs are missing.
        """
        lora_cfg = _require_lora_config(cfg)
        # The model arrives already quantized: the loader builds a
        # BitsAndBytesConfig from this same field and passes it to
        # from_pretrained, which is the only point at which linear layers can
        # be replaced. Re-reading it here is what makes that a checked
        # precondition rather than an assumption.
        _require_quantization_config(cfg)

        # Ready the quantized model BEFORE adapters are attached. This freezes
        # every parameter it finds and upcasts the non-4-bit half-precision
        # ones to fp32; running it after create_peft_model would freeze the
        # adapter too and leave the run with nothing trainable.
        prepared_base = Hooks.prepare_for_kbit_training(model)

        # Enable gradient checkpointing for memory efficiency if available
        Hooks.enable_gradient_checkpointing(prepared_base)

        peft_model = Hooks.create_peft_model(
            prepared_base,
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

        Only saves the adapter weights. The quantized base model is not saved;
        it must be recreated during loading.

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
        """Load LoRA adapter weights onto a quantized base model.

        The base_model should already be loaded with quantization config.

        Args:
            base_model: Quantized base model to apply adapters to.
            model_id: HuggingFace model ID.
            adapter_path: Path to saved adapter weights.

        Returns:
            AdaptedModel with adapters loaded on quantized base.

        Raises:
            FileNotFoundError: If adapter_path does not exist.
        """
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        loaded_model = Hooks.load_peft_model(base_model, adapter_path)

        return AdaptedModel(
            model=loaded_model,
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=True,
            lora_config=None,  # Config not preserved on load
        )


def create_qlora_strategy() -> QLoRAStrategy:
    """Factory function to create a QLoRAStrategy.

    Returns:
        New QLoRAStrategy instance.
    """
    return QLoRAStrategy()


__all__ = [
    "QLoRAStrategy",
    "create_qlora_strategy",
]
