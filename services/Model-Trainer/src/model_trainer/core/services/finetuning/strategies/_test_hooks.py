"""Test hooks for fine-tuning strategies.

Follows the covenant pattern: production code sets hooks to real implementations,
tests set hooks to fakes for isolation.
"""

from __future__ import annotations

from typing import Protocol

from model_trainer.core.types import LMModelProto


class PeftModelCreator(Protocol):
    """Protocol for creating PEFT models with LoRA adapters."""

    def __call__(
        self,
        model: LMModelProto,
        *,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple[str, ...],
        bias: str,
    ) -> LMModelProto:
        """Create a PEFT model with LoRA adapters.

        Args:
            model: Base model to wrap with adapters.
            r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability for LoRA layers.
            target_modules: Module names to apply LoRA.
            bias: Bias training mode.

        Returns:
            Model with LoRA adapters attached.
        """
        ...


class PeftModelSaver(Protocol):
    """Protocol for saving PEFT adapter weights."""

    def __call__(self, model: LMModelProto, out_dir: str) -> None:
        """Save adapter weights to directory.

        Args:
            model: PEFT model with adapters.
            out_dir: Output directory path.
        """
        ...


class PeftModelLoader(Protocol):
    """Protocol for loading PEFT adapters onto a base model."""

    def __call__(self, model: LMModelProto, adapter_path: str) -> LMModelProto:
        """Load adapter weights and apply to model.

        Args:
            model: Base model to apply adapters to.
            adapter_path: Path to saved adapter weights.

        Returns:
            Model with adapters loaded.
        """
        ...


class QuantizedModelLoader(Protocol):
    """Protocol for loading models with quantization."""

    def __call__(
        self,
        model_id: str,
        *,
        load_in_4bit: bool,
        load_in_8bit: bool,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
    ) -> LMModelProto:
        """Load a model with quantization config.

        Args:
            model_id: HuggingFace model ID.
            load_in_4bit: Enable 4-bit quantization.
            load_in_8bit: Enable 8-bit quantization.
            bnb_4bit_compute_dtype: Compute dtype for 4-bit.
            bnb_4bit_quant_type: Quantization type (nf4/fp4).

        Returns:
            Quantized model.
        """
        ...


class UnslothModelLoader(Protocol):
    """Protocol for loading models with Unsloth optimization."""

    def __call__(
        self,
        model_id: str,
        *,
        max_seq_length: int,
        dtype: str | None,
        load_in_4bit: bool,
    ) -> LMModelProto:
        """Load a model with Unsloth optimization.

        Args:
            model_id: HuggingFace model ID.
            max_seq_length: Maximum sequence length.
            dtype: Data type (float16/bfloat16) or None for auto.
            load_in_4bit: Enable 4-bit quantization.

        Returns:
            Unsloth-optimized model.
        """
        ...


class UnslothPeftApplier(Protocol):
    """Protocol for applying Unsloth's optimized LoRA."""

    def __call__(
        self,
        model: LMModelProto,
        *,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple[str, ...],
    ) -> LMModelProto:
        """Apply Unsloth's optimized LoRA adapters.

        Args:
            model: Unsloth-loaded model.
            r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability.
            target_modules: Module names to apply LoRA.

        Returns:
            Model with Unsloth LoRA adapters.
        """
        ...


class GradientCheckpointEnabler(Protocol):
    """Protocol for enabling gradient checkpointing on a model."""

    def __call__(self, model: LMModelProto) -> None:
        """Enable gradient checkpointing for memory efficiency.

        Args:
            model: Model to enable checkpointing on.
        """
        ...


class FullModelLoader(Protocol):
    """Protocol for loading a full model from a path."""

    def __call__(self, model_path: str) -> LMModelProto:
        """Load a model from disk.

        Args:
            model_path: Path to saved model.

        Returns:
            Loaded model instance.
        """
        ...


class Hooks:
    """Container for test hooks.

    Production code sets these to real implementations.
    Tests set these to fakes for isolation.
    """

    create_peft_model: PeftModelCreator | None = None
    save_peft_model: PeftModelSaver | None = None
    load_peft_model: PeftModelLoader | None = None
    load_quantized_model: QuantizedModelLoader | None = None
    load_unsloth_model: UnslothModelLoader | None = None
    apply_unsloth_peft: UnslothPeftApplier | None = None
    enable_gradient_checkpointing: GradientCheckpointEnabler | None = None
    load_full_model: FullModelLoader | None = None


def reset_hooks() -> None:
    """Reset all hooks to None (for test cleanup)."""
    Hooks.create_peft_model = None
    Hooks.save_peft_model = None
    Hooks.load_peft_model = None
    Hooks.load_quantized_model = None
    Hooks.load_unsloth_model = None
    Hooks.apply_unsloth_peft = None
    Hooks.enable_gradient_checkpointing = None
    Hooks.load_full_model = None


__all__ = [
    "FullModelLoader",
    "GradientCheckpointEnabler",
    "Hooks",
    "PeftModelCreator",
    "PeftModelLoader",
    "PeftModelSaver",
    "QuantizedModelLoader",
    "UnslothModelLoader",
    "UnslothPeftApplier",
    "reset_hooks",
]
