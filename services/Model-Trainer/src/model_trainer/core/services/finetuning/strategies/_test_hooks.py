"""Test hooks for fine-tuning strategies.

Follows the covenant pattern: production code sets hooks to real implementations,
tests set hooks to fakes for isolation.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

from model_trainer.core.types import LMModelProto

# ============================================================================
# Protocols for dynamic PEFT imports
# ============================================================================


class _LoraConfigProto(Protocol):
    """Protocol for PEFT LoraConfig class."""

    pass


class _LoraConfigClassProto(Protocol):
    """Protocol for LoraConfig constructor."""

    def __call__(
        self,
        *,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        bias: str,
        task_type: Literal["CAUSAL_LM"],
        fan_in_fan_out: bool = False,
    ) -> _LoraConfigProto:
        """Create LoRA configuration.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability.
            target_modules: Modules to apply LoRA to.
            bias: Bias training mode.
            task_type: Task type for PEFT.

        Returns:
            LoraConfig instance.
        """
        ...


class _GetPeftModelFn(Protocol):
    """Protocol for peft.get_peft_model function."""

    def __call__(self, model: LMModelProto, config: _LoraConfigProto) -> LMModelProto:
        """Wrap model with PEFT adapters.

        Args:
            model: Base model to wrap.
            config: LoRA configuration.

        Returns:
            PEFT-wrapped model.
        """
        ...


class _PeftModelClassProto(Protocol):
    """Protocol for PeftModel class with from_pretrained."""

    def from_pretrained(self, model: LMModelProto, adapter_path: str) -> LMModelProto:
        """Load PEFT adapters onto a model.

        Args:
            model: Base model to apply adapters to.
            adapter_path: Path to saved adapter weights.

        Returns:
            Model with adapters loaded.
        """
        ...


class _LoadPeftWeightsFn(Protocol):
    """Protocol for peft.load_peft_weights.

    Signature mirrors peft v0.14.0 ``utils/save_and_load.py`` L 489: one
    required argument, a path or hub id. It resolves safetensors-versus-bin
    itself, so callers do not name a filename.
    """

    def __call__(self, model_id: str) -> dict[str, torch.Tensor]:
        """Read an adapter's weights off disk.

        Args:
            model_id: Directory holding the saved adapter.

        Returns:
            The adapter's state dict.
        """
        ...


class _LoadStateDictResultProto(Protocol):
    """Protocol for the incompatible-keys result of a state-dict load.

    ``set_peft_model_state_dict`` returns whatever ``load_state_dict``
    returned (peft v0.14.0 ``utils/save_and_load.py`` L 451 and L 474), and
    that load runs with ``strict=False``. Under ``strict=False`` every base
    weight is reported missing, which is expected for an adapter and says
    nothing. An UNEXPECTED key is different: it means the file holds a
    parameter the live model has no slot for, so the adapter does not belong
    to this model.
    """

    unexpected_keys: list[str]


class _SetPeftModelStateDictFn(Protocol):
    """Protocol for peft.set_peft_model_state_dict.

    Signature mirrors peft v0.14.0 ``utils/save_and_load.py`` L 329: the
    model it takes is the already-wrapped PeftModel, not a base model, and
    the write happens in place.
    """

    def __call__(
        self, model: LMModelProto, peft_model_state_dict: dict[str, torch.Tensor]
    ) -> _LoadStateDictResultProto:
        """Write adapter weights into a live PEFT model.

        Args:
            model: The wrapped PEFT model to mutate.
            peft_model_state_dict: Weights to install.

        Returns:
            The load result, whose unexpected keys the caller checks.
        """
        ...


class _PrepareForKbitTrainingFn(Protocol):
    """Protocol for peft.prepare_model_for_kbit_training."""

    def __call__(self, model: LMModelProto) -> LMModelProto:
        """Ready a quantized model for adapter training.

        Args:
            model: The quantized base model.

        Returns:
            The same model, prepared in place and returned for chaining.
        """
        ...


class _AutoModelClassProto(Protocol):
    """Protocol for AutoModelForCausalLM class."""

    def from_pretrained(self, model_path: str) -> LMModelProto:
        """Load model from path.

        Args:
            model_path: Path to saved model.

        Returns:
            Loaded model instance.
        """
        ...


# ============================================================================
# Public Protocols for hooks
# ============================================================================


class AdapterWeightsReloader(Protocol):
    """Protocol for reloading an adapter's weights into a live PEFT model."""

    def __call__(self, model: LMModelProto, adapter_path: str) -> None:
        """Reload adapter weights in place.

        Args:
            model: The live PEFT model to write into.
            adapter_path: Directory holding the saved adapter.
        """
        ...


class KbitTrainingPreparer(Protocol):
    """Protocol for readying a quantized model for adapter training."""

    def __call__(self, model: LMModelProto) -> LMModelProto:
        """Prepare a quantized model.

        Args:
            model: The quantized base model.

        Returns:
            The prepared model.
        """
        ...


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


def _default_create_peft_model(
    model: LMModelProto,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    bias: str,
) -> LMModelProto:
    """Production implementation for creating PEFT LoRA models.

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
    peft = __import__("peft", fromlist=["get_peft_model", "LoraConfig"])
    lora_config_cls: _LoraConfigClassProto = peft.LoraConfig
    get_peft_model_fn: _GetPeftModelFn = peft.get_peft_model
    config: _LoraConfigProto = lora_config_cls(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias=bias,
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    peft_model: LMModelProto = get_peft_model_fn(model, config)
    return peft_model


def _default_save_peft_model(model: LMModelProto, out_dir: str) -> None:
    """Production implementation for saving PEFT models.

    Args:
        model: PEFT model with adapters.
        out_dir: Output directory path.
    """
    model.save_pretrained(out_dir)


def _default_load_peft_model(model: LMModelProto, adapter_path: str) -> LMModelProto:
    """Production implementation for loading PEFT adapters.

    Args:
        model: Base model to apply adapters to.
        adapter_path: Path to saved adapter weights.

    Returns:
        Model with adapters loaded.
    """
    peft = __import__("peft", fromlist=["PeftModel"])
    peft_model_cls: _PeftModelClassProto = peft.PeftModel
    loaded_model: LMModelProto = peft_model_cls.from_pretrained(model, adapter_path)
    return loaded_model


def _default_reload_adapter_weights(model: LMModelProto, adapter_path: str) -> None:
    """Production reload_adapter_weights - used as default hook.

    Reads the saved adapter and writes it into the model already in memory,
    rather than constructing a new one. ``PeftModel.from_pretrained`` cannot
    do this job: it takes a base model AND a path, because it builds a
    wrapper, and here the wrapper already exists and its optimizer, device
    placement and surrounding references must survive the reload.

    Args:
        model: The live PEFT model to write into.
        adapter_path: Directory holding the saved adapter.

    Raises:
        AppError: ``ADAPTER_RELOAD_MISMATCH`` when the saved adapter carries
            parameters this model has no slot for.
    """
    peft = __import__("peft", fromlist=["load_peft_weights", "set_peft_model_state_dict"])
    load_weights: _LoadPeftWeightsFn = peft.load_peft_weights
    set_state_dict: _SetPeftModelStateDictFn = peft.set_peft_model_state_dict
    result = set_state_dict(model, load_weights(adapter_path))
    if result.unexpected_keys:
        named = ", ".join(sorted(result.unexpected_keys)[:5])
        raise AppError(
            ModelTrainerErrorCode.ADAPTER_RELOAD_MISMATCH,
            (
                f"the adapter at {adapter_path} carries {len(result.unexpected_keys)} "
                f"parameter(s) this model has no slot for ({named}); it was saved from "
                f"a different model and loading it would score weights that were never "
                f"trained here"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.ADAPTER_RELOAD_MISMATCH),
        )


def _default_prepare_for_kbit_training(model: LMModelProto) -> LMModelProto:
    """Production prepare_for_kbit_training - used as default hook.

    Must run BEFORE adapters are attached. The function freezes every
    parameter it finds, so running it afterwards would freeze the adapter
    too and leave the run with nothing trainable.

    Args:
        model: The quantized base model.

    Returns:
        The prepared model.
    """
    peft = __import__("peft", fromlist=["prepare_model_for_kbit_training"])
    prepare: _PrepareForKbitTrainingFn = peft.prepare_model_for_kbit_training
    return prepare(model)


def _default_enable_gradient_checkpointing(model: LMModelProto) -> None:
    """Production implementation for enabling gradient checkpointing.

    Args:
        model: Model to enable checkpointing on.
    """
    model.gradient_checkpointing_enable()


def _default_load_full_model(model_path: str) -> LMModelProto:
    """Production implementation for loading a full model from path.

    Args:
        model_path: Path to saved model.

    Returns:
        Loaded model instance.
    """
    transformers = __import__("transformers", fromlist=["AutoModelForCausalLM"])
    model_cls: _AutoModelClassProto = transformers.AutoModelForCausalLM
    loaded_model: LMModelProto = model_cls.from_pretrained(model_path)
    return loaded_model


class Hooks:
    """Container for fine-tuning hooks, each bound to its real implementation.

    Production calls these without wiring anything first. Tests assign a fake
    and call reset() afterwards, which puts the real implementation back.
    """

    create_peft_model: PeftModelCreator = _default_create_peft_model
    save_peft_model: PeftModelSaver = _default_save_peft_model
    load_peft_model: PeftModelLoader = _default_load_peft_model
    reload_adapter_weights: AdapterWeightsReloader = _default_reload_adapter_weights
    prepare_for_kbit_training: KbitTrainingPreparer = _default_prepare_for_kbit_training
    enable_gradient_checkpointing: GradientCheckpointEnabler = (
        _default_enable_gradient_checkpointing
    )
    load_full_model: FullModelLoader = _default_load_full_model

    @classmethod
    def reset(cls) -> None:
        """Restore every hook to its real implementation.

        The restoration `reset_hooks()` performs, exposed as a classmethod so
        an autouse fixture can name the container it protects.
        """
        reset_hooks()


def reset_hooks() -> None:
    """Restore every hook to the production implementation it is bound to."""
    Hooks.create_peft_model = _default_create_peft_model
    Hooks.save_peft_model = _default_save_peft_model
    Hooks.load_peft_model = _default_load_peft_model
    Hooks.reload_adapter_weights = _default_reload_adapter_weights
    Hooks.prepare_for_kbit_training = _default_prepare_for_kbit_training
    Hooks.enable_gradient_checkpointing = _default_enable_gradient_checkpointing
    Hooks.load_full_model = _default_load_full_model


__all__ = [
    "FullModelLoader",
    "GradientCheckpointEnabler",
    "Hooks",
    "PeftModelCreator",
    "PeftModelLoader",
    "PeftModelSaver",
    "reset_hooks",
]
