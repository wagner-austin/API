"""Protocols for the libraries the fine-tuning strategies reach into.

Split out of ``_test_hooks`` when that module passed this package's file-size
ceiling. The division is by ROLE: this file says what the third-party surfaces
look like, and ``_test_hooks`` says which implementation is bound to each hook.
The same split ``hf_lm`` already makes between ``_hook_protocols`` and its own
``_test_hooks``.

Every signature here mirrors a real upstream one, at a pinned version, because
a protocol that spelled its argument list as ``**kwargs`` would accept a
misspelling -- and a misspelled quantization keyword loads an unquantized model
without complaint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

import torch

from model_trainer.core.contracts.cartridge import CartridgeGeometry
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.types import (
    CacheCapableLMProto,
    ForwardOutProto,
    KVCacheProto,
    LMModelProto,
)

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


class _DynamicCacheProto(KVCacheProto, Protocol):
    """Protocol for ``transformers.cache_utils.DynamicCache``.

    Signature mirrors transformers 4.46.3 ``cache_utils.py``: ``update`` takes
    the key and value states and the layer index positionally and returns the
    pair it now holds for that layer. ``cache_kwargs`` is declared by upstream
    and is not passed here, so it is not declared.

    Extends :class:`~model_trainer.core.types.KVCacheProto`, which is what a
    MODEL is handed. The extra method is what a BUILDER needs, and the two are
    separate because a cache is filled at one boundary and read at another.
    """

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Install one layer's key and value blocks.

        Args:
            key_states: The layer's keys.
            value_states: The layer's values.
            layer_idx: Zero-based layer index.

        Returns:
            The key and value tensors the cache now holds for that layer.
        """
        ...


class _DynamicCacheClassProto(Protocol):
    """Protocol for the ``DynamicCache`` class itself."""

    def __call__(self) -> _DynamicCacheProto:
        """Construct an empty cache.

        Returns:
            A cache with no layers installed yet.
        """
        ...


class CacheLayerProbe(Protocol):
    """Protocol for measuring a model's per-layer cached key tensors."""

    def __call__(self, model: CacheCapableLMProto) -> Sequence[torch.Tensor]:
        """Return one cached key tensor per attention layer.

        Args:
            model: The model to measure.

        Returns:
            The layer-zero-first key tensors, whose shapes carry the geometry.
        """
        ...


class PrefixCacheBuilder(Protocol):
    """Protocol for assembling per-layer blocks into a cache object."""

    def __call__(self, blocks: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> KVCacheProto:
        """Build a cache holding one key and value block per layer.

        Args:
            blocks: Key and value pairs, in layer order.

        Returns:
            A cache ready to pass as ``past_key_values``.
        """
        ...


class PrefixForward(Protocol):
    """Protocol for running a base model with a prefix cache installed."""

    def __call__(
        self,
        model: CacheCapableLMProto,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        past_key_values: KVCacheProto,
        attention_mask: torch.Tensor,
    ) -> ForwardOutProto:
        """Run one forward pass in front of a prefix.

        Args:
            model: The frozen base model.
            input_ids: Token ids.
            labels: Targets for the input's own positions.
            past_key_values: The prefix.
            attention_mask: Ones over the prefix and the input together.

        Returns:
            The model's output, whose loss carries gradients into the prefix.
        """
        ...


class CartridgeSaver(Protocol):
    """Protocol for writing a cartridge to a directory."""

    def __call__(self, slots: CartridgeSlots, out_dir: str) -> None:
        """Write the slots and their manifest.

        Args:
            slots: The blocks to write.
            out_dir: Directory to write into.
        """
        ...


class CartridgeLoader(Protocol):
    """Protocol for reading a cartridge back from a directory."""

    def __call__(self, cartridge_dir: str) -> CartridgeSlots:
        """Read the slots and their manifest.

        Args:
            cartridge_dir: Directory previously written by the saver.

        Returns:
            The rebuilt slots.
        """
        ...


class CartridgeStateReader(Protocol):
    """Protocol for rebuilding slots from an in-memory state dict."""

    def __call__(
        self, state: dict[str, torch.Tensor], geometry: CartridgeGeometry
    ) -> CartridgeSlots:
        """Rebuild slots from named tensors.

        Args:
            state: The tensors, by name.
            geometry: The shape they must match.

        Returns:
            The rebuilt slots.
        """
        ...
