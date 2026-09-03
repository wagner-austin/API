"""Test hooks for fine-tuning strategies.

Follows the covenant pattern: production code sets hooks to real
implementations, tests set hooks to fakes for isolation.

The PROTOCOLS these implementations satisfy live in ``_hook_protocols``, split
out when this module passed the package's file-size ceiling. The division is by
role: that file says what the third-party surfaces look like, this one says
which implementation is bound to each hook.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import dump_json_str, load_json_str

from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_MANIFEST_NAME,
    CARTRIDGE_WEIGHTS_NAME,
    CartridgeGeometry,
    decode_cartridge_geometry,
    encode_cartridge_geometry,
)
from model_trainer.core.services.finetuning.strategies._hook_protocols import (
    AdapterWeightsReloader,
    CacheLayerProbe,
    CartridgeLoader,
    CartridgeSaver,
    CartridgeStateReader,
    FullModelLoader,
    GradientCheckpointEnabler,
    KbitTrainingPreparer,
    PeftModelCreator,
    PeftModelLoader,
    PeftModelSaver,
    PrefixCacheBuilder,
    PrefixForward,
    _AutoModelClassProto,
    _DynamicCacheClassProto,
    _GetPeftModelFn,
    _LoadPeftWeightsFn,
    _LoraConfigClassProto,
    _LoraConfigProto,
    _PeftModelClassProto,
    _PrepareForKbitTrainingFn,
    _SetPeftModelStateDictFn,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    slots_from_state,
)
from model_trainer.core.types import (
    CacheCapableLMProto,
    ForwardOutProto,
    KVCacheProto,
    LMModelProto,
)


def _default_probe_cache_layers(model: CacheCapableLMProto) -> Sequence[torch.Tensor]:
    """Production implementation - runs a one-token cached forward.

    Under ``no_grad`` and on the model's own device, because this measures a
    shape and must neither build a graph nor move the model. No labels, since
    a loss would be computed and discarded.

    Args:
        model: The model to measure.

    Returns:
        One cached key tensor per attention layer.
    """
    device = next(iter(model.named_parameters()))[1].detach().device
    probe = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=probe, use_cache=True)
    return [pair[0] for pair in out.past_key_values]


def _default_build_prefix_cache(
    blocks: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> KVCacheProto:
    """Production implementation - builds a fresh DynamicCache per forward.

    Fresh rather than reused: ``update`` appends, so a cache carried across
    two forwards would grow a second copy of the prefix. Measured 2026-09-03
    against transformers 4.46.3 -- two consecutive forwards through freshly
    built caches produce identical losses and leave the blocks' shapes
    untouched.

    Args:
        blocks: Key and value pairs, in layer order.

    Returns:
        A cache holding exactly those blocks.
    """
    cache_utils = __import__("transformers.cache_utils", fromlist=["DynamicCache"])
    cache_cls: _DynamicCacheClassProto = cache_utils.DynamicCache
    cache = cache_cls()
    for layer, (keys, values) in enumerate(blocks):
        _ = cache.update(keys, values, layer)
    return cache


def _default_forward_with_prefix(
    model: CacheCapableLMProto,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    past_key_values: KVCacheProto,
    attention_mask: torch.Tensor,
) -> ForwardOutProto:
    """Production implementation - one forward pass in front of a prefix.

    ``use_cache=False`` so the run does not accumulate its own keys on top of
    the prefix, which would make the cache grow by the sequence length on
    every step.

    Args:
        model: The frozen base model.
        input_ids: Token ids.
        labels: Targets for the input's own positions.
        past_key_values: The prefix.
        attention_mask: Ones over the prefix and the input together.

    Returns:
        The model's output.
    """
    return model(
        input_ids=input_ids,
        labels=labels,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=False,
    )


def _default_save_cartridge(slots: CartridgeSlots, out_dir: str) -> None:
    """Production implementation - writes tensors and manifest to a directory.

    Two files rather than one. The manifest is JSON so a person can read what
    a cartridge is without loading torch, and the tensors are a torch file
    because that is what they are.

    Args:
        slots: The blocks to write.
        out_dir: Directory to write into.
    """
    directory = pathlib.Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CARTRIDGE_MANIFEST_NAME).write_text(
        dump_json_str(encode_cartridge_geometry(slots.geometry)),
        encoding="utf-8",
    )
    torch.save(slots.state_dict(), directory / CARTRIDGE_WEIGHTS_NAME)


def _default_load_cartridge(cartridge_dir: str) -> CartridgeSlots:
    """Production implementation - reads a cartridge back off disk.

    Args:
        cartridge_dir: Directory previously written by the saver.

    Returns:
        The rebuilt slots.

    Raises:
        FileNotFoundError: If either the manifest or the weights are absent.
    """
    directory = pathlib.Path(cartridge_dir)
    manifest = directory / CARTRIDGE_MANIFEST_NAME
    weights = directory / CARTRIDGE_WEIGHTS_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"cartridge manifest not found: {manifest}")
    if not weights.is_file():
        raise FileNotFoundError(f"cartridge weights not found: {weights}")
    geometry = decode_cartridge_geometry(load_json_str(manifest.read_text(encoding="utf-8")))
    loaded: dict[str, torch.Tensor] = torch.load(weights, weights_only=True)
    return slots_from_state(loaded, geometry)


def _default_slots_from_state(
    state: dict[str, torch.Tensor], geometry: CartridgeGeometry
) -> CartridgeSlots:
    """Production implementation - rebuilds slots from named tensors.

    Args:
        state: The tensors, by name.
        geometry: The shape they must match.

    Returns:
        The rebuilt slots.
    """
    return slots_from_state(state, geometry)


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

    probe_cache_layers: CacheLayerProbe = _default_probe_cache_layers
    build_prefix_cache: PrefixCacheBuilder = _default_build_prefix_cache
    forward_with_prefix: PrefixForward = _default_forward_with_prefix
    save_cartridge: CartridgeSaver = _default_save_cartridge
    load_cartridge: CartridgeLoader = _default_load_cartridge
    slots_from_state: CartridgeStateReader = _default_slots_from_state
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
    Hooks.probe_cache_layers = _default_probe_cache_layers
    Hooks.build_prefix_cache = _default_build_prefix_cache
    Hooks.forward_with_prefix = _default_forward_with_prefix
    Hooks.save_cartridge = _default_save_cartridge
    Hooks.load_cartridge = _default_load_cartridge
    Hooks.slots_from_state = _default_slots_from_state
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
