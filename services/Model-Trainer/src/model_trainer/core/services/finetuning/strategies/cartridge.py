"""Cartridge strategy - trains a key-value prefix over a frozen base model.

The fourth adaptation strategy, and the only one that changes no weight at
all. LoRA trains a low-rank delta ONTO the weights; a cartridge trains a block
of attention context that sits IN FRONT of them, so the base model is returned
from a run byte-identical to how it arrived.

Method: Eyuboglu et al. (2025), "Cartridges: Lightweight and general-purpose
long context representations via self-study". This implements the trained-KV
object the paper describes. It does not vendor the paper's code, which builds
its artifact store on a third-party experiment-tracking service and supports
two model classes; nothing here needs either, and the training loop, the
checkpointing and the measurement harness this plugs into already exist.

WHAT COMPOSES, AND WHAT IT COSTS. Two cartridges concatenate rather than being
summed, which is what distinguishes them from steering vectors in the residual
stream. ``cartridge_slots.compose`` does the joining. Measured on the tiny rung:
a composed pair retains about a quarter of what each was worth alone, and most
of that loss is DILUTION -- doubling the prefix with content-free padding costs
nearly as much as adding a real second cartridge. See
``tests/test_cartridge_composition.py`` for the arms and the attribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

from model_trainer.core.contracts.cartridge import CartridgeGeometry
from model_trainer.core.contracts.finetuning import (
    AdaptedModel,
    StrategyCapabilities,
)
from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.contracts.strategy_names import StrategyName
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    discover_geometry,
    initialise_slots,
    require_matching_geometry,
)
from model_trainer.core.types import CacheCapableLMProto, LMModelProto

#: Trainable fraction reported for capability discovery.
#:
#: A representative figure rather than an exact one, matching how the LoRA
#: strategy reports its own: the true fraction depends on the slot count and
#: the base model's size, and is knowable only once both are chosen. Measured
#: for the shape this was designed against -- 2048 slots on a 32-layer,
#: 8-key-value-head, 128-wide model is 134 million scalars against an 8
#: billion parameter base, or 1.7 percent. Callers needing the real number for
#: a specific run call
#: :func:`~model_trainer.core.contracts.cartridge.trainable_parameter_count`.
_REPRESENTATIVE_TRAINABLE_FRACTION = 0.017


def _require_cartridge_config(cfg: ModelTrainConfig) -> CartridgeConfig:
    """Read the cartridge settings a run must carry to use this strategy.

    Args:
        cfg: Training configuration.

    Returns:
        The cartridge configuration.

    Raises:
        AppError: With ``CARTRIDGE_CONFIG_MISSING`` when the config is absent
            or disabled. Absent and disabled are one failure here: both mean
            the caller selected this strategy and then did not describe the
            cartridge it needs, and inventing a slot count on their behalf
            would silently decide the run's capacity.
    """
    cartridge_cfg = cfg.get("cartridge")
    if cartridge_cfg is None:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING,
            (
                "the cartridge strategy was selected without a 'cartridge' config; "
                "the slot count decides how much the cartridge can hold and there is "
                "no defensible default, so it must be stated"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING),
        )
    if not cartridge_cfg["enabled"]:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING,
            (
                "the cartridge strategy was selected but its config carries "
                "enabled=false; the two disagree about whether this run trains a "
                "cartridge, and guessing which one is meant would train the wrong "
                "thing either way"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CONFIG_MISSING),
        )
    return cartridge_cfg


def require_cache_capable(model: LMModelProto) -> CacheCapableLMProto:
    """Narrow a model to one that can be run against a key-value cache.

    The single place the widening happens, so exactly one error message has to
    explain it. ``isinstance`` against a runtime-checkable protocol rather
    than a cast: a cast would assert the capability, and this establishes it.

    What the check can see is that the model is callable in the shape a cached
    forward needs. What it cannot see is whether that call actually returns a
    cache -- no runtime protocol check inspects signatures or behaviour -- so
    it is not the whole verification. :func:`measure_geometry` runs the model
    and refuses an empty or wrongly shaped cache, and that is the check with
    teeth. This one exists so the type is honest on the way there.

    Args:
        model: The model to narrow.

    Returns:
        The same model, typed as cache-capable.

    Raises:
        AppError: With ``CARTRIDGE_MODEL_REPORTS_NO_CACHE`` if the model does
            not present the callable surface a cached forward needs.
    """
    if isinstance(model, CacheCapableLMProto):
        return model
    raise AppError(
        ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE,
        (
            "this model cannot be called with a key-value cache, so there is nothing "
            "for a cartridge to sit in front of; the cartridge strategy needs a "
            "transformer whose attention layers cache their keys and values"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE),
    )


def probe_cache_layers(model: CacheCapableLMProto) -> Sequence[torch.Tensor]:
    """Ask a model for its per-layer cached key tensors.

    A one-token forward with caching on, under ``no_grad`` and on the model's
    own device: this measures a shape and must neither build a graph nor move
    anything. No labels, because a loss would be computed and discarded.

    Not a hook. The only thing it touches is the model, which every caller
    already supplies and every test already fakes -- routing it through an
    injection point would add a seam nothing needs and hide a five-line
    measurement behind a protocol.

    Args:
        model: The model to measure.

    Returns:
        One cached key tensor per attention layer, layer zero first.
    """
    device = next(iter(model.named_parameters()))[1].detach().device
    probe = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=probe, use_cache=True)
    return [pair[0] for pair in out.past_key_values]


def measure_geometry(model: LMModelProto, *, num_slots: int) -> CartridgeGeometry:
    """Measure the shape a cartridge for this model must be cut to.

    Args:
        model: The base model, already on its device.
        num_slots: Prefix positions the caller asked for.

    Returns:
        The geometry.

    Raises:
        AppError: With ``CARTRIDGE_MODEL_REPORTS_NO_CACHE`` if the model
            cannot be run against a cache, or returns none when it is.
    """
    layers = probe_cache_layers(require_cache_capable(model))
    return discover_geometry(layers, num_slots=num_slots)


class CartridgeStrategy:
    """Trains a key-value prefix while the base model stays frozen.

    Attributes:
        _name: Strategy identifier "cartridge".
    """

    def __init__(self) -> None:
        """Initialize the cartridge strategy."""
        self._name: StrategyName = "cartridge"

    def name(self) -> StrategyName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "cartridge".
        """
        return self._name

    def capabilities(self) -> StrategyCapabilities:
        """Return strategy capabilities for discovery.

        ``requires_peft`` is False and that is the substantive claim: this
        strategy has no third-party adapter library underneath it.

        ``supports_gradient_checkpointing`` is False, and it was True until it
        was measured. A checkpointed model discards the key-value cache it is
        handed (transformers 4.46.3, measured 2026-09-03), so the prefix never
        reaches attention -- the memory saving would be bought by not training
        the thing the run exists to train. Every other strategy here reports
        True, which is exactly why this one had to be checked rather than
        copied.

        Returns:
            Capabilities describing a prefix-only, PEFT-free strategy that
            cannot be checkpointed.
        """
        return StrategyCapabilities(
            supports_quantization=False,
            supports_gradient_checkpointing=False,
            requires_peft=False,
            trainable_param_fraction=_REPRESENTATIVE_TRAINABLE_FRACTION,
        )

    def adapt(
        self,
        model: LMModelProto,
        model_id: str,
        cfg: ModelTrainConfig,
    ) -> AdaptedModel:
        """Put a freshly drawn cartridge in front of a frozen base model.

        Args:
            model: Base model to prepend to.
            model_id: HuggingFace model ID (for metadata).
            cfg: Training configuration carrying the cartridge settings.

        Returns:
            AdaptedModel wrapping a :class:`CartridgeModel`.

        Raises:
            AppError: With ``CARTRIDGE_CONFIG_MISSING`` if the run does not
                describe a cartridge, ``CARTRIDGE_GEOMETRY_INVALID`` if the
                slot count is not positive, or
                ``CARTRIDGE_MODEL_REPORTS_NO_CACHE`` if the model cannot host
                a prefix.
        """
        cartridge_cfg = _require_cartridge_config(cfg)
        geometry = measure_geometry(model, num_slots=cartridge_cfg["num_slots"])
        slots = initialise_slots(geometry, seed=cartridge_cfg["init_seed"])
        return AdaptedModel(
            model=CartridgeModel(base=require_cache_capable(model), slots=slots),
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
        """Write the cartridge to disk.

        The base model is not written. A cartridge run leaves its base
        unchanged by construction, so saving it would store an unmodified copy
        of something already addressable by its hub id.

        Args:
            adapted: The adapted model to save.
            out_dir: Output directory path.
        """
        adapted.model.save_pretrained(out_dir)

    def load_adapted(
        self,
        base_model: LMModelProto,
        model_id: str,
        adapter_path: str,
    ) -> AdaptedModel:
        """Read a cartridge off disk and put it in front of a base model.

        The saved geometry is checked against the model actually supplied,
        rather than trusted. A cartridge is a block of one particular model's
        own attention keys and values, so attaching one to a differently
        shaped model produces confident nonsense rather than an error, which
        is exactly the failure worth refusing.

        Args:
            base_model: Base model to prepend to.
            model_id: HuggingFace model ID.
            adapter_path: Directory holding the saved cartridge.

        Returns:
            AdaptedModel with the cartridge loaded.

        Raises:
            FileNotFoundError: If the directory holds no cartridge.
            AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if the saved
                cartridge was cut for a differently shaped model.
        """
        slots = Hooks.load_cartridge(adapter_path)
        model_shape = measure_geometry(base_model, num_slots=slots.geometry["num_slots"])
        require_matching_geometry(slots.geometry, model_shape)
        return AdaptedModel(
            model=CartridgeModel(base=require_cache_capable(base_model), slots=slots),
            base_model_id=model_id,
            strategy_name=self._name,
            is_peft_model=False,
            lora_config=None,
        )


def create_cartridge_strategy() -> CartridgeStrategy:
    """Factory function to create a CartridgeStrategy.

    Returns:
        New CartridgeStrategy instance.
    """
    return CartridgeStrategy()


__all__ = [
    "CartridgeStrategy",
    "create_cartridge_strategy",
    "measure_geometry",
    "probe_cache_layers",
]
