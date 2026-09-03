"""A frozen base model with a trained key-value prefix in front of it.

Satisfies :class:`~model_trainer.core.types.LMModelProto` exactly, which is
why the training loop needs no change to train one: the loop calls
``forward(input_ids=..., labels=...)`` and this supplies the prefix itself. The
prefix is not something a caller passes in, it is part of what this model IS.

WHAT WAS MEASURED, 2026-09-03, transformers 4.46.3 + torch 2.6.0+cu124:

- A cache built from leaf tensors propagates gradients back to them through a
  full forward and backward. That is what makes the prefix trainable at all.
- Logits come back at the INPUT's length, not the input plus the prefix, so
  the labels the caller supplies line up without adjustment and the prefix
  never becomes a prediction target.
- An attention mask covering only the input, rather than the prefix and the
  input together, raises a shape error. The failure is loud, which is what
  makes it safe to build the mask here rather than to require one.
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
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.types import (
    CacheCapableLMProto,
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
)


class CartridgeLoadResult:
    """What loading a cartridge's tensors reported.

    Mirrors the shape torch returns from ``load_state_dict`` so this model
    satisfies the protocol, and carries nothing, for the reason
    :class:`~model_trainer.core.types.LoadStateDictResultProto` carries
    nothing: the load here is total, so there is no partial outcome to report.
    """


class CartridgeModel:
    """A base model whose attention is preceded by trained slots.

    The base is frozen on construction and never unfrozen. ``parameters`` and
    ``state_dict`` report the SLOTS ONLY, which is what makes the training
    loop train a cartridge without knowing it is doing so: the optimizer it
    builds can only reach the prefix, and the checkpoint it writes is the
    prefix. The base is reconstructed from its hub id, never from a checkpoint.

    Attributes:
        slots: The trainable key and value blocks.
    """

    slots: CartridgeSlots
    _base: CacheCapableLMProto

    def __init__(self, *, base: CacheCapableLMProto, slots: CartridgeSlots) -> None:
        """Freeze a base model and put slots in front of it.

        Args:
            base: The model to prepend to. Every one of its parameters has
                ``requires_grad`` cleared here, so a caller cannot end up
                training it by supplying an optimizer built elsewhere.
            slots: The trainable blocks.
        """
        self._base = base
        self.slots = slots
        for _, parameter in base.named_parameters():
            parameter.requires_grad = False

    @property
    def geometry(self) -> CartridgeGeometry:
        """Return the shape of the prefix this model carries.

        Returns:
            The slots' geometry.
        """
        return self.slots.geometry

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Run the base model with the trained prefix in front of the input.

        Args:
            input_ids: Token ids, shaped (batch, positions).
            labels: Targets for those same positions. Not widened for the
                prefix: the model emits logits only for the input's positions,
                so the caller's labels already line up.

        Returns:
            The base model's output, whose loss carries gradients back into
            the slots.
        """
        batch_size = int(input_ids.shape[0])
        blocks = [
            self.slots.layer_blocks(layer, batch_size=batch_size)
            for layer in range(self.geometry["num_layers"])
        ]
        cache = Hooks.build_prefix_cache(blocks)
        attended = int(input_ids.shape[1]) + self.geometry["num_slots"]
        mask = torch.ones((batch_size, attended), dtype=torch.long, device=input_ids.device)
        return Hooks.forward_with_prefix(
            self._base,
            input_ids=input_ids,
            labels=labels,
            past_key_values=cache,
            attention_mask=mask,
        )

    def train(self) -> None:
        """Put the base model in training mode."""
        self._base.train()

    def eval(self) -> None:
        """Put the base model in evaluation mode."""
        self._base.eval()

    def parameters(self) -> Sequence[ParameterLike]:
        """Return the trainable tensors, which are the slots and only the slots.

        Returns:
            The slot tensors.
        """
        return self.slots.parameters()

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return the trainable tensors with their names.

        Returns:
            The slot tensors, named.
        """
        return self.slots.named_parameters()

    def to(self, device: str) -> LMModelProto:
        """Move the base model and the slots onto a device.

        Args:
            device: Torch device string.

        Returns:
            This model, so the call chains the way the protocol's does.
        """
        self._base.to(device)
        self.slots.to(device)
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the slot tensors by name.

        The base model is absent deliberately. A cartridge checkpoint is the
        cartridge; the base is named by its hub id and reloaded from there,
        exactly as an adapter checkpoint does not carry the model it adapts.

        Returns:
            The slot tensors.
        """
        return self.slots.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> LoadStateDictResultProto:
        """Install slot tensors read from a checkpoint.

        Args:
            state_dict: Tensors previously returned by :meth:`state_dict`.

        Returns:
            An empty result, because the load is total.

        Raises:
            AppError: With ``CARTRIDGE_STATE_INCOMPLETE`` or
                ``CARTRIDGE_GEOMETRY_MISMATCH`` if the tensors do not match
                this model's geometry.
        """
        self.slots = Hooks.slots_from_state(state_dict, self.geometry)
        return CartridgeLoadResult()

    def save_pretrained(self, out_dir: str) -> None:
        """Write the cartridge to a directory.

        Args:
            out_dir: Directory to write into.
        """
        Hooks.save_cartridge(self.slots, out_dir)

    def gradient_checkpointing_enable(self) -> None:
        """Refuse gradient checkpointing, which silently discards the prefix.

        MEASURED, 2026-09-03, transformers 4.46.3 + torch 2.6.0+cu124. A
        checkpointed model forces ``use_cache=False`` and drops the cache it
        was handed, so the prefix never reaches attention. On GPT-2 the
        symptom is a shape error -- the attention mask still covers the prefix
        and the keys no longer do -- and a caller who supplied no mask would
        instead get a run that trains a prefix nothing attends to, converging
        to nothing while reporting a falling loss.

        So this refuses rather than delegating. The alternative, enabling it
        on the base and hoping, is the silent-wrong-answer case this whole
        strategy is measured to avoid.

        Raises:
            AppError: Always, with
                ``CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED``.
        """
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED,
            (
                "gradient checkpointing cannot be combined with a trained key-value "
                "prefix: a checkpointed model discards the cache it is handed, so the "
                "cartridge would never reach attention and the run would train a "
                "prefix nothing reads"
            ),
            model_trainer_status_for(
                ModelTrainerErrorCode.CARTRIDGE_GRADIENT_CHECKPOINTING_UNSUPPORTED
            ),
        )

    @property
    def config(self) -> ConfigLike:
        """Return the base model's configuration.

        Returns:
            The base's config, unchanged. A cartridge alters what the model
            attends to, not what it is.
        """
        return self._base.config


__all__ = [
    "CartridgeLoadResult",
    "CartridgeModel",
]
