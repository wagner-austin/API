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
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    SLOT_AXIS,
    CartridgeSlots,
    require_matching_geometry,
    slots_from_state,
)
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

    @property
    def base(self) -> CacheCapableLMProto:
        """Return the frozen model this prefix sits in front of.

        Exposed because scoring the same tokens WITHOUT the prefix is the
        control arm every claim about a cartridge needs, and the only model
        that is the right control is this one -- same weights, same device,
        same dtype. A measurement that loaded its own copy of the base would
        be comparing against a different object and calling the difference an
        effect of the cartridge.

        Returns:
            The base model, frozen. It is returned for READING: its
            parameters have ``requires_grad`` cleared and nothing here
            restores them.
        """
        return self._base

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
        attended = int(input_ids.shape[1]) + self.geometry["num_slots"]
        return self._base(
            input_ids=input_ids,
            labels=labels,
            past_key_values=Hooks.build_prefix_cache(blocks),
            attention_mask=torch.ones(
                (batch_size, attended), dtype=torch.long, device=input_ids.device
            ),
            # False so the run does not accumulate its own keys on top of the
            # prefix, which would grow the cache by the sequence length on
            # every step of every epoch.
            use_cache=False,
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
        self.slots = slots_from_state(state_dict, self.geometry)
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


class CompanionedCartridgeModel(CartridgeModel):
    """A cartridge trained in the presence of a frozen stranger.

    The composition-scaling measurement (board task ``a67d6038``) found that
    independently trained cartridges collapse when composed, and its
    untrained-composed control attributed the n2 cost to STRUCTURE: the base
    was never asked to read a prefix with company in it. This model is the
    intervention: during training, with a per-step probability, a frozen
    companion's blocks are concatenated in front of the trainee's, so the
    gradients teach the trainee to deliver its content beside a stranger.
    The published precedent is ICAE's multi-span finding, where concatenation
    of separately compressed spans failed until concatenation examples
    entered training.

    TRAINING ONLY. Scoring always builds a plain :class:`CartridgeModel`, so
    a companion can never leak into a measurement arm: what is scored is the
    trainee's slots, alone or explicitly composed.

    THE COMPANION IS FROZEN BY CONSTRUCTION, not convention: its blocks are
    ``detach()``-ed at every forward, so the optimizer built from
    ``parameters()`` -- which reports the trainee's slots only, inherited
    unchanged -- could not reach it even if a caller wired one that tried.

    THE PROBABILITY DRAW IS UNIFORM ACROSS ARMS. One draw from torch's global
    generator per forward, whatever the probability, including 1.0. Skipping
    the draw at 1.0 would give the p-sweep's arms different RNG streams for
    reasons that have nothing to do with the knob being swept, and the sweep
    exists to vary exactly one thing.
    """

    _companion: CartridgeSlots
    _companion_probability: float

    def __init__(
        self,
        *,
        base: CacheCapableLMProto,
        slots: CartridgeSlots,
        companion: CartridgeSlots,
        companion_probability: float,
    ) -> None:
        """Freeze a base, put trainee slots in front, and hold a companion.

        Args:
            base: The model to prepend to, frozen exactly as
                :class:`CartridgeModel` freezes it.
            slots: The trainable blocks.
            companion: The frozen stranger. Must be cut for the same model as
                ``slots``; its slot count may differ.
            companion_probability: Chance per forward that the companion is
                present. Must lie in (0, 1]: zero would be a
                :class:`CartridgeModel` wearing a knob that does nothing, and
                the plain class is the honest spelling of that configuration.

        Raises:
            ValueError: If the probability is outside (0, 1].
            AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if the companion
                was cut for a differently shaped model.
        """
        if not 0.0 < companion_probability <= 1.0:
            raise ValueError(
                f"a companion probability of {companion_probability} is outside (0, 1]; "
                f"zero companionship is the plain CartridgeModel and should be "
                f"constructed as one, so the record never carries a dead knob"
            )
        require_matching_geometry(companion.geometry, slots.geometry)
        super().__init__(base=base, slots=slots)
        self._companion = companion
        self._companion_probability = companion_probability
        # The companion joins the base's device HERE, not in a caller: a
        # provider draws noise slots on the CPU (initialise_slots draws
        # nowhere else), and the first forward on a CUDA base would
        # torch.cat across devices. A CPU-only suite cannot reach that
        # failure, so the invariant lives in the constructor where no code
        # path can skip it.
        self._companion.to(str(next(iter(base.named_parameters()))[1].detach().device))

    def to(self, device: str) -> LMModelProto:
        """Move the base, the trainee slots and the companion onto a device.

        Args:
            device: Torch device string.

        Returns:
            This model, so the call chains.
        """
        super().to(device)
        self._companion.to(device)
        return self

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Run the base with the companion sometimes present before the trainee.

        Args:
            input_ids: Token ids, shaped (batch, positions).
            labels: Targets for those same positions.

        Returns:
            The base model's output. Gradients reach the trainee's slots
            only; the companion's blocks are detached.
        """
        present = float(torch.rand(())) < self._companion_probability
        if not present:
            return super().forward(input_ids=input_ids, labels=labels)
        batch_size = int(input_ids.shape[0])
        blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.geometry["num_layers"]):
            companion_key, companion_value = self._companion.layer_blocks(
                layer, batch_size=batch_size
            )
            trainee_key, trainee_value = self.slots.layer_blocks(layer, batch_size=batch_size)
            blocks.append(
                (
                    torch.cat([companion_key.detach(), trainee_key], dim=SLOT_AXIS),
                    torch.cat([companion_value.detach(), trainee_value], dim=SLOT_AXIS),
                )
            )
        attended = (
            int(input_ids.shape[1])
            + self._companion.geometry["num_slots"]
            + self.geometry["num_slots"]
        )
        return self._base(
            input_ids=input_ids,
            labels=labels,
            past_key_values=Hooks.build_prefix_cache(blocks),
            attention_mask=torch.ones(
                (batch_size, attended), dtype=torch.long, device=input_ids.device
            ),
            use_cache=False,
        )


class MultiCompanionedCartridgeModel(CartridgeModel):
    """A cartridge trained beside a VARYING number of frozen strangers.

    The single-companion recipe's retention decays with deployment count
    (44.6% at four compartments, 26.5% at eight, both recorded), and this
    model is the intervention over that decay: each training forward draws
    how many of a fixed pool of frozen companions stand in front of the
    trainee, so gradients see the prefix at several lengths instead of one.

    TRAINING ONLY, and THE POOL IS FROZEN BY CONSTRUCTION, both exactly as
    :class:`CompanionedCartridgeModel`: scoring builds plain models, every
    companion block is detached at every forward, and ``parameters()``
    reports the trainee's slots only.

    THE RNG CONSUMPTION IS FIXED PER FORWARD. Three draws -- presence,
    count, and a full permutation of the pool -- are taken from torch's
    global generator on EVERY forward, whatever the outcome, extending the
    uniformity rule the single-companion model set: two configurations of
    this class share one RNG-consumption pattern, so a sweep over its knobs
    varies exactly the knob.
    """

    _companions: tuple[CartridgeSlots, ...]
    _companion_probability: float

    def __init__(
        self,
        *,
        base: CacheCapableLMProto,
        slots: CartridgeSlots,
        companions: tuple[CartridgeSlots, ...],
        companion_probability: float,
    ) -> None:
        """Freeze a base, put trainee slots in front, and hold a pool.

        Args:
            base: The model to prepend to, frozen exactly as
                :class:`CartridgeModel` freezes it.
            slots: The trainable blocks.
            companions: The frozen pool. At least two members: a pool of one
                is :class:`CompanionedCartridgeModel` wearing dead count and
                permutation draws, and that class is the honest spelling.
                Each member must be cut for the same model as ``slots``.
            companion_probability: Chance per forward that any companions
                are present, in (0, 1]; zero would be a plain
                :class:`CartridgeModel` wearing a dead knob. When present,
                the count is drawn uniformly from one to the pool size.

        Raises:
            ValueError: If the probability is outside (0, 1], or the pool
                holds fewer than two companions.
            AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if any companion
                was cut for a differently shaped model.
        """
        if not 0.0 < companion_probability <= 1.0:
            raise ValueError(
                f"a companion probability of {companion_probability} is outside (0, 1]; "
                f"zero companionship is the plain CartridgeModel and should be "
                f"constructed as one, so the record never carries a dead knob"
            )
        if len(companions) < 2:
            raise ValueError(
                f"a pool of {len(companions)} companion(s) cannot vary the count; "
                f"one companion is CompanionedCartridgeModel and should be "
                f"constructed as one, so the count and permutation draws are "
                f"never dead knobs"
            )
        for companion in companions:
            require_matching_geometry(companion.geometry, slots.geometry)
        super().__init__(base=base, slots=slots)
        self._companions = companions
        self._companion_probability = companion_probability
        # Same constructor-owned device invariant as the single-companion
        # model: a CPU-drawn pool meeting a CUDA base would torch.cat across
        # devices on the first present forward, and a CPU-only suite cannot
        # reach that failure.
        device = str(next(iter(base.named_parameters()))[1].detach().device)
        for companion in self._companions:
            companion.to(device)

    def to(self, device: str) -> LMModelProto:
        """Move the base, the trainee slots and the pool onto a device.

        Args:
            device: Torch device string.

        Returns:
            This model, so the call chains.
        """
        super().to(device)
        for companion in self._companions:
            companion.to(device)
        return self

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Run the base with a drawn number of companions before the trainee.

        Args:
            input_ids: Token ids, shaped (batch, positions).
            labels: Targets for those same positions.

        Returns:
            The base model's output. Gradients reach the trainee's slots
            only; every companion block is detached.
        """
        present = float(torch.rand(())) < self._companion_probability
        count = int(torch.randint(1, len(self._companions) + 1, ()))
        order = torch.randperm(len(self._companions))
        if not present:
            return super().forward(input_ids=input_ids, labels=labels)
        chosen = [self._companions[int(order[position].item())] for position in range(count)]
        batch_size = int(input_ids.shape[0])
        blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.geometry["num_layers"]):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for companion in chosen:
                companion_key, companion_value = companion.layer_blocks(
                    layer, batch_size=batch_size
                )
                keys.append(companion_key.detach())
                values.append(companion_value.detach())
            trainee_key, trainee_value = self.slots.layer_blocks(layer, batch_size=batch_size)
            keys.append(trainee_key)
            values.append(trainee_value)
            blocks.append((torch.cat(keys, dim=SLOT_AXIS), torch.cat(values, dim=SLOT_AXIS)))
        attended = (
            int(input_ids.shape[1])
            + sum(companion.geometry["num_slots"] for companion in chosen)
            + self.geometry["num_slots"]
        )
        return self._base(
            input_ids=input_ids,
            labels=labels,
            past_key_values=Hooks.build_prefix_cache(blocks),
            attention_mask=torch.ones(
                (batch_size, attended), dtype=torch.long, device=input_ids.device
            ),
            use_cache=False,
        )


__all__ = [
    "CartridgeLoadResult",
    "CartridgeModel",
    "CompanionedCartridgeModel",
    "MultiCompanionedCartridgeModel",
]
