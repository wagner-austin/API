"""The trainable half of a cartridge: key and value blocks, and their shape.

Holds the parameters and nothing else. Attaching them to a forward pass is
:mod:`cartridge_model`'s job, and deciding when to do it is the strategy's.
The split is what lets the parameters be tested against a real model without
a training loop, and the wrapper be tested without a real model.
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
from model_trainer.core.types import NamedParameter, ParameterLike

#: Standard deviation of the initial key and value blocks.
#:
#: Matches the initializer range transformers uses for a model's own weights
#: (``initializer_range`` defaults to 0.02 on both GPT-2 and Llama configs), so
#: a fresh cartridge enters attention at the same scale as the activations it
#: sits beside. Zeros were the alternative and are wrong here: an all-zero key
#: block gives every slot an identical attention logit, so the slots stay
#: symmetric and the cartridge collapses to a single learned position.
_INIT_STD = 0.02

#: Axis the slots live on in a ``(batch, kv_heads, slots, head_dim)`` block.
#:
#: Named because two operations index it and a wrong axis is silent: joining on
#: the head axis would produce a block of the right total size describing a
#: model with twice the heads, and attention would read it without complaint.
_SLOT_AXIS = 2


def discover_geometry(layers: Sequence[torch.Tensor], *, num_slots: int) -> CartridgeGeometry:
    """Read a cartridge's shape off a model's own cached keys.

    Takes the measured tensors rather than the model, so the measurement
    (a forward pass, injected) and its interpretation (this, pure arithmetic)
    are separately testable and this module stays free of hooks.

    This is the architecture-agnostic route: under grouped-query attention the
    key-value head count is smaller than the attention head count, and the two
    are spelled differently on every model family, so reading a config would
    mean a per-architecture table. The cache itself carries the answer for any
    architecture that has one.

    Args:
        layers: One cached key tensor per attention layer, layer zero first,
            as :meth:`Hooks.probe_cache_layers` returns them.
        num_slots: Prefix positions the caller wants. Carried into the result
            unchanged; it is the caller's choice, not the model's.

    Returns:
        The geometry a cartridge for this model must be cut to.

    Raises:
        AppError: With ``CARTRIDGE_MODEL_REPORTS_NO_CACHE`` if there are no
            layers, or if the keys are not four-dimensional. Such a model
            cannot host a prefix.
    """
    if not layers:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE,
            (
                "this model returned an empty key-value cache from a probe forward, "
                "so it has no attention cache for a cartridge to prepend to; the "
                "cartridge strategy needs a transformer with key-value caching"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE),
        )
    first_key = layers[0]
    if first_key.dim() != 4:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE,
            (
                f"this model's cached keys have {first_key.dim()} dimensions, and a "
                f"cartridge needs the standard (batch, kv_heads, positions, head_dim) "
                f"layout to know how many heads and how wide to cut its slots"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_MODEL_REPORTS_NO_CACHE),
        )
    return CartridgeGeometry(
        num_layers=len(layers),
        num_kv_heads=int(first_key.shape[1]),
        head_dim=int(first_key.shape[3]),
        num_slots=num_slots,
    )


def _slot_name(layer: int, *, is_key: bool) -> str:
    """Name one slot tensor for state-dict round-tripping.

    Args:
        layer: Zero-based layer index.
        is_key: Whether this is the key block rather than the value block.

    Returns:
        The tensor's name, stable across save and load.
    """
    kind = "key" if is_key else "value"
    return f"cartridge.layer_{layer}.{kind}"


class CartridgeSlots:
    """The key and value blocks a cartridge trains, one pair per layer.

    Deliberately NOT a ``torch.nn.Module``. The surface needed here is small
    and fully typed -- parameters, a state dict, a device move -- while
    ``nn.Module`` would bring an untyped base whose ``train`` returns itself
    where :class:`~model_trainer.core.types.LMModelProto` declares ``None``,
    and inheriting that conflict to gain three methods is a poor trade. The
    tensors are ordinary leaf tensors with ``requires_grad``, which is all
    autograd and any optimizer require.

    Attributes:
        geometry: The shape these blocks were cut to.
    """

    geometry: CartridgeGeometry
    _keys: list[torch.Tensor]
    _values: list[torch.Tensor]

    def __init__(
        self,
        *,
        geometry: CartridgeGeometry,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> None:
        """Hold pre-built blocks.

        Callers build blocks with :func:`initialise_slots` or
        :func:`slots_from_state`, which are the two ways a cartridge comes
        into existence: drawn fresh, or read back from disk.

        Args:
            geometry: The shape the blocks were cut to.
            keys: One key block per layer.
            values: One value block per layer.
        """
        self.geometry = geometry
        self._keys = keys
        self._values = values

    def parameters(self) -> Sequence[ParameterLike]:
        """Return every trainable tensor, keys and values interleaved by layer.

        This is what an optimizer is built from, and it holds the cartridge's
        whole trainable surface: the base model's weights are deliberately
        absent, so "only the prefix is trained" is a structural fact rather
        than a convention someone has to maintain.

        Returns:
            The slot tensors, in layer order.
        """
        ordered: list[ParameterLike] = []
        for layer in range(self.geometry["num_layers"]):
            ordered.append(self._keys[layer])
            ordered.append(self._values[layer])
        return ordered

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return every trainable tensor with its stable name.

        Returns:
            Name and tensor pairs, in layer order.
        """
        named: list[tuple[str, NamedParameter]] = []
        for layer in range(self.geometry["num_layers"]):
            named.append((_slot_name(layer, is_key=True), self._keys[layer]))
            named.append((_slot_name(layer, is_key=False), self._values[layer]))
        return named

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the slot tensors by name, for saving.

        Returns:
            Every block, keyed by its stable name.
        """
        saved: dict[str, torch.Tensor] = {}
        for layer in range(self.geometry["num_layers"]):
            saved[_slot_name(layer, is_key=True)] = self._keys[layer]
            saved[_slot_name(layer, is_key=False)] = self._values[layer]
        return saved

    def to(self, device: str) -> None:
        """Move every block onto a device, in place.

        Rebinds rather than mutating: ``Tensor.to`` returns a new tensor when
        the device differs, and the new one is the leaf the optimizer must
        see, so the references held here are replaced.

        Args:
            device: Torch device string.
        """
        self._keys = [tensor.to(device).detach().requires_grad_(True) for tensor in self._keys]
        self._values = [tensor.to(device).detach().requires_grad_(True) for tensor in self._values]

    def layer_blocks(self, layer: int, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one layer's blocks widened to a batch.

        The stored blocks carry a batch dimension of one, because a cartridge
        is one object shared by every sequence that uses it. ``expand`` widens
        that view without copying, so a batch of 32 does not allocate 32
        cartridges, and gradients from every row accumulate into the one set
        of parameters -- which is the training signal wanted.

        Args:
            layer: Zero-based layer index.
            batch_size: Rows in the batch being run.

        Returns:
            The key and value blocks for this layer, batch-shaped.
        """
        geometry = self.geometry
        shape = (
            batch_size,
            geometry["num_kv_heads"],
            geometry["num_slots"],
            geometry["head_dim"],
        )
        return self._keys[layer].expand(shape), self._values[layer].expand(shape)


def _empty_block(geometry: CartridgeGeometry, generator: torch.Generator) -> torch.Tensor:
    """Draw one initial block.

    Args:
        geometry: The shape to cut to.
        generator: Seeded source, so a run can say what it started from.

    Returns:
        A leaf tensor with gradients enabled, batch dimension one.
    """
    drawn = torch.empty(
        1,
        geometry["num_kv_heads"],
        geometry["num_slots"],
        geometry["head_dim"],
    )
    drawn.normal_(mean=0.0, std=_INIT_STD, generator=generator)
    return drawn.requires_grad_(True)


def initialise_slots(geometry: CartridgeGeometry, *, seed: int) -> CartridgeSlots:
    """Draw a fresh cartridge.

    Args:
        geometry: The shape to cut to.
        seed: Seed for the draw, recorded on the run so it can be repeated.

    Returns:
        Newly drawn slots, ready to train.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    layers = range(geometry["num_layers"])
    return CartridgeSlots(
        geometry=geometry,
        keys=[_empty_block(geometry, generator) for _ in layers],
        values=[_empty_block(geometry, generator) for _ in layers],
    )


def _require_block(
    state: dict[str, torch.Tensor],
    name: str,
    geometry: CartridgeGeometry,
) -> torch.Tensor:
    """Read one named block out of a loaded state dict, checking its shape.

    Args:
        state: The loaded tensors.
        name: The block's stable name.
        geometry: The shape the manifest declares.

    Returns:
        The block, as a leaf tensor with gradients enabled.

    Raises:
        AppError: With ``CARTRIDGE_STATE_INCOMPLETE`` if the block is absent,
            or ``CARTRIDGE_GEOMETRY_MISMATCH`` if it is present at a shape the
            manifest does not describe.
    """
    if name not in state:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_STATE_INCOMPLETE,
            (
                f"the saved cartridge declares {geometry['num_layers']} layers but "
                f"carries no tensor named {name!r}; the manifest and the weights "
                f"beside it describe different objects"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_STATE_INCOMPLETE),
        )
    block = state[name]
    expected = (1, geometry["num_kv_heads"], geometry["num_slots"], geometry["head_dim"])
    actual = tuple(int(size) for size in block.shape)
    if actual != expected:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH,
            (
                f"the saved tensor {name!r} has shape {actual}, but this cartridge's "
                f"manifest describes {expected}; loading it would attach a prefix "
                f"shaped for a different model"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH),
        )
    return block.detach().requires_grad_(True)


def slots_from_state(
    state: dict[str, torch.Tensor],
    geometry: CartridgeGeometry,
) -> CartridgeSlots:
    """Rebuild a cartridge from tensors read off disk.

    Args:
        state: The loaded tensors, by name.
        geometry: The shape its manifest declares.

    Returns:
        The rebuilt slots.

    Raises:
        AppError: With ``CARTRIDGE_STATE_INCOMPLETE`` or
            ``CARTRIDGE_GEOMETRY_MISMATCH`` if the tensors and the manifest
            disagree.
    """
    layers = range(geometry["num_layers"])
    return CartridgeSlots(
        geometry=geometry,
        keys=[_require_block(state, _slot_name(n, is_key=True), geometry) for n in layers],
        values=[_require_block(state, _slot_name(n, is_key=False), geometry) for n in layers],
    )


def compose(first: CartridgeSlots, second: CartridgeSlots) -> CartridgeSlots:
    """Join two cartridges into one prefix, laid end to end.

    WHY THIS IS CONCATENATION AND NOT ADDITION, which is the whole reason a
    cartridge is a different kind of object from a steering vector. A steering
    vector is a direction added into the residual stream, so combining two
    means summing them, and the sum is a third direction that is neither --
    measured as a 15.7 to 40.1 point loss of trait expression at two vectors
    (Subbiah et al. 2026). A cartridge is attention CONTEXT. Two contexts
    combine the way two documents in a prompt combine: both are present, at
    their own positions, and attention decides what to read. Nothing is
    averaged and nothing is displaced.

    The cost is the honest one: the prefix gets longer. Composition buys
    retention and pays attention cost, where summing buys constant cost and
    pays in interference.

    ORDER DOES NOT MATTER, AND THAT IS MEASURED RATHER THAN ASSUMED. This
    docstring used to say the opposite -- that the two orders give different
    objects because the slots sit at different positions -- and that was
    wrong. A cartridge's slots are raw parameters: unlike keys derived from
    tokens, they never pass through a position embedding on the way into the
    cache, so they carry no position at all. Softmax attention over a set of
    keys is permutation-equivariant when the keys and values are permuted
    together, which concatenating in the other order is.

    Measured on the tiny rung, 2026-09-03: ``compose(a, b)`` and
    ``compose(b, a)`` agree to 3.6e-07 in logits, as does reversing the slot
    order inside one composed cartridge -- float32 reduction noise, not a
    difference. The first attempt at this measurement reported 0.43 and was
    wrong: it ran the forwards in TRAIN mode, where GPT-2's three 0.1 dropouts
    make two runs of one input differ anyway. ``score_held_out`` calls
    ``eval`` for exactly that reason.

    This is a real difference from putting two documents in a prompt, where
    order changes the answer and the second document is often the one that
    wins. Callers need not think about it, and nothing here sorts.

    Args:
        first: The cartridge whose slots come first.
        second: The cartridge whose slots follow.

    Returns:
        A cartridge holding both, with the summed slot count. Its blocks are
        new tensors: the inputs are unchanged and remain independently usable.

    Raises:
        AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if the two were cut for
            differently shaped models. Two prefixes for different models
            cannot occupy one cache.
    """
    require_matching_geometry(first.geometry, second.geometry)
    layers = range(first.geometry["num_layers"])
    joined = CartridgeGeometry(
        num_layers=first.geometry["num_layers"],
        num_kv_heads=first.geometry["num_kv_heads"],
        head_dim=first.geometry["head_dim"],
        num_slots=first.geometry["num_slots"] + second.geometry["num_slots"],
    )
    return CartridgeSlots(
        geometry=joined,
        keys=[_join_blocks(first, second, layer, is_key=True) for layer in layers],
        values=[_join_blocks(first, second, layer, is_key=False) for layer in layers],
    )


def _join_blocks(
    first: CartridgeSlots, second: CartridgeSlots, layer: int, *, is_key: bool
) -> torch.Tensor:
    """Concatenate one layer's blocks along the slot axis.

    Args:
        first: The cartridge whose slots come first.
        second: The cartridge whose slots follow.
        layer: Zero-based layer index.
        is_key: Whether to join the key blocks rather than the value blocks.

    Returns:
        The joined block, a leaf tensor with gradients enabled so a composed
        cartridge can itself be trained further.
    """
    index = 2 * layer + (0 if is_key else 1)
    left = first.named_parameters()[index][1].detach()
    right = second.named_parameters()[index][1].detach()
    return torch.cat([left, right], dim=_SLOT_AXIS).requires_grad_(True)


def require_matching_geometry(
    saved: CartridgeGeometry,
    model_shape: CartridgeGeometry,
) -> None:
    """Refuse a cartridge that was cut for a different model.

    Compares only the three fields the MODEL decides. The slot count is the
    caller's and is carried by the cartridge, so a 512-slot cartridge is
    legitimate on a model a 2048-slot one was also trained against.

    Args:
        saved: Geometry the cartridge was trained at.
        model_shape: Geometry the base model reports now.

    Raises:
        AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if the layer count,
            head count or head width differ.
    """
    compared: list[tuple[str, int, int]] = [
        ("num_layers", saved["num_layers"], model_shape["num_layers"]),
        ("num_kv_heads", saved["num_kv_heads"], model_shape["num_kv_heads"]),
        ("head_dim", saved["head_dim"], model_shape["head_dim"]),
    ]
    mismatched = [(field, was, now) for field, was, now in compared if was != now]
    if not mismatched:
        return
    described = ", ".join(f"{field} {was} vs {now}" for field, was, now in mismatched)
    raise AppError(
        ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH,
        (
            f"this cartridge was trained against a differently shaped model ({described}); "
            f"a prefix is a block of that model's own attention keys and values, so it "
            f"carries no meaning on another one"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_MISMATCH),
    )


__all__ = [
    "CartridgeSlots",
    "compose",
    "discover_geometry",
    "initialise_slots",
    "require_matching_geometry",
    "slots_from_state",
]
