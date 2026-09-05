"""Base-side composition training: a LoRA that learns to read crowded prefixes.

THE MEASUREMENT CHAIN THIS ANSWERS. Three arms converged on one verdict
(board tasks ``684492dd``, ``7815a0fd``, ``d2c03dd4`` and the scale rung):
cartridge-side recipes can train content interference away, but the residual
many-compartment cost is STRUCTURAL -- the base was never trained to read a
512-slot foreign prefix -- and it inverts with depth (gpt2-medium's n8
noise-composition control is negative where gpt2-small's is positive). The
lever that remains is the base itself. This module trains a LoRA on the
base's attention while frozen composed cartridges crowd its prefix, so the
adaptation belongs to the MODEL and fixes every cartridge at once (board
task ``6c752568``).

WHAT IS TRAINABLE IS EXACTLY THE LORA. The pool cartridges are detached at
every forward, and :meth:`CrowdedPrefixModel.parameters` returns only the
parameters PEFT left trainable, so the optimizer cannot reach the base
weights or the pool even in principle -- the same freeze-by-construction
argument the companioned cartridge models make, pointed the other way.

THE COUNT IS DRAWN, NEVER FIXED. Each forward draws how many pool members
stand in the prefix (uniform over 1..max_drawn) and which ones (a
permutation), consuming the global generator identically on every step, so
training remains a pure function of its seed and the base meets the crowded
regime at every width it will serve.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    SLOT_AXIS,
    CartridgeSlots,
)
from model_trainer.core.services.model.cartridge_scoring import train_on
from model_trainer.core.types import (
    CacheCapableLMProto,
    ForwardOutProto,
    LMModelProto,
    ParameterLike,
)


class CrowdedPrefixModel:
    """A PEFT-adapted base doing language modeling behind drawn company.

    Shaped like :class:`CartridgeModel` from the trainer's point of view --
    ``parameters()``, ``train()``, ``forward(input_ids=, labels=)`` -- so
    :func:`~model_trainer.core.services.model.cartridge_scoring.train_on`
    drives it unchanged. What differs is which side learns: here the pool of
    cartridges is frozen scenery and the gradients land in the base's LoRA.
    """

    _adapted: CacheCapableLMProto
    _pool: tuple[CartridgeSlots, ...]
    _max_drawn: int
    _num_layers: int
    _slots_per_member: int

    def __init__(
        self,
        *,
        adapted: CacheCapableLMProto,
        pool: tuple[CartridgeSlots, ...],
        max_drawn: int,
    ) -> None:
        """Hold the adapted base and the frozen pool.

        Args:
            adapted: The PEFT-wrapped base. Its trainable parameters -- the
                LoRA's, by PEFT's own freezing -- are what training updates.
            pool: Frozen cartridges the prefix is drawn from. At least two,
                or the permutation draw is a dead knob; every member cut for
                the same model, with one slot count, so the attended length
                is a pure function of the drawn count.
            max_drawn: Largest count one forward may draw, at least two (a
                fixed count of one is a plain prefixed forward and should be
                spelled as one) and at most the pool size.

        Raises:
            ValueError: If the pool or ``max_drawn`` is out of contract.
        """
        if len(pool) < 2:
            raise ValueError(
                f"a pool of {len(pool)} cartridge(s) cannot crowd a prefix; "
                f"the permutation draw would be a dead knob"
            )
        if not 2 <= max_drawn <= len(pool):
            raise ValueError(
                f"max_drawn={max_drawn} is outside [2, {len(pool)}]: one drawn "
                f"cartridge is a plain prefixed forward, and more than the pool "
                f"holds cannot be drawn"
            )
        slot_counts = {member.geometry["num_slots"] for member in pool}
        if len(slot_counts) != 1:
            raise ValueError(
                f"pool members carry {sorted(slot_counts)} slots; mixed widths "
                f"would make the attended length depend on WHICH members were "
                f"drawn rather than how many"
            )
        self._adapted = adapted
        self._pool = pool
        self._max_drawn = max_drawn
        self._num_layers = pool[0].geometry["num_layers"]
        self._slots_per_member = pool[0].geometry["num_slots"]
        device = str(next(iter(adapted.named_parameters()))[1].detach().device)
        for member in self._pool:
            member.to(device)

    def parameters(self) -> Sequence[ParameterLike]:
        """Return the parameters PEFT left trainable -- the LoRA's.

        Returns:
            The trainable parameters, and only those: the base's own weights
            are frozen by PEFT and the pool never appears here at all.
        """
        return [parameter for parameter in self._adapted.parameters() if parameter.requires_grad]

    def train(self) -> None:
        """Put the adapted base in training mode."""
        self._adapted.train()

    def eval(self) -> None:
        """Put the adapted base in evaluation mode."""
        self._adapted.eval()

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Run the adapted base behind a drawn number of frozen cartridges.

        Args:
            input_ids: Token ids, shaped (batch, positions).
            labels: Targets for those same positions.

        Returns:
            The adapted base's output. Gradients reach the LoRA parameters
            only; every pool block is detached.
        """
        count = int(torch.randint(1, self._max_drawn + 1, ()))
        order = torch.randperm(len(self._pool))
        chosen = [self._pool[int(order[position].item())] for position in range(count)]
        batch_size = int(input_ids.shape[0])
        blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self._num_layers):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for member in chosen:
                member_key, member_value = member.layer_blocks(layer, batch_size=batch_size)
                keys.append(member_key.detach())
                values.append(member_value.detach())
            blocks.append((torch.cat(keys, dim=SLOT_AXIS), torch.cat(values, dim=SLOT_AXIS)))
        attended = int(input_ids.shape[1]) + count * self._slots_per_member
        return self._adapted(
            input_ids=input_ids,
            labels=labels,
            past_key_values=Hooks.build_prefix_cache(blocks),
            attention_mask=torch.ones(
                (batch_size, attended), dtype=torch.long, device=input_ids.device
            ),
            use_cache=False,
        )


def train_composition_lora(
    adapted: CacheCapableLMProto,
    pool: tuple[CartridgeSlots, ...],
    corpus: Sequence[torch.Tensor],
    *,
    max_drawn: int,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    """Train the base's LoRA to do language modeling behind drawn company.

    The base-side mirror of ``train_cartridge_with_companions``: identical
    seeding discipline (the seed is set immediately before the loop and
    covers dropout and every count and permutation draw), the plainest loop
    that is still real underneath, and the one difference is which side the
    gradients land on.

    Args:
        adapted: The PEFT-wrapped base whose LoRA learns.
        pool: Frozen cartridges the prefix is drawn from.
        corpus: Training windows, each shaped (1, positions). These must be
            held out from every corpus the adapted base will later be
            measured on -- the caller owns that wall and the CLI refuses
            breaches of it.
        max_drawn: Largest count one forward may draw.
        seed: Seed for dropout and the per-step draws.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW over the LoRA parameters.

    Returns:
        The mean loss of each epoch, in order, so the record can show the
        adaptation converged rather than asserting it did.

    Raises:
        ValueError: Propagated from :class:`CrowdedPrefixModel`.
    """
    model = CrowdedPrefixModel(adapted=adapted, pool=pool, max_drawn=max_drawn)
    torch.manual_seed(seed)
    return train_on(model, corpus, epochs=epochs, learning_rate=learning_rate)


def freeze_adapted(adapted: LMModelProto) -> None:
    """Clear ``requires_grad`` on every parameter of the adapted base.

    Run between the LoRA's training and the measurement phase: from here on
    the adapted base is a fixed serving artifact, and nothing downstream --
    not even a path that skips :class:`CartridgeModel`'s own freeze -- can
    move it.

    Args:
        adapted: The model to freeze in place.
    """
    for _name, parameter in adapted.named_parameters():
        parameter.requires_grad = False


__all__ = [
    "CrowdedPrefixModel",
    "freeze_adapted",
    "train_composition_lora",
]
