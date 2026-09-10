"""One frozen companion pool per replicate, built the recorded way.

THREE COPIES OF THIS CLASS EXISTED -- the diverse sweep's, the base-LoRA
sweep's, and the content-LoRA sweep's, the last of which imported the second
across module boundaries rather than defining a third. They differed in their
docstrings and in the TYPE of the plan they read four knobs out of, and in
nothing else.

WHY THAT IS WORTH ONE MODULE. The class is a seed formula:
``seed + (COMPANION_SEED_STRIDE + j) * len(seeds)`` for member ``j``. That
formula is what makes a sweep's diverse cells structurally identical to the
recorded diverse grid's, which is the whole basis on which their numbers are
subtracted from each other. Three copies of a seed formula do not fail when
one drifts -- they produce two records that look comparable, are labelled as
comparable, and are measuring different companions.

THE KNOBS ARE PARAMETERS, NOT A PLAN. The three sweeps carry two different
plan TypedDicts, and taking either one here would have forced the other to
convert. What the pool actually needs is four numbers, so it asks for four
numbers, and neither plan type is mentioned.

THE CACHE IS THE OTHER REASON THIS IS A CLASS. Several cells of one sweep run
at the same seed, and they must see the SAME pool -- by identity, not merely
by value -- or a composed arm and its control would be built over separately
trained companions. Caching per seed is what guarantees that, and it is also
what makes a resumed sweep cheap: pools are trained lazily, so only the cells
that still have to run pay for the pools they need.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.cli.cartridge_companion_sweep import COMPANION_SEED_STRIDE
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.types import CacheCapableLMProto


class SeededPoolProvider:
    """Deterministic per-seed companion pools, one member per corpus.

    Member ``j`` of seed ``s``'s pool trains ON THE j-TH companion corpus
    from ``s + (COMPANION_SEED_STRIDE + j) * len(seeds)`` -- the recorded
    formula, so a sweep's pools nest the recorded configuration with the
    recorded companion's corpus first while every later member carries a
    different voice.
    """

    _base: CacheCapableLMProto
    _companion_trains: Sequence[Sequence[torch.Tensor]]
    _slots: int
    _seeds: Sequence[int]
    _epochs: int
    _learning_rate: float
    _pools: dict[int, tuple[CartridgeSlots, ...]]

    def __init__(
        self,
        base: CacheCapableLMProto,
        companion_trains: Sequence[Sequence[torch.Tensor]],
        *,
        slots: int,
        seeds: Sequence[int],
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Hold what the pool builds need.

        Args:
            base: The frozen base every member trains in front of -- the
                plain one for the diverse sweep, the adapted one for the
                LoRA sweeps.
            companion_trains: One training-window sequence per pool member,
                in pool order.
            slots: Prefix positions per companion, from the plan.
            seeds: The sweep's measurement seeds. Only their COUNT enters the
                formula, which is what spaces one replicate's member seeds
                past the next replicate's.
            epochs: Passes over each companion's corpus, from the plan.
            learning_rate: Step size for AdamW, from the plan.
        """
        self._base = base
        self._companion_trains = companion_trains
        self._slots = slots
        self._seeds = seeds
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._pools = {}

    def pool(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """The frozen pool for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            One plain-trained companion per corpus, cached so every cell that
            shares this seed shares one pool by identity.
        """
        if seed not in self._pools:
            self._pools[seed] = tuple(
                train_cartridge(
                    self._base,
                    companion_train,
                    num_slots=self._slots,
                    seed=seed + (COMPANION_SEED_STRIDE + member) * len(self._seeds),
                    epochs=self._epochs,
                    learning_rate=self._learning_rate,
                )
                for member, companion_train in enumerate(self._companion_trains)
            )
        return self._pools[seed]


__all__ = ["SeededPoolProvider"]
