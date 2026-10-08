"""Varied-count companioned training, and the arms that measure it.

Sibling of :mod:`cartridge_companioned`, split for the same reason that
module was split from :mod:`cartridge_measurement`: the single-companion
grid's numbers are bit-identity-certified against their own history (record
sha256 ``9e87e816`` locally, ``6e63dad7`` cross-node on the cluster), and
keeping this intervention in its own module means no refactor here can
reorder an RNG-consuming call there. The measured target is the recipe's
retention decay with deployment count -- 44.6% at four compartments, 26.5%
at eight -- and the hypothesis is that training under a drawn number of
simultaneous companions teaches count-invariance a fixed single companion
cannot (board task ``7815a0fd``).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Protocol

import torch

from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    replicate,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import (
    CartridgeModel,
    MultiCompanionedCartridgeModel,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    compose,
)
from model_trainer.core.services.model.cartridge_measurement import (
    ComposedReplicate,
    ReplicateConsumerProto,
    fresh_cartridge,
    held_out_gain,
)
from model_trainer.core.services.model.cartridge_scoring import train_on
from model_trainer.core.types import CacheCapableLMProto


def train_cartridge_with_companions(
    base: CacheCapableLMProto,
    corpus: Sequence[torch.Tensor],
    *,
    num_slots: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    companions: tuple[CartridgeSlots, ...],
    companion_probability: float,
) -> CartridgeSlots:
    """Draw a cartridge and train it beside a drawn number of strangers.

    The varied-count variant of
    :func:`~model_trainer.core.services.model.cartridge_companioned.train_cartridge_with_companion`:
    identical draw, identical seeding discipline, and the one difference is
    that training forwards run through
    :class:`MultiCompanionedCartridgeModel`, which draws per forward how
    many of the frozen pool stand in front of the trainee. Whether this
    closes the count-decay is the measurement; nothing here asserts it does.

    THE SEED COVERS EVERY DRAW. Presence, count and permutation all consume
    the process-wide generator, and the seed is set after the geometry probe
    exactly where ``train_cartridge`` sets it, so a run remains a function
    of its seed with the pool machinery included.

    Args:
        base: The frozen base to train in front of.
        corpus: Training windows.
        num_slots: Prefix positions for the trainee.
        seed: Seed for the draw, the dropout stream, and every pool draw.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.
        companions: The frozen pool, at least two members. Never updated:
            gradients cannot reach any member by construction.
        companion_probability: Chance per training forward that companions
            are present, in (0, 1].

    Returns:
        The trained slots, detached from any model.

    Raises:
        ValueError: If the probability is outside (0, 1] or the pool holds
            fewer than two companions.
        AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if a pool member was
            cut for a differently shaped model.
    """
    drawn = fresh_cartridge(base, num_slots=num_slots, seed=seed)
    model = MultiCompanionedCartridgeModel(
        base=base,
        slots=drawn.slots,
        companions=companions,
        companion_probability=companion_probability,
    )
    torch.manual_seed(seed)
    _losses = train_on(model, corpus, epochs=epochs, learning_rate=learning_rate)
    return model.slots


class CompanionPoolProviderProto(Protocol):
    """One frozen companion pool per replicate, keyed by the replicate's seed.

    A parameter for the reason
    :class:`~model_trainer.core.services.model.cartridge_companioned.CompanionProviderProto`
    is one: the pool is built from things the measurement has no business
    holding (a held-out corpus and a training schedule for someone else's
    cartridges). The caller builds the pool; this module measures with it.
    """

    def __call__(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """Return the frozen pool for one replicate.

        Args:
            seed: The replicate's base seed. A provider must be a pure
                function of it, or the arm stops being a function of its
                seeds.

        Returns:
            The pool's slots, at least two members. Never trained by the
            measurement.
        """
        ...


def companioned_replicates(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[torch.Tensor],
    other_trains: Sequence[Sequence[torch.Tensor]],
    num_slots: int,
    seeds: Sequence[int],
    seed_stride: int,
    epochs: int,
    learning_rate: float,
    pool_for_seed: CompanionPoolProviderProto,
    companion_probability: float,
    consume: ReplicateConsumerProto,
) -> None:
    """Build one replicate per seed where EVERY cartridge trained beside a pool.

    THE VARIED-COMPANIONED GEOMETRY, WITH ONE OWNER, for the reason
    :func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`
    gives for the plain one: the seed offsets, the fold order and the
    untrained control are what two measurements must share to be subtracted,
    and the corpus grid and the trait grid both drive this one. It hands each
    replicate to ``consume`` before building the next, because scoring flips
    the base to evaluation mode and the geometry probe's RNG consumption
    depends on that mode -- a caller that collected every replicate first
    would train the next one from a different RNG state.

    Args:
        base: The frozen base.
        first_train: Training items for the cartridge whose retention is the
            finding.
        other_trains: One training-item sequence per additional cartridge.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each.
        seed_stride: Spacing between a replicate's cartridge draws: the
            number of seeds in the whole plan, for the reason
            :func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`
            gives.
        epochs: Passes over each training set.
        learning_rate: Step size for AdamW.
        pool_for_seed: Builds the replicate's frozen companion pool.
        companion_probability: Chance per training forward that companions
            are present, in (0, 1].
        consume: Scores one replicate. Called once per seed, in seed order,
            and before the next replicate is built.

    Raises:
        ValueError: If the probability is outside (0, 1] or a pool holds
            fewer than two companions.
        AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if a provider returns a
            pool cut for another model.
    """
    for seed in seeds:
        pool = pool_for_seed(seed)
        first = train_cartridge_with_companions(
            base,
            first_train,
            num_slots=num_slots,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            companions=pool,
            companion_probability=companion_probability,
        )
        others = [
            train_cartridge_with_companions(
                base,
                other_train,
                num_slots=num_slots,
                seed=seed + (position + 1) * seed_stride,
                epochs=epochs,
                learning_rate=learning_rate,
                companions=pool,
                companion_probability=companion_probability,
            )
            for position, other_train in enumerate(other_trains)
        ]
        untrained_others = [
            fresh_cartridge(
                base, num_slots=num_slots, seed=seed + (position + 1) * seed_stride
            ).slots
            for position in range(len(other_trains))
        ]
        consume(
            ComposedReplicate(
                seed=seed,
                alone=first,
                composed=functools.reduce(compose, others, first),
                untrained_composed=functools.reduce(compose, untrained_others, first),
                others=tuple(others),
            )
        )


def measure_varied_companioned_scaling(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[torch.Tensor],
    other_trains: Sequence[Sequence[torch.Tensor]],
    held_out: Sequence[torch.Tensor],
    arm: str,
    num_slots: int,
    seeds: Sequence[int],
    epochs: int,
    learning_rate: float,
    pool_for_seed: CompanionPoolProviderProto,
    companion_probability: float,
) -> tuple[ReplicatedGain, ReplicatedGain, ReplicatedGain, tuple[ReplicatedGain, ...]]:
    """Measure composition where EVERY cartridge trained beside a drawn pool.

    Identical arms and identical seed-offset rules to
    :func:`~model_trainer.core.services.model.cartridge_companioned.measure_companioned_scaling`,
    with one difference -- every cartridge trains through
    :func:`train_cartridge_with_companions`, sharing one frozen pool per
    replicate. All cartridges train varied-companioned because that is the
    deployment shape under test: a library where every compartment was
    built count-invariant.

    Args:
        base: The frozen base.
        first_train: Training windows for the cartridge whose retention is
            the finding.
        other_trains: One training-window sequence per additional cartridge.
        held_out: Items to score every arm on, drawn from the first corpus.
        arm: Name for this configuration, e.g. ``"varied-K3-n8"``.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.
        pool_for_seed: Builds the replicate's frozen companion pool.
        companion_probability: Chance per training forward that companions
            are present, in (0, 1].

    Returns:
        ``(alone, composed, untrained_composed, cross)``, exactly as the
        sibling measurements return them. The alone arm is the solo-cost
        axis, mandatory beside every composed number.

    Raises:
        ValueError: If the probability is outside (0, 1] or a pool holds
            fewer than two companions.
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            the minimum seeds are given, or ``CARTRIDGE_GEOMETRY_MISMATCH``
            if a provider returns a pool cut for another model.
    """
    alone: list[tuple[int, float]] = []
    composed: list[tuple[int, float]] = []
    untrained_composed: list[tuple[int, float]] = []
    cross: list[list[tuple[int, float]]] = [[] for _ in other_trains]

    def _gain(slots: CartridgeSlots) -> float:
        """Score one slot block on the primary held-out items.

        Args:
            slots: The block to put in front of the base.

        Returns:
            The held-out gain.
        """
        return held_out_gain(CartridgeModel(base=base, slots=slots), held_out)

    def _score(built: ComposedReplicate, /) -> None:
        """Score one replicate's arms, in the order the record has always used.

        Args:
            built: The replicate just constructed.
        """
        seed = built["seed"]
        alone.append((seed, _gain(built["alone"])))
        composed.append((seed, _gain(built["composed"])))
        untrained_composed.append((seed, _gain(built["untrained_composed"])))
        for position, other in enumerate(built["others"]):
            cross[position].append((seed, _gain(other)))

    companioned_replicates(
        base,
        first_train=first_train,
        other_trains=other_trains,
        num_slots=num_slots,
        seeds=seeds,
        seed_stride=len(seeds),
        epochs=epochs,
        learning_rate=learning_rate,
        pool_for_seed=pool_for_seed,
        companion_probability=companion_probability,
        consume=_score,
    )
    return (
        replicate(f"{arm}-alone", alone),
        replicate(f"{arm}-composed", composed),
        replicate(f"{arm}-untrained-composed", untrained_composed),
        tuple(
            replicate(f"{arm}-cross-{position}", results) for position, results in enumerate(cross)
        ),
    )


__all__ = [
    "CompanionPoolProviderProto",
    "companioned_replicates",
    "measure_varied_companioned_scaling",
    "train_cartridge_with_companions",
]
