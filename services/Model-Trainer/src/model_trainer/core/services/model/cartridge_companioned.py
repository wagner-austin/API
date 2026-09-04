"""Composition-aware cartridge training, and the arms that measure it.

Split from :mod:`cartridge_measurement` when that module passed the 600-line
ceiling, along the seam the work itself has: that module measures what
PLAIN-trained cartridges do, and its numbers are bit-identity-certified
against their own history (sha256-certified record, cited on the wiki); this
module holds the INTERVENTION over those numbers -- training every cartridge
with a frozen companion present so composition stops being an untrained
capability (board task ``bc29dc3e``, attacking the two-compartment ceiling
measured in ``a67d6038``). Keeping the intervention out of the baseline's
module means no refactor here can reorder an RNG-consuming call there.

The published precedent for the intervention is ICAE's multi-span finding:
concatenation of separately compressed spans failed until concatenation
examples entered training (Ge et al. 2023 p. 8, personal wiki page
``ge-2023-in-context-autoencoder``).
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
    CompanionedCartridgeModel,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    compose,
)
from model_trainer.core.services.model.cartridge_measurement import (
    fresh_cartridge,
    held_out_gain,
)
from model_trainer.core.services.model.cartridge_scoring import train_on
from model_trainer.core.types import CacheCapableLMProto


def train_cartridge_with_companion(
    base: CacheCapableLMProto,
    corpus: Sequence[torch.Tensor],
    *,
    num_slots: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    companion: CartridgeSlots,
    companion_probability: float,
) -> CartridgeSlots:
    """Draw a cartridge and train it with a frozen stranger sometimes present.

    The composition-aware variant of
    :func:`~model_trainer.core.services.model.cartridge_measurement.train_cartridge`:
    identical draw, identical seeding discipline, and the one difference is
    that training forwards run through :class:`CompanionedCartridgeModel`,
    which concatenates the companion's detached blocks in front of the
    trainee's with the given per-step probability. Whether this lifts
    composed retention is the measurement; nothing here asserts that it does.

    THE SEED COVERS THE PRESENCE DRAWS TOO. The companion-presence draw
    consumes the same process-wide generator dropout does, and the seed is
    set at the same point ``train_cartridge`` sets it -- after the geometry
    probe -- so a run remains a function of its seed with the companion
    machinery included.

    Args:
        base: The frozen base to train in front of.
        corpus: Training windows.
        num_slots: Prefix positions for the trainee.
        seed: Seed for the draw, the dropout stream, and the presence draws.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.
        companion: The frozen stranger's slots. Not updated: gradients cannot
            reach it by construction.
        companion_probability: Chance per training forward that the companion
            is present, in (0, 1].

    Returns:
        The trained slots, detached from any model.

    Raises:
        ValueError: If the probability is outside (0, 1].
        AppError: With ``CARTRIDGE_GEOMETRY_MISMATCH`` if the companion was
            cut for a differently shaped model.
    """
    drawn = fresh_cartridge(base, num_slots=num_slots, seed=seed)
    model = CompanionedCartridgeModel(
        base=base,
        slots=drawn.slots,
        companion=companion,
        companion_probability=companion_probability,
    )
    torch.manual_seed(seed)
    _losses = train_on(model, corpus, epochs=epochs, learning_rate=learning_rate)
    return model.slots


class CompanionProviderProto(Protocol):
    """One frozen companion per replicate, keyed by the replicate's seed.

    A parameter rather than a kind-enum because the two kinds a sweep runs --
    a fresh noise draw and a plain-trained stranger -- are built from things
    the measurement function has no business holding (a corpus, a training
    schedule for someone else's cartridge). The caller builds the companion;
    this module measures with it.
    """

    def __call__(self, seed: int) -> CartridgeSlots:
        """Return the frozen companion for one replicate.

        Args:
            seed: The replicate's base seed. A provider must be a pure
                function of it, or the arm stops being a function of its
                seeds.

        Returns:
            The companion's slots. Never trained by the measurement.
        """
        ...


def measure_companioned_scaling(
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
    companion_for_seed: CompanionProviderProto,
    companion_probability: float,
) -> tuple[ReplicatedGain, ReplicatedGain, ReplicatedGain, tuple[ReplicatedGain, ...]]:
    """Measure composition where EVERY cartridge trained beside a companion.

    Identical arms and identical seed-offset rules to
    :func:`~model_trainer.core.services.model.cartridge_measurement.measure_composition_scaling`,
    with one difference -- the primary and every other-corpus cartridge are
    trained through :func:`train_cartridge_with_companion`, sharing one
    frozen companion per replicate. All cartridges train companioned because
    that is the deployment shape: a library where every compartment was
    built composition-aware, not one hardened compartment among naive ones.

    Args:
        base: The frozen base.
        first_train: Training windows for the cartridge whose retention is
            the finding.
        other_trains: One training-window sequence per additional cartridge.
        held_out: Items to score every arm on, drawn from the first corpus.
        arm: Name for this configuration, e.g. ``"noise-p0.5-n4"``.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.
        companion_for_seed: Builds the replicate's frozen training companion.
        companion_probability: Chance per training forward that the
            companion is present, in (0, 1].

    Returns:
        ``(alone, composed, untrained_composed, cross)``, exactly as the
        plain scaling measurement returns them. The alone arm is the
        solo-cost axis: what companioned training did to the primary's own
        gain is as much the finding as what it did to retention.

    Raises:
        ValueError: If the probability is outside (0, 1].
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            the minimum seeds are given, or ``CARTRIDGE_GEOMETRY_MISMATCH``
            if a provider returns a companion cut for another model.
    """
    alone: list[tuple[int, float]] = []
    composed: list[tuple[int, float]] = []
    untrained_composed: list[tuple[int, float]] = []
    cross: list[list[tuple[int, float]]] = [[] for _ in other_trains]
    for seed in seeds:
        companion = companion_for_seed(seed)
        first = train_cartridge_with_companion(
            base,
            first_train,
            num_slots=num_slots,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            companion=companion,
            companion_probability=companion_probability,
        )
        others = [
            train_cartridge_with_companion(
                base,
                other_train,
                num_slots=num_slots,
                seed=seed + (position + 1) * len(seeds),
                epochs=epochs,
                learning_rate=learning_rate,
                companion=companion,
                companion_probability=companion_probability,
            )
            for position, other_train in enumerate(other_trains)
        ]
        joined = functools.reduce(compose, others, first)
        untrained_others = [
            fresh_cartridge(
                base, num_slots=num_slots, seed=seed + (position + 1) * len(seeds)
            ).slots
            for position in range(len(other_trains))
        ]
        untrained_joined = functools.reduce(compose, untrained_others, first)
        alone.append((seed, held_out_gain(CartridgeModel(base=base, slots=first), held_out)))
        composed.append((seed, held_out_gain(CartridgeModel(base=base, slots=joined), held_out)))
        untrained_composed.append(
            (seed, held_out_gain(CartridgeModel(base=base, slots=untrained_joined), held_out))
        )
        for position, other in enumerate(others):
            cross[position].append(
                (seed, held_out_gain(CartridgeModel(base=base, slots=other), held_out))
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
    "CompanionProviderProto",
    "measure_companioned_scaling",
    "train_cartridge_with_companion",
]
