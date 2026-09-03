"""The arms a cartridge measurement runs, each replicated across seeds.

WHAT THIS IS FOR. ``tests/test_cartridge_*.py`` measure a two-layer,
two-head model with randomly initialised weights, because a test suite must
run in seconds without a model cache. That model answered the questions it was
asked and three of its answers did not survive contact with a real one:

    finding                       tiny (2L/2H)      gpt2 (12L/12H)
    an untrained prefix costs     ~nothing          -0.7612, 0 of 14 items
    more slots stop paying at     ~8 slots          not within 2..512
    past that, slots              actively hurt     still pay, slowly
    composition retains           ~26%              ~59%
    the loss is mostly            dilution          interference

The gpt2 sweep, three seeds, noise floor 0.0202: 2 -> +0.7358, 8 -> +0.8224,
32 -> +0.8809, 128 -> +0.9104, 512 -> +0.9381. Every 4x step clears the floor,
including the last, so this measurement found NO saturation point -- returns
diminish sharply (+0.087, +0.058, +0.030, +0.028 per step) and never vanish.
The tiny model saturates by eight slots and then reverses.

The last row is the one that mattered. On the tiny model an untrained prefix
was near-neutral, so padding a cartridge with untrained slots isolated the cost
of LENGTH and the loss looked like dilution. On a real model untrained slots do
active damage, so the same control measures damage instead -- and the slot
sweep shows length costing nothing at all. Same code, same arms, opposite
conclusion, and nothing but running it on a real base could have found that.

So this module exists to make the real-base run a first-class, recorded thing
rather than a script somebody had once. It builds cartridges from
:func:`measure_geometry`, :func:`initialise_slots` and :class:`CartridgeModel`
directly rather than through :meth:`CartridgeStrategy.adapt`, because ``adapt``
takes a full :class:`ModelTrainConfig` and a measurement has no run to
configure -- twenty-odd fields would have to be invented, and each invented
field is a knob whose value nobody chose but which is in the record anyway.

EVERY ARM IS REPLICATED. Each returns a
:class:`~model_trainer.core.contracts.replicated_measurement.ReplicatedGain`,
which cannot be built from one seed. That is deliberate and it is the whole
lesson of the first pass: the single-seed sweep reported differences of 0.02
as findings, and 0.02 is what this measurement's own noise turned out to be.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    replicate,
)
from model_trainer.core.services.finetuning.strategies.cartridge import measure_geometry
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    compose,
    initialise_slots,
)
from model_trainer.core.services.model.cartridge_scoring import score_held_out, train_on
from model_trainer.core.types import CacheCapableLMProto


def fresh_cartridge(base: CacheCapableLMProto, *, num_slots: int, seed: int) -> CartridgeModel:
    """Put a newly drawn cartridge in front of a base model, on its device.

    Args:
        base: The frozen base. Its device decides the cartridge's: the slots
            are drawn on the CPU by :func:`initialise_slots` and moved here,
            so a caller never has to state a device the base already knows.
        num_slots: Prefix positions.
        seed: Seed for the draw.

    Returns:
        The cartridge-wrapped model, ready to train.
    """
    geometry = measure_geometry(base, num_slots=num_slots)
    model = CartridgeModel(base=base, slots=initialise_slots(geometry, seed=seed))
    # The base is already where it belongs; this moves the slots to join it.
    model.to(str(next(iter(base.named_parameters()))[1].detach().device))
    return model


def _gain(model: CartridgeModel, held_out: Sequence[torch.Tensor]) -> float:
    """Score held-out items with a prefix and without it, and take the difference.

    Args:
        model: The cartridge-wrapped model. Its own base is the control.
        held_out: Items the cartridge was not trained on.

    Returns:
        How much lower the loss is with the prefix. Negative when the prefix
        makes the model worse, which an untrained one does on a real base.
    """
    comparison, _outcomes = score_held_out(model, held_out)
    return comparison["mean_baseline"] - comparison["mean_treatment"]


def train_cartridge(
    base: CacheCapableLMProto,
    corpus: Sequence[torch.Tensor],
    *,
    num_slots: int,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> CartridgeSlots:
    """Draw a cartridge and train it over a corpus.

    Args:
        base: The frozen base to train in front of.
        corpus: Training windows.
        num_slots: Prefix positions.
        seed: Seed for the draw.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.

    Returns:
        The trained slots, detached from any model, so a caller can compose
        them or place them in front of the base again.
    """
    model = fresh_cartridge(base, num_slots=num_slots, seed=seed)
    # SEEDS THE GLOBAL RNG, AND DOES IT HERE RATHER THAN A LINE EARLIER.
    #
    # Why it is needed at all: `train_on` puts the base in training mode, and
    # GPT-2 carries three dropouts at 0.1 (attention, residual, embedding)
    # that draw from torch's PROCESS-WIDE generator. `initialise_slots` seeds
    # only its own generator, so without this the seed names the cartridge's
    # starting point and nothing else. MEASURED 2026-09-03: two runs of one
    # plan, identical settings, determinism pinned, reported the eight-slot
    # arm's spread as 0.0049 and then 0.0268 -- both labelled "across seeds 7,
    # 8, 9", and neither actually that. The noise floor every claim here is
    # judged against was changing when nothing had changed.
    #
    # Why AFTER `fresh_cartridge`: that call measures the geometry, and
    # measuring it runs a real forward pass through the base
    # (`probe_cache_layers`). If the base is already in training mode -- which
    # it is for every arm after the first, because `train_on` left it there --
    # that probe applies dropout and CONSUMES global RNG. Seeding before it
    # therefore hands training a state that depends on how many arms ran
    # earlier, which is the same defect one level down and harder to see.
    # Seeding here makes an arm a function of its seed and nothing else.
    torch.manual_seed(seed)
    _losses = train_on(model, corpus, epochs=epochs, learning_rate=learning_rate)
    return model.slots


def measure_untrained(
    base: CacheCapableLMProto,
    held_out: Sequence[torch.Tensor],
    *,
    num_slots: int,
    seeds: Sequence[int],
) -> ReplicatedGain:
    """Measure what a cartridge is worth before it is trained.

    The control every other arm is read against, and the one whose answer
    differs most between a random model and a real one. It is separate from
    the trained arms rather than folded into them because it needs no
    training: a run that skipped it would save nothing and lose the only
    number that says whether a prefix helps because it was TRAINED or because
    it is there.

    Args:
        base: The frozen base.
        held_out: Items to score.
        num_slots: Prefix positions.
        seeds: Seeds to draw, one arm each.

    Returns:
        The replicated gain of an untrained prefix.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            two seeds are given.
    """
    return replicate(
        f"untrained-slots-{num_slots}",
        [
            (seed, _gain(fresh_cartridge(base, num_slots=num_slots, seed=seed), held_out))
            for seed in seeds
        ],
    )


def measure_slot_count(
    base: CacheCapableLMProto,
    train: Sequence[torch.Tensor],
    held_out: Sequence[torch.Tensor],
    *,
    num_slots: int,
    seeds: Sequence[int],
    epochs: int,
    learning_rate: float,
) -> ReplicatedGain:
    """Train and score one point of the capacity sweep.

    Args:
        base: The frozen base.
        train: Training windows.
        held_out: Items to score, which the cartridge never sees.
        num_slots: Prefix positions.
        seeds: Seeds to draw, one arm each.
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.

    Returns:
        The replicated held-out gain at this slot count.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            two seeds are given.
    """
    results: list[tuple[int, float]] = []
    for seed in seeds:
        slots = train_cartridge(
            base,
            train,
            num_slots=num_slots,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        results.append((seed, _gain(CartridgeModel(base=base, slots=slots), held_out)))
    return replicate(f"slots-{num_slots}", results)


def measure_composition(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[torch.Tensor],
    second_train: Sequence[torch.Tensor],
    held_out: Sequence[torch.Tensor],
    arm: str,
    num_slots: int,
    seeds: Sequence[int],
    epochs: int,
    learning_rate: float,
) -> tuple[ReplicatedGain, ReplicatedGain]:
    """Measure one cartridge alone, and the same one with another in front.

    Both arms are scored on the SAME held-out items, and the second cartridge
    is trained on ``second_train`` -- which the caller should draw from a
    genuinely different corpus. Composing two cartridges trained on two halves
    of one wiki measured 94% retention here, and the number was an artifact:
    each half already predicted the other, so the second cartridge was close
    to a copy of the first. Against an unrelated corpus the same measurement
    gives 59%.

    Args:
        base: The frozen base.
        first_train: Training windows for the cartridge being retained.
        second_train: Training windows for the cartridge composed in front of
            it.
        held_out: Items to score both arms on, drawn from the first corpus.
        arm: Name for this pairing, e.g. ``"me-with-civic"``.
        num_slots: Prefix positions for EACH cartridge; the composed prefix is
            twice this long.
        seeds: Seeds to draw, one arm each.
        epochs: Passes over each corpus.
        learning_rate: Step size for AdamW.

    Returns:
        ``(alone, composed)``, both replicated across the same seeds.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            two seeds are given.
    """
    alone: list[tuple[int, float]] = []
    composed: list[tuple[int, float]] = []
    for seed in seeds:
        first = train_cartridge(
            base,
            first_train,
            num_slots=num_slots,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        # A different seed for the second cartridge, so the pair are not the
        # same draw trained twice -- which would make their concatenation two
        # copies of one starting point rather than two cartridges.
        second = train_cartridge(
            base,
            second_train,
            num_slots=num_slots,
            seed=seed + len(seeds),
            epochs=epochs,
            learning_rate=learning_rate,
        )
        alone.append((seed, _gain(CartridgeModel(base=base, slots=first), held_out)))
        composed.append(
            (seed, _gain(CartridgeModel(base=base, slots=compose(first, second)), held_out))
        )
    return replicate(f"{arm}-alone", alone), replicate(f"{arm}-composed", composed)


__all__ = [
    "fresh_cartridge",
    "measure_composition",
    "measure_slot_count",
    "measure_untrained",
    "train_cartridge",
]
