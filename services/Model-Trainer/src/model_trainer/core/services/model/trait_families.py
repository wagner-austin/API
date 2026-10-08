"""The corpus grid's training recipes, bound to trait pairs.

WHY A MODULE OF BINDINGS AND NOT A MODULE OF RECIPES. The question on board
task ``83c25b86`` is whether the levers that repaired composition between
CORPUS compartments -- a content-diverse companion pool, a base-side LoRA
trained behind a crowd, crowd-invariance distillation -- transfer to
compartments that carry a DISPOSITION. That question is only asked if the
levers are the recorded ones. So nothing here trains anything new: each
builder hands the recorded geometry
(:func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`,
:func:`~model_trainer.core.services.model.cartridge_varied.companioned_replicates`)
the expressing members of a trait's training pairs, and
:func:`~model_trainer.core.services.model.trait_arms.measure_trait_composition`
scores what comes back with the trait instrument instead of held-out loss.

THE COMPANION POOL IS THE CORPUS GRID'S POOL, NOT A POOL OF TRAITS. The
recorded diverse pool is three cartridges on three held-out corpora, and the
trait roster cannot supply three more: four of the six admissible traits are
the roster, and the other two are one short of the pool size. More to the
point, a pool of the recorded corpora is the recipe of record applied
unchanged, which is what "does the lever transfer" means.
"""

from __future__ import annotations

from collections.abc import Sequence

from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.cartridge_measurement import (
    ReplicateBuilderProto,
    ReplicateConsumerProto,
    composed_replicates,
)
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.cartridge_varied import (
    CompanionPoolProviderProto,
    companioned_replicates,
)
from model_trainer.core.services.model.trait_arms import (
    TraitArm,
    TraitGains,
    trait_arm,
    trait_gains,
)
from model_trainer.core.services.model.trait_corpus import training_items
from model_trainer.core.types import CacheCapableLMProto


def plain_trait_build(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[TraitPair],
    other_trains: Sequence[Sequence[TraitPair]],
    num_slots: int,
    seeds: Sequence[int],
    seed_stride: int,
    epochs: int,
    learning_rate: float,
) -> ReplicateBuilderProto:
    """Bind the naive recipe to one trait cell's training pairs.

    Args:
        base: The frozen base every cartridge trains in front of -- the plain
            one for the naive grid, the adapted one for a LoRA family.
        first_train: Training pairs of the trait whose expression is the
            finding.
        other_trains: One training-pair sequence per additional trait, in
            roster order. Composing N compartments takes ``N - 1`` entries.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each -- the whole plan's, or one
            block of them.
        seed_stride: The whole plan's seed count, which spaces a replicate's
            partner draws the same way whichever block it is measured in.
        epochs: Passes over each trait's training pairs.
        learning_rate: Step size for AdamW.

    Returns:
        The builder.
    """
    first_items = training_items(first_train)
    other_items = [training_items(other) for other in other_trains]

    def _build(consume: ReplicateConsumerProto, /) -> None:
        """Run the naive recipe, handing each replicate to the scorer.

        Args:
            consume: Scores one replicate.
        """
        composed_replicates(
            base,
            first_train=first_items,
            other_trains=other_items,
            num_slots=num_slots,
            seeds=seeds,
            seed_stride=seed_stride,
            epochs=epochs,
            learning_rate=learning_rate,
            consume=consume,
        )

    return _build


def companioned_trait_build(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[TraitPair],
    other_trains: Sequence[Sequence[TraitPair]],
    num_slots: int,
    seeds: Sequence[int],
    seed_stride: int,
    epochs: int,
    learning_rate: float,
    pool_for_seed: CompanionPoolProviderProto,
    companion_probability: float,
) -> ReplicateBuilderProto:
    """Bind the diverse-companion recipe to one trait cell's training pairs.

    Every cartridge in the cell -- the primary trait's and every partner's --
    trains beside the replicate's frozen pool, because that is the deployment
    shape the recorded diverse grid measured: a library where every
    compartment was built companioned.

    Args:
        base: The frozen base every cartridge trains in front of.
        first_train: Training pairs of the trait whose expression is the
            finding.
        other_trains: One training-pair sequence per additional trait.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each -- the whole plan's, or one
            block of them.
        seed_stride: The whole plan's seed count, which spaces a replicate's
            partner draws the same way whichever block it is measured in.
        epochs: Passes over each trait's training pairs.
        learning_rate: Step size for AdamW.
        pool_for_seed: Builds the replicate's frozen companion pool.
        companion_probability: Chance per training forward that companions
            are present, in (0, 1].

    Returns:
        The builder.
    """
    first_items = training_items(first_train)
    other_items = [training_items(other) for other in other_trains]

    def _build(consume: ReplicateConsumerProto, /) -> None:
        """Run the companioned recipe, handing each replicate to the scorer.

        Args:
            consume: Scores one replicate.
        """
        companioned_replicates(
            base,
            first_train=first_items,
            other_trains=other_items,
            num_slots=num_slots,
            seeds=seeds,
            seed_stride=seed_stride,
            epochs=epochs,
            learning_rate=learning_rate,
            pool_for_seed=pool_for_seed,
            companion_probability=companion_probability,
            consume=consume,
        )

    return _build


def measure_trait_companion_cross(
    base: CacheCapableLMProto,
    *,
    pool_for_seed: CompanionPoolProviderProto,
    pool_size: int,
    seeds: Sequence[int],
    held_out: Sequence[TraitPair],
    arm: str,
) -> tuple[TraitArm, ...]:
    """Score every pool member ALONE on the primary trait's held-out pairs.

    LEAKAGE MEASURED, NOT ASSUMED. A companion whose corpus happens to lean
    toward the trait would inflate every companioned arm by what it carries,
    and the record could not tell that from the recipe working. The corpus
    grid added this arm for the same reason; here it asks whether a
    cartridge on an unrelated corpus shifts the model's PREFERENCE toward the
    trait, which is a different leak from predicting the primary corpus.

    Args:
        base: The base the pool was trained in front of, and the control
            every member is differenced against.
        pool_for_seed: The replicate's frozen pool, the one the companioned
            cells train beside.
        pool_size: How many members a pool has.
        seeds: The sweep's measurement seeds, in order.
        held_out: The primary trait's held-out pairs.
        arm: Name prefix; member ``j`` is ``f"{arm}-{j}"``.

    Returns:
        One arm per pool member, in pool order.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            the minimum seeds are given.
    """
    gains: list[list[tuple[int, TraitGains]]] = [[] for _ in range(pool_size)]
    for seed in seeds:
        for member, slots in enumerate(pool_for_seed(seed)):
            gains[member].append(
                (seed, trait_gains(CartridgeModel(base=base, slots=slots), held_out))
            )
    return tuple(trait_arm(f"{arm}-{member}", results) for member, results in enumerate(gains))


__all__ = [
    "companioned_trait_build",
    "measure_trait_companion_cross",
    "plain_trait_build",
]
