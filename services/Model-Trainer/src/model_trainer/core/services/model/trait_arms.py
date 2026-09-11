"""The arms a trait-composition measurement runs, and what each is for.

THE GRID IS THE CORPUS GRID'S, CELL FOR CELL, with one substitution: the
dependent variable. Where that one asks whether held-out text from a corpus
became easier to predict, this asks whether the prefix carries a DISPOSITION.
Everything else is deliberately identical -- the seed offsets, the composition
order, the untrained-composed control, the cross arms -- and it is identical
because it is the SAME code:
:func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`
builds the cartridges and this module scores them. Two measurements that
differ in exactly one thing are comparable; two that were written separately
and happen to look alike are not.

EVERY ARM CARRIES TWO NUMBERS AND NEITHER IS OPTIONAL. Expression says how far
toward the trait the arm leans; coherence says what it did to ordinary text. A
cartridge can express a trait by degrading fluency, and the expression reading
is blind to that by construction -- it differences the two members of a pair,
so a prefix that wrecks both equally scores zero. A record carrying only the
first cannot be read at all, which is why they are produced together rather
than as two passes somebody might run one of.

THE STEERING ARM HAS NO SEEDS, AND THAT IS REPORTED RATHER THAN PAPERED OVER.
A contrastive activation vector is the mean difference over a fixed pair set:
nothing is drawn, so running it three times produces one number three times.
Wrapping that in a replicated gain would report a spread of exactly zero and
invite a reader to compare it against the cartridge arms' spreads as though
the two were estimates of the same thing. They are not -- one is seed noise
and the other is the absence of a seed -- so the steering arm is emitted as a
single reading and the record says so in its names.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.run_record import Observation
from typing_extensions import TypedDict

from model_trainer.core.contracts.paired_comparison import PairedComparison, summarise_pairs
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    per_seed_observations,
    replicate,
    retention,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.cartridge_measurement import (
    ComposedReplicate,
    composed_replicates,
    fresh_cartridge,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_scoring import (
    TraitPair,
    coherence_outcomes,
    expression_outcomes,
    read_trait_pairs,
)
from model_trainer.core.services.model.steering_vectors import (
    compose_directions,
    extract_steering_vector,
    steered_trait_losses,
    unit_direction,
)
from model_trainer.core.services.model.trait_corpus import training_items
from model_trainer.core.types import CacheCapableLMProto, SteerableLMProto


class TraitArm(TypedDict):
    """One arm's two readings, replicated across seeds.

    Attributes:
        expression: How much further toward the trait this arm leans than the
            plain base, per seed.
        coherence: What this arm did to the loss on trait-free text, per seed.
    """

    expression: ReplicatedGain
    coherence: ReplicatedGain


class TraitCompositionArms(TypedDict):
    """One compartment count's arms, with the controls that attribute them.

    Attributes:
        alone: The trait cartridge by itself -- the solo cost of composing.
        composed: The same cartridge with the other traits' in front of it.
        untrained_composed: The same cartridge composed with freshly drawn
            slots of identical shape, which separates the STRUCTURAL cost of a
            longer foreign prefix from the INTERFERENCE of what the others
            learned.
        cross: One arm per other trait, that trait's cartridge scored ALONE on
            the primary trait's pairs. The leakage detector: a positive cross
            arm means the traits are not independent, and the composed
            retention is inflated by whatever they share.
    """

    alone: TraitArm
    composed: TraitArm
    untrained_composed: TraitArm
    cross: tuple[TraitArm, ...]


def trait_gains(model: CartridgeModel, pairs: Sequence[TraitPair]) -> tuple[float, float]:
    """Score one model on one trait's pairs, as two gains.

    The one place a reading becomes a gain, so both numbers get the SIGN
    CONVENTION the whole arc uses: positive is better, meaning more
    trait-preferring for expression and lower loss on ordinary text for
    coherence. Two call sites computing ``baseline - treatment`` separately
    is exactly how a sign flips in one arm and nothing notices.

    Args:
        model: The cartridge-wrapped model. Its own base is the control.
        pairs: The held-out pairs of the trait being measured.

    Returns:
        ``(expression, coherence)``.
    """
    reading = read_trait_pairs(model, pairs)
    return (
        reading["expression"]["mean_baseline"] - reading["expression"]["mean_treatment"],
        reading["coherence"]["mean_baseline"] - reading["coherence"]["mean_treatment"],
    )


def _arm(name: str, results: Sequence[tuple[int, tuple[float, float]]]) -> TraitArm:
    """Reduce one arm's per-seed pairs of gains to two replicated gains.

    Args:
        name: The arm's name, without a reading suffix.
        results: ``(seed, (expression, coherence))`` in the order run.

    Returns:
        The arm.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if too few seeds
            were run, propagated from :func:`replicate`.
    """
    return TraitArm(
        expression=replicate(f"{name}-expression", [(seed, gains[0]) for seed, gains in results]),
        coherence=replicate(f"{name}-coherence", [(seed, gains[1]) for seed, gains in results]),
    )


def measure_trait_solo(
    base: CacheCapableLMProto,
    *,
    train: Sequence[TraitPair],
    held_out: Sequence[TraitPair],
    arm: str,
    num_slots: int,
    seeds: Sequence[int],
    epochs: int,
    learning_rate: float,
) -> tuple[TraitArm, TraitArm]:
    """Measure a single trait cartridge, and an untrained prefix beside it.

    THE PRECONDITION ARM, AND THE ARC STOPS HERE IF IT FAILS. At the 7B rung
    the corpus programme's solo gain nearly vanished -- +0.068 against ~0.81
    on every GPT-2 rung, with per-seed spans the same order as the means --
    and every retention ratio computed on those records became a division
    artefact. A composition question asked before its solo arm clears its own
    noise is unanswerable in exactly that way, so this runs first and its
    spread is what the composed cells are read against.

    THE UNTRAINED PREFIX IS NOT DECORATION. Without it, "attaching any prefix
    shifts preference" fits the numbers as well as "training put the trait in
    the prefix", and the corpus arc records this control INVERTING between
    model scales -- harmless on a tiny random-weight model, -0.7612 on a real
    base. It is measured per rung rather than assumed once.

    Args:
        base: The frozen base.
        train: The trait's training pairs; only their expressing members are
            trained on, which
            :func:`~model_trainer.core.services.model.trait_corpus.training_items`
            decides.
        held_out: The trait's held-out pairs, which both arms are scored on.
        arm: Name for this trait's solo cell, e.g. ``"bullets-solo"``.
        num_slots: Prefix positions.
        seeds: Seeds to draw, one replicate each.
        epochs: Passes over the training pairs.
        learning_rate: Step size for AdamW.

    Returns:
        ``(trained, untrained)``, both replicated across the same seeds.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            the minimum seeds are given.
    """
    items = training_items(train)
    trained: list[tuple[int, tuple[float, float]]] = []
    untrained: list[tuple[int, tuple[float, float]]] = []
    for seed in seeds:
        # THE UNTRAINED DRAW FIRST, at the same seed, so the two arms differ
        # in training and in nothing else. Scored before training runs,
        # because `train_cartridge` documents that a trained-into base leaves
        # the module in training mode and the geometry probe then consumes
        # process-wide randomness -- measuring the control afterwards would
        # draw it from a different state than the trained arm's own draw.
        untrained.append(
            (
                seed,
                trait_gains(fresh_cartridge(base, num_slots=num_slots, seed=seed), held_out),
            )
        )
        slots = train_cartridge(
            base,
            items,
            num_slots=num_slots,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        trained.append((seed, trait_gains(CartridgeModel(base=base, slots=slots), held_out)))
    return _arm(arm, trained), _arm(f"{arm}-untrained", untrained)


def measure_trait_composition(
    base: CacheCapableLMProto,
    *,
    first_train: Sequence[TraitPair],
    other_trains: Sequence[Sequence[TraitPair]],
    held_out: Sequence[TraitPair],
    arm: str,
    num_slots: int,
    seeds: Sequence[int],
    epochs: int,
    learning_rate: float,
) -> TraitCompositionArms:
    """Measure one trait with other traits' cartridges composed in front of it.

    THE COMPOSITION GEOMETRY IS NOT THIS MODULE'S, deliberately. The seed
    offsets, the fold order and the untrained control come from
    :func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`,
    the same function the corpus grid uses, so a trait cell and a corpus cell
    differ in what is trained and scored and in nothing structural. That is
    the property that makes the two records comparable, and it cannot be
    obtained by writing similar code twice.

    Args:
        base: The frozen base.
        first_train: Training pairs of the trait whose expression is the
            finding.
        other_trains: One training-pair sequence per additional trait, in
            roster order. Composing N compartments takes ``N - 1`` entries.
        held_out: Held-out pairs of the FIRST trait. Every arm here is scored
            on these, including the cross arms -- a cross arm asks what
            another trait's cartridge does to THIS trait's pairs.
        arm: Name for this cell, e.g. ``"bullets-n4"``.
        num_slots: Prefix positions for EACH cartridge.
        seeds: Seeds to draw, one replicate each.
        epochs: Passes over each trait's training pairs.
        learning_rate: Step size for AdamW.

    Returns:
        The cell's arms and its controls.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if fewer than
            the minimum seeds are given.
    """
    alone: list[tuple[int, tuple[float, float]]] = []
    composed: list[tuple[int, tuple[float, float]]] = []
    untrained_composed: list[tuple[int, tuple[float, float]]] = []
    cross: list[list[tuple[int, tuple[float, float]]]] = [[] for _ in other_trains]

    def _scored(slots: CartridgeSlots) -> tuple[float, float]:
        """Score one composed or solo slot block on the primary trait.

        Args:
            slots: The block to put in front of the base.

        Returns:
            ``(expression, coherence)``.
        """
        return trait_gains(CartridgeModel(base=base, slots=slots), held_out)

    def _score(built: ComposedReplicate, /) -> None:
        """Score one replicate's four arms on the primary trait's pairs.

        Args:
            built: The replicate just constructed.
        """
        seed = built["seed"]
        alone.append((seed, _scored(built["alone"])))
        composed.append((seed, _scored(built["composed"])))
        untrained_composed.append((seed, _scored(built["untrained_composed"])))
        for position, other in enumerate(built["others"]):
            cross[position].append((seed, _scored(other)))

    composed_replicates(
        base,
        first_train=training_items(first_train),
        other_trains=[training_items(other) for other in other_trains],
        num_slots=num_slots,
        seeds=seeds,
        epochs=epochs,
        learning_rate=learning_rate,
        consume=_score,
    )
    return TraitCompositionArms(
        alone=_arm(f"{arm}-alone", alone),
        composed=_arm(f"{arm}-composed", composed),
        untrained_composed=_arm(f"{arm}-untrained-composed", untrained_composed),
        cross=tuple(
            _arm(f"{arm}-cross-{position}", results) for position, results in enumerate(cross)
        ),
    )


class SteeringReading(TypedDict):
    """One steering configuration, measured once because it is deterministic.

    Attributes:
        arm: What was measured, e.g. ``"steer-n4"``.
        expression: The paired comparison for trait preference.
        coherence: The paired comparison for the loss on trait-free text.
    """

    arm: str
    expression: PairedComparison
    coherence: PairedComparison


def measure_trait_steering(
    base: SteerableLMProto,
    *,
    trait_trains: Sequence[Sequence[TraitPair]],
    held_out: Sequence[TraitPair],
    arm: str,
    module_name: str,
    strength: float,
) -> SteeringReading:
    """Measure the published intervention at one trait count.

    THE ARM THAT MAKES THE RESULT A COMPARISON. Without it a trait grid
    reports a number about cartridges; with it the run sits beside the only
    measured account of dispositional composition, at matched trait counts on
    the same base and the same pairs.

    DIRECTIONS ARE READ FROM THE TRAINING PAIRS, not the held-out ones, so the
    steering arm is held to the same wall the cartridge arms are: a direction
    read off the pairs it is scored on measures memorisation and the two arms
    would no longer be answering the same question.

    Args:
        base: The plain base, steerable. Never cartridge-wrapped: this is an
            alternative to a prefix, not an addition to one.
        trait_trains: Training pairs per trait, in roster order. The first is
            the trait whose expression is the finding; a one-entry sequence is
            the solo steering arm and more entries compose.
        held_out: Held-out pairs of the FIRST trait, which the arm is scored
            on.
        arm: Name for this configuration.
        module_name: Dotted path of the site to read and perturb.
        strength: Multiplier applied to the unit direction.

    Returns:
        The reading.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if a trait supplies no pairs
            or its direction is degenerate, or ``EDIT_MODULE_NOT_FOUND`` if
            the site does not exist.
    """
    directions = [
        unit_direction(extract_steering_vector(base, pairs, module_name=module_name))
        for pairs in trait_trains
    ]
    losses = steered_trait_losses(
        base,
        held_out,
        compose_directions(directions),
        module_name=module_name,
        strength=strength,
    )
    return SteeringReading(
        arm=arm,
        expression=summarise_pairs(expression_outcomes(losses)),
        coherence=summarise_pairs(coherence_outcomes(losses)),
    )


def trait_arm_observations(measured: TraitArm) -> tuple[Observation, ...]:
    """Name one arm's two readings for the record.

    BOTH READINGS, ALWAYS. Emitting them from one function is what makes it
    impossible for an arm to reach a record with an expression number and no
    coherence number beside it -- the failure that would let a trait gain
    bought by wrecked fluency read as a clean result.

    Args:
        measured: The arm.

    Returns:
        Each reading's mean, spread and per-seed gains.
    """
    named: list[Observation] = []
    for reading in (measured["expression"], measured["coherence"]):
        named.extend(gain_observations(reading))
        named.extend(per_seed_observations(reading))
    return tuple(named)


def trait_cell_observations(arm: str, cell: TraitCompositionArms) -> tuple[Observation, ...]:
    """Name one composed cell's arms, controls and retention.

    THE RETENTION EXISTS ONLY WHERE THE ALONE ARM IMPROVED ON THE BASE, the
    same condition
    :func:`~model_trainer.core.contracts.replicated_measurement.retention`
    itself refuses on. A trait cartridge that failed to express its own trait
    is a RESULT this grid exists to find -- the 7B precondition failure is
    exactly that shape -- and an absent retention reads as "alone did not
    improve on base", checkable from the alone mean's sign in the same record.
    A ratio against a non-gain would report a number whose sign and size both
    mean nothing.

    RETENTION IS EXPRESSION-ONLY, AND DELIBERATELY. A ratio of coherence gains
    would divide one fluency change by another, which is not a retained
    fraction of anything: coherence is a cost, and costs are read as
    differences against the controls in the same record.

    Args:
        arm: The cell's name.
        cell: The cell's arms.

    Returns:
        Every arm's rows, and the expression retention where it is readable.
    """
    named: list[Observation] = []
    for measured in (cell["alone"], cell["composed"], cell["untrained_composed"], *cell["cross"]):
        named.extend(trait_arm_observations(measured))
    if cell["alone"]["expression"]["mean"] > 0.0:
        named.append(
            Observation(
                name=f"{arm}_expression_retention",
                value=retention(cell["alone"]["expression"], cell["composed"]["expression"]),
            )
        )
    return tuple(named)


def steering_observations(reading: SteeringReading) -> tuple[Observation, ...]:
    """Name a steering configuration's numbers for the record.

    NAMED ``_once`` RATHER THAN ``_mean``, and that is the whole point of the
    suffix. Every cartridge arm's row is a mean over seeds beside a spread;
    this one is a single measurement with no spread to report, because nothing
    was drawn. A row called ``_mean`` sitting next to rows that are means
    would invite a reader to compare its (absent) spread with theirs, and the
    name is the only place that distinction can survive into the record.

    Args:
        reading: The configuration.

    Returns:
        Both readings' shifts, item counts, directional splits and p-values.
    """
    named: list[Observation] = []
    for label, comparison in (
        ("expression", reading["expression"]),
        ("coherence", reading["coherence"]),
    ):
        prefix = f"{reading['arm']}-{label}"
        named.append(
            Observation(
                name=f"{prefix}_once",
                value=comparison["mean_baseline"] - comparison["mean_treatment"],
            )
        )
        named.append(Observation(name=f"{prefix}_items", value=float(comparison["items"])))
        named.append(Observation(name=f"{prefix}_improved", value=float(comparison["improved"])))
        named.append(Observation(name=f"{prefix}_worsened", value=float(comparison["worsened"])))
        named.append(Observation(name=f"{prefix}_p_value", value=comparison["p_value"]))
    return tuple(named)


__all__ = [
    "SteeringReading",
    "TraitArm",
    "TraitCompositionArms",
    "measure_trait_composition",
    "measure_trait_solo",
    "measure_trait_steering",
    "steering_observations",
    "trait_arm_observations",
    "trait_cell_observations",
    "trait_gains",
]
