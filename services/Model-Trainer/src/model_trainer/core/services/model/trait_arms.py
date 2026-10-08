"""The arms a trait-composition measurement runs, and what each is for.

THE GRID IS THE CORPUS GRID'S, CELL FOR CELL, with one substitution: the
dependent variable. Where that one asks whether held-out text from a corpus
became easier to predict, this asks whether the prefix carries a DISPOSITION.
Everything else is deliberately identical -- the seed offsets, the composition
order, the untrained-composed control, the cross arms -- and it is identical
because it is the SAME code: the corpus grid's replicate builders, bound in
:mod:`~model_trainer.core.services.model.trait_families`, build the cartridges
and this module scores them. Two measurements that differ in exactly one thing
are comparable; two that were written separately and happen to look alike are
not.

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
    ReplicateBuilderProto,
    fresh_cartridge,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_scoring import (
    TraitPair,
    coherence_outcomes,
    expression_outcomes,
    read_trait_pairs,
    style_outcomes,
)
from model_trainer.core.services.model.steering_vectors import (
    SteeringTuning,
    compose_directions,
    extract_steering_vector,
    steered_trait_losses,
    tune_steering_strength,
    unit_direction,
)
from model_trainer.core.services.model.trait_corpus import training_items
from model_trainer.core.types import CacheCapableLMProto, SteerableLMProto


class TraitGains(TypedDict):
    """One arm's three readings at one seed, as gains.

    Attributes:
        expression: How much further toward the trait the arm leans than its
            base does.
        coherence: How much easier the arm made the trait-free member.
        style: How much easier the arm made the trait-expressing member.
            ``expression == style - coherence`` exactly.
    """

    expression: float
    coherence: float
    style: float


class TraitArm(TypedDict):
    """One arm's three readings, replicated across seeds.

    Attributes:
        expression: How much further toward the trait this arm leans than the
            plain base, per seed.
        coherence: What this arm did to the loss on trait-free text, per seed.
        style: What this arm did to the loss on the trait-expressing text,
            per seed -- the corpus arc's own variable, carried so a persona
            result that is really a style-corpus result shows as one.
    """

    expression: ReplicatedGain
    coherence: ReplicatedGain
    style: ReplicatedGain


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


def trait_gains(model: CartridgeModel, pairs: Sequence[TraitPair]) -> TraitGains:
    """Score one model on one trait's pairs, as three gains.

    The one place a reading becomes a gain, so every number gets the SIGN
    CONVENTION the whole arc uses: positive is better, meaning more
    trait-preferring for expression and lower loss for coherence and style.
    Two call sites computing ``baseline - treatment`` separately is exactly
    how a sign flips in one arm and nothing notices.

    Args:
        model: The cartridge-wrapped model. Its own base is the control.
        pairs: The held-out pairs of the trait being measured.

    Returns:
        The arm's three gains.
    """
    reading = read_trait_pairs(model, pairs)
    return TraitGains(
        expression=reading["expression"]["mean_baseline"] - reading["expression"]["mean_treatment"],
        coherence=reading["coherence"]["mean_baseline"] - reading["coherence"]["mean_treatment"],
        style=reading["style"]["mean_baseline"] - reading["style"]["mean_treatment"],
    )


def trait_arm(name: str, results: Sequence[tuple[int, TraitGains]]) -> TraitArm:
    """Reduce one arm's per-seed gains to three replicated gains.

    Public because every module that scores a trait arm needs the same
    reduction, and two copies of it would be two places the reading suffixes
    are spelled.

    Args:
        name: The arm's name, without a reading suffix.
        results: ``(seed, gains)`` in the order run.

    Returns:
        The arm.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if too few seeds
            were run, propagated from :func:`replicate`.
    """
    return TraitArm(
        expression=replicate(
            f"{name}-expression", [(seed, gains["expression"]) for seed, gains in results]
        ),
        coherence=replicate(
            f"{name}-coherence", [(seed, gains["coherence"]) for seed, gains in results]
        ),
        style=replicate(f"{name}-style", [(seed, gains["style"]) for seed, gains in results]),
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
    trained: list[tuple[int, TraitGains]] = []
    untrained: list[tuple[int, TraitGains]] = []
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
    return trait_arm(arm, trained), trait_arm(f"{arm}-untrained", untrained)


def solo_precondition_cleared(expression: ReplicatedGain) -> bool:
    """Decide whether the solo arm cleared its own noise.

    THE BAR IS THE ARM'S OWN SPREAD, and it is deliberately the weakest
    defensible one. A solo gain smaller than the range its own seeds produced
    cannot be told from which cartridge happened to be drawn; clearing it is
    not evidence the effect is large, only that there is an effect for every
    composed retention to divide by. The 7B rung of the corpus programme
    failed exactly here -- +0.068 against per-seed spans of the same order --
    and every retention computed beside it was a division artefact.

    A VERDICT, NOT A REFUSAL. A failed precondition is the result the task
    names first (a null about the SUBSTRATE), so it has to reach a record the
    way every other result does; a raise would leave it in a job log.

    TAKES THE EXPRESSION READING RATHER THAN THE ARM because the caller
    rebuilds it from checkpointed rows, and a resumed run never held the arm.

    Args:
        expression: The trained solo arm's expression reading.

    Returns:
        True when the mean expression gain exceeds its own per-seed spread.
    """
    return expression["mean"] > expression["spread"]


def measure_trait_composition(
    base: CacheCapableLMProto,
    *,
    build: ReplicateBuilderProto,
    partners: int,
    held_out: Sequence[TraitPair],
    arm: str,
) -> TraitCompositionArms:
    """Measure one trait with other traits' cartridges composed in front of it.

    THE COMPOSITION GEOMETRY IS NOT THIS MODULE'S, deliberately. The seed
    offsets, the fold order and the untrained control come from the builder
    -- :func:`~model_trainer.core.services.model.cartridge_measurement.composed_replicates`
    for the naive grid, its companioned sibling for the repair families --
    the same functions the corpus grid uses, so a trait cell and a corpus
    cell differ in what is trained and scored and in nothing structural. That
    is the property that makes the two records comparable, and it cannot be
    obtained by writing similar code twice.

    Args:
        base: The frozen base every replicate was trained in front of. Its own
            preference is the control each arm is differenced against.
        build: The training recipe, bound to the trait's training pairs and
            schedule, that emits one replicate per seed.
        partners: How many other traits the builder composes, which is how
            many cross arms the cell carries.
        held_out: Held-out pairs of the FIRST trait. Every arm here is scored
            on these, including the cross arms -- a cross arm asks what
            another trait's cartridge does to THIS trait's pairs.
        arm: Name for this cell, e.g. ``"bullets-n4"``.

    Returns:
        The cell's arms and its controls.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if the builder
            emitted fewer than the minimum seeds.
    """
    alone: list[tuple[int, TraitGains]] = []
    composed: list[tuple[int, TraitGains]] = []
    untrained_composed: list[tuple[int, TraitGains]] = []
    cross: list[list[tuple[int, TraitGains]]] = [[] for _ in range(partners)]

    def _scored(slots: CartridgeSlots) -> TraitGains:
        """Score one composed or solo slot block on the primary trait.

        Args:
            slots: The block to put in front of the base.

        Returns:
            The block's three gains.
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

    build(_score)
    return TraitCompositionArms(
        alone=trait_arm(f"{arm}-alone", alone),
        composed=trait_arm(f"{arm}-composed", composed),
        untrained_composed=trait_arm(f"{arm}-untrained-composed", untrained_composed),
        cross=tuple(
            trait_arm(f"{arm}-cross-{position}", results) for position, results in enumerate(cross)
        ),
    )


class SteeringReading(TypedDict):
    """One steering configuration, measured once because it is deterministic.

    Attributes:
        arm: What was measured, e.g. ``"steer-n4"``.
        expression: The paired comparison for trait preference.
        coherence: The paired comparison for the loss on trait-free text.
        style: The paired comparison for the loss on trait-expressing text.
        tuning: The strength the arm was applied at, and the training-pair
            trials it was chosen from.
    """

    arm: str
    expression: PairedComparison
    coherence: PairedComparison
    style: PairedComparison
    tuning: SteeringTuning


def measure_trait_steering(
    base: SteerableLMProto,
    *,
    trait_trains: Sequence[Sequence[TraitPair]],
    held_out: Sequence[TraitPair],
    arm: str,
    module_name: str,
    strengths: Sequence[float],
    coherence_bar: float,
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

    THE STRENGTH IS TUNED ON THE SOLO DIRECTION AND CARRIED TO EVERY COUNT,
    which is how the published measurement composes: each vector's
    coefficient is tuned alone, and the cost of composing is read at those
    coefficients. Re-tuning the composed direction would tune the
    composition's cost away and measure the tuner.

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
        strengths: The plan's candidate strengths for the unit direction.
        coherence_bar: Largest coherence cost, in nats, the tuned strength
            may incur on the first trait's training pairs.

    Returns:
        The reading, with the tuning that fixed its strength.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` if a trait supplies no pairs
            or its direction is degenerate, ``EDIT_MODULE_NOT_FOUND`` if the
            site does not exist, or ``TRAIT_STEERING_STRENGTH_UNREACHABLE``
            if no strength stays within the bar.
    """
    directions = [
        unit_direction(extract_steering_vector(base, pairs, module_name=module_name))
        for pairs in trait_trains
    ]
    tuning = tune_steering_strength(
        base,
        trait_trains[0],
        directions[0],
        module_name=module_name,
        strengths=strengths,
        coherence_bar=coherence_bar,
    )
    losses = steered_trait_losses(
        base,
        held_out,
        compose_directions(directions),
        module_name=module_name,
        strength=tuning["strength"],
    )
    return SteeringReading(
        arm=arm,
        expression=summarise_pairs(expression_outcomes(losses)),
        coherence=summarise_pairs(coherence_outcomes(losses)),
        style=summarise_pairs(style_outcomes(losses)),
        tuning=tuning,
    )


def trait_arm_observations(measured: TraitArm) -> tuple[Observation, ...]:
    """Name one arm's three readings for the record.

    EVERY READING, ALWAYS. Emitting them from one function is what makes it
    impossible for an arm to reach a record with an expression number and no
    coherence number beside it -- the failure that would let a trait gain
    bought by wrecked fluency read as a clean result -- or without the style
    number that says whether the gain is just a style corpus learned.

    Args:
        measured: The arm.

    Returns:
        Each reading's mean, spread and per-seed gains.
    """
    named: list[Observation] = []
    for reading in (measured["expression"], measured["coherence"], measured["style"]):
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

    NO COHERENCE RETENTION, AND DELIBERATELY. A ratio of coherence gains
    would divide one fluency change by another, which is not a retained
    fraction of anything: coherence is a cost, and costs are read as
    differences against the controls in the same record. The STYLE retention
    is emitted beside the expression one because it is the corpus arc's own
    retention on a style corpus: where the two agree arm for arm, the persona
    result is a style-corpus result and has to be named as one.

    Args:
        arm: The cell's name.
        cell: The cell's arms.

    Returns:
        Every arm's rows, and each retention where its alone arm gained.
    """
    named: list[Observation] = []
    for measured in (cell["alone"], cell["composed"], cell["untrained_composed"], *cell["cross"]):
        named.extend(trait_arm_observations(measured))
    for reading in ("expression", "style"):
        if cell["alone"][reading]["mean"] > 0.0:
            named.append(
                Observation(
                    name=f"{arm}_{reading}_retention",
                    value=retention(cell["alone"][reading], cell["composed"][reading]),
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
        The tuned strength and every training trial it was chosen from, then
        every reading's shift, item count, directional split and p-value.
    """
    named: list[Observation] = [
        Observation(name=f"{reading['arm']}-strength", value=reading["tuning"]["strength"])
    ]
    for trial in reading["tuning"]["trials"]:
        prefix = f"{reading['arm']}-tune-s{trial['strength']:g}"
        named.append(Observation(name=f"{prefix}-expression_train", value=trial["expression"]))
        named.append(Observation(name=f"{prefix}-coherence_train", value=trial["coherence"]))
    for label, comparison in (
        ("expression", reading["expression"]),
        ("coherence", reading["coherence"]),
        ("style", reading["style"]),
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
    "TraitGains",
    "measure_trait_composition",
    "measure_trait_solo",
    "measure_trait_steering",
    "solo_precondition_cleared",
    "steering_observations",
    "trait_arm",
    "trait_arm_observations",
    "trait_cell_observations",
    "trait_gains",
]
