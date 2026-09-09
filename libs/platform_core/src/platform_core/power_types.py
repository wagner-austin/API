"""The record SHAPES a power statement is published as.

Split from :mod:`platform_core.minimum_detectable_effect` on 2026-09-09 when
that module passed the 600-line cap, along the layer boundary this package
already uses: :mod:`platform_core.power_distributions` holds the mathematics,
this module holds the shapes, the instruments module computes them, and
:mod:`platform_core.power_records` moves them across the JSON boundary.

The split is by LAYER rather than by instrument family deliberately. Splitting
by family would have put ``RequiredReplicates`` beside its own function and
left ``power_records`` importing from four modules to serialise five records;
one module of shapes keeps the JSON boundary pointed at one place, and keeps
the validators that guard construction private to the module that constructs.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import TypedDict


class PowerInstrument(StrEnum):
    """Which power calculation produced a record.

    Published beside every number so a reader can tell which arithmetic was
    applied without inferring it from the field names.
    """

    PAIRED_CONTINUOUS = "paired_continuous"
    MCNEMAR = "mcnemar"
    ZERO_FAILURE_PROPORTION = "zero_failure_proportion"
    RATE_FLOOR = "rate_floor"


class PowerVerdict(StrEnum):
    """Whether a null was actually tested, or merely reported.

    ``TESTED`` means the instrument could have resolved an effect as small as
    the one anyone would act on. ``NOT_TESTED`` means it could not, and the
    null therefore says nothing about the world.
    """

    TESTED = "TESTED"
    NOT_TESTED = "NOT_TESTED"


class PairedContinuousPower(TypedDict):
    """Power of a paired continuous comparison.

    Attributes:
        instrument: Always :attr:`PowerInstrument.PAIRED_CONTINUOUS`.
        replicates: Number of paired differences.
        degrees_of_freedom: ``replicates - 1``.
        alpha: Two-sided significance level the MDE is computed at.
        mean_difference: Mean of the paired differences, in outcome units.
        sample_sd: Sample standard deviation of the paired differences.
        t_critical: Two-sided critical t at these df and alpha.
        minimum_detectable_effect: Smallest true effect this instrument would
            call significant, in outcome units.
        smallest_effect_of_interest: The effect worth acting on, supplied by
            the caller.
        verdict: :class:`PowerVerdict` for this comparison.
    """

    instrument: str
    replicates: int
    degrees_of_freedom: int
    alpha: float
    mean_difference: float
    sample_sd: float
    t_critical: float
    minimum_detectable_effect: float
    smallest_effect_of_interest: float
    verdict: str


class McNemarPower(TypedDict):
    """Power of a paired BINARY comparison, conditioned on discordant pairs.

    THIS RECORD CARRIES NO :class:`PowerVerdict`, DELIBERATELY, and that is
    the one asymmetry in this module worth understanding before using it.

    The other two instruments take the effect anyone would act on and answer
    "could this have detected it?". McNemar conditions on the discordant
    pairs alone, so from ``discordant_pairs`` and ``alpha`` the only question
    answerable is "can any attainable split reject?" -- falsifiability, not
    practical detectability. Those are different questions, and giving them
    one vocabulary is how a reader ends up reporting the first as though it
    were the second.

    Concretely, on real code-style data: at d=6 under mid-p the comparison
    CAN reject (at a perfect 6:0), while the project's own classification
    against a +5 pp threshold was NOT TESTED, because the detectable
    difference sat above the base rate it applied to. A ``verdict: TESTED``
    here would have been true of the instrument and false about the world --
    the exact confusion this whole sweep exists to end, one level up.

    So the truth is carried by :attr:`can_ever_reject`, a boolean that cannot
    be mistaken for a classification. To classify a binary null against a
    stated threshold, convert the returned split into your outcome's units
    (the smallest detectable net difference is ``discordant_pairs - 2 *
    most_balanced_rejecting_minority`` over your total pairs) and compare
    that to the effect you care about.

    Attributes:
        instrument: Always :attr:`PowerInstrument.MCNEMAR`.
        test: Which :class:`McNemarTest` the report uses. Carried because the
            two variants have different rejection regions, so an MDE computed
            against the wrong one describes a test nobody ran.
        discordant_pairs: Number of pairs that disagreed.
        alpha: Two-sided significance level.
        smallest_attainable_p: p of the most extreme split. No result from
            this many discordant pairs can beat it.
        can_ever_reject: Whether ``smallest_attainable_p <= alpha``.
        most_balanced_rejecting_minority: Largest minority count whose split
            still rejects, or -1 when none does. This is the informative
            bound; a search returning the trivial extreme instead is the bug
            the tests pin.
    """

    instrument: str
    test: str
    discordant_pairs: int
    alpha: float
    smallest_attainable_p: float
    can_ever_reject: bool
    most_balanced_rejecting_minority: int


class ZeroFailurePower(TypedDict):
    """Bound on a rate that was observed to be zero.

    Attributes:
        instrument: Always :attr:`PowerInstrument.ZERO_FAILURE_PROPORTION`.
        trials: Number of independent trials, all of which succeeded.
        confidence: One-sided Clopper-Pearson confidence level.
        upper_bound: Largest true failure rate consistent with observing zero
            failures at this confidence, ``1 - (1 - c) ** (1/n)``.
        largest_rate_of_interest: The failure rate worth acting on.
        verdict: :class:`PowerVerdict`. ``TESTED`` only when the bound is at
            or below the rate anyone would care about.
    """

    instrument: str
    trials: int
    confidence: float
    upper_bound: float
    largest_rate_of_interest: float
    verdict: str


class RequiredReplicates(TypedDict):
    """How many paired replicates would make a comparison conclusive.

    The companion to :class:`PairedContinuousPower`: that record says an
    instrument could not resolve the effect of interest, this one says what it
    would take to fix that. A ``NOT_TESTED`` verdict published without this
    number tells a reader the experiment failed but not what to run instead.

    ``observed_sample_sd`` is carried because the answer is only valid while
    the spread holds. The required count is computed from the sd the pilot
    actually measured, so a rerun whose spread turns out larger will need more
    replicates than this record names -- it is a planning estimate from one
    sample, not a guarantee.

    Attributes:
        instrument: Always :attr:`PowerInstrument.PAIRED_CONTINUOUS`.
        observed_replicates: Replicate count the pilot measurement used.
        observed_sample_sd: Sample standard deviation of the observed paired
            differences, the spread the projection assumes.
        alpha: Two-sided significance level the projection is made at.
        smallest_effect_of_interest: The effect worth acting on.
        required_replicates: Fewest replicates whose minimum detectable effect
            is at or below ``smallest_effect_of_interest``.
        additional_replicates: ``required_replicates - observed_replicates``,
            floored at zero -- what still has to be run. Zero means the
            observed design was already adequate.
    """

    instrument: str
    observed_replicates: int
    observed_sample_sd: float
    alpha: float
    smallest_effect_of_interest: float
    required_replicates: int
    additional_replicates: int


class RateFloorPower(TypedDict):
    """Whether an observed pass-rate is distinguishable from the floor it must beat.

    The instrument for a gate that gives a claim a PASS when
    ``successes / trials`` reaches some floor. Such a gate tests a RATE and
    usually puts no condition on ``trials``, so a flawless six-of-six clears an
    0.85 floor while carrying almost no evidence that the true rate exceeds it:
    at the floor itself, a perfect six-of-six happens 37.7% of the time.

    Neither sibling instrument fits. :func:`zero_failure_power` handles only a
    spotless record and most of these have failures; :func:`paired_continuous_power`
    needs a spread and a rate has none. The statistic here is the one-sided exact
    binomial ``P(X >= successes | trials, floor)``, evaluated through the
    regularized incomplete beta rather than a sum of binomial coefficients --
    the sum is exact but infeasible at the tens of thousands of trials real
    archives produce.

    Attributes:
        instrument: Always :attr:`PowerInstrument.RATE_FLOOR`.
        successes: Trials meeting the claim.
        trials: Trials attempted; the denominator the gate divided by.
        observed_rate: ``successes / trials``.
        floor: The rate the claim must beat for its gate to mean anything.
        alpha: One-sided significance level.
        p_value: ``P(X >= successes | trials, floor)`` -- the chance a record
            this good arises when the true rate is exactly the floor.
        perfect_record_trials: Fewest trials at which a FLAWLESS record clears
            the floor, ``ceil(log(alpha) / log(floor))``. Below this, no
            outcome whatsoever can pass, so the gate cannot be failed OR
            passed on evidence.
        verdict: :class:`PowerVerdict`. ``TESTED`` only when ``p_value`` is at
            or below ``alpha``.
    """

    instrument: str
    successes: int
    trials: int
    observed_rate: float
    floor: float
    alpha: float
    p_value: float
    perfect_record_trials: int
    verdict: str


__all__ = [
    "McNemarPower",
    "PairedContinuousPower",
    "PowerInstrument",
    "PowerVerdict",
    "RateFloorPower",
    "RequiredReplicates",
    "ZeroFailurePower",
]
