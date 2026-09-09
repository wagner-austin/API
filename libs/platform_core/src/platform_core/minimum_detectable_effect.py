"""What your instrument could have seen, stated beside what it did see.

WHY THIS EXISTS. The GPT-2 corpus-extraction ablation published, for four
days, that hub-slug markers were indistinguishable from noise at 124M, 355M
and 774M -- three rungs, three seeds each, paired by seed. The page argued the
null was REAL rather than underpowered. The 1.5B rung landed 2026-09-08 at
-0.5588 pp, t = -11.13, all three seeds negative.

The sign flip is not the lesson. This is: the sd of the paired difference fell
1.1235 -> 0.7165 -> 0.2278 -> 0.0870 down the ladder, so the minimum
detectable effect fell 2.79 -> 1.78 -> 0.57 -> 0.22 pp. A CONSTANT -0.56 pp
penalty present at every rung would have been invisible at the first three and
significant only at the fourth -- which is exactly what was observed. The
earlier point estimates (-0.79, +0.07, -0.23) all sit within about one sd of
-0.56. The data cannot distinguish "the effect emerged at scale" from "the
effect was always there and only the last rung could resolve it".

**A null reported without its minimum detectable effect is not a result. It is
a sentence about your instrument that reads like a sentence about the world.**
The specific trap: when the error bar is IMPROVING across your conditions, a
sequence of point estimates looks like a shrinking effect when it is actually
a shrinking spread. Reading a trend in noisy estimates whose noise is
collapsing is reading the noise.

WHY THREE INSTRUMENTS AND NOT ONE FORMULA. The obvious helper computes
``t_crit(df) * sd / sqrt(n)`` and stops. That serves exactly one of the four
audits this module was written for; the other three were measured to need
something else:

  paired continuous   the ladder above. sd and n both enter.

  paired binary       code-style's guard-pass strata. McNemar conditions on
                      the DISCORDANT pairs; n and sd do not enter at all. Its
                      power analogue is which splits can reject: at d=5 the
                      smallest attainable two-sided exact p is 0.0625, so NO
                      ATTAINABLE RESULT rejects at alpha=0.05. Two of that
                      project's three published strata sat at d=5 --
                      unfalsifiable by the very test they printed.

  zero-failure        mi-cu128's identity checks: "n of n records
  proportion          bit-identical, 0 diverged". The sample variance is
                      exactly zero, so there is no sd for any formula. What
                      bounds the claim is Clopper-Pearson: with 0 failures in
                      n trials the true rate could still be as high as
                      ``1 - (1-c)**(1/n)``.

A helper answering only the first would have sent three projects back to
hand-rolling, which is the drift it exists to prevent.

WHY NO ``_test_hooks.py``. The convention is ``testing.py`` for ``libs/`` and
``_test_hooks.py`` for services' injection seams. This module has no clock,
network, filesystem or randomness -- every function is a pure transformation
of its arguments. A hooks file here would be a seam with nothing behind it,
which is the placeholder code the standards forbid. Tests call the real
functions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from statistics import fmean, stdev

from typing_extensions import TypedDict

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.power_distributions import (
    McNemarTest,
    mcnemar_p,
    require_alpha,
    t_critical,
)

#: Fewest paired differences a continuous power statement may be built from.
#:
#: Two replicates give a spread that is a single ``|a - b|`` -- a range
#: estimate from one draw. :mod:`model_trainer.core.contracts.
#: replicated_measurement` measured that swinging by 70% with the replicate
#: count and raised its own floor from two to three; this module keeps the two
#: consistent rather than inventing a second number.
MIN_REPLICATES: int = 3


class PowerInstrument(StrEnum):
    """Which power calculation produced a record.

    Published beside every number so a reader can tell which arithmetic was
    applied without inferring it from the field names.
    """

    PAIRED_CONTINUOUS = "paired_continuous"
    MCNEMAR = "mcnemar"
    ZERO_FAILURE_PROPORTION = "zero_failure_proportion"


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


def _require_effect_of_interest(value: float, field: str) -> None:
    """Reject a non-positive smallest-effect-of-interest.

    Args:
        value: Candidate effect size.
        field: Field name, for the message.

    Raises:
        AppError: ``POWER_EFFECT_OF_INTEREST_INVALID`` when not positive.
    """
    if not value > 0.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_EFFECT_OF_INTEREST_INVALID,
            f"{field} must be positive to make a verdict meaningful; got {value!r}",
        )


def paired_continuous_power(
    differences: Sequence[float],
    alpha: float,
    smallest_effect_of_interest: float,
) -> PairedContinuousPower:
    """Compute the MDE of a paired continuous comparison.

    ``MDE = t_crit(n - 1, alpha) * sd / sqrt(n)`` on the ACTUAL spread and the
    ACTUAL replicate count, never a nominal design value.

    Args:
        differences: Per-replicate paired differences, in outcome units.
        alpha: Two-sided significance level in ``(0, 1)``.
        smallest_effect_of_interest: The effect worth acting on.

    Returns:
        A populated :class:`PairedContinuousPower`.

    Raises:
        AppError: ``POWER_TOO_FEW_REPLICATES`` below :data:`MIN_REPLICATES`;
            ``POWER_ALPHA_OUT_OF_RANGE`` or
            ``POWER_EFFECT_OF_INTEREST_INVALID`` on bad parameters.
    """
    replicates = len(differences)
    if replicates < MIN_REPLICATES:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TOO_FEW_REPLICATES,
            f"a power statement needs at least {MIN_REPLICATES} replicates; got {replicates}",
        )
    require_alpha(alpha)
    _require_effect_of_interest(smallest_effect_of_interest, "smallest_effect_of_interest")
    degrees_of_freedom = replicates - 1
    sample_sd = stdev(differences)
    critical = t_critical(degrees_of_freedom, alpha)
    mde = critical * sample_sd / math.sqrt(replicates)
    verdict = PowerVerdict.TESTED if mde <= smallest_effect_of_interest else PowerVerdict.NOT_TESTED
    return PairedContinuousPower(
        instrument=PowerInstrument.PAIRED_CONTINUOUS.value,
        replicates=replicates,
        degrees_of_freedom=degrees_of_freedom,
        alpha=alpha,
        mean_difference=fmean(differences),
        sample_sd=sample_sd,
        t_critical=critical,
        minimum_detectable_effect=mde,
        smallest_effect_of_interest=smallest_effect_of_interest,
        verdict=verdict.value,
    )


def mcnemar_power(
    discordant_pairs: int,
    alpha: float,
    test: McNemarTest,
) -> McNemarPower:
    """Compute what a paired BINARY comparison could ever resolve.

    McNemar conditions on the discordant pairs alone, so neither sample size
    nor standard deviation enters. "What is this test's MDE" becomes "which
    splits of ``discordant_pairs`` can reject at all", and when none can, the
    comparison is unfalsifiable however the data fall.

    Args:
        discordant_pairs: Number of pairs that disagreed, non-negative.
        alpha: Two-sided significance level in ``(0, 1)``.
        test: Which McNemar variant the report uses. Required, not defaulted:
            the exact and mid-p rejection regions differ, and an MDE against
            the wrong one is an MDE for a test nobody ran.

    Returns:
        A populated :class:`McNemarPower`.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when negative;
            ``POWER_ALPHA_OUT_OF_RANGE`` when alpha is out of range.
    """
    if discordant_pairs < 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"discordant pairs cannot be negative; got {discordant_pairs!r}",
        )
    require_alpha(alpha)
    smallest_attainable_p = mcnemar_p(0, discordant_pairs, test)
    can_ever_reject = smallest_attainable_p <= alpha
    # ASCENDING from the most extreme split, keeping the LAST that rejects. A
    # scan that returns the FIRST rejecting split returns the trivial extreme
    # -- a number that type-checks, sorts correctly and is wrong. That bug was
    # written on this machine on 2026-09-08 and caught before publication.
    most_balanced = -1
    for minority in range(discordant_pairs // 2 + 1):
        if mcnemar_p(minority, discordant_pairs, test) <= alpha:
            most_balanced = minority
    return McNemarPower(
        instrument=PowerInstrument.MCNEMAR.value,
        test=test.value,
        discordant_pairs=discordant_pairs,
        alpha=alpha,
        smallest_attainable_p=smallest_attainable_p,
        can_ever_reject=can_ever_reject,
        most_balanced_rejecting_minority=most_balanced,
    )


def zero_failure_power(
    trials: int,
    confidence: float,
    largest_rate_of_interest: float,
) -> ZeroFailurePower:
    """Bound a failure rate that was observed to be zero.

    With zero failures the sample variance is exactly zero, so every
    spread-based instrument reports perfect precision and says nothing. The
    Clopper-Pearson one-sided bound is what actually constrains the claim:
    ``1 - (1 - confidence) ** (1 / trials)``.

    Args:
        trials: Independent trials, all successful. Must be positive.
        confidence: One-sided confidence level in ``(0, 1)``.
        largest_rate_of_interest: The failure rate worth acting on.

    Returns:
        A populated :class:`ZeroFailurePower`.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when trials is not positive;
            ``POWER_CONFIDENCE_OUT_OF_RANGE`` or
            ``POWER_EFFECT_OF_INTEREST_INVALID`` on bad parameters.
    """
    if trials < 1:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"trials must be positive; got {trials!r}",
        )
    if not 0.0 < confidence < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_CONFIDENCE_OUT_OF_RANGE,
            f"confidence must lie in (0, 1); got {confidence!r}",
        )
    _require_effect_of_interest(largest_rate_of_interest, "largest_rate_of_interest")
    # ``math.pow`` rather than ``**``: the operator is typed ``Any``, because
    # a negative base with a fractional exponent is complex in general. Both
    # operands are known-positive floats here, and ``math.pow`` is typed
    # ``(float, float) -> float``, so this keeps the module free of ``Any``
    # without a cast or an ignore.
    upper_bound = 1.0 - math.pow(1.0 - confidence, 1.0 / float(trials))
    verdict = (
        PowerVerdict.TESTED if upper_bound <= largest_rate_of_interest else PowerVerdict.NOT_TESTED
    )
    return ZeroFailurePower(
        instrument=PowerInstrument.ZERO_FAILURE_PROPORTION.value,
        trials=trials,
        confidence=confidence,
        upper_bound=upper_bound,
        largest_rate_of_interest=largest_rate_of_interest,
        verdict=verdict.value,
    )


__all__ = [
    "MIN_REPLICATES",
    "McNemarPower",
    "PairedContinuousPower",
    "PowerInstrument",
    "PowerVerdict",
    "ZeroFailurePower",
    "mcnemar_power",
    "paired_continuous_power",
    "zero_failure_power",
]
