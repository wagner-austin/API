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
from statistics import fmean, stdev

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.power_distributions import (
    McNemarTest,
    mcnemar_p,
    regularized_incomplete_beta,
    require_alpha,
    t_critical,
)
from platform_core.power_types import (
    McNemarPower,
    PairedContinuousPower,
    PowerInstrument,
    PowerVerdict,
    RateFloorPower,
    RequiredReplicates,
    ZeroFailurePower,
)

#: Fewest paired differences a continuous power statement may be built from.
#:
#: Two replicates give a spread that is a single ``|a - b|`` -- a range
#: estimate from one draw. :mod:`model_trainer.core.contracts.
#: replicated_measurement` measured that swinging by 70% with the replicate
#: count and raised its own floor from two to three; this module keeps the two
#: consistent rather than inventing a second number.
MIN_REPLICATES: int = 3

#: Largest replicate count :func:`required_replicates` will search to.
#:
#: Not a statistical limit but a design one. The minimum detectable effect
#: falls as ``1/sqrt(n)``, so halving it costs four times the runs; a design
#: needing more than this many replicates to resolve the effect its own
#: authors called interesting is not short of replicates, it is measuring the
#: wrong thing or measuring it too noisily. Returning the ceiling instead of
#: raising would hand that design a number it could schedule against.
MAX_SEARCH_REPLICATES: int = 10_000


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


def required_replicates(
    differences: Sequence[float],
    alpha: float,
    smallest_effect_of_interest: float,
) -> RequiredReplicates:
    """Find the fewest paired replicates that would resolve the effect of interest.

    Answers the question a ``NOT_TESTED`` verdict raises and does not settle:
    how many replicates would have been enough. Searches upward from
    :data:`MIN_REPLICATES` for the first ``n`` whose minimum detectable effect
    ``t_crit(n - 1, alpha) * sd / sqrt(n)`` is at or below the effect of
    interest, holding the spread fixed at the observed sample sd.

    The first hit IS the smallest: ``t_crit`` falls with degrees of freedom and
    ``sqrt(n)`` rises, so the MDE is strictly decreasing in ``n`` and the
    search cannot step over a solution.

    The search starts at :data:`MIN_REPLICATES` rather than at the observed
    count, because the question is what the design needs, not what it happens
    to have run. A design already adequate therefore reports a required count
    at or below its observed one, and ``additional_replicates`` of zero.

    Args:
        differences: Per-replicate paired differences from the pilot run, in
            outcome units. Supplies the spread the projection rests on.
        alpha: Two-sided significance level in ``(0, 1)``.
        smallest_effect_of_interest: The effect worth acting on.

    Returns:
        A populated :class:`RequiredReplicates`.

    Raises:
        AppError: ``POWER_TOO_FEW_REPLICATES`` below :data:`MIN_REPLICATES`;
            ``POWER_ALPHA_OUT_OF_RANGE`` or
            ``POWER_EFFECT_OF_INTEREST_INVALID`` on bad parameters;
            ``POWER_REQUIRED_REPLICATES_UNREACHABLE`` when no count at or below
            :data:`MAX_SEARCH_REPLICATES` suffices.
    """
    observed = len(differences)
    if observed < MIN_REPLICATES:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TOO_FEW_REPLICATES,
            f"a power statement needs at least {MIN_REPLICATES} replicates; got {observed}",
        )
    require_alpha(alpha)
    _require_effect_of_interest(smallest_effect_of_interest, "smallest_effect_of_interest")
    sample_sd = stdev(differences)

    def resolves(candidate: int) -> bool:
        """Report whether ``candidate`` replicates resolve the effect of interest.

        Args:
            candidate: A replicate count at or above :data:`MIN_REPLICATES`.

        Returns:
            True when the MDE at this count is at or below the effect of interest.
        """
        return t_critical(candidate - 1, alpha) * sample_sd / math.sqrt(candidate) <= (
            smallest_effect_of_interest
        )

    if not resolves(MAX_SEARCH_REPLICATES):
        raise AppError(
            StatisticalPowerErrorCode.POWER_REQUIRED_REPLICATES_UNREACHABLE,
            f"no replicate count at or below {MAX_SEARCH_REPLICATES} brings the minimum "
            f"detectable effect down to {smallest_effect_of_interest!r} at a paired sd of "
            f"{sample_sd!r}; more replicates are the wrong remedy at this spread",
        )
    # Bisection, not a scan: the MDE is strictly decreasing in ``n``, so the
    # predicate is monotone and the boundary is unique. A scan would cost
    # MAX_SEARCH_REPLICATES evaluations of ``t_critical``, which is itself a
    # bisection -- measured at four minutes for one unreachable case.
    low, high = MIN_REPLICATES, MAX_SEARCH_REPLICATES
    while low < high:
        midpoint = (low + high) // 2
        if resolves(midpoint):
            high = midpoint
        else:
            low = midpoint + 1
    return RequiredReplicates(
        instrument=PowerInstrument.PAIRED_CONTINUOUS.value,
        observed_replicates=observed,
        observed_sample_sd=sample_sd,
        alpha=alpha,
        smallest_effect_of_interest=smallest_effect_of_interest,
        required_replicates=low,
        additional_replicates=max(0, low - observed),
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


def _require_rate_floor(floor: float) -> None:
    """Reject a pass-rate floor outside the open unit interval.

    Args:
        floor: Candidate floor.

    Raises:
        AppError: ``POWER_RATE_FLOOR_OUT_OF_RANGE`` when not in ``(0, 1)``.
    """
    if not 0.0 < floor < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_RATE_FLOOR_OUT_OF_RANGE,
            f"a pass-rate floor must lie in (0, 1) to be beatable; got {floor!r}",
        )


def rate_floor_power(
    successes: int,
    trials: int,
    floor: float,
    alpha: float,
) -> RateFloorPower:
    """Test an observed pass-rate against the floor its gate requires.

    Args:
        successes: Trials meeting the claim; ``0 <= successes <= trials``.
        trials: Trials attempted; must be positive.
        floor: The rate the claim must beat, in ``(0, 1)``.
        alpha: One-sided significance level in ``(0, 1)``.

    Returns:
        A populated :class:`RateFloorPower`.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when ``trials`` is not positive
            or ``successes`` is outside ``[0, trials]``;
            ``POWER_RATE_FLOOR_OUT_OF_RANGE`` or ``POWER_ALPHA_OUT_OF_RANGE``
            on bad parameters.
    """
    if trials <= 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"a rate needs at least one trial; got {trials!r}",
        )
    if not 0 <= successes <= trials:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"successes must lie in [0, {trials}]; got {successes!r}",
        )
    _require_rate_floor(floor)
    require_alpha(alpha)
    # P(X >= k | n, p) = I_p(k, n - k + 1). At k = 0 every outcome qualifies,
    # which the beta form does not cover, so it is stated rather than computed.
    p_value = (
        1.0
        if successes == 0
        else regularized_incomplete_beta(floor, successes, trials - successes + 1)
    )
    verdict = PowerVerdict.TESTED if p_value <= alpha else PowerVerdict.NOT_TESTED
    return RateFloorPower(
        instrument=PowerInstrument.RATE_FLOOR.value,
        successes=successes,
        trials=trials,
        observed_rate=successes / trials,
        floor=floor,
        alpha=alpha,
        p_value=p_value,
        perfect_record_trials=math.ceil(math.log(alpha) / math.log(floor)),
        verdict=verdict.value,
    )


__all__ = [
    "MAX_SEARCH_REPLICATES",
    "MIN_REPLICATES",
    "mcnemar_power",
    "paired_continuous_power",
    "rate_floor_power",
    "required_replicates",
    "zero_failure_power",
]
