"""How many pairs the NEXT paired BINARY run needs, before *d* is known.

THE UNCONDITIONAL SIBLING OF :mod:`platform_core.mcnemar_detectability`, and
the difference is which quantity is random. That module is handed the
discordant count a comparison ACTUALLY produced and asks what that comparison
could have caught. This one is asked before the run, when the discordant
count does not exist yet and is itself random -- *d* ~ Binomial(*n*, rate) --
so power has to be averaged across every *d* the design might produce.

THE TWO ARE NUMERICALLY DIFFERENT AND THE GAP IS NOT SMALL. Measured
2026-09-10 against ``code-style``'s published power table -- exact test, alpha
0.05, a 70:30 split, 5.6% discordant:

    n      published   conditional at round(n x rate)   averaged over d
    226      0.21                0.2026                     0.2061
    800      0.73                0.7462                     0.7335

The averaged column is this module and it reproduces the published figures;
the conditional column is the other module and it does not. That table was
computed the unconditional way, and until this module existed nothing in the
package could reproduce it -- ``mcnemar_detectability`` said so in its own
docstring rather than implying coverage it did not have. This is that gap
closed.

POWER IS NOT MONOTONE IN THE SAMPLE SIZE, WHICH IS THE FINDING THAT SHAPED
THIS MODULE. The obvious implementation bisects for the smallest *n* reaching
the target, exactly as
:func:`~platform_core.minimum_detectable_effect.required_replicates`
legitimately does for the continuous case. That is INVALID here and it was
measured before it was relied on: 216 configurations swept over *n* = 1..600
produced 14,214 sample sizes at which adding one more pair LOWERS the averaged
power, the largest single drop being 7.8 percentage points. The cause is
discreteness -- the rejection boundary moves in integer steps -- and it is
worst at high discordant rates and extreme splits. ``code-style``'s own
configuration is clean, which is exactly why a narrow probe of it saw nothing
and a wide sweep was necessary.

So this module SCANS rather than bisects, and reports TWO sample sizes: the
first that reaches the target, and the one past which the target stays
reached. A design scheduled at the first can be invalidated by running one
extra item, which is not advice anyone should ship.

WHAT THIS MODULE ASSUMES AND DOES NOT CHECK. The discordant rate is an
ASSUMPTION carried from a pilot, and the whole answer is proportional to it in
the way sample sizes usually are -- halving the rate roughly doubles the pairs
needed. It is not estimated here and no interval is put around it, so a design
sized from a pilot with few discordant pairs inherits that pilot's noise
silently. The honest companion is to size the run at the rate's lower
confidence bound as well and read both numbers.

IT ALSO ASSUMES THE PAIRS ARE INDEPENDENT DRAWS. Files from a shared monorepo
are not, and :mod:`platform_core.clustering` measures the design effect that
correction needs. Every count here is therefore a floor: the clustered
requirement is larger, by the design effect.
"""

from __future__ import annotations

import math

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.mcnemar_detectability import power_at_split
from platform_core.minimum_detectable_effect import mcnemar_power
from platform_core.power_distributions import (
    McNemarTest,
    binomial_point_vector,
    require_alpha,
)
from platform_core.power_types import McNemarDesignSize, PowerInstrument
from platform_core.power_validators import (
    require_design_split,
    require_discordant_rate,
    require_target_power,
)

#: Largest ``search_ceiling`` this module will accept.
#:
#: A COST BOUND, MEASURED, not a statistical one. The work is quadratic in the
#: ceiling twice over -- once to build a rejection boundary for every
#: discordant count, once to average over every count at every candidate size
#: -- and the whole search was timed on 2026-09-10 at 0.21 s to 512 pairs,
#: 1.04 s to 1,024 and 10.60 s to 2,048. Past this the instrument stops being
#: something a person runs while thinking.
#:
#: It is a ceiling on the ARGUMENT rather than a silent clamp: a caller who
#: needs more is told, and can ask the question in a form that does not need
#: a per-item search. Sized so the sample sizes ``code-style`` actually needs
#: are inside it -- its 5.6% discordant stratum reaches 80% power at roughly a
#: thousand items, against a corpus of 392.
MAX_SEARCH_PAIRS: int = 2048


def require_search_ceiling(search_ceiling: int) -> None:
    """Reject a search ceiling that is not a usable bound.

    Args:
        search_ceiling: Largest sample size the caller will consider.

    Raises:
        AppError: ``POWER_SEARCH_CEILING_INVALID`` when not positive or above
            :data:`MAX_SEARCH_PAIRS`.
    """
    if search_ceiling < 1 or search_ceiling > MAX_SEARCH_PAIRS:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SEARCH_CEILING_INVALID,
            f"the search ceiling must lie in [1, {MAX_SEARCH_PAIRS}]; got {search_ceiling!r}. "
            "The upper bound is a measured cost limit: the search is quadratic in the "
            "ceiling and was timed at 10.60 s to 2,048 pairs.",
        )


def rejection_boundaries(max_discordant: int, alpha: float, test: McNemarTest) -> tuple[int, ...]:
    """Tabulate the rejection boundary for every discordant count up to a limit.

    Public because it is the expensive half of every answer here, and a
    caller sizing several designs against one alpha should be able to build it
    once and read it rather than paying for it per call.

    Each entry comes from
    :func:`~platform_core.minimum_detectable_effect.mcnemar_power` rather than
    being re-derived, so the design sizes and the falsifiability floors can
    never disagree about which splits reject.

    Args:
        max_discordant: Largest discordant count to tabulate, non-negative.
        alpha: Two-sided significance level in ``(0, 1)``.
        test: Which McNemar variant the design will be reported under.

    Returns:
        A tuple of length ``max_discordant + 1`` whose ``d``-th entry is the
        largest minority count that still rejects at ``d`` discordant pairs,
        or ``-1`` where no split rejects.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when ``max_discordant`` is
            negative; ``POWER_ALPHA_OUT_OF_RANGE`` on a bad level.
    """
    if max_discordant < 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"max_discordant cannot be negative; got {max_discordant!r}",
        )
    require_alpha(alpha)
    return tuple(
        mcnemar_power(discordant, alpha, test)["most_balanced_rejecting_minority"]
        for discordant in range(max_discordant + 1)
    )


def conditional_power_table(boundaries: tuple[int, ...], split: float) -> tuple[float, ...]:
    """Tabulate the conditional rejection chance at every discordant count.

    Args:
        boundaries: Rejection boundaries as returned by
            :func:`rejection_boundaries`.
        split: True probability that a discordant pair falls to the candidate.

    Returns:
        A tuple the same length as ``boundaries`` whose ``d``-th entry is the
        chance the test rejects given exactly ``d`` discordant pairs.
    """
    return tuple(
        power_at_split(discordant, boundary, split)
        for discordant, boundary in enumerate(boundaries)
    )


def unconditional_power(
    total_pairs: int,
    discordant_rate: float,
    split: float,
    alpha: float,
    test: McNemarTest,
) -> float:
    """Average the rejection chance over every discordant count the run might produce.

    THE POWER A DESIGN ACTUALLY HAS, as opposed to the power it would have if
    the discordant count came out exactly at its expectation. Both are
    defensible numbers and they are not the same one; this is the one that
    belongs beside a sample size.

    Args:
        total_pairs: Items both arms would answer, non-negative.
        discordant_rate: Expected share of pairs that disagree, in ``(0, 1]``.
        split: The effect being designed for, in ``(1/2, 1]``.
        alpha: Two-sided significance level in ``(0, 1)``.
        test: Which McNemar variant the design will be reported under.

    Returns:
        The averaged rejection probability in ``[0, 1]``.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when ``total_pairs`` is
            negative; ``POWER_ALPHA_OUT_OF_RANGE``,
            ``POWER_DISCORDANT_RATE_OUT_OF_RANGE`` or
            ``POWER_DESIGN_SPLIT_OUT_OF_RANGE`` on a bad parameter.
    """
    if total_pairs < 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"total_pairs cannot be negative; got {total_pairs!r}",
        )
    require_alpha(alpha)
    require_discordant_rate(discordant_rate)
    require_design_split(split)
    boundaries = rejection_boundaries(total_pairs, alpha, test)
    conditional = conditional_power_table(boundaries, split)
    weights = binomial_point_vector(total_pairs, discordant_rate)
    return math.fsum(weight * chance for weight, chance in zip(weights, conditional, strict=True))


def mcnemar_design_size(
    discordant_rate: float,
    split: float,
    alpha: float,
    target_power: float,
    test: McNemarTest,
    search_ceiling: int,
) -> McNemarDesignSize:
    """Find how many pairs the next run needs to reach a target power.

    Scans every sample size up to ``search_ceiling`` rather than bisecting,
    because the averaged power is NOT monotone in the sample size -- see this
    module's docstring for the measurement. The scan is what makes
    ``durably_reaching_pairs`` computable at all: it is read off the whole
    curve, from the top down.

    Args:
        discordant_rate: Expected share of pairs that disagree, in ``(0, 1]``.
            Normally the pilot's observed rate, and the assumption the answer
            rests on.
        split: The effect being designed for, as the probability that a
            discordant pair falls to the candidate; in ``(1/2, 1]``.
        alpha: Two-sided significance level in ``(0, 1)``.
        target_power: The power the design must reach, in ``(0, 1)``.
        test: Which McNemar variant. Required rather than defaulted: the
            variants have different rejection regions, so a design sized
            against the wrong one is sized for a test nobody will run.
        search_ceiling: Largest sample size to examine, in
            ``[1, MAX_SEARCH_PAIRS]``. Required rather than defaulted because
            both answers are relative to it, and a caller who did not choose
            it cannot read ``durably_reaching_pairs`` correctly.

    Returns:
        A populated :class:`~platform_core.power_types.McNemarDesignSize`.

    Raises:
        AppError: ``POWER_ALPHA_OUT_OF_RANGE``,
            ``POWER_DISCORDANT_RATE_OUT_OF_RANGE``,
            ``POWER_DESIGN_SPLIT_OUT_OF_RANGE``,
            ``POWER_TARGET_POWER_OUT_OF_RANGE`` or
            ``POWER_SEARCH_CEILING_INVALID`` on a bad parameter; and
            ``POWER_DESIGN_SIZE_UNREACHABLE`` when no size at or below the
            ceiling reaches the target. That last is raised rather than
            returned as the ceiling, for the reason
            ``POWER_REQUIRED_REPLICATES_UNREACHABLE`` is: a caller reading a
            returned ceiling as an answer would schedule a run that cannot
            answer its own question.
    """
    require_alpha(alpha)
    require_discordant_rate(discordant_rate)
    require_design_split(split)
    require_target_power(target_power)
    require_search_ceiling(search_ceiling)

    # Built ONCE to the ceiling and reused at every candidate size. The
    # boundary depends on the discordant count and the alpha alone -- never on
    # the sample size -- so rebuilding it per candidate would multiply the
    # expensive half of the work by the length of the scan.
    boundaries = rejection_boundaries(search_ceiling, alpha, test)
    conditional = conditional_power_table(boundaries, split)

    curve: list[float] = []
    for candidate in range(search_ceiling + 1):
        weights = binomial_point_vector(candidate, discordant_rate)
        curve.append(
            math.fsum(
                weight * chance
                for weight, chance in zip(weights, conditional[: candidate + 1], strict=True)
            )
        )

    # THE CEILING ITSELF MUST MEET THE TARGET, and that is a stricter test
    # than "some size did". Because power dips, a curve can cross the target
    # and fall back below it before the ceiling; reporting the crossing then
    # would be exactly the advice this module exists to refuse -- a size that
    # running a few more items invalidates. The two failures are told apart in
    # the message because their remedies differ: never reaching it means the
    # design is far off, reaching and losing it means the ceiling landed in a
    # trough and a slightly larger one settles the question.
    if curve[search_ceiling] < target_power:
        crossed = [size for size, power in enumerate(curve) if power >= target_power]
        detail = (
            f"no sample size at or below {search_ceiling} reaches it at all"
            if not crossed
            else (
                f"sizes from {crossed[0]} do reach it but {search_ceiling} itself falls "
                f"back below, so no size here can be certified to HOLD the target"
            )
        )
        raise AppError(
            StatisticalPowerErrorCode.POWER_DESIGN_SIZE_UNREACHABLE,
            f"power {target_power} is not durably reached within {search_ceiling} pairs "
            f"at a {discordant_rate} discordant rate and a {split} split under the "
            f"{test.value} test: {detail}. The power at the ceiling is "
            f"{curve[search_ceiling]}. Raising the ceiling may help -- unlike "
            "POWER_TARGET_UNREACHABLE, this bounds where the search looked and not "
            "what the test can do.",
        )
    first = next(size for size, power in enumerate(curve) if power >= target_power)
    # DOWNWARD FROM THE CEILING, because the question is where the target
    # stops being lost again rather than where it is first met. The walk
    # cannot run off the bottom: no pairs means no discordant pairs means no
    # rejection, so ``curve[0]`` is exactly zero and every legal target power
    # is strictly above it.
    durable = search_ceiling
    while curve[durable - 1] >= target_power:
        durable -= 1
    return McNemarDesignSize(
        instrument=PowerInstrument.MCNEMAR_DESIGN_SIZE.value,
        test=test.value,
        discordant_rate=discordant_rate,
        split=split,
        alpha=alpha,
        target_power=target_power,
        search_ceiling=search_ceiling,
        first_reaching_pairs=first,
        durably_reaching_pairs=durable,
        sawtooth_gap_pairs=durable - first,
        power_at_first_reaching=curve[first],
        power_at_durably_reaching=curve[durable],
        expected_discordant_pairs=durable * discordant_rate,
    )


__all__ = [
    "MAX_SEARCH_PAIRS",
    "conditional_power_table",
    "mcnemar_design_size",
    "rejection_boundaries",
    "require_search_ceiling",
    "unconditional_power",
]
