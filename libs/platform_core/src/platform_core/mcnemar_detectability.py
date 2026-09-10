"""How lopsided a paired BINARY comparison had to be to reach a target power.

THE POWER SIBLING OF :func:`~platform_core.minimum_detectable_effect.mcnemar_power`,
AND THE TWO ANSWER DIFFERENT QUESTIONS. That one asks "can ANY split of these
discordant pairs reject?" -- falsifiability, a yes/no about the design. This
one asks "how far from even would the split have had to fall before the test
called it, 80% of the time?" -- detectability, a magnitude. A comparison can
pass the first and be hopeless on the second, which is exactly the state
``code-style``'s published strata are in: at *d* = 6 under mid-p the null CAN
be rejected, and only by a 6:0 split, so the smallest detectable difference
sits above the base rate it would have to move.

CONDITIONED ON THE OBSERVED DISCORDANT COUNT, WHICH IS A LIMIT AND NOT AN
OVERSIGHT. McNemar conditions on the discordant pairs, so this instrument
takes the *d* a comparison actually produced and answers a question about
THAT comparison: given these pairs disagreed, how lopsided did they need to
be. It does NOT answer "how many items should the next run score", because
that question ranges over a RANDOM *d* ~ Binomial(n, discordant rate) and
must average power across it.

Those two are numerically different and the gap is not small. Measured
2026-09-10 against ``code-style``'s own published power table -- exact test,
alpha 0.05, a 70:30 split, 5.6% discordant:

    n      published   conditional at round(n x rate)   averaged over d
    226      0.21                0.2026                     0.2061
    800      0.73                0.7462                     0.7335

The averaged column reproduces the published figures and the conditional one
does not, so that table was computed the other way and this module would give
a different answer for it. **Anyone sizing a FUTURE run wants the averaged
form, which is not here.** It is named rather than implied, because an
instrument that silently answers a neighbouring question is the defect this
whole module family exists to prevent.

THE REJECTION REGION IS NOT RE-DERIVED HERE. It comes from
:func:`~platform_core.minimum_detectable_effect.mcnemar_power`'s
``most_balanced_rejecting_minority``, so the detectable effect and the
falsifiability floor are always computed against the same boundary. A second
expression of "which splits reject" is a second thing that can drift from the
p-value function, and the two would then disagree about one comparison while
both looking right.

NO :class:`~platform_core.power_types.PowerVerdict`, for the reason
:class:`~platform_core.power_types.RequiredReplicates` carries none: this
record ANSWERS "what would it have taken" rather than classifying a result
against a threshold. Nothing is handed to it to compare against -- a caller
holding a smallest-effect-of-interest compares it to
``minimum_detectable_rate_difference`` themselves, and a caller without one
(``code-style`` has no declared SEI) still gets a usable number.
"""

from __future__ import annotations

import math

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.minimum_detectable_effect import mcnemar_power
from platform_core.power_distributions import McNemarTest, require_alpha
from platform_core.power_types import McNemarDetectableEffect, PowerInstrument

#: Halvings of the split interval. The bracket is ``[0.5, 1.0]``, so after 60
#: halvings its width is below the spacing of a double in that range and
#: further iterations cannot move the answer. Fixed rather than
#: tolerance-driven so the result is reproducible to the bit on every machine
#: -- a published minimum detectable effect that moves with a convergence
#: threshold is not a number anyone can re-derive.
_SPLIT_HALVINGS = 60


def require_target_power(target_power: float) -> None:
    """Refuse a target power outside ``(0, 1)``.

    Its own check rather than reuse of the Clopper-Pearson confidence one:
    a target POWER is the chance of detecting a real effect and a CONFIDENCE
    is how sure an interval is, so a caller passing ``0.95`` meaning either
    would get no error from a shared validator.

    Args:
        target_power: The power the design must reach.

    Raises:
        AppError: ``POWER_TARGET_POWER_OUT_OF_RANGE`` when outside ``(0, 1)``.
            Both endpoints are excluded: power 0 is reached by every design
            and asks nothing, and power 1 is unreachable by any finite split.
    """
    if not 0.0 < target_power < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TARGET_POWER_OUT_OF_RANGE,
            f"target power must lie strictly inside (0, 1); got {target_power!r}. "
            "0 is met by every design and 1 by none, so neither states a design goal.",
        )


def _log_binomial_point(successes: int, trials: int, probability: float) -> float:
    """Log of the binomial point probability, at ANY success probability.

    IN LOGS BECAUSE THE DIRECT FORM OVERFLOWS, and this module's own family
    has already paid for that lesson:
    :func:`~platform_core.power_distributions.binomial_point_probability`
    carries a comment recording an ``OverflowError`` at 1,024 discordant
    pairs, fixed there by dividing integers rather than casting. That fix
    does not transfer, because the weight here is ``C(n, k) * p**k *
    (1-p)**(n-k)`` with a real ``p`` -- the coefficient is a huge int and the
    powers underflow, so their product is representable while neither factor
    is. Reproduced while building this module.

    ``lgamma`` rather than ``log(comb(...))`` for the same reason: the
    coefficient itself exceeds a double long before the probability does.

    Args:
        successes: Number of successes, ``0 <= successes <= trials``.
        trials: Number of independent trials, non-negative.
        probability: Success probability in ``[0, 1]``.

    Returns:
        The natural log of the point probability, or ``-inf`` where the point
        has probability zero. ``-inf`` is a real answer here and not an error:
        at ``probability`` exactly 1 every outcome but ``successes == trials``
        is impossible, and the caller sums exponentials so a zero term is
        simply absent.
    """
    if probability <= 0.0:
        return 0.0 if successes == 0 else -math.inf
    if probability >= 1.0:
        return 0.0 if successes == trials else -math.inf
    coefficient = (
        math.lgamma(trials + 1) - math.lgamma(successes + 1) - math.lgamma(trials - successes + 1)
    )
    return (
        coefficient
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )


def power_at_split(
    discordant_pairs: int, most_balanced_rejecting_minority: int, split: float
) -> float:
    """Chance the conditional test rejects, at a stated true split.

    Public because it is what makes a published minimum detectable effect
    auditable: a reader who doubts the number can evaluate the power at the
    returned split and see it clear the target.

    Args:
        discordant_pairs: The *d* the comparison produced.
        most_balanced_rejecting_minority: Largest minority count that still
            rejects, as reported by
            :func:`~platform_core.minimum_detectable_effect.mcnemar_power`.
            ``-1`` means no split rejects.
        split: True probability that one discordant pair falls to the
            candidate, in ``[0, 1]``. ``0.5`` is the null.

    Returns:
        The rejection probability in ``[0, 1]``. Exactly ``0.0`` when no split
        rejects, which is the unfalsifiable design rather than a failure.
    """
    if most_balanced_rejecting_minority < 0:
        return 0.0
    lower = range(most_balanced_rejecting_minority + 1)
    upper = range(discordant_pairs - most_balanced_rejecting_minority, discordant_pairs + 1)
    # THE TAILS CAN GENUINELY OVERLAP, AND ONLY UNDER MID-P. They share an
    # outcome when the boundary reaches the middle -- even ``d`` with
    # ``m == d/2`` -- which needs the PERFECTLY EVEN split to reject. Under
    # the exact test that split's p is 1.0 and no legal alpha admits it, so
    # reasoning from the exact variant alone says this cannot happen. It can:
    # mid-p gives the even split the tie form instead, measured 0.75 at d=2,
    # 0.8125 at d=4 and 0.84375 at d=6 -- the same 3:3 value
    # :func:`~platform_core.power_distributions.mid_p_mcnemar_p` documents --
    # so a permissive alpha reaches the middle and both ranges then contain
    # ``d/2``. Summing them directly returns a probability above 1.
    #
    # I removed this guard once on the strength of the exact-test argument
    # and put it back after measuring the mid-p case.
    outcomes = set(lower) | set(upper)
    return math.fsum(
        math.exp(_log_binomial_point(count, discordant_pairs, split)) for count in outcomes
    )


def mcnemar_detectable_effect(
    discordant_pairs: int,
    total_pairs: int,
    alpha: float,
    target_power: float,
    test: McNemarTest,
) -> McNemarDetectableEffect:
    """Smallest true difference this comparison could have detected.

    Args:
        discordant_pairs: The *d* the comparison produced, non-negative.
        total_pairs: Items both arms answered. Carried so the answer can be
            expressed as a rate; it does NOT enter the power arithmetic,
            because McNemar conditions on the discordant pairs alone.
        alpha: Two-sided significance level in ``(0, 1)``.
        target_power: Power the design must reach, in ``(0, 1)``.
        test: Which McNemar variant. Required rather than defaulted: the two
            variants have different rejection regions, so an effect computed
            against the wrong one describes a test nobody ran.

    Returns:
        A populated :class:`~platform_core.power_types.McNemarDetectableEffect`.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when a count is negative or
            more pairs disagreed than were compared;
            ``POWER_ALPHA_OUT_OF_RANGE`` / ``POWER_TARGET_POWER_OUT_OF_RANGE``
            on a bad level; and ``POWER_TARGET_UNREACHABLE`` when NO split of
            these discordant pairs rejects at this alpha. That last is raised
            rather than returned as a maximal effect, for the reason
            ``POWER_REQUIRED_REPLICATES_UNREACHABLE`` is: a caller reading a
            returned ceiling as an answer would report a detectable effect for
            a comparison in which nothing is detectable at all.
    """
    if discordant_pairs < 0 or total_pairs < 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"counts cannot be negative; got discordant_pairs={discordant_pairs!r} "
            f"and total_pairs={total_pairs!r}",
        )
    if discordant_pairs > total_pairs:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"{discordant_pairs} pairs cannot disagree out of {total_pairs} compared; "
            "the discordant count is a subset of the pairs, not a separate sample",
        )
    require_alpha(alpha)
    require_target_power(target_power)

    floor = mcnemar_power(discordant_pairs, alpha, test)
    boundary = floor["most_balanced_rejecting_minority"]
    if boundary < 0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TARGET_UNREACHABLE,
            f"no split of {discordant_pairs} discordant pairs rejects at alpha={alpha} "
            f"under the {test.value} test, so NO true effect reaches power "
            f"{target_power} however large it is. The smallest attainable p is "
            f"{floor['smallest_attainable_p']}. More discordant pairs are the remedy, "
            "not a larger effect.",
        )

    # Power rises monotonically as the split leaves 1/2, so the smallest
    # detectable split is a bisection rather than a search over outcomes.
    low, high = 0.5, 1.0
    for _ in range(_SPLIT_HALVINGS):
        middle = (low + high) / 2
        if power_at_split(discordant_pairs, boundary, middle) >= target_power:
            high = middle
        else:
            low = middle

    net_pairs = discordant_pairs * (2 * high - 1)
    return McNemarDetectableEffect(
        instrument=PowerInstrument.MCNEMAR_DETECTABLE_EFFECT.value,
        test=test.value,
        discordant_pairs=discordant_pairs,
        total_pairs=total_pairs,
        alpha=alpha,
        target_power=target_power,
        most_balanced_rejecting_minority=boundary,
        minimum_detectable_split=high,
        minimum_detectable_net_pairs=net_pairs,
        # Zero pairs compared is the empty comparison, and it cannot reach
        # here: d > n is refused above and d < 0 with it, so n == 0 implies
        # d == 0, which has no rejecting split and raised already.
        minimum_detectable_rate_difference=net_pairs / total_pairs,
        achieved_power=power_at_split(discordant_pairs, boundary, high),
    )


__all__ = [
    "mcnemar_detectable_effect",
    "power_at_split",
    "require_target_power",
]
