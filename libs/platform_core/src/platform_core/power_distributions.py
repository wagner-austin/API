"""The distributions the power helper inverts, in the standard library alone.

WHY THIS IS SEPARATE FROM :mod:`platform_core.minimum_detectable_effect`.
Two roles: this module answers "what does the distribution say", the other
answers "what could this instrument have resolved". Keeping them apart is
what lets a consumer check a published p-value against the same arithmetic
that produced the MDE beside it, without importing the record types.

WHY PURE STDLIB. ``platform_core``'s runtime dependencies are fastapi, httpx
and rich -- numpy is dev-only, scipy absent. There is no
``scipy.stats.t.ppf``, and that absence is exactly why a hardcoded
t-critical TABLE is the tempting shortcut. A table is a second source of
truth that goes stale silently, so :func:`t_critical` inverts the
distribution exactly instead.

WHY McNEMAR APPEARS HERE AT ALL. A paired BINARY comparison has no standard
deviation to put in a t-formula: McNemar conditions on the discordant pairs
and tests their split against 50/50. Three sessions hand-rolled that
arithmetic on 2026-09-08 and two of them shipped a wrong number that looked
right, so it belongs beside the t-distribution rather than in each project.

THE SEMANTICS HERE ARE DELIBERATELY IDENTICAL TO A SHIPPED IMPLEMENTATION.
``tools/code-style-eval/src/code_style_eval/core/scoring.py`` already carries
correct ``exact_mcnemar_p`` and ``mid_p_mcnemar_p`` (blob
be9442c3047ea93eb16c955704a8e51a86c19907). That package is under ``tools/``
and cannot be imported by ``libs/``, so this is the shared home and that copy
should migrate to consume it. Until it does, two implementations exist, and
:mod:`tests.test_power_distributions` pins this one to the published values of
that one to the digit -- so they cannot drift apart unnoticed while both live.
"""

from __future__ import annotations

import math
from enum import StrEnum

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError

#: Bisection bracket for :func:`t_critical`, in units of t.
#:
#: t is unbounded as alpha approaches 0, so this is a refusal boundary rather
#: than a guess at any answer: an alpha needing t beyond it is outside what
#: this helper will certify, and :func:`t_critical` raises instead of
#: returning the endpoint.
_T_SEARCH_CEILING: float = 1.0e6

#: Bisection iterations for :func:`t_critical`.
#:
#: The bracket halves each pass, so 200 passes resolve 1e6 to far below
#: floating-point resolution. Fixed rather than tolerance-based: a fixed count
#: cannot fail to terminate.
_T_SEARCH_ITERATIONS: int = 200

#: Continued-fraction iterations for the incomplete beta.
#:
#: FIXED, NOT A TOLERANCE, and the classic Lentz guards are deliberately
#: absent. Numerical Recipes' version breaks on convergence and floors any
#: denominator that underflows toward zero. Both were removed here after
#: measurement, for reasons the standards in this repo already state:
#:
#:   * The floor is a SILENT SUBSTITUTION. Replacing a zero denominator with
#:     1e-300 softens a genuine numerical failure into a plausible number,
#:     which is the best-effort behaviour the codebase forbids. Without it a
#:     true zero raises ``ZeroDivisionError`` and propagates, which is what a
#:     failure should do.
#:   * Neither the floor nor the early break is REACHABLE from this module's
#:     inputs. ``regularized_incomplete_beta`` is called only with
#:     ``a = df/2`` and ``b = 0.5``, validated ``df >= 1``, and ``x`` strictly
#:     inside ``(0, 1)``. Coverage over the full test matrix (df 1-120, alpha
#:     0.01-0.10, the symmetry and closed-form identities) never entered any
#:     of them. Unreachable branches are dead code, and dead code cannot be
#:     covered -- so keeping them would fail this package's own 100% gate
#:     while adding no protection anyone can demonstrate.
#:
#: The fraction converges to machine precision in roughly ten passes for
#: these shapes; 300 is far beyond that, costs microseconds, and cannot fail
#: to terminate. Same reasoning as :data:`_T_SEARCH_ITERATIONS`.
_BETA_CF_ITERATIONS: int = 300


class McNemarTest(StrEnum):
    """Which McNemar variant a comparison is reported under.

    NOT A STYLE CHOICE, AND NOT OPTIONAL. The two variants have different
    rejection regions, so an MDE computed against one is an MDE for a test
    nobody ran if the page reports the other. Measured on 2026-09-08: at
    d=29 the exact test rejects when the minority is at most 8 while mid-p
    rejects at 9, and at d=60 the boundary moves 21 -> 22. Numbers published
    from the wrong pairing moved 1.80 -> 1.59 pp and 8.17 -> 7.49 pp on
    recompute, with neither half individually wrong.

    ``MID_P`` is the better default and that is a measured result, not a
    preference: Fagerland, Lydersen and Laake (BMC Med Res Methodol 2013)
    compared type I error and power across 9,595 scenarios and found the
    exact conditional test "did not perform well for any of the considered
    scenarios" -- its guarantee of the nominal level buys conservativeness
    that costs real detections, worst in the small-discordant regime where
    these audits live. ``EXACT`` is kept because a project that reports it
    must have its MDE computed against it.
    """

    EXACT = "exact"
    MID_P = "mid_p"


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Evaluate the incomplete-beta continued fraction by Lentz's method.

    Args:
        x: Point in ``[0, 1]`` at which the fraction is evaluated.
        a: First beta shape parameter.
        b: Second beta shape parameter.

    Returns:
        The continued fraction's value at ``x``.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 / (1.0 - qab * x / qap)
    h = d
    for m in range(1, _BETA_CF_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / (1.0 + aa * d)
        c = 1.0 + aa / c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / (1.0 + aa * d)
        c = 1.0 + aa / c
        h *= d * c
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute the regularized incomplete beta function ``I_x(a, b)``.

    Public because it is the one numerical routine behind every instrument
    here, and a consumer checking a published number should be able to reach
    it rather than re-derive it.

    Args:
        x: Point in ``[0, 1]``.
        a: First shape parameter, positive.
        b: Second shape parameter, positive.

    Returns:
        ``I_x(a, b)`` in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def two_sided_t_survival(t: float, degrees_of_freedom: int) -> float:
    """Compute ``P(|T| > t)`` for Student's t.

    Args:
        t: Non-negative t statistic.
        degrees_of_freedom: Degrees of freedom, at least 1.

    Returns:
        The two-sided tail probability.
    """
    df = float(degrees_of_freedom)
    return regularized_incomplete_beta(df / (df + t * t), df / 2.0, 0.5)


def t_critical(degrees_of_freedom: int, alpha: float) -> float:
    """Find the two-sided critical value of Student's t.

    Inverts :func:`two_sided_t_survival` by bisection. The survival function
    is strictly decreasing in ``t``, so the root is unique and bisection
    cannot converge on the wrong one.

    Args:
        degrees_of_freedom: Degrees of freedom, at least 1.
        alpha: Two-sided significance level in ``(0, 1)``.

    Returns:
        The ``t`` at which ``P(|T| > t) == alpha``.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when df < 1, or
            ``POWER_ALPHA_OUT_OF_RANGE`` when alpha is out of range or
            demands a critical value beyond the search ceiling.
    """
    if degrees_of_freedom < 1:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"degrees of freedom must be at least 1; got {degrees_of_freedom!r}",
        )
    require_alpha(alpha)
    if two_sided_t_survival(_T_SEARCH_CEILING, degrees_of_freedom) > alpha:
        raise AppError(
            StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE,
            f"alpha {alpha!r} at {degrees_of_freedom} df needs a critical value "
            f"beyond t={_T_SEARCH_CEILING}; this helper will not certify it",
        )
    low = 0.0
    high = _T_SEARCH_CEILING
    for _ in range(_T_SEARCH_ITERATIONS):
        mid = (low + high) / 2.0
        if two_sided_t_survival(mid, degrees_of_freedom) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def require_alpha(alpha: float) -> None:
    """Reject a two-sided alpha outside ``(0, 1)``.

    Args:
        alpha: Candidate significance level.

    Raises:
        AppError: ``POWER_ALPHA_OUT_OF_RANGE`` when out of range.
    """
    if not 0.0 < alpha < 1.0:
        raise AppError(
            StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE,
            f"two-sided alpha must lie in (0, 1); got {alpha!r}",
        )


def _require_discordant(minority: int, discordant_pairs: int) -> None:
    """Reject a nonsensical discordant split.

    Args:
        minority: Count in the smaller discordant cell.
        discordant_pairs: Total discordant pairs.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` when either is negative or
            the minority exceeds the total.
    """
    if discordant_pairs < 0 or minority < 0 or minority > discordant_pairs:
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            f"need 0 <= minority <= discordant_pairs; got minority={minority!r}, "
            f"discordant_pairs={discordant_pairs!r}",
        )


def binomial_point_probability(minority: int, discordant_pairs: int) -> float:
    """Compute the binomial point probability of the observed split.

    Args:
        minority: Count in one discordant cell.
        discordant_pairs: Total discordant pairs.

    Returns:
        The point probability under p = 1/2.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` on a nonsensical split.
    """
    _require_discordant(minority, discordant_pairs)
    extreme = min(minority, discordant_pairs - minority)
    # ``1 << n`` rather than ``2 ** n``: integer arithmetic all the way to the
    # final division is what keeps the value exact rather than accumulated.
    return math.comb(discordant_pairs, extreme) / float(1 << discordant_pairs)


def exact_mcnemar_p(minority: int, discordant_pairs: int) -> float:
    """Compute the two-sided exact conditional McNemar p-value.

    Args:
        minority: Count in one discordant cell.
        discordant_pairs: Total discordant pairs.

    Returns:
        The p-value, capped at 1.0. Returns 1.0 when there are no discordant
        pairs -- the correct answer rather than a sentinel, since identical
        arms contain no evidence of any difference.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` on a nonsensical split.
    """
    _require_discordant(minority, discordant_pairs)
    if discordant_pairs == 0:
        return 1.0
    extreme = min(minority, discordant_pairs - minority)
    tail = sum(math.comb(discordant_pairs, k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / float(1 << discordant_pairs))


def mid_p_mcnemar_p(minority: int, discordant_pairs: int) -> float:
    """Compute the two-sided McNemar mid-p value.

    The correction is one term: subtract the point probability of the
    observed statistic from the two-sided exact value. THE TIE NEEDS ITS OWN
    FORM: when the two discordant cells are equal the observation sits at the
    centre of the distribution, so doubling a tail double-counts it. A
    hand-rolled version that doubled anyway returned 1.0 at a 3:3 split
    instead of 0.84375, agreed with this one on every unequal split, and was
    trusted for exactly that reason.

    Args:
        minority: Count in one discordant cell.
        discordant_pairs: Total discordant pairs.

    Returns:
        The mid-p value, clamped to ``[0, 1]``. Returns 1.0 when there are no
        discordant pairs.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` on a nonsensical split.
    """
    _require_discordant(minority, discordant_pairs)
    if discordant_pairs == 0:
        return 1.0
    point = binomial_point_probability(minority, discordant_pairs)
    if minority * 2 == discordant_pairs:
        return 1.0 - 0.5 * point
    return max(0.0, min(1.0, exact_mcnemar_p(minority, discordant_pairs) - point))


def mcnemar_p(minority: int, discordant_pairs: int, test: McNemarTest) -> float:
    """Compute the McNemar p-value under the named test.

    The entry point every consumer should use: passing the test explicitly is
    what stops an MDE being computed against a rejection region the report
    does not use.

    Args:
        minority: Count in one discordant cell.
        discordant_pairs: Total discordant pairs.
        test: Which variant the comparison is reported under.

    Returns:
        The p-value under ``test``.

    Raises:
        AppError: ``POWER_SAMPLE_SIZE_INVALID`` on a nonsensical split.
    """
    if test is McNemarTest.EXACT:
        return exact_mcnemar_p(minority, discordant_pairs)
    return mid_p_mcnemar_p(minority, discordant_pairs)


__all__ = [
    "McNemarTest",
    "binomial_point_probability",
    "exact_mcnemar_p",
    "mcnemar_p",
    "mid_p_mcnemar_p",
    "regularized_incomplete_beta",
    "require_alpha",
    "t_critical",
    "two_sided_t_survival",
]
