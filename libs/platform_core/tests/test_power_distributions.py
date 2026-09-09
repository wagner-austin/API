"""Tests for the distributions the power helper inverts.

Every value asserted here comes from OUTSIDE this module: published
textbook t-critical values, closed forms the incomplete beta must reduce
to, and the digits a shipped McNemar implementation prints. A test
seeded from the implementation's own output passes against any
arithmetic, including wrong arithmetic.
"""

from __future__ import annotations

import math

import pytest

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.power_distributions import (
    McNemarTest,
    binomial_point_probability,
    exact_mcnemar_p,
    mcnemar_p,
    mid_p_mcnemar_p,
    regularized_incomplete_beta,
    require_alpha,
    t_critical,
    two_sided_t_survival,
)


class TestTCritical:
    """The inverse Student-t, against values this module did not produce."""

    @pytest.mark.parametrize(
        ("degrees_of_freedom", "expected"),
        [
            (1, 12.7062),
            (2, 4.302653),
            (3, 3.182446),
            (10, 2.228139),
            (30, 2.042272),
            (120, 1.979930),
        ],
    )
    def test_matches_published_two_sided_five_percent_values(
        self, degrees_of_freedom: int, expected: float
    ) -> None:
        assert t_critical(degrees_of_freedom, 0.05) == pytest.approx(expected, abs=1e-4)

    def test_matches_published_one_percent_values(self) -> None:
        assert t_critical(2, 0.01) == pytest.approx(9.92484, abs=1e-4)
        assert t_critical(10, 0.01) == pytest.approx(3.169273, abs=1e-4)

    def test_is_the_actual_inverse_of_the_survival_function(self) -> None:
        for degrees_of_freedom in (1, 2, 7, 40):
            for alpha in (0.10, 0.05, 0.01):
                critical = t_critical(degrees_of_freedom, alpha)
                assert two_sided_t_survival(critical, degrees_of_freedom) == pytest.approx(
                    alpha, rel=1e-9
                )

    def test_refuses_degrees_of_freedom_below_one(self) -> None:
        with pytest.raises(AppError) as excinfo:
            t_critical(0, 0.05)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_refuses_alpha_outside_the_open_unit_interval(self, alpha: float) -> None:
        with pytest.raises(AppError) as excinfo:
            t_critical(5, alpha)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    def test_refuses_an_alpha_beyond_the_search_ceiling(self) -> None:
        # Refusal, not a silently-returned endpoint.
        with pytest.raises(AppError) as excinfo:
            t_critical(1, 1e-300)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    def test_require_alpha_accepts_the_interior(self) -> None:
        require_alpha(0.5)


class TestRegularizedIncompleteBeta:
    """The numerical core, including both continued-fraction branches."""

    def test_boundaries_are_exact(self) -> None:
        assert regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0
        assert regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0
        assert regularized_incomplete_beta(-0.5, 2.0, 3.0) == 0.0
        assert regularized_incomplete_beta(1.5, 2.0, 3.0) == 1.0

    def test_is_symmetric_under_the_standard_identity(self) -> None:
        # Exercises BOTH branches of the continued-fraction switch: x and
        # 1-x fall on opposite sides of the (a+1)/(a+b+2) pivot.
        for x, a, b in ((0.25, 2.0, 5.0), (0.75, 2.0, 5.0), (0.5, 3.0, 3.0)):
            assert regularized_incomplete_beta(x, a, b) == pytest.approx(
                1.0 - regularized_incomplete_beta(1.0 - x, b, a), rel=1e-12
            )

    def test_reduces_to_the_closed_form_for_a_equals_one(self) -> None:
        # I_x(1, b) == 1 - (1-x)**b, independent of the fraction.
        for x, b in ((0.3, 4.0), (0.7, 2.0)):
            assert regularized_incomplete_beta(x, 1.0, b) == pytest.approx(
                1.0 - math.pow(1.0 - x, b), rel=1e-12
            )


class TestMcNemarPValues:
    """Against the digits a shipped implementation already prints."""

    def test_exact_matches_the_shipped_reference(self) -> None:
        # tools/code-style-eval .../core/scoring.py, exact_mcnemar_p.
        # A 4:3 table has 7 discordant pairs and a minority of 3, where
        # the doubled tail exceeds 1 and the cap is what makes it a
        # probability. The 0.7265625 that table is known by is its MID-P
        # value, asserted below -- not this one.
        assert exact_mcnemar_p(3, 7) == 1.0
        assert exact_mcnemar_p(0, 6) == pytest.approx(0.03125)
        assert exact_mcnemar_p(1, 10) == pytest.approx(0.021484375)
        assert exact_mcnemar_p(2, 10) == pytest.approx(0.109375)

    def test_mid_p_tie_does_not_double_count_the_centre(self) -> None:
        # THE TIE BUG, from a live implementation. Doubling a tail at a
        # 3:3 split double-counts the centre and yields 1.0; the correct
        # form is 1 - 0.5 * point = 0.84375.
        assert mid_p_mcnemar_p(3, 6) == pytest.approx(0.84375)
        assert mid_p_mcnemar_p(3, 6) != 1.0

    def test_mid_p_non_tie_agrees_with_the_shipped_reference(self) -> None:
        # The half a wrong implementation still passes -- which is why the
        # tie assertion above has to exist beside it. 4:3 is the table the
        # reference session checked theirs against and got to the digit.
        assert mid_p_mcnemar_p(3, 7) == pytest.approx(0.7265625)

    def test_no_discordant_pairs_is_certainty_not_a_sentinel(self) -> None:
        assert exact_mcnemar_p(0, 0) == 1.0
        assert mid_p_mcnemar_p(0, 0) == 1.0

    def test_exact_is_capped_at_one(self) -> None:
        assert exact_mcnemar_p(1, 2) == 1.0

    def test_point_probability_is_symmetric_about_the_centre(self) -> None:
        assert binomial_point_probability(2, 10) == pytest.approx(binomial_point_probability(8, 10))

    def test_mid_p_is_never_larger_than_exact(self) -> None:
        # The whole point of the correction: mid-p is less conservative.
        for discordant in (4, 9, 15):
            for minority in range(discordant // 2 + 1):
                assert mid_p_mcnemar_p(minority, discordant) <= exact_mcnemar_p(
                    minority, discordant
                )

    def test_dispatch_selects_the_named_variant(self) -> None:
        assert mcnemar_p(3, 6, McNemarTest.EXACT) == exact_mcnemar_p(3, 6)
        assert mcnemar_p(3, 6, McNemarTest.MID_P) == mid_p_mcnemar_p(3, 6)

    @pytest.mark.parametrize(("minority", "discordant"), [(-1, 5), (3, 2), (0, -1)])
    def test_refuses_a_nonsensical_split(self, minority: int, discordant: int) -> None:
        with pytest.raises(AppError) as excinfo:
            exact_mcnemar_p(minority, discordant)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID


def _tail_from_the_definition(minority: int, discordant_pairs: int) -> float:
    """Compute the exact two-sided p from the textbook sum, term by term.

    The definition the optimised implementation must agree with, written the
    obvious way: every binomial coefficient built independently by
    :func:`math.comb`. Slow and unmistakable, which is what an oracle is for.

    Args:
        minority: Count in one discordant cell.
        discordant_pairs: Total discordant pairs.

    Returns:
        The two-sided exact conditional p-value.
    """
    extreme = min(minority, discordant_pairs - minority)
    tail = sum(math.comb(discordant_pairs, k) for k in range(extreme + 1))
    return min(1.0, 2 * tail / (1 << discordant_pairs))


class TestLargeDiscordantCounts:
    """The regime these helpers are used to ARGUE FOR, and could not enter.

    Both failures were measured on 2026-09-09 while sizing a cartridge
    question set against this module. They matter because the sizes involved
    are not hypothetical: the workspace's own corpus-extraction ablation ran
    2,627 items, and that is the number the cartridge sets are being scaled
    toward precisely because the small ones resolved nothing.
    """

    @pytest.mark.parametrize("discordant", [1024, 1200, 2627])
    def test_a_thousand_or_more_pairs_returns_a_probability(self, discordant: int) -> None:
        """``float(1 << n)`` exceeded the largest double at n = 1024.

        The denominator was converted BEFORE the division, so every call at
        or above 1,024 discordant pairs raised ``OverflowError`` rather than
        returning a probability. Asserted AT THE CENTRE of the distribution,
        where the value is comfortably representable -- the balanced split's
        point probability is about ``sqrt(2 / (pi * n))`` -- so the test
        proves the arithmetic ran rather than merely that nothing was raised.

        Args:
            discordant: Total discordant pairs.
        """
        centre = binomial_point_probability(discordant // 2, discordant)

        assert math.isfinite(centre)
        assert centre == pytest.approx(math.sqrt(2.0 / (math.pi * discordant)), rel=1e-3)
        assert exact_mcnemar_p(discordant // 2, discordant) == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("discordant", [1200, 2627])
    def test_the_most_extreme_split_underflows_to_zero_rather_than_raising(
        self, discordant: int
    ) -> None:
        """``2 ** -1200`` is below the smallest denormal, and that is correct.

        The rounded value of a 0-versus-1200 split really is zero to double
        precision, and zero is the right answer for ``can_ever_reject`` to
        read -- such a split rejects at any alpha. Pinned so that a later
        change cannot turn a correct underflow into an exception, or a
        sentinel, without a test noticing.

        Args:
            discordant: Total discordant pairs.
        """
        assert binomial_point_probability(0, discordant) == 0.0
        assert exact_mcnemar_p(0, discordant) == 0.0

    def test_the_tail_recurrence_equals_the_definition_at_every_split(self) -> None:
        """The optimisation must be an optimisation and nothing else.

        Exhaustive over every split of every size up to 120: the recurrence
        ``comb(n, k+1) == comb(n, k) * (n - k) // (k + 1)`` is exact in
        integers, so agreement should be bit-for-bit rather than approximate,
        and that is what is asserted.
        """
        for discordant in range(0, 121):
            for minority in range(discordant + 1):
                assert exact_mcnemar_p(minority, discordant) == _tail_from_the_definition(
                    minority, discordant
                )

    def test_the_resolvable_difference_shrinks_as_the_set_grows(self) -> None:
        """Sanity on the direction, since the whole gate depends on it.

        More items must resolve smaller differences, or a plan could be
        refused for having too much data.
        """
        smallest = [
            exact_mcnemar_p(discordant // 2 - 40, discordant) for discordant in (1024, 2048)
        ]

        assert smallest[1] > smallest[0]
