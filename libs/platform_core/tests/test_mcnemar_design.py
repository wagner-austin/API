"""Tests for the unconditional (design-sizing) McNemar instrument.

The load-bearing tests here are the ones that would have caught the mistake
this module was nearly built on: that averaged power is monotone in the sample
size. It is not, and ``test_power_is_not_monotone_in_the_sample_size`` pins a
measured counterexample so nobody re-introduces a bisection.
"""

from __future__ import annotations

import math

import pytest

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.mcnemar_design import (
    MAX_SEARCH_PAIRS,
    conditional_power_table,
    mcnemar_design_size,
    rejection_boundaries,
    require_search_ceiling,
    unconditional_power,
)
from platform_core.power_distributions import McNemarTest, binomial_point_vector
from platform_core.power_records import (
    decode_mcnemar_design_size,
    encode_mcnemar_design_size,
)
from platform_core.power_types import PowerInstrument

ALPHA = 0.05


def _independent_unconditional_power(
    total_pairs: int, rate: float, split: float, alpha: float, test: McNemarTest
) -> float:
    """Average conditional power over d, written independently of the module.

    Deliberately built from ``mcnemar_p`` and ``math.comb`` rather than from
    the module's own helpers, so agreement is evidence rather than a tautology.
    """
    from platform_core.power_distributions import mcnemar_p

    total = 0.0
    for discordant in range(total_pairs + 1):
        boundary = -1
        for minority in range(discordant // 2 + 1):
            if mcnemar_p(minority, discordant, test) <= alpha:
                boundary = minority
        if boundary < 0:
            continue
        conditional = 0.0
        for count in range(discordant + 1):
            if count <= boundary or count >= discordant - boundary:
                conditional += (
                    math.comb(discordant, count)
                    * split**count
                    * (1.0 - split) ** (discordant - count)
                )
        weight = (
            math.comb(total_pairs, discordant)
            * rate**discordant
            * (1.0 - rate) ** (total_pairs - discordant)
        )
        total += weight * conditional
    return total


class TestTheAveragedPowerItself:
    """The quantity the whole module rests on."""

    @pytest.mark.parametrize("total_pairs", [0, 1, 6, 7, 20, 45])
    @pytest.mark.parametrize("test", [McNemarTest.EXACT, McNemarTest.MID_P])
    def test_it_matches_an_independently_written_average(
        self, total_pairs: int, test: McNemarTest
    ) -> None:
        got = unconditional_power(total_pairs, 0.4, 0.75, ALPHA, test)
        want = _independent_unconditional_power(total_pairs, 0.4, 0.75, ALPHA, test)
        assert got == pytest.approx(want, abs=1e-12)

    def test_no_pairs_can_never_reject(self) -> None:
        assert unconditional_power(0, 0.5, 0.9, ALPHA, McNemarTest.MID_P) == 0.0

    def test_too_few_pairs_to_reach_the_smallest_rejecting_count_give_zero(self) -> None:
        # Under mid-p at alpha 0.05 no split of fewer than 5 discordant pairs
        # rejects, so a design that cannot produce 5 has no power at all.
        assert unconditional_power(4, 1.0, 1.0, ALPHA, McNemarTest.MID_P) == 0.0

    def test_it_rises_with_the_discordant_rate(self) -> None:
        low = unconditional_power(60, 0.20, 0.8, ALPHA, McNemarTest.MID_P)
        high = unconditional_power(60, 0.60, 0.8, ALPHA, McNemarTest.MID_P)
        assert low < high

    def test_it_rises_with_the_split(self) -> None:
        near = unconditional_power(60, 0.5, 0.60, ALPHA, McNemarTest.MID_P)
        far = unconditional_power(60, 0.5, 0.95, ALPHA, McNemarTest.MID_P)
        assert near < far

    def test_it_stays_a_probability(self) -> None:
        for pairs in range(0, 40):
            value = unconditional_power(pairs, 0.9, 0.99, ALPHA, McNemarTest.MID_P)
            assert 0.0 <= value <= 1.0


class TestThePublishedTableItReproduces:
    """``code-style``'s own power table, which the conditional module could not give.

    These are the figures the wiki page published, and they are the reason
    this module exists: they were computed by averaging over a random
    discordant count, and until now nothing in the package did that.
    """

    @pytest.mark.parametrize(
        ("total_pairs", "rate", "split", "published"),
        [
            (226, 0.056, 0.70, 0.21),
            (800, 0.056, 0.70, 0.73),
            (226, 0.11, 0.70, 0.44),
            (226, 0.33, 0.60, 0.37),
        ],
    )
    def test_it_reproduces_the_published_figure(
        self, total_pairs: int, rate: float, split: float, published: float
    ) -> None:
        computed = unconditional_power(total_pairs, rate, split, ALPHA, McNemarTest.EXACT)
        assert round(computed, 2) == published

    def test_the_conditional_form_gives_a_different_answer(self) -> None:
        """The gap that made this module necessary, pinned as a number."""
        from platform_core.mcnemar_detectability import power_at_split
        from platform_core.minimum_detectable_effect import mcnemar_power

        averaged = unconditional_power(226, 0.056, 0.70, ALPHA, McNemarTest.EXACT)
        expected_discordant = round(226 * 0.056)
        boundary = mcnemar_power(expected_discordant, ALPHA, McNemarTest.EXACT)
        conditional = power_at_split(
            expected_discordant, boundary["most_balanced_rejecting_minority"], 0.70
        )
        assert round(averaged, 2) == 0.21
        assert round(conditional, 4) == 0.2026
        assert averaged != conditional


class TestPowerIsNotMonotoneInTheSampleSize:
    """The measured fact that forbids a bisection. Do not delete this."""

    def test_adding_a_pair_can_lower_the_power(self) -> None:
        # Measured 2026-09-10: with every pair discordant and a 90:10 split,
        # eight pairs beat nine. A bisection for "the smallest n reaching the
        # target" is invalid on a curve that does this.
        at_eight = unconditional_power(8, 1.0, 0.9, ALPHA, McNemarTest.MID_P)
        at_nine = unconditional_power(9, 1.0, 0.9, ALPHA, McNemarTest.MID_P)
        assert at_nine < at_eight

    def test_the_drop_is_large_enough_to_change_a_verdict(self) -> None:
        at_eight = unconditional_power(8, 1.0, 0.9, ALPHA, McNemarTest.MID_P)
        at_nine = unconditional_power(9, 1.0, 0.9, ALPHA, McNemarTest.MID_P)
        assert at_eight >= 0.80 > at_nine


class TestTheTables:
    """The two halves that every answer is assembled from."""

    def test_boundaries_come_from_mcnemar_power(self) -> None:
        from platform_core.minimum_detectable_effect import mcnemar_power

        built = rejection_boundaries(30, ALPHA, McNemarTest.MID_P)
        assert built == tuple(
            mcnemar_power(d, ALPHA, McNemarTest.MID_P)["most_balanced_rejecting_minority"]
            for d in range(31)
        )

    def test_boundaries_are_minus_one_where_nothing_rejects(self) -> None:
        built = rejection_boundaries(5, ALPHA, McNemarTest.MID_P)
        assert built == (-1, -1, -1, -1, -1, 0)

    def test_the_table_length_is_the_limit_plus_one(self) -> None:
        assert len(rejection_boundaries(0, ALPHA, McNemarTest.EXACT)) == 1

    def test_conditional_table_is_zero_where_nothing_rejects(self) -> None:
        table = conditional_power_table(rejection_boundaries(5, ALPHA, McNemarTest.MID_P), 0.9)
        assert table[:5] == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert table[5] > 0.0

    def test_boundaries_refuse_a_negative_limit(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rejection_boundaries(-1, ALPHA, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_boundaries_refuse_a_bad_alpha(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rejection_boundaries(10, 0.0, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE


class TestTheDesignSize:
    """The answer someone schedules a run against."""

    def test_it_reports_both_sizes_and_their_gap(self) -> None:
        record = mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 40)
        assert record["first_reaching_pairs"] == 8
        assert record["durably_reaching_pairs"] == 11
        assert record["sawtooth_gap_pairs"] == 3

    def test_the_gap_is_zero_on_a_clean_crossing(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert record["first_reaching_pairs"] == record["durably_reaching_pairs"] == 21
        assert record["sawtooth_gap_pairs"] == 0

    def test_the_durable_size_really_does_hold_the_target(self) -> None:
        record = mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 40)
        durable = record["durably_reaching_pairs"]
        for pairs in range(durable, 41):
            assert unconditional_power(pairs, 1.0, 0.9, ALPHA, McNemarTest.MID_P) >= 0.80

    def test_the_first_size_does_not_hold_it(self) -> None:
        """The whole reason two numbers are reported rather than one."""
        record = mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 40)
        first = record["first_reaching_pairs"]
        assert unconditional_power(first, 1.0, 0.9, ALPHA, McNemarTest.MID_P) >= 0.80
        assert unconditional_power(first + 1, 1.0, 0.9, ALPHA, McNemarTest.MID_P) < 0.80

    def test_the_durable_size_can_equal_the_ceiling(self) -> None:
        # The downward walk must handle terminating immediately.
        record = mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 8)
        assert record["durably_reaching_pairs"] == 8
        assert record["sawtooth_gap_pairs"] == 0

    def test_the_reported_powers_are_the_curve_at_those_sizes(self) -> None:
        record = mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 40)
        assert record["power_at_first_reaching"] == pytest.approx(
            unconditional_power(record["first_reaching_pairs"], 1.0, 0.9, ALPHA, McNemarTest.MID_P)
        )
        assert record["power_at_durably_reaching"] == pytest.approx(
            unconditional_power(
                record["durably_reaching_pairs"], 1.0, 0.9, ALPHA, McNemarTest.MID_P
            )
        )

    def test_the_expected_discordant_count_is_the_durable_size_times_the_rate(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert record["expected_discordant_pairs"] == pytest.approx(
            record["durably_reaching_pairs"] * 0.5
        )

    def test_a_harder_target_never_needs_fewer_pairs(self) -> None:
        easy = mcnemar_design_size(0.5, 0.9, ALPHA, 0.70, McNemarTest.MID_P, 60)
        hard = mcnemar_design_size(0.5, 0.9, ALPHA, 0.90, McNemarTest.MID_P, 60)
        assert hard["durably_reaching_pairs"] >= easy["durably_reaching_pairs"]

    def test_the_two_variants_can_need_different_sizes(self) -> None:
        """Why the test is a required argument rather than a default."""
        mid = mcnemar_design_size(1.0, 0.9, ALPHA, 0.90, McNemarTest.MID_P, 40)
        exact = mcnemar_design_size(1.0, 0.9, ALPHA, 0.90, McNemarTest.EXACT, 40)
        assert mid["durably_reaching_pairs"] != exact["durably_reaching_pairs"]


class TestWhenNoSizeWillDo:
    """Both refusals, told apart by their remedy."""

    def test_it_refuses_when_the_target_is_never_reached(self) -> None:
        with pytest.raises(AppError) as excinfo:
            mcnemar_design_size(0.056, 0.70, ALPHA, 0.80, McNemarTest.EXACT, 20)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_DESIGN_SIZE_UNREACHABLE
        assert "reaches it at all" in str(excinfo.value)

    def test_it_refuses_when_the_target_is_reached_but_not_held(self) -> None:
        # Measured: at a 90:10 split with every pair discordant, eight pairs
        # clear 0.80 and nine fall back below it. A ceiling of nine therefore
        # has a crossing but cannot certify one.
        with pytest.raises(AppError) as excinfo:
            mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 9)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_DESIGN_SIZE_UNREACHABLE
        message = str(excinfo.value)
        assert "falls back below" in message
        assert "sizes from 8" in message

    def test_the_two_refusals_carry_different_explanations(self) -> None:
        with pytest.raises(AppError) as never:
            mcnemar_design_size(0.056, 0.70, ALPHA, 0.80, McNemarTest.EXACT, 20)
        with pytest.raises(AppError) as lost:
            mcnemar_design_size(1.0, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 9)
        assert str(never.value) != str(lost.value)


class TestTheRefusals:
    """Every parameter that can be wrong, refused by its own code."""

    def test_negative_pairs(self) -> None:
        with pytest.raises(AppError) as excinfo:
            unconditional_power(-1, 0.5, 0.9, ALPHA, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_bad_alpha(self, alpha: float) -> None:
        with pytest.raises(AppError) as excinfo:
            unconditional_power(10, 0.5, 0.9, alpha, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.01])
    def test_bad_discordant_rate(self, rate: float) -> None:
        with pytest.raises(AppError) as excinfo:
            unconditional_power(10, rate, 0.9, ALPHA, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_DISCORDANT_RATE_OUT_OF_RANGE

    def test_a_rate_of_one_is_allowed(self) -> None:
        assert unconditional_power(10, 1.0, 0.9, ALPHA, McNemarTest.MID_P) > 0.0

    @pytest.mark.parametrize("split", [0.5, 0.4, 0.0, 1.01])
    def test_bad_design_split(self, split: float) -> None:
        with pytest.raises(AppError) as excinfo:
            unconditional_power(10, 0.5, split, ALPHA, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_DESIGN_SPLIT_OUT_OF_RANGE

    def test_a_split_of_one_is_allowed(self) -> None:
        assert unconditional_power(10, 1.0, 1.0, ALPHA, McNemarTest.MID_P) > 0.0

    @pytest.mark.parametrize("target", [0.0, 1.0, -0.5, 2.0])
    def test_bad_target_power(self, target: float) -> None:
        with pytest.raises(AppError) as excinfo:
            mcnemar_design_size(0.5, 0.9, ALPHA, target, McNemarTest.EXACT, 20)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TARGET_POWER_OUT_OF_RANGE

    @pytest.mark.parametrize("ceiling", [0, -1, MAX_SEARCH_PAIRS + 1])
    def test_bad_search_ceiling(self, ceiling: int) -> None:
        with pytest.raises(AppError) as excinfo:
            mcnemar_design_size(0.5, 0.9, ALPHA, 0.8, McNemarTest.EXACT, ceiling)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SEARCH_CEILING_INVALID

    def test_the_ceiling_bound_is_inclusive(self) -> None:
        require_search_ceiling(1)
        require_search_ceiling(MAX_SEARCH_PAIRS)

    def test_the_ceiling_message_names_the_measured_cost(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_search_ceiling(MAX_SEARCH_PAIRS + 1)
        assert "10.60 s" in str(excinfo.value)


class TestTheRecord:
    """Shape, codec and the verdict it deliberately does not carry."""

    def test_the_instrument_names_itself(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert record["instrument"] == PowerInstrument.MCNEMAR_DESIGN_SIZE.value
        assert record["test"] == McNemarTest.MID_P.value

    def test_it_round_trips(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert decode_mcnemar_design_size(encode_mcnemar_design_size(record)) == record

    def test_decoding_refuses_another_instrument(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        payload = encode_mcnemar_design_size(record)
        payload["instrument"] = PowerInstrument.MCNEMAR.value
        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_design_size(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decoding_refuses_an_unknown_test(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        payload = encode_mcnemar_design_size(record)
        payload["test"] = "chi_squared"
        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_design_size(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TEST_UNKNOWN

    def test_it_carries_no_verdict(self) -> None:
        """It reports what a design needs; it classifies nothing."""
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert "verdict" not in record
        assert "verdict" not in encode_mcnemar_design_size(record)

    def test_the_encoding_carries_every_field(self) -> None:
        record = mcnemar_design_size(0.5, 0.9, ALPHA, 0.80, McNemarTest.MID_P, 60)
        assert set(encode_mcnemar_design_size(record)) == set(record)


class TestTheWeightsItAveragesOver:
    """The binomial vector, which both McNemar modules now share."""

    @pytest.mark.parametrize("trials", [0, 1, 5, 40])
    @pytest.mark.parametrize("probability", [0.05, 0.5, 0.93])
    def test_it_matches_the_closed_form(self, trials: int, probability: float) -> None:
        got = binomial_point_vector(trials, probability)
        want = [
            math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
            for k in range(trials + 1)
        ]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-300)

    @pytest.mark.parametrize("trials", [0, 1, 17, 200])
    def test_it_sums_to_one(self, trials: int) -> None:
        assert math.fsum(binomial_point_vector(trials, 0.37)) == pytest.approx(1.0, abs=1e-12)

    def test_it_survives_a_size_that_overflows_the_direct_form(self) -> None:
        """1,024 pairs is where the integer form raised OverflowError."""
        points = binomial_point_vector(1200, 0.5)
        assert len(points) == 1201
        assert math.fsum(points) == pytest.approx(1.0, abs=1e-9)
        assert all(math.isfinite(value) for value in points)
