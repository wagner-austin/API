"""Tests for the shared minimum-detectable-effect helper.

Every test calls the real functions. There is nothing to mock -- the
module has no clock, network, filesystem or randomness -- so a fake here
would be testing itself.

Numbers asserted here come from outside this module: the campaign's own
published MDE ladder, and the discordant-split boundaries measured on
code-style. A test seeded from the implementation's output would pass
against any arithmetic, including wrong arithmetic.

The distributions themselves are covered in
:mod:`tests.test_power_distributions`.
"""

from __future__ import annotations

import math

import pytest

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError
from platform_core.minimum_detectable_effect import (
    MAX_SEARCH_REPLICATES,
    MIN_REPLICATES,
    mcnemar_power,
    paired_continuous_power,
    rate_floor_power,
    required_replicates,
    zero_failure_power,
)
from platform_core.power_distributions import McNemarTest, t_critical
from platform_core.power_records import (
    decode_mcnemar_power,
    decode_paired_continuous_power,
    decode_rate_floor_power,
    decode_required_replicates,
    decode_zero_failure_power,
    encode_mcnemar_power,
    encode_paired_continuous_power,
    encode_rate_floor_power,
    encode_required_replicates,
    encode_zero_failure_power,
)
from platform_core.power_types import PowerInstrument, PowerVerdict


class TestPairedContinuousPower:
    """The ladder instrument, on the campaign's own published numbers."""

    def test_reproduces_the_ablation_ladder_mde(self) -> None:
        # The receipt in task 9d34f1bb: at n=3 the MDE fell
        # 2.79 -> 1.78 -> 0.57 -> 0.22 pp as sd fell
        # 1.1235 -> 0.7165 -> 0.2278 -> 0.0870. Those are published
        # numbers computed elsewhere; this asserts the helper agrees
        # with them rather than with itself.
        for sample_sd, published_mde in (
            (1.1235, 2.79),
            (0.7165, 1.78),
            (0.2278, 0.57),
            (0.0870, 0.22),
        ):
            # Three differences with exactly this sample sd and zero mean.
            offset = sample_sd
            differences = [-offset, 0.0, offset]
            record = paired_continuous_power(differences, 0.05, 1.0)
            assert record["sample_sd"] == pytest.approx(sample_sd, rel=1e-12)
            assert record["minimum_detectable_effect"] == pytest.approx(published_mde, abs=0.005)

    def test_carries_the_inputs_that_make_the_number_reproducible(self) -> None:
        record = paired_continuous_power([-1.0, 0.0, 1.0], 0.05, 5.0)
        assert record["instrument"] == PowerInstrument.PAIRED_CONTINUOUS.value
        assert record["replicates"] == 3
        assert record["degrees_of_freedom"] == 2
        assert record["alpha"] == 0.05
        assert record["mean_difference"] == pytest.approx(0.0)
        assert record["sample_sd"] == pytest.approx(1.0)
        assert record["t_critical"] == pytest.approx(4.302653, abs=1e-5)
        assert record["minimum_detectable_effect"] == pytest.approx(
            4.302653 / math.sqrt(3.0), abs=1e-5
        )

    def test_verdict_is_tested_when_the_mde_clears_the_bar(self) -> None:
        record = paired_continuous_power([-1.0, 0.0, 1.0], 0.05, 5.0)
        assert record["verdict"] == PowerVerdict.TESTED.value

    def test_verdict_is_not_tested_when_the_mde_exceeds_the_bar(self) -> None:
        # The ablation's own failure: a 0.5 pp effect against a 2.48 MDE.
        record = paired_continuous_power([-1.0, 0.0, 1.0], 0.05, 0.5)
        assert record["verdict"] == PowerVerdict.NOT_TESTED.value

    def test_refuses_fewer_than_three_replicates(self) -> None:
        with pytest.raises(AppError) as excinfo:
            paired_continuous_power([1.0, 2.0], 0.05, 1.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TOO_FEW_REPLICATES
        assert MIN_REPLICATES == 3

    def test_refuses_a_bad_alpha(self) -> None:
        with pytest.raises(AppError) as excinfo:
            paired_continuous_power([-1.0, 0.0, 1.0], 0.0, 1.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    @pytest.mark.parametrize("effect", [0.0, -1.0])
    def test_refuses_a_non_positive_effect_of_interest(self, effect: float) -> None:
        with pytest.raises(AppError) as excinfo:
            paired_continuous_power([-1.0, 0.0, 1.0], 0.05, effect)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_EFFECT_OF_INTEREST_INVALID


class TestMcNemarPower:
    """The paired-binary instrument, and the bug it was written against."""

    def test_five_discordant_pairs_can_never_reject(self) -> None:
        # code-style's measured failure: two of three published strata
        # sat at d=5, where the smallest attainable two-sided exact p is
        # 0.0625 -- above 0.05 however the pairs fall.
        record = mcnemar_power(5, 0.05, McNemarTest.EXACT)
        assert record["smallest_attainable_p"] == pytest.approx(0.0625)
        assert record["can_ever_reject"] is False
        assert record["most_balanced_rejecting_minority"] == -1

    def test_carries_no_power_verdict_because_it_answers_a_different_question(
        self,
    ) -> None:
        # THE ASYMMETRY, PINNED. The other two instruments take the effect
        # worth acting on and answer "could this detect it?". McNemar, from
        # discordant pairs and alpha alone, can only answer "can any split
        # reject?" -- falsifiability, not practical detectability.
        #
        # Real disagreement, from code-style: at d=6 mid-p CAN reject (at a
        # perfect 6:0), while that project's classification against its +5 pp
        # threshold was NOT TESTED. A PowerVerdict here would have been true
        # of the instrument and false about the world.
        #
        # can_ever_reject carries the truth as a boolean that cannot be
        # mistaken for a classification. If a verdict field ever returns,
        # this fails.
        record = mcnemar_power(6, 0.05, McNemarTest.MID_P)
        assert record["can_ever_reject"] is True
        assert "verdict" not in record
        assert PowerVerdict.TESTED.value not in record.values()

    def test_six_discordant_pairs_reject_only_at_the_extreme(self) -> None:
        record = mcnemar_power(6, 0.05, McNemarTest.EXACT)
        assert record["smallest_attainable_p"] == pytest.approx(0.03125)
        assert record["can_ever_reject"] is True
        assert record["most_balanced_rejecting_minority"] == 0

    def test_ten_discordant_pairs_reject_at_a_non_trivial_split(self) -> None:
        # THE ANTI-BUG TEST. A search returning the FIRST rejecting split
        # yields the trivial 0 here, because 0 rejects and is met first.
        # The informative answer is 1: a 9:1 split still rejects at
        # alpha=0.05 (p = 0.0215), and 8:2 does not (p = 0.109).
        record = mcnemar_power(10, 0.05, McNemarTest.EXACT)
        assert record["most_balanced_rejecting_minority"] == 1
        assert record["most_balanced_rejecting_minority"] != 0

    def test_the_reported_test_changes_the_rejection_boundary(self) -> None:
        # BUG 2 FROM THE SWEEP: an MDE computed against the exact test,
        # published on a page that reports mid-p, describes a test nobody
        # ran. Measured at d=29: exact rejects at minority <= 8, mid-p at
        # <= 9. The helper takes the test as a parameter for this reason,
        # and this asserts the two genuinely disagree.
        exact = mcnemar_power(29, 0.05, McNemarTest.EXACT)
        mid_p = mcnemar_power(29, 0.05, McNemarTest.MID_P)
        assert exact["most_balanced_rejecting_minority"] == 8
        assert mid_p["most_balanced_rejecting_minority"] == 9
        assert exact["test"] == McNemarTest.EXACT.value
        assert mid_p["test"] == McNemarTest.MID_P.value

    def test_the_boundary_also_moves_at_sixty_discordant_pairs(self) -> None:
        # The second measured pair from the same sweep: 21 -> 22.
        assert mcnemar_power(60, 0.05, McNemarTest.EXACT)["most_balanced_rejecting_minority"] == 21
        assert mcnemar_power(60, 0.05, McNemarTest.MID_P)["most_balanced_rejecting_minority"] == 22

    def test_zero_discordant_pairs_cannot_reject(self) -> None:
        record = mcnemar_power(0, 0.05, McNemarTest.EXACT)
        assert record["smallest_attainable_p"] == 1.0
        assert record["can_ever_reject"] is False
        assert record["most_balanced_rejecting_minority"] == -1

    def test_refuses_negative_discordant_pairs(self) -> None:
        with pytest.raises(AppError) as excinfo:
            mcnemar_power(-1, 0.05, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_refuses_a_bad_alpha(self) -> None:
        with pytest.raises(AppError) as excinfo:
            mcnemar_power(10, 1.0, McNemarTest.EXACT)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE


class TestZeroFailurePower:
    """The instrument for a rate observed to be exactly zero."""

    def test_matches_the_clopper_pearson_closed_form(self) -> None:
        record = zero_failure_power(28, 0.95, 0.10)
        assert record["upper_bound"] == pytest.approx(1.0 - math.pow(0.05, 1.0 / 28.0), rel=1e-12)

    def test_approximates_the_rule_of_three(self) -> None:
        # The familiar 3/n rule for zero failures at 95% confidence.
        record = zero_failure_power(300, 0.95, 0.05)
        assert record["upper_bound"] == pytest.approx(3.0 / 300.0, rel=0.02)

    def test_verdict_is_not_tested_when_the_bound_exceeds_the_bar(self) -> None:
        # mi-cu128's shape: a handful of identical records bounds the
        # divergence rate very loosely, however clean the observation.
        record = zero_failure_power(3, 0.95, 0.01)
        assert record["upper_bound"] > 0.5
        assert record["verdict"] == PowerVerdict.NOT_TESTED.value

    def test_verdict_is_tested_when_the_bound_clears_the_bar(self) -> None:
        record = zero_failure_power(1000, 0.95, 0.10)
        assert record["verdict"] == PowerVerdict.TESTED.value

    def test_refuses_non_positive_trials(self) -> None:
        with pytest.raises(AppError) as excinfo:
            zero_failure_power(0, 0.95, 0.1)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0])
    def test_refuses_a_bad_confidence(self, confidence: float) -> None:
        with pytest.raises(AppError) as excinfo:
            zero_failure_power(10, confidence, 0.1)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_CONFIDENCE_OUT_OF_RANGE

    def test_refuses_a_non_positive_rate_of_interest(self) -> None:
        with pytest.raises(AppError) as excinfo:
            zero_failure_power(10, 0.95, 0.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_EFFECT_OF_INTEREST_INVALID


class TestRequiredReplicates:
    """The projection instrument, against published Student-t critical values.

    Every expectation below is derived from a t table and closed-form
    arithmetic, never from this module's own output. The differences
    ``[-1.0, 0.0, 1.0]`` have a sample sd of exactly 1.0 (mean 0, squared
    deviations 1 + 0 + 1, divided by 2 df), so ``MDE(n) = t_crit(n - 1) /
    sqrt(n)`` and the ladder is readable straight off the table:

        n = 3   4.302653 / 1.732051 = 2.484519
        n = 4   3.182446 / 2.000000 = 1.591223
        n = 5   2.776445 / 2.236068 = 1.241664
        n = 6   2.570582 / 2.449490 = 1.049435
        n = 7   2.446912 / 2.645751 = 0.924849
    """

    def test_finds_the_first_n_that_clears_the_effect_of_interest(self) -> None:
        # delta = 1.0 sits between MDE(6) = 1.0494 and MDE(7) = 0.9248,
        # so 7 is the smallest adequate design and 4 more runs are owed.
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0)
        assert record["required_replicates"] == 7
        assert record["observed_replicates"] == 3
        assert record["additional_replicates"] == 4

    def test_lands_on_the_lower_rung_when_the_effect_is_larger(self) -> None:
        # delta = 1.6 clears MDE(4) = 1.5912 but not MDE(3) = 2.4845.
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 1.6)
        assert record["required_replicates"] == 4
        assert record["additional_replicates"] == 1

    def test_reports_the_floor_when_the_pilot_was_already_adequate(self) -> None:
        # delta = 3.0 exceeds MDE(3) = 2.4845, so nothing more is owed.
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 3.0)
        assert record["required_replicates"] == MIN_REPLICATES
        assert record["additional_replicates"] == 0

    def test_floors_additional_at_zero_when_observed_exceeds_required(self) -> None:
        # Five differences, same sd of 1.0: squared deviations 1+1+0+1+1 = 4
        # over 4 df. Required stays 3, so the surplus must not go negative.
        record = required_replicates([-1.0, -1.0, 0.0, 1.0, 1.0], 0.05, 3.0)
        assert record["observed_replicates"] == 5
        assert record["required_replicates"] == MIN_REPLICATES
        assert record["additional_replicates"] == 0

    def test_a_spreadless_pilot_needs_only_the_floor(self) -> None:
        # Identical differences have sd 0, so the MDE is 0 at every n and
        # the smallest legal design already resolves any positive effect.
        record = required_replicates([5.0, 5.0, 5.0], 0.05, 1e-6)
        assert record["observed_sample_sd"] == 0.0
        assert record["required_replicates"] == MIN_REPLICATES

    def test_the_returned_count_is_the_boundary_and_not_merely_sufficient(self) -> None:
        # The contract is "FEWEST", which a merely-sufficient answer would
        # also satisfy. Pin both sides: n resolves the effect and n - 1 does
        # not. At sd 1.0 the MDE is t_crit(n - 1) / sqrt(n).
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0)
        needed = record["required_replicates"]
        assert record["observed_sample_sd"] == 1.0
        assert needed > MIN_REPLICATES
        assert t_critical(needed - 1, 0.05) / math.sqrt(needed) <= 1.0
        assert t_critical(needed - 2, 0.05) / math.sqrt(needed - 1) > 1.0

    def test_publishes_the_instrument_it_projects_for(self) -> None:
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0)
        assert record["instrument"] == PowerInstrument.PAIRED_CONTINUOUS.value

    def test_refuses_a_pilot_below_the_replicate_floor(self) -> None:
        with pytest.raises(AppError) as excinfo:
            required_replicates([1.0, 2.0], 0.05, 1.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TOO_FEW_REPLICATES

    def test_refuses_an_alpha_outside_the_unit_interval(self) -> None:
        with pytest.raises(AppError) as excinfo:
            required_replicates([-1.0, 0.0, 1.0], 1.5, 1.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    def test_refuses_a_non_positive_effect_of_interest(self) -> None:
        with pytest.raises(AppError) as excinfo:
            required_replicates([-1.0, 0.0, 1.0], 0.05, 0.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_EFFECT_OF_INTEREST_INVALID

    def test_refuses_to_certify_a_design_replicates_cannot_rescue(self) -> None:
        # At sd 1.0 the MDE is still about 0.0196 at the search ceiling, so
        # an effect of interest of 1e-9 is unreachable by replication. The
        # module raises rather than returning the ceiling, because a caller
        # would otherwise schedule 10,000 runs that still would not resolve it.
        with pytest.raises(AppError) as excinfo:
            required_replicates([-1.0, 0.0, 1.0], 0.05, 1e-9)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_REQUIRED_REPLICATES_UNREACHABLE
        assert str(MAX_SEARCH_REPLICATES) in excinfo.value.message


class TestRateFloorPower:
    """The pass-rate instrument, on TankpitBot's own `make audit` counts.

    Expectations come from the exact binomial, computed independently as a sum
    of binomial coefficients (`sum(C(n,i) p^i (1-p)^(n-i) for i in k..n)`) and
    cross-checked against this module's beta form. They are NOT read back from
    this module's output.
    """

    def test_a_flawless_six_of_six_does_not_clear_an_085_floor(self) -> None:
        # 0.85 ** 6 = 0.37714951... A perfect record this short arises 37.7%
        # of the time when the true rate is exactly the floor, so the gate's
        # PASS carries no evidence the claim exceeds it.
        record = rate_floor_power(6, 6, 0.85, 0.05)
        assert record["p_value"] == pytest.approx(0.85**6, abs=1e-12)
        assert record["rate_exceeds_floor"] is False
        assert record["design_can_clear_floor"] is False
        assert record["observed_rate"] == 1.0

    def test_a_large_flawless_record_clears_the_floor(self) -> None:
        # 18,649 of 18,649 -- the capacity claim. A comb-sum is infeasible here,
        # which is why the beta form is used.
        record = rate_floor_power(18_649, 18_649, 0.85, 0.05)
        assert record["p_value"] < 1e-12
        assert record["rate_exceeds_floor"] is True
        assert record["design_can_clear_floor"] is True

    def test_a_rate_close_to_the_floor_is_not_distinguishable_from_it(self) -> None:
        # walk: 204/232 = 87.9%, above 0.85 by eye and not separable from it.
        record = rate_floor_power(204, 232, 0.85, 0.05)
        assert record["p_value"] == pytest.approx(0.1215724091, abs=1e-9)
        assert record["rate_exceeds_floor"] is False

    def test_a_rate_far_above_the_floor_is_separable(self) -> None:
        # homing: 487/522 = 93.3%.
        record = rate_floor_power(487, 522, 0.85, 0.05)
        assert record["p_value"] == pytest.approx(4.1e-9, rel=0.05)
        assert record["rate_exceeds_floor"] is True

    def test_zero_successes_cannot_be_evidence_against_the_floor(self) -> None:
        # Every outcome is at least as good as the worst one, so the one-sided
        # p is exactly 1. The beta form does not cover k = 0.
        record = rate_floor_power(0, 40, 0.85, 0.05)
        assert record["p_value"] == 1.0
        assert record["observed_rate"] == 0.0
        assert record["rate_exceeds_floor"] is False

    def test_publishes_the_shortest_flawless_record_that_could_pass(self) -> None:
        # 0.85 ** 19 = 0.04559 <= 0.05 < 0.05386 = 0.85 ** 18. Below 19 trials
        # NO outcome can clear the floor, so the gate cannot be passed on
        # evidence however clean the record.
        record = rate_floor_power(6, 6, 0.85, 0.05)
        assert record["perfect_record_trials"] == 19
        assert 0.85**19 <= 0.05
        assert 0.85**18 > 0.05

    def test_it_carries_no_power_verdict(self) -> None:
        """The abstention, pinned in the same change that made it.

        This instrument SHIPPED with a ``PowerVerdict`` for four hours on
        2026-09-09. That enum means "the design could resolve an effect worth
        acting on"; this record set it from ``p_value <= alpha``, which is
        whether THIS RESULT is separable from the floor. On this question the
        two are anti-correlated -- 120 of 200 against an 0.85 floor is a
        decisively measured failure and read NOT_TESTED, while 20 of 20, able
        to resolve only a perfect record, read TESTED.

        The ``.values()`` half catches a verdict reintroduced under a renamed
        key, which ``"verdict" not in record`` alone would miss.
        """
        record = rate_floor_power(120, 200, 0.85, 0.05)
        assert "verdict" not in record
        assert PowerVerdict.TESTED.value not in record.values()
        assert PowerVerdict.NOT_TESTED.value not in record.values()

    def test_significance_and_resolution_are_reported_separately(self) -> None:
        """The two questions the single verdict used to conflate.

        120/200 is decisively BELOW the floor: the design could have cleared
        it (200 trials against 19 needed) and the rate does not. 20/20 is the
        mirror: the rate clears alpha, and the design could never have
        resolved a single failure.
        """
        decisive_failure = rate_floor_power(120, 200, 0.85, 0.05)
        assert decisive_failure["design_can_clear_floor"] is True
        assert decisive_failure["rate_exceeds_floor"] is False

        flimsy_pass = rate_floor_power(20, 20, 0.85, 0.05)
        assert flimsy_pass["rate_exceeds_floor"] is True
        assert flimsy_pass["design_can_clear_floor"] is True

        too_short = rate_floor_power(6, 6, 0.85, 0.05)
        assert too_short["design_can_clear_floor"] is False

    def test_publishes_the_instrument_it_used(self) -> None:
        record = rate_floor_power(6, 6, 0.85, 0.05)
        assert record["instrument"] == PowerInstrument.RATE_FLOOR.value

    def test_refuses_a_trial_count_that_is_not_positive(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rate_floor_power(0, 0, 0.85, 0.05)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_refuses_more_successes_than_trials(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rate_floor_power(7, 6, 0.85, 0.05)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_refuses_a_negative_success_count(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rate_floor_power(-1, 6, 0.85, 0.05)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_refuses_a_floor_outside_the_unit_interval(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rate_floor_power(6, 6, 1.0, 0.05)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_RATE_FLOOR_OUT_OF_RANGE

    def test_refuses_an_alpha_outside_the_unit_interval(self) -> None:
        with pytest.raises(AppError) as excinfo:
            rate_floor_power(6, 6, 0.85, 0.0)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE


class TestRoundTrips:
    """Encode/decode, and the validation that makes decode meaningful."""

    def test_paired_continuous_round_trip_is_lossless(self) -> None:
        record = paired_continuous_power([-1.0, 0.5, 1.0], 0.05, 2.0)
        assert decode_paired_continuous_power(encode_paired_continuous_power(record)) == record

    def test_mcnemar_round_trip_is_lossless(self) -> None:
        record = mcnemar_power(12, 0.05, McNemarTest.MID_P)
        assert decode_mcnemar_power(encode_mcnemar_power(record)) == record

    def test_zero_failure_round_trip_is_lossless(self) -> None:
        record = zero_failure_power(40, 0.95, 0.2)
        assert decode_zero_failure_power(encode_zero_failure_power(record)) == record

    def test_required_replicates_survives_a_round_trip(self) -> None:
        record = required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0)
        assert decode_required_replicates(encode_required_replicates(record)) == record

    def test_decode_rejects_a_mismatched_instrument_on_required_replicates(self) -> None:
        payload = encode_required_replicates(required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0))
        payload["instrument"] = PowerInstrument.MCNEMAR.value
        with pytest.raises(AppError) as excinfo:
            decode_required_replicates(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_a_missing_required_replicates_field(self) -> None:
        payload = encode_required_replicates(required_replicates([-1.0, 0.0, 1.0], 0.05, 1.0))
        del payload["required_replicates"]
        with pytest.raises(JSONTypeError):
            decode_required_replicates(payload)

    def test_rate_floor_survives_a_round_trip(self) -> None:
        record = rate_floor_power(204, 232, 0.85, 0.05)
        assert decode_rate_floor_power(encode_rate_floor_power(record)) == record

    def test_decode_rejects_a_mismatched_instrument_on_rate_floor(self) -> None:
        payload = encode_rate_floor_power(rate_floor_power(6, 6, 0.85, 0.05))
        payload["instrument"] = PowerInstrument.MCNEMAR.value
        with pytest.raises(AppError) as excinfo:
            decode_rate_floor_power(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_a_missing_rate_floor_field(self) -> None:
        payload = encode_rate_floor_power(rate_floor_power(6, 6, 0.85, 0.05))
        del payload["p_value"]
        with pytest.raises(JSONTypeError):
            decode_rate_floor_power(payload)

    def test_decode_rejects_an_unknown_verdict(self) -> None:
        payload = encode_paired_continuous_power(
            paired_continuous_power([-1.0, 0.0, 1.0], 0.05, 2.0)
        )
        payload["verdict"] = "PROBABLY_FINE"
        with pytest.raises(AppError) as excinfo:
            decode_paired_continuous_power(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_VERDICT_UNKNOWN

    def test_decode_rejects_an_unknown_mcnemar_test(self) -> None:
        # Its own code, not POWER_INSTRUMENT_UNKNOWN: this is the right
        # record type carrying a rejection region nobody computed against,
        # which is the failure that moved published numbers by 0.2 pp.
        payload = encode_mcnemar_power(mcnemar_power(12, 0.05, McNemarTest.MID_P))
        payload["test"] = "chi_squared"
        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_power(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TEST_UNKNOWN

    def test_decode_rejects_a_mismatched_instrument(self) -> None:
        payload = encode_mcnemar_power(mcnemar_power(12, 0.05, McNemarTest.EXACT))
        payload["instrument"] = PowerInstrument.PAIRED_CONTINUOUS.value
        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_power(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_a_mismatched_instrument_on_zero_failure(self) -> None:
        payload = encode_zero_failure_power(zero_failure_power(40, 0.95, 0.2))
        payload["instrument"] = PowerInstrument.MCNEMAR.value
        with pytest.raises(AppError) as excinfo:
            decode_zero_failure_power(payload)
        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_a_missing_field(self) -> None:
        payload = encode_zero_failure_power(zero_failure_power(40, 0.95, 0.2))
        del payload["trials"]
        with pytest.raises(JSONTypeError):
            decode_zero_failure_power(payload)
