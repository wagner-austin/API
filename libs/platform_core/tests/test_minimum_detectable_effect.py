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
    MIN_REPLICATES,
    PowerInstrument,
    PowerVerdict,
    mcnemar_power,
    paired_continuous_power,
    zero_failure_power,
)
from platform_core.power_distributions import McNemarTest
from platform_core.power_records import (
    decode_mcnemar_power,
    decode_paired_continuous_power,
    decode_zero_failure_power,
    encode_mcnemar_power,
    encode_paired_continuous_power,
    encode_zero_failure_power,
)


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
        assert record["verdict"] == PowerVerdict.NOT_TESTED.value

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
