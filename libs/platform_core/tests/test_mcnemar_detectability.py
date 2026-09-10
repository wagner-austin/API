"""How lopsided a split had to be, and the design where nothing is detectable.

WHAT IS CHECKED AGAINST WHAT. The power arithmetic is not compared to this
module's own output. Where the rejection region is the extreme split alone --
which is every case ``code-style`` publishes -- the power has a closed form,
``p**d + (1-p)**d``, and the tests assert against THAT, written out here in a
different expression from the implementation's log-space sum. A test that
asserted the implementation's own number would pass a rewrite that changed
the meaning.

The three published minimum detectable effects are pinned as literals for the
same reason ``test_published_comparisons`` pins its p-values: they are on a
wiki page, and a change that moves them should fail here rather than be
discovered by a reader.
"""

from __future__ import annotations

import math

import pytest

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.mcnemar_detectability import (
    mcnemar_detectable_effect,
    power_at_split,
    require_target_power,
)
from platform_core.minimum_detectable_effect import mcnemar_power
from platform_core.power_distributions import McNemarTest
from platform_core.power_records import (
    decode_mcnemar_detectable_effect,
    encode_mcnemar_detectable_effect,
)
from platform_core.power_types import PowerInstrument


def _extreme_only_power(discordant: int, split: float) -> float:
    """Closed-form power when ONLY the ``d:0`` split rejects.

    Independent expression of the same quantity: every discordant pair falls
    one way, or every one falls the other. Written as arithmetic rather than
    as a sum over a rejection set so it shares no structure with the code
    under test.

    Args:
        discordant: Number of discordant pairs.
        split: True probability one pair falls to the candidate.

    Returns:
        The rejection probability.
    """
    return split**discordant + (1.0 - split) ** discordant


class TestTheTargetPowerCheck:
    """A level that states no design goal is refused rather than clamped."""

    @pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.5])
    def test_a_target_outside_the_open_unit_interval_is_refused(self, target: float) -> None:
        """Both endpoints excluded: 0 is met by every design and 1 by none.

        Args:
            target: The out-of-range level under test.
        """
        with pytest.raises(AppError) as excinfo:
            require_target_power(target)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TARGET_POWER_OUT_OF_RANGE

    def test_an_ordinary_target_passes_silently(self) -> None:
        """The check is invisible when it holds."""
        require_target_power(0.80)


class TestThePowerArithmetic:
    """Checked against the closed form, not against this module."""

    @pytest.mark.parametrize("discordant", [5, 6, 13])
    @pytest.mark.parametrize("split", [0.5, 0.7, 0.9, 0.99])
    def test_extreme_only_power_matches_the_closed_form(
        self, discordant: int, split: float
    ) -> None:
        """The case every published stratum is in.

        Args:
            discordant: Number of discordant pairs.
            split: True split probability.
        """
        assert power_at_split(discordant, 0, split) == pytest.approx(
            _extreme_only_power(discordant, split)
        )

    def test_the_null_split_returns_about_alpha_worth_of_power(self) -> None:
        """At p = 1/2 the test rejects at its own size, not more.

        d=6, extreme-only: 2 of 64 outcomes reject, so 0.03125 -- which is
        the exact two-sided p at that split and below alpha, as it must be.
        """
        assert power_at_split(6, 0, 0.5) == pytest.approx(2 / 64)

    def test_a_certain_split_always_rejects(self) -> None:
        """At p = 1 every pair falls one way, which is the rejecting split."""
        assert power_at_split(6, 0, 1.0) == pytest.approx(1.0)

    def test_a_certain_split_the_other_way_also_always_rejects(self) -> None:
        """p = 0 is the mirror extreme, and the test is two-sided.

        Worth pinning rather than assuming: the rejection region is symmetric
        about the middle, so an effect that favours the BASELINE completely is
        exactly as detectable as one that favours the candidate completely. A
        one-sided implementation would return 0 here and still pass every
        test above.
        """
        assert power_at_split(6, 0, 0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("split", [0.0, 0.1, 0.3, 0.5])
    def test_power_is_symmetric_about_an_even_split(self, split: float) -> None:
        """``power(p) == power(1-p)``, which is what two-sided MEANS here.

        Args:
            split: The split whose mirror is compared against it.
        """
        assert power_at_split(6, 0, split) == pytest.approx(power_at_split(6, 0, 1.0 - split))

    def test_a_design_with_no_rejecting_split_has_zero_power(self) -> None:
        """``-1`` is the unfalsifiable design, and its power is 0 at ANY split.

        Returned rather than raised here: this function is the primitive, and
        a caller sizing a design wants the zero. The REFUSAL belongs one level
        up, where a detectable effect would otherwise be invented.
        """
        assert power_at_split(5, -1, 0.99) == 0.0

    def test_overlapping_tails_cannot_exceed_one(self) -> None:
        """THE DOUBLE-COUNT GUARD, and it bites ONLY under mid-p.

        The tails share an outcome when the boundary reaches the middle --
        even ``d`` with ``m == d/2`` -- which needs the perfectly even split
        to reject. Under the EXACT test that split's p is 1.0, so no legal
        alpha admits it and this case looks impossible. Under mid-p the even
        split takes the tie form instead (0.8125 at d=4), a permissive alpha
        reaches it, and [0,1,2] and [2,3,4] then share the outcome 2.

        Written first against the exact variant, where it wrongly passed by
        never reaching the condition at all.
        """
        boundary = mcnemar_power(4, 0.99, McNemarTest.MID_P)["most_balanced_rejecting_minority"]

        assert boundary == 2
        assert power_at_split(4, boundary, 0.7) == pytest.approx(1.0)

    def test_the_exact_variant_never_reaches_the_overlap(self) -> None:
        """The contrast that makes the guard's condition legible.

        Pinned because it is the reason the guard reads as dead code: from
        the exact test alone the overlap is unreachable at every alpha.
        """
        boundary = mcnemar_power(4, 0.99, McNemarTest.EXACT)["most_balanced_rejecting_minority"]

        assert boundary == 1
        assert boundary < 4 - boundary

    def test_a_large_discordant_count_does_not_overflow(self) -> None:
        """The regime that broke this module family once already.

        ``binomial_point_probability`` carries an OverflowError at 1,024
        pairs; the direct ``C(n,k) * p**k`` form here fails the same way for
        a different reason, which is why the implementation works in logs.
        """
        value = power_at_split(2627, 0, 0.9)

        assert math.isfinite(value)
        assert value == pytest.approx(_extreme_only_power(2627, 0.9))


class TestTheDetectableEffect:
    """The record, and the arithmetic that produced it."""

    def test_the_reported_split_actually_reaches_the_target(self) -> None:
        """The property that makes the number mean anything."""
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)

        assert record["achieved_power"] >= 0.80
        assert record["achieved_power"] == pytest.approx(
            _extreme_only_power(6, record["minimum_detectable_split"])
        )

    def test_the_split_is_the_smallest_that_reaches_it(self) -> None:
        """Not merely A split that works -- the boundary.

        A hair below the reported split must fall short, or the search
        returned something larger than necessary and every published effect
        would be overstated.
        """
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)
        just_below = record["minimum_detectable_split"] - 1e-9

        assert power_at_split(6, 0, just_below) < 0.80

    def test_the_net_and_the_rate_are_the_same_fact(self) -> None:
        """``d(2p-1)`` items, and that over n."""
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)
        split = record["minimum_detectable_split"]

        assert record["minimum_detectable_net_pairs"] == pytest.approx(6 * (2 * split - 1))
        assert record["minimum_detectable_rate_difference"] == pytest.approx(
            record["minimum_detectable_net_pairs"] / 226
        )

    def test_the_boundary_comes_from_the_falsifiability_record(self) -> None:
        """The two instruments must never disagree about which splits reject."""
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)
        floor = mcnemar_power(6, 0.05, McNemarTest.MID_P)

        assert (
            record["most_balanced_rejecting_minority"] == floor["most_balanced_rejecting_minority"]
        )

    def test_a_harder_target_needs_a_more_lopsided_split(self) -> None:
        """Monotonicity, which is what licenses the bisection."""
        easier = mcnemar_detectable_effect(6, 226, 0.05, 0.50, McNemarTest.MID_P)
        harder = mcnemar_detectable_effect(6, 226, 0.05, 0.95, McNemarTest.MID_P)

        assert harder["minimum_detectable_split"] > easier["minimum_detectable_split"]

    def test_the_variant_alone_can_change_the_answer(self) -> None:
        """d=5 is detectable under mid-p and unfalsifiable under exact.

        The starkest case for why ``test`` is required rather than defaulted:
        the same comparison yields a number under one variant and a refusal
        under the other.
        """
        under_mid_p = mcnemar_detectable_effect(5, 90, 0.05, 0.80, McNemarTest.MID_P)

        assert under_mid_p["minimum_detectable_net_pairs"] == pytest.approx(4.5635, abs=1e-4)
        with pytest.raises(AppError):
            mcnemar_detectable_effect(5, 90, 0.05, 0.80, McNemarTest.EXACT)


class TestThePublishedFigures:
    """The three effects on `code-style-guard-pass-instrument-limits`.

    Pinned as literals so a change that moves a published number fails here
    rather than being found by a reader of the page.
    """

    @pytest.mark.parametrize(
        ("discordant", "total", "expected_pp"),
        [(6, 226, 2.4610), (5, 90, 5.0706), (5, 49, 9.3133)],
    )
    def test_each_stratum_reproduces(self, discordant: int, total: int, expected_pp: float) -> None:
        """Mid-p, alpha 0.05, 80% power, conditioned on the observed d.

        Args:
            discordant: The stratum's discordant count.
            total: Items both arms answered.
            expected_pp: The effect in percentage points.
        """
        record = mcnemar_detectable_effect(discordant, total, 0.05, 0.80, McNemarTest.MID_P)

        assert 100 * record["minimum_detectable_rate_difference"] == pytest.approx(
            expected_pp, abs=5e-4
        )

    def test_the_two_five_pair_strata_share_a_net_and_differ_only_by_denominator(
        self,
    ) -> None:
        """Why 5.07 and 9.31 are the same measurement read over different n.

        Both strata produced d=5, so the detectable NET is identical; the
        percentage points differ only because 90 items and 49 items divide it
        differently. Worth pinning because the two numbers look independent
        on the page and are not.
        """
        ninety = mcnemar_detectable_effect(5, 90, 0.05, 0.80, McNemarTest.MID_P)
        forty_nine = mcnemar_detectable_effect(5, 49, 0.05, 0.80, McNemarTest.MID_P)

        assert ninety["minimum_detectable_net_pairs"] == pytest.approx(
            forty_nine["minimum_detectable_net_pairs"]
        )


class TestTheRefusals:
    """Every way a caller can be told no, and why it is not a returned value."""

    def test_a_design_where_nothing_is_detectable_raises(self) -> None:
        """Not a maximal effect, which a reader would take as an answer."""
        with pytest.raises(AppError) as excinfo:
            mcnemar_detectable_effect(5, 90, 0.05, 0.80, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TARGET_UNREACHABLE
        assert "more discordant pairs are the remedy" in excinfo.value.message.lower()

    @pytest.mark.parametrize(("discordant", "total"), [(-1, 10), (5, -1)])
    def test_a_negative_count_is_refused(self, discordant: int, total: int) -> None:
        """Neither count can be negative.

        Args:
            discordant: Discordant pairs under test.
            total: Total pairs under test.
        """
        with pytest.raises(AppError) as excinfo:
            mcnemar_detectable_effect(discordant, total, 0.05, 0.80, McNemarTest.MID_P)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_more_disagreement_than_comparison_is_refused(self) -> None:
        """The discordant pairs are a SUBSET, not a separate sample.

        Without this, a transposed call would divide a net by the smaller
        number and report an effect several times its true size.
        """
        with pytest.raises(AppError) as excinfo:
            mcnemar_detectable_effect(90, 5, 0.05, 0.80, McNemarTest.MID_P)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID
        assert "subset" in excinfo.value.message

    def test_a_bad_alpha_is_refused(self) -> None:
        """Delegated to the shared validator rather than re-checked."""
        with pytest.raises(AppError) as excinfo:
            mcnemar_detectable_effect(6, 226, 1.5, 0.80, McNemarTest.MID_P)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    def test_a_bad_target_power_is_refused(self) -> None:
        """And with its OWN code, not the confidence one."""
        with pytest.raises(AppError) as excinfo:
            mcnemar_detectable_effect(6, 226, 0.05, 1.0, McNemarTest.MID_P)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TARGET_POWER_OUT_OF_RANGE


class TestTheJSONBoundary:
    """Round trip, and the instrument this record refuses to be mistaken for."""

    def test_the_record_survives_a_round_trip(self) -> None:
        """Every field, including the split that carries the precision."""
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)

        assert decode_mcnemar_detectable_effect(encode_mcnemar_detectable_effect(record)) == record

    def test_decode_rejects_a_mismatched_instrument(self) -> None:
        """A falsifiability payload read as a detectability one.

        The two share ``discordant_pairs`` and ``alpha``, so a decoder that
        merely checked the field names would accept the wrong record and
        report a floor as an effect.
        """
        obj = encode_mcnemar_detectable_effect(
            mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)
        )
        obj["instrument"] = PowerInstrument.MCNEMAR.value

        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_detectable_effect(obj)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_an_unknown_test_variant(self) -> None:
        """The variant is load-bearing, so it is checked on the way in."""
        obj = encode_mcnemar_detectable_effect(
            mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)
        )
        obj["test"] = "chi_square"

        with pytest.raises(AppError) as excinfo:
            decode_mcnemar_detectable_effect(obj)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TEST_UNKNOWN

    def test_the_record_carries_no_verdict(self) -> None:
        """The fourth abstention in this module, pinned rather than stated.

        If a ``verdict`` field ever returns to this record, this fails. The
        ``.values()`` half catches one smuggled in under a renamed key.
        """
        record = mcnemar_detectable_effect(6, 226, 0.05, 0.80, McNemarTest.MID_P)

        assert "verdict" not in record
        assert "TESTED" not in record.values()
        assert "NOT_TESTED" not in record.values()
