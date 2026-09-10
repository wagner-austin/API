"""The floor an OBSERVED net difference has to clear to mean anything.

Its own module rather than more of :mod:`tests.test_minimum_detectable_effect`,
which was already at 504 lines against this repository's 600-line ceiling.

WHERE THE ASSERTED NUMBERS COME FROM, since a test seeded from the
implementation's own output passes against wrong arithmetic too:

* The four ``(test, alpha) -> floor`` pairs were derived independently by a
  second session against the real module on 2026-09-09, and separately by
  hand from ``2 * 0.5 ** d`` and ``0.5 ** d``, before either was written here.
* The code-style row (net 4 of 875 at d=54) and the cartridge rows (net 2 and
  net 1 of 32) are published numbers that this instrument exists because of.

THE PROPERTY TEST IS THE LOAD-BEARING ONE. Everything else checks a value;
:meth:`TestTheFloorIsWhereTheProofSaysItIs.test_no_arrangement_beats_the_extreme_split`
checks the LEMMA the instrument is built on, exhaustively, and it is here
because a sibling gate was built on the wrong end of this surface on
2026-09-09 -- the margin needed to reject is not monotone in the discordant
count, so nothing about this shape may be assumed from a direction.
"""

from __future__ import annotations

import pytest

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError
from platform_core.minimum_detectable_effect import (
    MAX_SEARCH_NET_DIFFERENCE,
    net_difference_power,
    smallest_resolvable_net_difference,
)
from platform_core.power_distributions import McNemarTest, mcnemar_p
from platform_core.power_records import (
    decode_net_difference_power,
    encode_net_difference_power,
)
from platform_core.power_types import PowerInstrument, PowerVerdict


class TestSmallestResolvableNetDifference:
    """The number to quote beside a table of paired comparisons."""

    @pytest.mark.parametrize(
        ("test", "alpha", "expected"),
        [
            (McNemarTest.EXACT, 0.05, 6),
            (McNemarTest.MID_P, 0.05, 5),
            (McNemarTest.EXACT, 0.01, 8),
            (McNemarTest.MID_P, 0.01, 7),
        ],
    )
    def test_it_matches_the_hand_computed_floor(
        self, test: McNemarTest, alpha: float, expected: int
    ) -> None:
        """Exact's best case is ``2 * 0.5 ** d`` and mid-p's is ``0.5 ** d``.

        At alpha 0.05 that is 0.0625 against 0.03125 at d=5 -- which is the
        whole reason the variant is a parameter here and never a default.
        """
        assert smallest_resolvable_net_difference(alpha, test) == expected

    def test_the_two_variants_differ_by_exactly_one_item(self) -> None:
        """The off-by-one that sends a floor to the wrong lane.

        Stated as its own test because it was published to five sessions as
        'six items regardless' before anyone named the variant.
        """
        for alpha in (0.05, 0.01, 0.1):
            exact = smallest_resolvable_net_difference(alpha, McNemarTest.EXACT)
            mid_p = smallest_resolvable_net_difference(alpha, McNemarTest.MID_P)

            assert exact - mid_p == 1

    def test_the_floor_is_the_smallest_that_actually_rejects(self) -> None:
        """Ties the returned integer back to the p-values it is derived from,
        so the search cannot drift from the distribution it searches.
        """
        for test in (McNemarTest.EXACT, McNemarTest.MID_P):
            floor = smallest_resolvable_net_difference(0.05, test)

            assert mcnemar_p(0, floor, test) <= 0.05
            assert mcnemar_p(0, floor - 1, test) > 0.05

    def test_an_alpha_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            smallest_resolvable_net_difference(1.0, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE

    def test_an_unreachable_alpha_raises_rather_than_returning_the_ceiling(self) -> None:
        """Returning ``MAX_SEARCH_NET_DIFFERENCE`` would hand a caller a floor
        no arrangement actually meets, which is the failure mode this whole
        module exists to end.
        """
        with pytest.raises(AppError) as excinfo:
            smallest_resolvable_net_difference(1e-320, McNemarTest.MID_P)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE
        assert str(MAX_SEARCH_NET_DIFFERENCE) in excinfo.value.message


class TestNetDifferencePower:
    """Whether one observed margin could ever have been significant."""

    def test_the_retracted_cartridge_headline_could_never_have_been_significant(self) -> None:
        """+0.052 accuracy on 32 items is 1.7 items, published as beating
        three retrievers and retracted the same day. Rounded to 2 in favour
        of the claim; it still fails.
        """
        record = net_difference_power(2, 32, 0.05, McNemarTest.EXACT)

        assert record["net_could_ever_be_significant"] is False
        assert record["smallest_resolvable_net_difference"] == 6
        assert record["best_case_p"] == pytest.approx(0.5)

    def test_the_code_style_mypy_row_fails_this_while_passing_the_other_check(self) -> None:
        """d=54 makes ``can_ever_reject`` true and correct. A net of 4 still
        cannot be significant at any d, which is the gap between the two
        questions and the reason both exist.
        """
        record = net_difference_power(4, 875, 0.05, McNemarTest.MID_P)

        assert record["net_could_ever_be_significant"] is False
        assert record["smallest_resolvable_net_difference"] == 5

    def test_a_margin_at_the_floor_is_resolvable(self) -> None:
        record = net_difference_power(6, 32, 0.05, McNemarTest.EXACT)

        assert record["net_could_ever_be_significant"] is True
        assert record["best_case_p"] == pytest.approx(0.03125)

    def test_the_variant_alone_can_flip_the_answer(self) -> None:
        """At net 5 the two tests disagree, which is exactly the band the
        cartridge and code-style comparisons live in.
        """
        assert net_difference_power(5, 32, 0.05, McNemarTest.MID_P)["net_could_ever_be_significant"]
        assert not net_difference_power(5, 32, 0.05, McNemarTest.EXACT)[
            "net_could_ever_be_significant"
        ]

    def test_it_carries_the_denominator_it_was_read_from(self) -> None:
        """``total_pairs`` does not enter the arithmetic -- McNemar conditions
        on the discordant pairs alone -- and is carried so the record reads
        '4 of 875'. A margin without its denominator is the form that made
        +0.052 look like a number.
        """
        record = net_difference_power(4, 875, 0.05, McNemarTest.MID_P)

        assert record["net_difference"] == 4
        assert record["total_pairs"] == 875
        assert record["instrument"] == PowerInstrument.NET_DIFFERENCE.value
        assert record["test"] == McNemarTest.MID_P.value
        assert record["alpha"] == 0.05

    def test_it_carries_no_power_verdict(self) -> None:
        """The abstention this record's docstring declares, made checkable.

        ``PowerVerdict`` is defined as DETECTABILITY -- "the instrument could
        have resolved an effect as small as the one anyone would act on" --
        and this instrument answers FALSIFIABILITY: could any ``d`` consistent
        with this net reject at all. A verdict here would be true of the
        instrument and false about the world, which is why the record carries
        ``net_could_ever_be_significant`` instead: a boolean named after its
        own question cannot be mistaken for a classification.

        ``McNemarPower`` has pinned the identical abstention since it was
        written and ``rate_floor_power`` was pinned the same way in
        ``a4c64ef9``; this record was the one left where the invariant lived
        in a docstring and a commit message and was enforced by nothing -- a
        condition stated as though it were self-executing, in the module
        written to stop that. If a verdict field ever returns here, this
        fails.
        """
        record = net_difference_power(4, 875, 0.05, McNemarTest.MID_P)

        assert record["net_could_ever_be_significant"] is False
        assert "verdict" not in record
        assert PowerVerdict.TESTED.value not in record.values()
        assert PowerVerdict.NOT_TESTED.value not in record.values()

    def test_a_tie_carries_no_evidence_and_says_so(self) -> None:
        record = net_difference_power(0, 32, 0.05, McNemarTest.MID_P)

        assert record["net_could_ever_be_significant"] is False
        assert record["best_case_p"] == pytest.approx(0.75)

    def test_a_tie_with_too_few_pairs_to_arrange_falls_back_to_no_evidence(self) -> None:
        """Under one pair there is no ``d = 2`` arrangement to dip into, so
        the answer is the definitional 1.0 rather than the tie form's 0.75.
        """
        record = net_difference_power(0, 1, 0.05, McNemarTest.MID_P)

        assert record["best_case_p"] == 1.0

    def test_a_tie_under_the_exact_test_is_flat_at_one(self) -> None:
        assert net_difference_power(0, 32, 0.05, McNemarTest.EXACT)["best_case_p"] == 1.0

    def test_a_negative_net_difference_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            net_difference_power(-1, 32, 0.05, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_a_net_larger_than_the_pairs_it_came_from_is_refused(self) -> None:
        """Unreachable by construction from real counts, and refused anyway:
        it means the caller transposed two arguments, and a record built from
        transposed arguments is a number about nothing.
        """
        with pytest.raises(AppError) as excinfo:
            net_difference_power(40, 32, 0.05, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_negative_pairs_are_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            net_difference_power(0, -1, 0.05, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID

    def test_an_alpha_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            net_difference_power(4, 32, 0.0, McNemarTest.EXACT)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_ALPHA_OUT_OF_RANGE


class TestTheFloorIsWhereTheProofSaysItIs:
    """The lemma the instrument stands on, checked rather than assumed."""

    def test_no_arrangement_beats_the_extreme_split(self) -> None:
        """For a net of k, the arrangements are d = k, k+2, k+4, ... and the
        claim is that d = k -- every discordant pair one way -- attains the
        smallest p.

        Exhaustive over both variants, k up to 40 and d up to 400. The
        algebra proves it for the exact test (the step term is proportional
        to ``C(d, m+1) - C(d, m)``, non-negative exactly when k >= 1); mid-p
        subtracts a point probability and is pinned here instead.
        """
        for test in (McNemarTest.EXACT, McNemarTest.MID_P):
            for net in range(1, 41):
                extreme = mcnemar_p(0, net, test)
                for discordant in range(net, 401, 2):
                    minority = (discordant - net) // 2

                    assert mcnemar_p(minority, discordant, test) >= extreme

    def test_the_tie_is_the_documented_exception(self) -> None:
        """k=0 is the one case where the extreme split is NOT the minimum,
        and it is handled by naming both candidates rather than by pretending
        the lemma covers it. Recorded as a test so the exception cannot be
        quietly removed.
        """
        assert mcnemar_p(1, 2, McNemarTest.MID_P) < mcnemar_p(0, 0, McNemarTest.MID_P)


class TestRoundTrip:
    """Encode/decode, and the validation that makes decode meaningful."""

    def test_the_record_survives_a_round_trip(self) -> None:
        record = net_difference_power(4, 875, 0.05, McNemarTest.MID_P)

        assert decode_net_difference_power(encode_net_difference_power(record)) == record

    def test_decode_rejects_a_mismatched_instrument(self) -> None:
        payload = encode_net_difference_power(net_difference_power(4, 875, 0.05, McNemarTest.MID_P))
        payload["instrument"] = PowerInstrument.MCNEMAR.value

        with pytest.raises(AppError) as excinfo:
            decode_net_difference_power(payload)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN

    def test_decode_rejects_an_unknown_mcnemar_test(self) -> None:
        """The field that decides whether the floor is 5 or 6. A record
        carrying a test nobody ran is a floor for a different instrument.
        """
        payload = encode_net_difference_power(net_difference_power(4, 875, 0.05, McNemarTest.MID_P))
        payload["test"] = "chi_squared"

        with pytest.raises(AppError) as excinfo:
            decode_net_difference_power(payload)

        assert excinfo.value.code is StatisticalPowerErrorCode.POWER_TEST_UNKNOWN

    def test_decode_rejects_a_missing_field(self) -> None:
        payload = encode_net_difference_power(net_difference_power(4, 875, 0.05, McNemarTest.MID_P))
        del payload["best_case_p"]

        with pytest.raises(JSONTypeError):
            decode_net_difference_power(payload)
