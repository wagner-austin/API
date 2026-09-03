"""Two arms on the same items, and the test that decides whether it means anything.

The p-values below are computed by hand in the docstrings rather than taken
from a library, because the whole point of an EXACT test is that its value is a
finite sum anyone can check. A test that asserted whatever the implementation
happened to return would confirm the code agrees with itself.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from model_trainer.core.contracts.paired_comparison import (
    PairedItemOutcome,
    decode_paired_comparison,
    decode_paired_item_outcome,
    decode_paired_item_outcomes,
    encode_paired_comparison,
    encode_paired_item_outcome,
    exact_mcnemar_p,
    outcomes_digest,
    summarise_pairs,
)


def outcome(index: int, *, baseline: float, treatment: float) -> PairedItemOutcome:
    """Build one per-item outcome.

    Args:
        index: Item position.
        baseline: Loss under the control arm.
        treatment: Loss under the arm being tested.

    Returns:
        The outcome.
    """
    return PairedItemOutcome(index=index, baseline=baseline, treatment=treatment)


class TestTheExactTest:
    """McNemar's exact conditional test, checked against hand arithmetic."""

    def test_no_discordant_pairs_cannot_reject(self) -> None:
        """With no evidence of direction there is nothing to reject."""
        assert exact_mcnemar_p(improved=0, worsened=0) == 1.0

    def test_a_clean_sweep_of_twelve(self) -> None:
        """12 improved, 0 worsened: 2 * C(12,0) / 2^12 = 2/4096."""
        assert exact_mcnemar_p(improved=12, worsened=0) == pytest.approx(2.0 / 4096.0)

    def test_a_clean_sweep_of_five_is_not_significant_at_one_percent(self) -> None:
        """2 * C(5,0) / 2^5 = 2/32 = 0.0625.

        Worth pinning: five items all moving the same way is the strongest
        result a five-item set can produce, and it still does not clear 0.01.
        A held-out set has to be big enough to be able to say anything.
        """
        assert exact_mcnemar_p(improved=5, worsened=0) == pytest.approx(0.0625)

    def test_an_even_split_is_capped_at_one(self) -> None:
        """2 * (C(2,0) + C(2,1)) / 2^2 = 6/4 = 1.5, which is not a probability."""
        assert exact_mcnemar_p(improved=1, worsened=1) == 1.0

    def test_a_lopsided_split(self) -> None:
        """3 improved, 1 worsened: 2 * (C(4,0) + C(4,1)) / 2^4 = 10/16."""
        assert exact_mcnemar_p(improved=3, worsened=1) == pytest.approx(0.625)

    def test_direction_does_not_change_the_p_value(self) -> None:
        """Two-sided: the test asks whether the split is a coin flip, not which
        way it fell."""
        assert exact_mcnemar_p(improved=9, worsened=2) == exact_mcnemar_p(improved=2, worsened=9)

    def test_it_never_exceeds_one(self) -> None:
        """The doubling can overshoot near an even split, at every size."""
        assert all(exact_mcnemar_p(improved=n, worsened=n) <= 1.0 for n in range(1, 12))


class TestSummarising:
    """Reducing per-item outcomes to the comparison they support."""

    def test_it_counts_the_three_directions(self) -> None:
        """Improved, worsened and tied partition the items."""
        summary = summarise_pairs(
            [
                outcome(0, baseline=2.0, treatment=1.0),
                outcome(1, baseline=1.0, treatment=2.0),
                outcome(2, baseline=1.5, treatment=1.5),
            ]
        )
        assert (summary["improved"], summary["worsened"], summary["tied"]) == (1, 1, 1)

    def test_the_three_counts_always_sum_to_the_item_count(self) -> None:
        """A fourth direction would mean an item was scored and not counted."""
        summary = summarise_pairs(
            [outcome(i, baseline=float(i), treatment=float(i % 2)) for i in range(9)]
        )
        assert summary["improved"] + summary["worsened"] + summary["tied"] == summary["items"]

    def test_it_reports_both_means(self) -> None:
        """The aggregate a report shows, alongside the pairing that justifies it."""
        summary = summarise_pairs(
            [outcome(0, baseline=4.0, treatment=1.0), outcome(1, baseline=2.0, treatment=3.0)]
        )
        assert summary["mean_baseline"] == pytest.approx(3.0)
        assert summary["mean_treatment"] == pytest.approx(2.0)

    def test_ties_are_excluded_from_the_test(self) -> None:
        """A tie carries no direction, so it must not weaken a clean split.

        Two sets with the same discordant pairs and different numbers of ties
        must produce the same p-value; if ties entered the test, adding
        identical items would change the conclusion.
        """
        discordant = [outcome(i, baseline=2.0, treatment=1.0) for i in range(6)]
        padded = [*discordant, *[outcome(9 + i, baseline=1.0, treatment=1.0) for i in range(20)]]
        assert summarise_pairs(discordant)["p_value"] == summarise_pairs(padded)["p_value"]

    def test_an_empty_set_reports_nothing_rather_than_dividing_by_zero(self) -> None:
        """ "Nothing was measured" is a legitimate state and must not raise."""
        summary = summarise_pairs([])
        assert (summary["items"], summary["mean_baseline"], summary["p_value"]) == (0, 0.0, 1.0)


class TestTheOutcomesDigest:
    """Which items moved, not by how much."""

    def test_two_runs_agreeing_on_direction_digest_the_same(self) -> None:
        """The float noise between two machines must not read as a difference."""
        first = [outcome(0, baseline=2.0, treatment=1.0)]
        second = [outcome(0, baseline=2.0000001, treatment=1.0000002)]
        assert outcomes_digest(first) == outcomes_digest(second)

    def test_two_runs_disagreeing_on_one_item_digest_differently(self) -> None:
        """Identical counts, different items: the case the digest exists for."""
        first = [
            outcome(0, baseline=2.0, treatment=1.0),
            outcome(1, baseline=1.0, treatment=2.0),
        ]
        second = [
            outcome(0, baseline=1.0, treatment=2.0),
            outcome(1, baseline=2.0, treatment=1.0),
        ]
        assert summarise_pairs(first)["improved"] == summarise_pairs(second)["improved"]
        assert outcomes_digest(first) != outcomes_digest(second)

    def test_a_tie_is_distinguished_from_both_directions(self) -> None:
        """Three states, three digests."""
        digests = {
            outcomes_digest([outcome(0, baseline=2.0, treatment=1.0)]),
            outcomes_digest([outcome(0, baseline=1.0, treatment=2.0)]),
            outcomes_digest([outcome(0, baseline=1.0, treatment=1.0)]),
        }
        assert len(digests) == 3

    def test_an_empty_set_still_digests(self) -> None:
        """A comparison of nothing is still a comparison and must be recordable."""
        assert len(outcomes_digest([])) == 64


class TestTheCodecs:
    """Both records travel, so both are validated on the way back in."""

    def test_an_outcome_round_trips(self) -> None:
        """Through a real serialise and parse."""
        original = outcome(3, baseline=2.5, treatment=1.25)
        parsed = load_json_str(dump_json_str(encode_paired_item_outcome(original)))
        assert decode_paired_item_outcome(parsed) == original

    def test_a_comparison_round_trips(self) -> None:
        """Every field survives, including the digest."""
        original = summarise_pairs([outcome(0, baseline=2.0, treatment=1.0)])
        parsed = load_json_str(dump_json_str(encode_paired_comparison(original)))
        assert decode_paired_comparison(parsed) == original

    def test_a_list_of_outcomes_round_trips(self) -> None:
        """Order is the pairing, so it has to survive."""
        originals = [outcome(i, baseline=float(i), treatment=float(i) / 2.0) for i in range(4)]
        encoded = [encode_paired_item_outcome(o) for o in originals]
        parsed = load_json_str(dump_json_str(encoded))
        assert decode_paired_item_outcomes(parsed) == originals

    def test_a_non_object_outcome_is_refused(self) -> None:
        """A truncated record must not decode to a partial outcome."""
        with pytest.raises(JSONTypeError):
            decode_paired_item_outcome([1, 2])

    def test_a_non_object_comparison_is_refused(self) -> None:
        """Same, for the summary."""
        with pytest.raises(JSONTypeError):
            decode_paired_comparison("summary")

    def test_a_non_list_of_outcomes_is_refused(self) -> None:
        """An object where an array belongs would silently score nothing."""
        with pytest.raises(JSONTypeError):
            decode_paired_item_outcomes({"index": 0})

    def test_a_missing_field_is_refused(self) -> None:
        """Absent is not zero: a missing baseline would read as an improvement."""
        encoded = encode_paired_item_outcome(outcome(0, baseline=1.0, treatment=2.0))
        del encoded["baseline"]
        with pytest.raises(JSONTypeError):
            decode_paired_item_outcome(encoded)
