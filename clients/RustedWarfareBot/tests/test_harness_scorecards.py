"""What one match is, read off the scorecard it was filed as.

The reader that ``scripts/analyze_sweep`` used to carry its own copy of. Two
readers of one format is one edit away from a table and an aggregate that
report different numbers for the same batch, with neither of them failing.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.scorecards import (
    MatchRow,
    arrow_end,
    decode_match_row,
    encode_match_row,
    parse_match_row,
    row_order,
    split_stem,
)
from rw_bot.validation import DecodeError

#: A real scorecard's shape: a lowercase label padded to the report's column
#: width, then the value. Written out rather than generated, because the
#: reader is defined by this shape and a generated one would agree with
#: whatever the reader does.
CARD = "\n".join(
    (
        "### attack-s777",
        "verdict        survived (sample_limit)",
        "extractors     4 -> 2",
        "total worth    3500 -> 4700",
        "best rival     5400 -> 6100 (peak 6100, worst dip 250)",
        "enemies seen   17 -> 20 (3 engageable)",
        "intercepted    5",
        "income         18/s",
        "  indented line that is not a figure",
    )
)


def _row(arm: str = "attack", **numbers: int) -> MatchRow:
    """Build a row with everything at zero, overriding what a test cares about.

    Args:
        arm: Which arm played it.
        **numbers: Figures to set.

    Returns:
        The row.
    """
    row = MatchRow(
        arm=arm,
        seed=1,
        verdict="won",
        extr_end=0,
        peak=0,
        dropped=0,
        worth_end=0,
        rival_end=0,
        dip=0,
        targets_end=0,
        engageable=0,
        intercepted=0,
        income="18/s",
    )
    for name, value in numbers.items():
        if name == "seed":
            row["seed"] = value
        if name == "dropped":
            row["dropped"] = value
        if name == "worth_end":
            row["worth_end"] = value
    return row


class TestReadingTheFilename:
    def test_the_arm_and_the_seed_are_read_apart(self) -> None:
        assert split_stem("attack-s12345") == ("attack", 12345)

    def test_an_arm_containing_the_marker_survives(self) -> None:
        """``aa-counter-s2`` is a real arm name. Splitting from the left would
        file it under ``aa-counter`` with a seed of 2."""
        assert split_stem("aa-counter-s2-s777") == ("aa-counter-s2", 777)

    def test_a_negative_seed_is_read_because_the_engine_takes_one(self) -> None:
        assert split_stem("attack-s-3") == ("attack", -3)

    def test_a_filename_with_no_seed_is_refused(self) -> None:
        """A result whose arm cannot be read would be aggregated into the
        wrong one, and an arm is what a verdict is about."""
        with pytest.raises(ValueError, match="before its seed"):
            split_stem("attack")

    def test_a_filename_with_a_non_numeric_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ends with a whole seed"):
            split_stem("attack-sLATER")


class TestReadingAnArrowValue:
    def test_the_end_figure_is_taken_not_the_start(self) -> None:
        assert arrow_end("3500 -> 4700") == 4700

    def test_a_negative_end_is_read(self) -> None:
        assert arrow_end("10 -> -4") == -4

    def test_a_value_with_no_arrow_reads_as_zero(self) -> None:
        """A scorecard from an older build legitimately lacks lines a newer
        one writes, and a missing figure is zero rather than a crash."""
        assert arrow_end("") == 0
        assert arrow_end("18/s") == 0


class TestReadingAWholeCard:
    def test_every_figure_comes_off_the_card(self) -> None:
        row = parse_match_row("attack-s777", CARD, peak=14, dropped=12)
        assert row == MatchRow(
            arm="attack",
            seed=777,
            verdict="survived",
            extr_end=2,
            peak=14,
            dropped=12,
            worth_end=4700,
            rival_end=6100,
            dip=250,
            targets_end=20,
            engageable=3,
            intercepted=5,
            income="18/s",
        )

    def test_the_verdict_is_the_first_word(self) -> None:
        """``survived (sample_limit)`` is one verdict with a reason; the
        aggregate counts verdicts, not reasons."""
        assert parse_match_row("a-s1", CARD, 0, 0)["verdict"] == "survived"

    def test_the_peak_and_drop_come_from_the_trace_not_the_card(self) -> None:
        """A match reporting ``extractors 0 -> 0`` had held a peak of
        fourteen before collapsing -- the card cannot carry that."""
        row = parse_match_row("a-s1", CARD, peak=14, dropped=12)
        assert (row["peak"], row["dropped"]) == (14, 12)

    def test_an_indented_line_is_not_a_figure(self) -> None:
        """The planner's commentary is indented; only a label padded to the
        report width is a figure."""
        assert parse_match_row("a-s1", CARD, 0, 0)["income"] == "18/s"

    def test_a_card_missing_a_line_reads_as_zero_rather_than_failing(self) -> None:
        row = parse_match_row("a-s1", "verdict        won", 0, 0)
        assert row["worth_end"] == 0
        assert row["verdict"] == "won"


class TestOrdering:
    def test_rows_order_by_arm_then_seed(self) -> None:
        rows = [_row("b", seed=1), _row("a", seed=9), _row("a", seed=2)]
        ordered = [(row["arm"], row["seed"]) for row in sorted(rows, key=row_order)]
        assert ordered == [("a", 2), ("a", 9), ("b", 1)]


class TestTheCodec:
    def test_a_row_round_trips(self) -> None:
        assert decode_match_row(encode_match_row(_row())) == _row()

    def test_every_field_survives(self) -> None:
        row = parse_match_row("attack-s777", CARD, 14, 12)
        payload = encode_match_row(row)
        assert sorted(payload) == sorted(row)
        assert decode_match_row(payload) == row

    def test_a_missing_field_is_refused(self) -> None:
        payload = encode_match_row(_row())
        del payload["dropped"]
        with pytest.raises(DecodeError) as caught:
            decode_match_row(payload)
        assert caught.value.code == "RW-DECODE-001"

    def test_a_field_of_the_wrong_type_is_refused_rather_than_coerced(self) -> None:
        payload = encode_match_row(_row())
        payload["dropped"] = "12"
        with pytest.raises(DecodeError) as caught:
            decode_match_row(payload)
        assert caught.value.code == "RW-DECODE-002"

    def test_a_blank_arm_is_refused(self) -> None:
        """A row filed under no arm would be aggregated into no verdict."""
        payload = encode_match_row(_row())
        payload["arm"] = "  "
        with pytest.raises(DecodeError) as caught:
            decode_match_row(payload)
        assert caught.value.code == "RW-DECODE-003"
