"""Tests for the comparison record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.comparison import (
    COMPARISON_BANNER,
    decode_comparison_record,
    encode_comparison_record,
)
from navprobe.codecs.run import encode_run_record
from navprobe.records import ComparisonRecord, RunRecord, RunSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> ComparisonRecord:
    """Build a valid comparison record.

    Returns:
        A record reporting a divergence at step one.
    """
    return ComparisonRecord(
        left_label="repetition-0",
        right_label="repetition-1",
        digests_match=False,
        first_divergent_step=1,
        compared_step_count=4,
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_comparison_record(_record()).strip("\n").split("\n")


class TestEncodeComparisonRecord:
    """Tests for :func:`encode_comparison_record`."""

    def test_starts_with_the_comparison_banner(self) -> None:
        """The banner distinguishes this record from a run record."""
        assert _lines()[0] == COMPARISON_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Header field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:]] == [
            "left_label",
            "right_label",
            "digests_match",
            "first_divergent_step",
            "compared_step_count",
        ]

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal verdicts encode identically, so files can be compared."""
        assert encode_comparison_record(_record()) == encode_comparison_record(_record())


class TestComparisonRecordRoundTrip:
    """Encoding and decoding a comparison record compose to the identity."""

    def test_round_trips_a_divergent_verdict(self) -> None:
        """A verdict with a divergence point survives the round trip."""
        assert decode_comparison_record(encode_comparison_record(_record())) == _record()

    def test_round_trips_an_agreeing_verdict(self) -> None:
        """Absence of a divergence point survives as ``None``, not zero."""
        record = ComparisonRecord(
            left_label="a",
            right_label="b",
            digests_match=True,
            first_divergent_step=None,
            compared_step_count=5,
        )
        assert decode_comparison_record(encode_comparison_record(record)) == record

    def test_round_trips_a_divergence_at_step_zero(self) -> None:
        """Step zero is a value, and must not decode as absence."""
        record = ComparisonRecord(
            left_label="a",
            right_label="b",
            digests_match=False,
            first_divergent_step=0,
            compared_step_count=5,
        )
        decoded = decode_comparison_record(encode_comparison_record(record))
        assert decoded["first_divergent_step"] == 0


class TestComparisonRecordRejections:
    """Malformed comparison records the decoder refuses."""

    def test_rejects_another_record_types_banner(self) -> None:
        """A run record must not decode as a comparison record."""
        other = RunRecord(
            spec=RunSpec(label="x", seed=1, step_count=0, world_count=1),
            steps=(),
            digest="cc",
        )
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record(encode_run_record(other))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every header line is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record(f"{COMPARISON_BANNER}\nleft_label\ta\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_trailing_lines(self) -> None:
        """A comparison declares no rows, so trailing content is refused."""
        text = encode_comparison_record(_record()) + "step\t0\taa\n"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record(text)
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_an_unspelled_boolean(self) -> None:
        """The match flag must be one of the two spelled tokens."""
        lines = _lines()
        lines[3] = f"digests_match{SEPARATOR}1"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-012"

    def test_rejects_an_empty_left_label(self) -> None:
        """A verdict must name the condition on the left."""
        lines = _lines()
        lines[1] = f"left_label{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_an_empty_right_label(self) -> None:
        """A verdict must name the condition on the right."""
        lines = _lines()
        lines[2] = f"right_label{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_a_negative_divergence_step(self) -> None:
        """A divergence before step zero is not a position."""
        lines = _lines()
        lines[4] = f"first_divergent_step{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-002"

    def test_rejects_a_negative_compared_step_count(self) -> None:
        """A negative count of compared steps is not a quantity."""
        lines = _lines()
        lines[5] = f"compared_step_count{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_comparison_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-002"
