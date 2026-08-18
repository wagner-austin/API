"""Tests for the run record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.comparison import encode_comparison_record
from navprobe.codecs.run import (
    RUN_BANNER,
    decode_run_record,
    decode_run_spec,
    decode_step_record,
    encode_run_record,
    encode_run_spec,
    encode_step_record,
)
from navprobe.records import ComparisonRecord, RunRecord, RunSpec, StepRecord
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> RunRecord:
    """Build a valid run record.

    Returns:
        A record with two contiguous steps.
    """
    return RunRecord(
        spec=RunSpec(label="same-process", seed=7, step_count=2, world_count=3),
        steps=(
            StepRecord(step_index=0, digest="aa"),
            StepRecord(step_index=1, digest="bb"),
        ),
        digest="cc",
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_run_record(_record()).strip("\n").split("\n")


class TestRunSpecCodec:
    """Tests for :func:`encode_run_spec` and :func:`decode_run_spec`."""

    def test_field_order_is_fixed(self) -> None:
        """Spec field order is part of the format."""
        spec = RunSpec(label="x", seed=1, step_count=2, world_count=3)
        keys = [line.split(SEPARATOR)[0] for line in encode_run_spec(spec)]
        assert keys == ["label", "seed", "step_count", "world_count"]

    def test_round_trips(self) -> None:
        """A spec survives encoding and decoding."""
        spec = RunSpec(label="fresh process", seed=1, step_count=2, world_count=3)
        assert decode_run_spec(encode_run_spec(spec)) == spec


class TestStepRecordCodec:
    """Tests for :func:`encode_step_record` and :func:`decode_step_record`."""

    def test_encodes_a_tagged_row(self) -> None:
        """A step becomes its tag, index, and digest."""
        assert encode_step_record(StepRecord(step_index=1, digest="bb")) == "step\t1\tbb"

    def test_round_trips_at_its_position(self) -> None:
        """A step survives encoding and decoding at its own index."""
        step = StepRecord(step_index=1, digest="bb")
        assert decode_step_record(encode_step_record(step), 1) == step


class TestEncodeRunRecord:
    """Tests for :func:`encode_run_record`."""

    def test_starts_with_the_run_banner(self) -> None:
        """The first line identifies the record type and version."""
        assert _lines()[0] == RUN_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Header field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:6]] == [
            "label",
            "seed",
            "step_count",
            "world_count",
            "digest",
        ]

    def test_writes_one_row_per_step(self) -> None:
        """Each step becomes one tagged row."""
        assert _lines()[6:] == ["step\t0\taa", "step\t1\tbb"]

    def test_is_newline_terminated(self) -> None:
        """The file ends with a newline, as text files should."""
        assert encode_run_record(_record()).endswith("\n")

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal records encode identically, so files can be compared."""
        assert encode_run_record(_record()) == encode_run_record(_record())


class TestRunRecordRoundTrip:
    """Encoding and decoding a run record compose to the identity."""

    def test_round_trips_a_record_with_steps(self) -> None:
        """A populated record survives the round trip."""
        assert decode_run_record(encode_run_record(_record())) == _record()

    def test_round_trips_an_empty_record(self) -> None:
        """A zero-step rollout is the base case, not a special case."""
        empty = RunRecord(
            spec=RunSpec(label="empty", seed=0, step_count=0, world_count=1),
            steps=(),
            digest="dd",
        )
        assert decode_run_record(encode_run_record(empty)) == empty

    def test_decoded_steps_are_a_tuple(self) -> None:
        """Decoded steps are immutable.

        Compared against a tuple by value: a list of the same steps compares
        unequal to a tuple, so this asserts the container type too.
        """
        assert decode_run_record(encode_run_record(_record()))["steps"] == (
            StepRecord(step_index=0, digest="aa"),
            StepRecord(step_index=1, digest="bb"),
        )

    def test_preserves_a_label_containing_spaces(self) -> None:
        """Tab separation is what makes a spaced label safe."""
        spaced = RunRecord(
            spec=RunSpec(label="fresh process run", seed=1, step_count=0, world_count=1),
            steps=(),
            digest="dd",
        )
        decoded = decode_run_record(encode_run_record(spaced))
        assert decoded["spec"]["label"] == "fresh process run"


class TestRunRecordRejections:
    """Malformed run records the decoder refuses."""

    def test_rejects_a_missing_banner(self) -> None:
        """Text without the banner is not a run record."""
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("something else\n")
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_another_record_types_banner(self) -> None:
        """A comparison record must not decode as a run record."""
        other = ComparisonRecord(
            left_label="a",
            right_label="b",
            digests_match=True,
            first_divergent_step=None,
            compared_step_count=1,
        )
        with pytest.raises(WireFormatError) as caught:
            decode_run_record(encode_comparison_record(other))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """A record needs its banner and all five header lines."""
        with pytest.raises(WireFormatError) as caught:
            decode_run_record(f"{RUN_BANNER}\nlabel\tx\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_a_header_field_out_of_order(self) -> None:
        """Field order is fixed, so a swapped key is refused."""
        lines = _lines()
        lines[1] = f"seed{SEPARATOR}7"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-006"

    def test_rejects_a_malformed_step_row(self) -> None:
        """A step row is the tag, an index, and a digest."""
        lines = _lines()
        lines[6] = f"step{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-007"

    def test_rejects_a_step_row_with_the_wrong_tag(self) -> None:
        """Rows after the header must be tagged as steps."""
        lines = _lines()
        lines[6] = f"note{SEPARATOR}0{SEPARATOR}aa"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-007"

    def test_rejects_non_contiguous_step_indices(self) -> None:
        """Steps must be in order and gapless."""
        lines = _lines()
        lines[7] = f"step{SEPARATOR}5{SEPARATOR}bb"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-008"

    def test_rejects_a_step_count_disagreeing_with_the_rows(self) -> None:
        """A truncated file must not pass as complete."""
        lines = _lines()
        del lines[7]
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-011"

    def test_rejects_a_zero_world_count(self) -> None:
        """A record claiming no worlds is refused."""
        lines = _lines()
        lines[4] = f"world_count{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_an_empty_digest(self) -> None:
        """An empty digest is a construction bug, not a value."""
        lines = _lines()
        lines[5] = f"digest{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_an_empty_step_digest(self) -> None:
        """Per-step digests are validated the same way."""
        lines = _lines()
        lines[6] = f"step{SEPARATOR}0{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_run_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"
