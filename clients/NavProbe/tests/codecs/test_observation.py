"""Tests for the observation record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.observation import (
    OBSERVATION_BANNER,
    VALUE_TAG,
    decode_observation_record,
    decode_value_row,
    encode_observation_record,
)
from navprobe.codecs.scene import encode_scene_spec
from navprobe.records import ObservationRecord, SceneSpec
from navprobe.wireformat import SEPARATOR, WireFormatError, encode_float_field


def _record() -> ObservationRecord:
    """Build an observation record.

    Returns:
        A record whose values span negative, zero and inexact-in-decimal, so a
        codec that narrowed the range or rounded would fail the round trip.
    """
    return ObservationRecord(
        label="wsl-cuda",
        seed=7,
        step_count=200,
        values=(-1.5, 0.0, 0.055, 6.5796294212),
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_observation_record(_record()).strip("\n").split("\n")


class TestEncodeObservationRecord:
    """Tests for :func:`encode_observation_record`."""

    def test_starts_with_the_observation_banner(self) -> None:
        """The banner distinguishes it from every other record."""
        assert _lines()[0] == OBSERVATION_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:5]] == [
            "label",
            "seed",
            "step_count",
            "value_count",
        ]

    def test_writes_one_line_per_value(self) -> None:
        """Line-per-value is what makes two environments' output diff-able."""
        assert len(_lines()[5:]) == 4

    def test_each_value_row_carries_its_index(self) -> None:
        """Element order is the observation's contract, so it is written down."""
        assert [line.split(SEPARATOR)[1] for line in _lines()[5:]] == ["0", "1", "2", "3"]

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal observations encode identically, so files can be compared."""
        assert encode_observation_record(_record()) == encode_observation_record(_record())


class TestObservationRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_round_trips(self) -> None:
        """A record survives encoding and decoding exactly."""
        assert decode_observation_record(encode_observation_record(_record())) == _record()

    def test_decoded_values_are_a_tuple(self) -> None:
        """Values are immutable, so a later stage cannot append to them."""
        decoded = decode_observation_record(encode_observation_record(_record()))
        assert decoded["values"] == (-1.5, 0.0, 0.055, 6.5796294212)

    def test_round_trips_an_empty_observation(self) -> None:
        """A zero-length observation is the base case, not a special case.

        The measurement layers refuse to compare one, but the codec's job is the
        format and a record is not malformed by a value a producer would not
        emit.
        """
        empty = ObservationRecord(label="x", seed=0, step_count=1, values=())
        assert decode_observation_record(encode_observation_record(empty)) == empty

    def test_preserves_a_label_containing_spaces(self) -> None:
        """Tab separation is what makes a spaced environment name safe."""
        spaced = ObservationRecord(label="wsl cuda 12", seed=1, step_count=1, values=(1.0,))
        decoded = decode_observation_record(encode_observation_record(spaced))
        assert decoded["label"] == "wsl cuda 12"


class TestObservationRejections:
    """Malformed observation records the decoder refuses."""

    def test_rejects_another_record_types_banner(self) -> None:
        """A scene must not decode as an observation."""
        scene = SceneSpec(body_count=1, lattice_width=1, spacing=0.055, radius=0.03, timestep=0.005)
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record(encode_scene_spec(scene))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every header field is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record(f"{OBSERVATION_BANNER}\nlabel\tx\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_a_count_disagreeing_with_the_rows(self) -> None:
        """A truncated file must not pass as a complete observation."""
        lines = _lines()
        del lines[-1]
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-020"

    def test_rejects_a_malformed_value_row(self) -> None:
        """A value row is the tag, an index, and a value."""
        with pytest.raises(WireFormatError) as caught:
            decode_value_row(f"{VALUE_TAG}{SEPARATOR}0", 0)
        assert caught.value.code == "NP-WIRE-018"

    def test_rejects_a_value_row_with_the_wrong_tag(self) -> None:
        """Rows after the header must be tagged as values."""
        with pytest.raises(WireFormatError) as caught:
            decode_value_row(f"note{SEPARATOR}0{SEPARATOR}{encode_float_field(1.0)}", 0)
        assert caught.value.code == "NP-WIRE-018"

    def test_rejects_non_contiguous_value_indices(self) -> None:
        """A reordered file would compare position three against position four."""
        lines = _lines()
        lines[6] = f"{VALUE_TAG}{SEPARATOR}9{SEPARATOR}{encode_float_field(1.0)}"
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-019"

    def test_rejects_an_empty_label(self) -> None:
        """An observation must name the environment that produced it."""
        lines = _lines()
        lines[1] = f"label{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_a_negative_seed(self) -> None:
        """A seed is drawn from the non-negative integers."""
        lines = _lines()
        lines[2] = f"seed{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_observation_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-002"
