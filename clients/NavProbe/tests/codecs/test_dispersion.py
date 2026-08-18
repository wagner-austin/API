"""Tests for the dispersion record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.dispersion import (
    DISPERSION_BANNER,
    decode_dispersion_record,
    encode_dispersion_record,
    require_non_negative_float_field,
)
from navprobe.codecs.scene import encode_float_field, encode_scene_spec
from navprobe.records import DispersionRecord, SceneSpec
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> DispersionRecord:
    """Build a dispersion record.

    Returns:
        A record whose spreads are not exactly representable in decimal.
    """
    return DispersionRecord(
        repetitions=8,
        observation_length=8192,
        max_spread=1.466274261475e-05,
        mean_spread=3.246042501440e-07,
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_dispersion_record(_record()).strip("\n").split("\n")


class TestNonNegativeFloatField:
    """Tests for :func:`require_non_negative_float_field`."""

    def test_accepts_zero(self) -> None:
        """A deterministic configuration has a spread of exactly zero.

        This is the most important value the record carries, so it must decode
        rather than being rejected as out of range.
        """
        assert require_non_negative_float_field(encode_float_field(0.0), "f") == 0.0

    def test_round_trips_a_tiny_spread(self) -> None:
        """A last-bit spread survives, which is the value that matters."""
        assert require_non_negative_float_field(encode_float_field(4.47e-08), "f") == 4.47e-08

    def test_rejects_a_negative_spread(self) -> None:
        """A range cannot be below zero."""
        with pytest.raises(WireFormatError) as caught:
            require_non_negative_float_field(encode_float_field(-1.0), "f")
        assert caught.value.code == "NP-WIRE-016"

    def test_rejects_a_decimal_token(self) -> None:
        """Decimal text is refused rather than silently parsed."""
        with pytest.raises(WireFormatError) as caught:
            require_non_negative_float_field("0.5", "f")
        assert caught.value.code == "NP-WIRE-014"


class TestDispersionRecordCodec:
    """Tests for the record's document form."""

    def test_starts_with_the_dispersion_banner(self) -> None:
        """The banner distinguishes it from every other record."""
        assert _lines()[0] == DISPERSION_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:]] == [
            "repetitions",
            "observation_length",
            "max_spread",
            "mean_spread",
        ]

    def test_round_trips(self) -> None:
        """A record survives encoding and decoding exactly."""
        assert decode_dispersion_record(encode_dispersion_record(_record())) == _record()

    def test_round_trips_a_zero_spread(self) -> None:
        """The deterministic case survives, spreads and all."""
        record = DispersionRecord(
            repetitions=4, observation_length=6, max_spread=0.0, mean_spread=0.0
        )
        assert decode_dispersion_record(encode_dispersion_record(record)) == record

    def test_allows_a_zero_observation_length_to_decode(self) -> None:
        """Length is a count, so zero is in range for the codec.

        The measurement layer refuses an empty observation under its own code;
        the codec's job is the format, and a record is not made invalid by a
        field the producer would never emit.
        """
        record = DispersionRecord(
            repetitions=2, observation_length=0, max_spread=0.0, mean_spread=0.0
        )
        assert decode_dispersion_record(encode_dispersion_record(record))["observation_length"] == 0

    def test_rejects_another_record_types_banner(self) -> None:
        """A scene must not decode as a dispersion record."""
        scene = SceneSpec(body_count=1, lattice_width=1, spacing=0.055, radius=0.03, timestep=0.005)
        with pytest.raises(WireFormatError) as caught:
            decode_dispersion_record(encode_scene_spec(scene))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every field is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_dispersion_record(f"{DISPERSION_BANNER}\nrepetitions\t8\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_trailing_lines(self) -> None:
        """A dispersion record declares no rows."""
        with pytest.raises(WireFormatError) as caught:
            decode_dispersion_record(encode_dispersion_record(_record()) + "extra\n")
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_zero_repetitions(self) -> None:
        """A spread over no rollouts is not a measurement."""
        lines = _lines()
        lines[1] = f"repetitions{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_dispersion_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_negative_max_spread(self) -> None:
        """A negative range is refused."""
        lines = _lines()
        lines[3] = f"max_spread{SEPARATOR}{encode_float_field(-1.0)}"
        with pytest.raises(WireFormatError) as caught:
            decode_dispersion_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-016"
