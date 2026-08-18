"""Tests for the divergence record codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.dispersion import encode_dispersion_record
from navprobe.codecs.divergence import (
    DIVERGENCE_BANNER,
    decode_divergence_record,
    encode_divergence_record,
)
from navprobe.codecs.scene import encode_float_field
from navprobe.records import DispersionRecord, DivergenceRecord
from navprobe.wireformat import SEPARATOR, WireFormatError


def _record() -> DivergenceRecord:
    """Build a divergence record.

    Returns:
        The measured cross-device depth comparison: 2,794 of 8,192 pixels
        differing, with magnitudes that have no exact decimal form.
    """
    return DivergenceRecord(
        observation_length=8192,
        differing_elements=2794,
        max_absolute_difference=1.466274261475e-05,
        mean_absolute_difference=3.246042501440e-07,
    )


def _lines() -> list[str]:
    """Encode the reference record and split it into lines.

    Returns:
        The encoded lines with the trailing blank removed.
    """
    return encode_divergence_record(_record()).strip("\n").split("\n")


class TestEncodeDivergenceRecord:
    """Tests for :func:`encode_divergence_record`."""

    def test_starts_with_the_divergence_banner(self) -> None:
        """The banner distinguishes it from the dispersion record it resembles."""
        assert _lines()[0] == DIVERGENCE_BANNER

    def test_writes_the_header_in_fixed_order(self) -> None:
        """Field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in _lines()[1:]] == [
            "observation_length",
            "differing_elements",
            "max_absolute_difference",
            "mean_absolute_difference",
        ]

    def test_is_byte_stable_for_equal_records(self) -> None:
        """Equal records encode identically, so results files can be compared."""
        assert encode_divergence_record(_record()) == encode_divergence_record(_record())


class TestDivergenceRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_round_trips(self) -> None:
        """A record survives encoding and decoding exactly."""
        assert decode_divergence_record(encode_divergence_record(_record())) == _record()

    def test_round_trips_perfect_agreement(self) -> None:
        """Zero differences and zero magnitudes survive.

        This is the value most worth storing — two configurations that agree —
        so it must not be rejected as out of range.
        """
        record = DivergenceRecord(
            observation_length=64,
            differing_elements=0,
            max_absolute_difference=0.0,
            mean_absolute_difference=0.0,
        )
        assert decode_divergence_record(encode_divergence_record(record)) == record

    def test_preserves_a_last_bit_magnitude(self) -> None:
        """A difference at the last bit survives, which decimal text need not."""
        record = DivergenceRecord(
            observation_length=8,
            differing_elements=1,
            max_absolute_difference=4.47e-08,
            mean_absolute_difference=4.47e-08,
        )
        decoded = decode_divergence_record(encode_divergence_record(record))
        assert decoded["max_absolute_difference"] == 4.47e-08


class TestDivergenceRejections:
    """Malformed divergence records the decoder refuses."""

    def test_rejects_another_record_types_banner(self) -> None:
        """A dispersion record must not decode as a divergence record.

        The two carry the same number of header lines and similar field names,
        so the banner is the only thing separating them.
        """
        other = DispersionRecord(
            repetitions=8, observation_length=8192, max_spread=0.0, mean_spread=0.0
        )
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record(encode_dispersion_record(other))
        assert caught.value.code == "NP-WIRE-009"

    def test_rejects_a_truncated_header(self) -> None:
        """Every field is required."""
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record(f"{DIVERGENCE_BANNER}\nobservation_length\t8\n")
        assert caught.value.code == "NP-WIRE-010"

    def test_rejects_trailing_lines(self) -> None:
        """A divergence record declares no rows."""
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record(encode_divergence_record(_record()) + "extra\n")
        assert caught.value.code == "NP-WIRE-013"

    def test_rejects_a_zero_observation_length(self) -> None:
        """A comparison over no elements cannot have happened."""
        lines = _lines()
        lines[1] = f"observation_length{SEPARATOR}0"
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-003"

    def test_rejects_a_negative_differing_count(self) -> None:
        """A count of differing elements cannot be below zero."""
        lines = _lines()
        lines[2] = f"differing_elements{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-002"

    def test_rejects_a_negative_magnitude(self) -> None:
        """An absolute difference cannot be negative."""
        lines = _lines()
        lines[3] = f"max_absolute_difference{SEPARATOR}{encode_float_field(-1.0)}"
        with pytest.raises(WireFormatError) as caught:
            decode_divergence_record("\n".join(lines))
        assert caught.value.code == "NP-WIRE-016"
