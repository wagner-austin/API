"""Tests for the shared device run-conditions codec."""

from __future__ import annotations

import pytest

from navprobe.codecs.device_conditions import (
    DEVICE_CONDITIONS_FIELD_COUNT,
    decode_device_conditions,
    encode_device_conditions,
)
from navprobe.records import DeviceRunConditions
from navprobe.wireformat import SEPARATOR, WireFormatError


def _conditions() -> DeviceRunConditions:
    """Build run conditions.

    Returns:
        Conditions whose resolved device differs from the requested one, which
        is the case that distinguishes the two fields.
    """
    return DeviceRunConditions(
        mode="RUN_TO_RUN",
        device="NVIDIA GeForce RTX 3090 Ti",
        device_request="cuda:0",
        max_records=64,
        linesearch_block_dim=64,
    )


class TestEncodeDeviceConditions:
    """Tests for :func:`encode_device_conditions`."""

    def test_emits_the_declared_number_of_lines(self) -> None:
        """The count is what lets an enclosing record slice its header."""
        assert len(encode_device_conditions(_conditions())) == DEVICE_CONDITIONS_FIELD_COUNT

    def test_writes_the_fields_in_fixed_order(self) -> None:
        """Field order is part of the format."""
        assert [line.split(SEPARATOR)[0] for line in encode_device_conditions(_conditions())] == [
            "mode",
            "device",
            "device_request",
            "max_records",
            "linesearch_block_dim",
        ]


class TestDeviceConditionsRoundTrip:
    """Encoding and decoding compose to the identity."""

    def test_round_trips(self) -> None:
        """Conditions survive encoding and decoding exactly."""
        assert decode_device_conditions(encode_device_conditions(_conditions())) == _conditions()

    def test_keeps_the_resolved_device_distinct_from_the_request(self) -> None:
        """Collapsing the two would lose which card actually ran."""
        decoded = decode_device_conditions(encode_device_conditions(_conditions()))
        assert decoded["device"] != decoded["device_request"]

    def test_round_trips_a_zero_record_bound(self) -> None:
        """Zero means Warp's own bound and is not an absent value."""
        conditions = DeviceRunConditions(
            mode="NOT_GUARANTEED",
            device="cpu",
            device_request="cpu",
            max_records=0,
            linesearch_block_dim=None,
        )
        assert decode_device_conditions(encode_device_conditions(conditions)) == conditions


class TestDeviceConditionsRejections:
    """Malformed conditions the decoder refuses."""

    def test_rejects_a_negative_record_bound(self) -> None:
        """A negative bound is not a bound."""
        lines = list(encode_device_conditions(_conditions()))
        lines[3] = f"max_records{SEPARATOR}-1"
        with pytest.raises(WireFormatError) as caught:
            decode_device_conditions(tuple(lines))
        assert caught.value.code == "NP-WIRE-002"

    def test_rejects_an_empty_mode(self) -> None:
        """An unlabelled mode makes every verdict in the record unreadable."""
        lines = list(encode_device_conditions(_conditions()))
        lines[0] = f"mode{SEPARATOR}"
        with pytest.raises(WireFormatError) as caught:
            decode_device_conditions(tuple(lines))
        assert caught.value.code == "NP-WIRE-004"

    def test_rejects_fields_in_the_wrong_order(self) -> None:
        """Order is pinned by key, so a swapped pair is caught rather than read."""
        lines = list(encode_device_conditions(_conditions()))
        lines[0], lines[1] = lines[1], lines[0]
        with pytest.raises(WireFormatError) as caught:
            decode_device_conditions(tuple(lines))
        assert caught.value.code == "NP-WIRE-006"
