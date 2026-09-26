"""Tests for provenance chain types and encode/decode."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.facts.provenance import (
    decode_provenance,
    decode_source_ref,
    encode_provenance,
    encode_source_ref,
    make_provenance,
    make_source_ref,
)
from tankpit_bot.facts.source import FactSource


def test_source_ref_round_trip() -> None:
    """SourceRefDict survives encode/decode unchanged."""
    ref = make_source_ref(FactSource.WIRE_0X4F_RADAR_RESPONSE, 1500)
    assert decode_source_ref(encode_source_ref(ref)) == ref


def test_source_ref_decode_rejects_bad_source() -> None:
    """Decoding a ref with an unknown source raises JSONTypeError naming it."""
    with pytest.raises(JSONTypeError, match="Invalid source 'nope'"):
        decode_source_ref({"source": "nope", "observed_ms": 0})


def test_provenance_round_trip_with_derivations() -> None:
    """A derived chain survives encode/decode unchanged."""
    chain = make_provenance(
        FactSource.CLIENT_SIDE_INFERENCE,
        [
            make_source_ref(FactSource.WIRE_0X3D_MOVEMENT, 100),
            make_source_ref(FactSource.WIRE_0X4F_RADAR_RESPONSE, 250),
        ],
    )
    assert decode_provenance(encode_provenance(chain)) == chain


def test_provenance_round_trip_observation() -> None:
    """An observation chain with no derivations round-trips."""
    chain = make_provenance(FactSource.WIRE_0X5A_VIEWPORT_PATCH, [])
    encoded = encode_provenance(chain)
    assert encoded == {"origin": "wire_0x5A_viewport_patch", "derived_from": []}
    assert decode_provenance(encoded)["origin"] is FactSource.WIRE_0X5A_VIEWPORT_PATCH


def test_provenance_decode_rejects_unknown_origin() -> None:
    """An origin outside FactSource raises JSONTypeError naming it."""
    with pytest.raises(JSONTypeError, match="Invalid origin 'wire_0xFF_unknown'"):
        decode_provenance({"origin": "wire_0xFF_unknown", "derived_from": []})


def test_provenance_decode_rejects_non_object_ref() -> None:
    """A derivation entry that is not an object raises JSONTypeError."""
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_provenance({"origin": "wire_0x3D_movement", "derived_from": [5]})


def test_provenance_decode_rejects_missing_origin() -> None:
    """A chain without an origin raises JSONTypeError."""
    with pytest.raises(JSONTypeError):
        decode_provenance({"derived_from": []})
