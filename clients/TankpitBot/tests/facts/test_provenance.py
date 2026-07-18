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


def test_source_ref_round_trip() -> None:
    """SourceRefDict survives encode/decode unchanged."""
    ref = make_source_ref("wire_0x4F_radar_response", 1500)
    assert decode_source_ref(encode_source_ref(ref)) == ref


def test_source_ref_decode_rejects_bad_source() -> None:
    """Decoding a ref with an unknown source raises JSONTypeError."""
    with pytest.raises(JSONTypeError, match="source must be one of"):
        decode_source_ref({"source": "nope", "observed_ms": 0})


def test_provenance_round_trip_with_derivations() -> None:
    """A derived chain survives encode/decode unchanged."""
    chain = make_provenance(
        "client_side_inference",
        [
            make_source_ref("wire_0x3D_movement", 100),
            make_source_ref("wire_0x4F_radar_response", 250),
        ],
    )
    assert decode_provenance(encode_provenance(chain)) == chain


def test_provenance_round_trip_observation() -> None:
    """An observation chain with no derivations round-trips."""
    chain = make_provenance("wire_0x5A_viewport_patch", [])
    encoded = encode_provenance(chain)
    assert encoded == {"origin": "wire_0x5A_viewport_patch", "derived_from": []}
    assert decode_provenance(encoded) == chain


def test_provenance_decode_rejects_non_object_ref() -> None:
    """A derivation entry that is not an object raises JSONTypeError."""
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_provenance({"origin": "wire_0x3D_movement", "derived_from": [5]})


def test_provenance_decode_rejects_missing_origin() -> None:
    """A chain without an origin raises JSONTypeError."""
    with pytest.raises(JSONTypeError):
        decode_provenance({"derived_from": []})
