"""Coverage tests for sniffer/world_state_dispatch.py: encode/decode round-trip
for MapPositionsParsedDiagnosticDict."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from tankpit_bot.sniffer.world_state_dispatch_position import (
    MapPositionsParsedDiagnosticDict,
    decode_map_positions_parsed_diagnostic,
    encode_map_positions_parsed_diagnostic,
)


def test_encode_decode_round_trip() -> None:
    """Encode then decode preserves every field."""
    payload = MapPositionsParsedDiagnosticDict(
        tank_count=7,
        blob_bytes=1024,
        fuel_dot_count=42,
    )
    encoded = encode_map_positions_parsed_diagnostic(payload)
    decoded = decode_map_positions_parsed_diagnostic(encoded)

    assert decoded == payload


def test_encode_produces_json_compatible_dict() -> None:
    """Encoded output uses plain Python types suitable for JSON serialization."""
    payload = MapPositionsParsedDiagnosticDict(
        tank_count=0,
        blob_bytes=0,
        fuel_dot_count=0,
    )
    encoded = encode_map_positions_parsed_diagnostic(payload)

    assert encoded["tank_count"] == 0
    assert encoded["blob_bytes"] == 0
    assert encoded["fuel_dot_count"] == 0
    assert set(encoded.keys()) == {"tank_count", "blob_bytes", "fuel_dot_count"}


def test_decode_validates_all_fields() -> None:
    """Decoded result matches a hand-built JSON payload."""
    raw: JSONObject = {"tank_count": 3, "blob_bytes": 512, "fuel_dot_count": 15}
    decoded = decode_map_positions_parsed_diagnostic(raw)

    assert decoded["tank_count"] == 3
    assert decoded["blob_bytes"] == 512
    assert decoded["fuel_dot_count"] == 15
