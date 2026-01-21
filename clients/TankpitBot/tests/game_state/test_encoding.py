"""Tests for game state encode/decode functions."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.game_state import (
    GameStateSnapshot,
    decode_game_state_snapshot,
    decode_location_state,
    decode_nearby_entity,
    decode_session_info,
    encode_game_state_snapshot,
    encode_location_state,
    encode_nearby_entity,
    encode_session_info,
)
from tests.game_state.conftest import (
    make_sample_game_state_snapshot,
    make_sample_inventory,
    make_sample_location,
    make_sample_nearby_entity,
    make_sample_session_info,
)

# =============================================================================
# LocationState encode/decode Tests
# =============================================================================


def test_encode_location_state() -> None:
    """Test encoding location state."""
    state = make_sample_location()
    encoded = encode_location_state(state)

    assert encoded["x"] == 123
    assert encoded["y"] == 456
    assert encoded["raw"] == "123,456"


def test_decode_location_state() -> None:
    """Test decoding location state."""
    obj: JSONObject = {"x": 100, "y": 200, "raw": "100,200"}
    decoded = decode_location_state(obj)

    assert decoded["x"] == 100
    assert decoded["y"] == 200
    assert decoded["raw"] == "100,200"


def test_decode_location_state_missing_field() -> None:
    """Test decoding location state with missing field."""
    with pytest.raises(JSONTypeError):
        decode_location_state({"x": 100, "y": 200})


def test_decode_location_state_wrong_type() -> None:
    """Test decoding location state with wrong type."""
    with pytest.raises(JSONTypeError):
        decode_location_state({"x": "not_int", "y": 200, "raw": "test"})


def test_location_state_roundtrip() -> None:
    """Test encode/decode roundtrip for location state."""
    original = make_sample_location()
    encoded = encode_location_state(original)
    decoded = decode_location_state(encoded)

    assert decoded == original


# =============================================================================
# NearbyEntity encode/decode Tests
# =============================================================================


def test_encode_nearby_entity() -> None:
    """Test encoding nearby entity."""
    entity = make_sample_nearby_entity()
    encoded = encode_nearby_entity(entity)

    assert encoded["name"] == "blue-7"
    assert encoded["direction"] == "W"
    assert encoded["coordinates"] == "57,135"
    assert encoded["is_private"] is True


def test_decode_nearby_entity() -> None:
    """Test decoding nearby entity."""
    obj: JSONObject = {
        "name": "red-1",
        "direction": "N",
        "coordinates": "10,20",
        "is_private": False,
    }
    decoded = decode_nearby_entity(obj)

    assert decoded["name"] == "red-1"
    assert decoded["direction"] == "N"
    assert decoded["coordinates"] == "10,20"
    assert decoded["is_private"] is False


def test_decode_nearby_entity_missing_field() -> None:
    """Test decoding nearby entity with missing field."""
    with pytest.raises(JSONTypeError):
        decode_nearby_entity({"name": "test", "direction": "N"})


def test_nearby_entity_roundtrip() -> None:
    """Test encode/decode roundtrip for nearby entity."""
    original = make_sample_nearby_entity()
    encoded = encode_nearby_entity(original)
    decoded = decode_nearby_entity(encoded)

    assert decoded == original


# =============================================================================
# SessionInfo encode/decode Tests
# =============================================================================


def test_encode_session_info() -> None:
    """Test encoding session info."""
    info = make_sample_session_info()
    encoded = encode_session_info(info)

    assert encoded["session_id"] == "test-session-123"
    assert encoded["start_timestamp_ms"] == 1234567890000
    assert encoded["magic_key"] == "abcdef123"
    assert encoded["tank_name"] == "TestTank"


def test_decode_session_info() -> None:
    """Test decoding session info."""
    obj: JSONObject = {
        "session_id": "abc",
        "start_timestamp_ms": 1000,
        "magic_key": "key123",
        "tank_name": "Tank1",
    }
    decoded = decode_session_info(obj)

    assert decoded["session_id"] == "abc"
    assert decoded["start_timestamp_ms"] == 1000
    assert decoded["magic_key"] == "key123"
    assert decoded["tank_name"] == "Tank1"


def test_decode_session_info_missing_field() -> None:
    """Test decoding session info with missing field."""
    with pytest.raises(JSONTypeError):
        decode_session_info({"session_id": "test"})


def test_session_info_roundtrip() -> None:
    """Test encode/decode roundtrip for session info."""
    original = make_sample_session_info()
    encoded = encode_session_info(original)
    decoded = decode_session_info(encoded)

    assert decoded == original


# =============================================================================
# GameStateSnapshot encode/decode Tests
# =============================================================================


def test_encode_game_state_snapshot() -> None:
    """Test encoding game state snapshot."""
    snapshot = make_sample_game_state_snapshot()
    encoded = encode_game_state_snapshot(snapshot)

    # Verify by round-tripping through decode (validates structure)
    decoded = decode_game_state_snapshot(encoded)

    # Check session
    assert decoded["session"]["session_id"] == "test-session-123"
    assert decoded["session"]["start_timestamp_ms"] == 1234567890000

    # Check location
    assert decoded["location"]["x"] == 123
    assert decoded["location"]["y"] == 456

    # Check inventory
    assert decoded["inventory"]["armor_shields"]["count"] == 10

    # Check combat stats
    assert len(decoded["combat_stats"]) == 1
    assert decoded["combat_stats"][0]["name"] == "red-1"

    # Check entity pair stats
    assert len(decoded["entity_pair_stats"]) == 1
    assert decoded["entity_pair_stats"][0]["attacker"] == "blue-3"

    # Check nearby entities
    assert len(decoded["nearby_entities"]) == 1
    assert decoded["nearby_entities"][0]["name"] == "blue-7"

    # Check unknown hits
    assert decoded["unknown_hits_received"] == 3


def test_decode_game_state_snapshot() -> None:
    """Test decoding game state snapshot."""
    snapshot = make_sample_game_state_snapshot()
    encoded = encode_game_state_snapshot(snapshot)
    decoded = decode_game_state_snapshot(encoded)

    assert decoded["session"]["session_id"] == "test-session-123"
    assert decoded["location"]["x"] == 123
    assert decoded["unknown_hits_received"] == 3
    assert len(decoded["combat_stats"]) == 1
    assert len(decoded["entity_pair_stats"]) == 1
    assert len(decoded["nearby_entities"]) == 1


def test_decode_game_state_snapshot_session_not_dict() -> None:
    """Test decoding snapshot with non-dict session."""
    with pytest.raises(ValueError, match="session must be a dict"):
        decode_game_state_snapshot({"session": "not_a_dict"})


def test_decode_game_state_snapshot_location_not_dict() -> None:
    """Test decoding snapshot with non-dict location."""
    snapshot = encode_game_state_snapshot(make_sample_game_state_snapshot())
    snapshot["location"] = "not_a_dict"
    with pytest.raises(ValueError, match="location must be a dict"):
        decode_game_state_snapshot(snapshot)


def test_decode_game_state_snapshot_inventory_not_dict() -> None:
    """Test decoding snapshot with non-dict inventory."""
    snapshot = encode_game_state_snapshot(make_sample_game_state_snapshot())
    snapshot["inventory"] = "not_a_dict"
    with pytest.raises(ValueError, match="inventory must be a dict"):
        decode_game_state_snapshot(snapshot)


def test_decode_game_state_snapshot_combat_stats_item_not_dict() -> None:
    """Test decoding snapshot with non-dict combat stats item."""
    snapshot = encode_game_state_snapshot(make_sample_game_state_snapshot())
    snapshot["combat_stats"] = ["not_a_dict"]
    with pytest.raises(ValueError, match=r"combat_stats\[0\] must be a dict"):
        decode_game_state_snapshot(snapshot)


def test_decode_game_state_snapshot_entity_pair_stats_item_not_dict() -> None:
    """Test decoding snapshot with non-dict entity pair stats item."""
    snapshot = encode_game_state_snapshot(make_sample_game_state_snapshot())
    snapshot["entity_pair_stats"] = ["not_a_dict"]
    with pytest.raises(ValueError, match=r"entity_pair_stats\[0\] must be a dict"):
        decode_game_state_snapshot(snapshot)


def test_decode_game_state_snapshot_nearby_entities_item_not_dict() -> None:
    """Test decoding snapshot with non-dict nearby entities item."""
    snapshot = encode_game_state_snapshot(make_sample_game_state_snapshot())
    snapshot["nearby_entities"] = ["not_a_dict"]
    with pytest.raises(ValueError, match=r"nearby_entities\[0\] must be a dict"):
        decode_game_state_snapshot(snapshot)


def test_game_state_snapshot_roundtrip() -> None:
    """Test encode/decode roundtrip for game state snapshot."""
    original = make_sample_game_state_snapshot()
    encoded = encode_game_state_snapshot(original)
    decoded = decode_game_state_snapshot(encoded)

    assert decoded["session"] == original["session"]
    assert decoded["location"] == original["location"]
    assert decoded["unknown_hits_received"] == original["unknown_hits_received"]


def test_game_state_snapshot_empty_lists() -> None:
    """Test encoding snapshot with empty lists."""
    snapshot = GameStateSnapshot(
        session=make_sample_session_info(),
        location=make_sample_location(),
        inventory=make_sample_inventory(),
        combat_stats=[],
        entity_pair_stats=[],
        nearby_entities=[],
        unknown_hits_received=0,
    )
    encoded = encode_game_state_snapshot(snapshot)
    decoded = decode_game_state_snapshot(encoded)

    assert decoded["combat_stats"] == []
    assert decoded["entity_pair_stats"] == []
    assert decoded["nearby_entities"] == []
