"""Tests for tankpit_bot.game_state module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.combat import CombatEvent, CombatStats, CombatTracker, EntityPairStats
from tankpit_bot.dom_scraper import GameLogEntry, GameLogScraper
from tankpit_bot.game_state import (
    GameStateManager,
    GameStateSnapshot,
    LocationState,
    NearbyEntity,
    SessionInfo,
    decode_game_state_snapshot,
    decode_location_state,
    decode_nearby_entity,
    decode_session_info,
    encode_game_state_snapshot,
    encode_location_state,
    encode_nearby_entity,
    encode_session_info,
    parse_location,
    parse_radar_detection,
)
from tankpit_bot.inventory import InventoryItem, InventoryScraper, InventoryState

# =============================================================================
# Test Data Factories
# =============================================================================


def make_sample_location() -> LocationState:
    """Create a sample location state.

    Returns:
        Sample LocationState.
    """
    return LocationState(x=123, y=456, raw="123,456")


def make_sample_nearby_entity() -> NearbyEntity:
    """Create a sample nearby entity.

    Returns:
        Sample NearbyEntity.
    """
    return NearbyEntity(
        name="blue-7",
        direction="W",
        coordinates="57,135",
        is_private=True,
    )


def make_sample_session_info() -> SessionInfo:
    """Create a sample session info.

    Returns:
        Sample SessionInfo.
    """
    return SessionInfo(
        session_id="test-session-123",
        start_timestamp_ms=1234567890000,
        magic_key="abcdef123",
        tank_name="TestTank",
    )


def make_sample_inventory() -> InventoryState:
    """Create a sample inventory state.

    Returns:
        Sample InventoryState.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=10, enabled=True),
        dual_shots=InventoryItem(count=5, enabled=False),
        missile_shots=InventoryItem(count=3, enabled=True),
        homing_shots=InventoryItem(count=2, enabled=True),
        extra_radars=InventoryItem(count=1, enabled=False),
    )


def make_sample_combat_stats() -> CombatStats:
    """Create a sample combat stats.

    Returns:
        Sample CombatStats.
    """
    return CombatStats(
        name="red-1",
        hits_given=5,
        hits_received=2,
        deactivated=False,
        destroyed=False,
    )


def make_sample_entity_pair_stats() -> EntityPairStats:
    """Create a sample entity pair stats.

    Returns:
        Sample EntityPairStats.
    """
    return EntityPairStats(
        attacker="blue-3",
        target="red-5",
        hits=3,
        deactivated=False,
        destroyed=False,
    )


def make_sample_game_state_snapshot() -> GameStateSnapshot:
    """Create a sample game state snapshot.

    Returns:
        Sample GameStateSnapshot.
    """
    return GameStateSnapshot(
        session=make_sample_session_info(),
        location=make_sample_location(),
        inventory=make_sample_inventory(),
        combat_stats=[make_sample_combat_stats()],
        entity_pair_stats=[make_sample_entity_pair_stats()],
        nearby_entities=[make_sample_nearby_entity()],
        unknown_hits_received=3,
    )


# =============================================================================
# parse_location Tests
# =============================================================================


def test_parse_location_valid() -> None:
    """Test parsing valid location string."""
    result = parse_location("123,456")
    assert result["x"] == 123
    assert result["y"] == 456
    assert result["raw"] == "123,456"


def test_parse_location_with_spaces() -> None:
    """Test parsing location with spaces."""
    result = parse_location(" 100 , 200 ")
    assert result["x"] == 100
    assert result["y"] == 200


def test_parse_location_empty_string() -> None:
    """Test parsing empty location string."""
    result = parse_location("")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == ""


def test_parse_location_no_comma() -> None:
    """Test parsing location without comma."""
    result = parse_location("12345")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == "12345"


def test_parse_location_too_many_parts() -> None:
    """Test parsing location with too many parts."""
    result = parse_location("1,2,3")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == "1,2,3"


def test_parse_location_non_numeric() -> None:
    """Test parsing location with non-numeric values."""
    result = parse_location("abc,def")
    assert result["x"] == 0
    assert result["y"] == 0


def test_parse_location_mixed_numeric() -> None:
    """Test parsing location with mixed values."""
    result = parse_location("123,abc")
    assert result["x"] == 0
    assert result["y"] == 0


# =============================================================================
# parse_radar_detection Tests
# =============================================================================


def test_parse_radar_detection_basic() -> None:
    """Test parsing basic radar detection."""
    result = parse_radar_detection("blue-7 detected to W [57,135]")
    # Verify all fields are correctly parsed
    assert result == NearbyEntity(
        name="blue-7",
        direction="W",
        coordinates="57,135",
        is_private=False,
    )


def test_parse_radar_detection_with_private() -> None:
    """Test parsing radar detection with private flag."""
    result = parse_radar_detection("blue-7 (private) detected to W [57,135]")
    # Verify private flag is correctly parsed
    assert result == NearbyEntity(
        name="blue-7",
        direction="W",
        coordinates="57,135",
        is_private=True,
    )


def test_parse_radar_detection_cardinal_directions() -> None:
    """Test parsing all cardinal directions."""
    for direction in ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]:
        result = parse_radar_detection(f"enemy detected to {direction} [0,0]")
        # Each direction should parse to correct NearbyEntity
        assert result == NearbyEntity(
            name="enemy",
            direction=direction,
            coordinates="0,0",
            is_private=False,
        )


def test_parse_radar_detection_with_spaces() -> None:
    """Test parsing radar detection with extra spaces."""
    result = parse_radar_detection("  red-1 detected to NE [100,200]  ")
    # Verify spaces are trimmed correctly
    assert result == NearbyEntity(
        name="red-1",
        direction="NE",
        coordinates="100,200",
        is_private=False,
    )


def test_parse_radar_detection_invalid_format() -> None:
    """Test parsing invalid radar detection format."""
    result = parse_radar_detection("some random text")
    assert result is None


def test_parse_radar_detection_missing_brackets() -> None:
    """Test parsing radar detection without brackets."""
    result = parse_radar_detection("blue-7 detected to W 57,135")
    assert result is None


def test_parse_radar_detection_empty_string() -> None:
    """Test parsing empty string."""
    result = parse_radar_detection("")
    assert result is None


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


# =============================================================================
# GameStateManager Tests
# =============================================================================


def test_game_state_manager_init() -> None:
    """Test GameStateManager initialization."""
    manager = GameStateManager()

    location = manager.get_location()
    assert location["x"] == 0
    assert location["y"] == 0

    session = manager.get_session_info()
    assert session["session_id"] == ""

    entities = manager.get_nearby_entities()
    assert entities == []


def test_game_state_manager_update_session_full() -> None:
    """Test updating all session fields."""
    manager = GameStateManager()
    manager.update_session(
        session_id="test-id",
        start_timestamp_ms=1000,
        magic_key="key123",
        tank_name="MyTank",
    )

    session = manager.get_session_info()
    assert session["session_id"] == "test-id"
    assert session["start_timestamp_ms"] == 1000
    assert session["magic_key"] == "key123"
    assert session["tank_name"] == "MyTank"


def test_game_state_manager_update_session_partial() -> None:
    """Test updating only some session fields."""
    manager = GameStateManager()
    manager.update_session(session_id="first-id", tank_name="Tank1")
    manager.update_session(magic_key="new-key")

    session = manager.get_session_info()
    assert session["session_id"] == "first-id"
    assert session["magic_key"] == "new-key"
    assert session["tank_name"] == "Tank1"


def test_game_state_manager_process_location_entry() -> None:
    """Test processing location log entry."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="LOCATION: 100,200",
        category="location",
    )
    manager.process_game_log_entry(entry)

    location = manager.get_location()
    assert location["x"] == 100
    assert location["y"] == 200


def test_game_state_manager_process_radar_entry() -> None:
    """Test processing radar detection log entry."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="red-5 (private) detected to NE [50,75]",
        category="action",
    )
    manager.process_game_log_entry(entry)

    entities = manager.get_nearby_entities()
    assert len(entities) == 1
    assert entities[0]["name"] == "red-5"
    assert entities[0]["direction"] == "NE"
    assert entities[0]["is_private"] is True


def test_game_state_manager_updates_existing_entity() -> None:
    """Test that radar updates existing entity by name."""
    manager = GameStateManager()

    # First detection
    entry1 = GameLogEntry(
        text="blue-3 detected to N [10,20]",
        category="action",
    )
    manager.process_game_log_entry(entry1)

    # Second detection of same entity
    entry2 = GameLogEntry(
        text="blue-3 detected to S [30,40]",
        category="action",
    )
    manager.process_game_log_entry(entry2)

    entities = manager.get_nearby_entities()
    assert len(entities) == 1
    assert entities[0]["direction"] == "S"
    assert entities[0]["coordinates"] == "30,40"


def test_game_state_manager_clear_nearby_entities() -> None:
    """Test clearing nearby entities."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="red-1 detected to W [0,0]",
        category="action",
    )
    manager.process_game_log_entry(entry)

    assert len(manager.get_nearby_entities()) == 1

    manager.clear_nearby_entities()

    assert len(manager.get_nearby_entities()) == 0


def test_game_state_manager_ignores_non_location_category() -> None:
    """Test that LOCATION entries with wrong category are ignored."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="LOCATION: 100,200",
        category="action",  # Wrong category
    )
    manager.process_game_log_entry(entry)

    location = manager.get_location()
    assert location["x"] == 0  # Not updated


def test_game_state_manager_ignores_non_radar_action() -> None:
    """Test that non-radar action entries are ignored."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="You fired at enemy",
        category="action",
    )
    manager.process_game_log_entry(entry)

    assert len(manager.get_nearby_entities()) == 0


def test_game_state_manager_ignores_malformed_radar() -> None:
    """Test that malformed radar entries (with 'detected to' but invalid format) are ignored."""
    manager = GameStateManager()
    # Contains "detected to" but doesn't match the expected pattern
    entry = GameLogEntry(
        text="Something detected to somewhere invalid",
        category="action",
    )
    manager.process_game_log_entry(entry)

    assert len(manager.get_nearby_entities()) == 0


def test_game_state_manager_set_combat_tracker() -> None:
    """Test setting combat tracker."""
    manager = GameStateManager()
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_hit", attacker="red-1", target="blue-1")
    tracker.record_event(event)

    manager.set_combat_tracker(tracker)
    snapshot = manager.get_snapshot()

    assert len(snapshot["entity_pair_stats"]) == 1


def test_game_state_manager_set_inventory_scraper() -> None:
    """Test setting inventory scraper."""

    class FakeInventoryScraper(InventoryScraper):
        """Fake inventory scraper for testing."""

        def __init__(self) -> None:
            """Initialize without CDP session."""
            pass

        def scrape(self) -> InventoryState:
            """Return sample inventory."""
            return make_sample_inventory()

    manager = GameStateManager()
    manager.set_inventory_scraper(FakeInventoryScraper())
    snapshot = manager.get_snapshot()

    assert snapshot["inventory"]["armor_shields"]["count"] == 10


def test_game_state_manager_set_game_log_scraper() -> None:
    """Test setting game log scraper."""
    manager = GameStateManager()

    class FakeGameLogScraper(GameLogScraper):
        """Fake game log scraper for testing."""

        def __init__(self) -> None:
            """Initialize without CDP session."""
            pass

    manager.set_game_log_scraper(FakeGameLogScraper())
    # Verify snapshot can still be created with scraper set
    snapshot = manager.get_snapshot()
    # Snapshot should have default empty values
    assert snapshot["location"]["x"] == 0
    assert snapshot["nearby_entities"] == []


def test_game_state_manager_snapshot_without_trackers() -> None:
    """Test getting snapshot without any trackers set."""
    manager = GameStateManager()
    snapshot = manager.get_snapshot()

    assert snapshot["combat_stats"] == []
    assert snapshot["entity_pair_stats"] == []
    assert snapshot["unknown_hits_received"] == 0
    assert snapshot["inventory"]["armor_shields"]["count"] == 0


def test_game_state_manager_snapshot_with_combat_tracker() -> None:
    """Test snapshot includes combat tracker data."""
    manager = GameStateManager()
    tracker = CombatTracker()

    # Record some entity hits
    event1 = CombatEvent(event_type="entity_hit", attacker="red-1", target="blue-1")
    tracker.record_event(event1)
    event2 = CombatEvent(event_type="entity_hit", attacker="blue-1", target="red-1")
    tracker.record_event(event2)

    manager.set_combat_tracker(tracker)
    snapshot = manager.get_snapshot()

    # Should have entity pair stats
    assert len(snapshot["entity_pair_stats"]) == 2

    # Combat stats is player-centric, these are entity-to-entity
    # so combat_stats would be empty unless player involved
    assert len(snapshot["combat_stats"]) == 0


def test_game_state_manager_snapshot_with_unknown_hits() -> None:
    """Test snapshot includes unknown hits count."""
    manager = GameStateManager()
    tracker = CombatTracker()

    # Record hit from unknown attacker
    event = CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")
    tracker.record_event(event)

    manager.set_combat_tracker(tracker)
    snapshot = manager.get_snapshot()

    assert snapshot["unknown_hits_received"] == 1


def test_game_state_manager_nearby_entities_copied() -> None:
    """Test that get_nearby_entities returns a copy."""
    manager = GameStateManager()
    entry = GameLogEntry(
        text="enemy detected to N [0,0]",
        category="action",
    )
    manager.process_game_log_entry(entry)

    entities1 = manager.get_nearby_entities()
    entities2 = manager.get_nearby_entities()

    # Should be equal but not the same list object
    assert entities1 == entities2
    assert entities1 is not entities2
