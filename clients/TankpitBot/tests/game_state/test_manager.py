"""Tests for GameStateManager class."""

from __future__ import annotations

from tankpit_bot.browser import GameLogEntry, GameLogScraper
from tankpit_bot.combat import CombatEvent
from tankpit_bot.combat_tracker import CombatTracker
from tankpit_bot.game_state import GameStateManager
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol

# =============================================================================
# GameStateManager Basic Tests
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


# =============================================================================
# GameStateManager Log Entry Processing Tests
# =============================================================================


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


# =============================================================================
# GameStateManager Tracker Integration Tests
# =============================================================================


def test_game_state_manager_set_combat_tracker() -> None:
    """Test setting combat tracker."""
    manager = GameStateManager()
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_hit", attacker="red-1", target="blue-1")
    tracker.record_event(event)

    manager.set_combat_tracker(tracker)
    snapshot = manager.get_snapshot()

    assert len(snapshot["entity_pair_stats"]) == 1


def test_game_state_manager_uses_binary_inventory() -> None:
    """Test snapshot reads inventory from binary protocol tracking."""
    reset_world_state()
    update_inventory_from_protocol(
        counts=[10, 5, 3, 2, 1],
        enabled=[True, False, True, True, False],
    )

    manager = GameStateManager()
    snapshot = manager.get_snapshot()

    assert snapshot["inventory"]["armor_shields"]["count"] == 10
    assert snapshot["inventory"]["dual_shots"]["enabled"] is False

    reset_world_state()


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


# =============================================================================
# GameStateManager Snapshot Tests
# =============================================================================


def test_game_state_manager_snapshot_without_trackers() -> None:
    """Test getting snapshot without any trackers set."""
    reset_world_state()
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
