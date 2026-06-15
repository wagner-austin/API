"""Tests for tankpit_bot.combat module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.combat import (
    VALID_COMBAT_EVENT_TYPES,
    CombatEvent,
    CombatStats,
    EntityPairStats,
    decode_combat_event,
    decode_combat_stats,
    decode_entity_pair_stats,
    encode_combat_event,
    encode_combat_stats,
    encode_entity_pair_stats,
    make_entity_pair_key,
    parse_combat_line,
    validate_combat_event_type,
)
from tankpit_bot.combat_tracker import CombatTracker

# =============================================================================
# parse_combat_line Tests - Player-centric
# =============================================================================


def test_parse_combat_line_hit_by_player() -> None:
    """Test parsing 'You hit {target}' line."""
    result = parse_combat_line("You hit blue-7")
    assert result == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")


def test_parse_combat_line_hit_by_enemy() -> None:
    """Test parsing '{attacker} hit you' line."""
    result = parse_combat_line("blue-7 hit you")
    assert result == CombatEvent(event_type="hit_by_enemy", attacker="blue-7", target="player")


def test_parse_combat_line_hit_by_unknown() -> None:
    """Test parsing 'You are hit' line (off-screen attacker)."""
    result = parse_combat_line("You are hit")
    assert result == CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")


def test_parse_combat_line_deactivated() -> None:
    """Test parsing '{target} has been deactivated by you' line."""
    result = parse_combat_line("red-9 has been deactivated by you")
    assert result == CombatEvent(event_type="deactivated", attacker="player", target="red-9")


def test_parse_combat_line_destroyed() -> None:
    """Test parsing '{target} has been destroyed by you' line."""
    result = parse_combat_line("green-3 has been destroyed by you")
    assert result == CombatEvent(event_type="destroyed", attacker="player", target="green-3")


def test_parse_combat_line_non_combat() -> None:
    """Test parsing non-combat line returns None."""
    result = parse_combat_line("LOCATION: 1,2")
    assert result is None


def test_parse_combat_line_strips_whitespace() -> None:
    """Test parsing line with leading/trailing whitespace."""
    result = parse_combat_line("  You hit blue-7  ")
    assert result == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")


def test_parse_combat_line_empty() -> None:
    """Test parsing empty line returns None."""
    result = parse_combat_line("")
    assert result is None


def test_parse_combat_line_whitespace_only() -> None:
    """Test parsing whitespace-only line returns None."""
    result = parse_combat_line("   ")
    assert result is None


# =============================================================================
# parse_combat_line Tests - Entity-to-Entity
# =============================================================================


def test_parse_combat_line_entity_hit() -> None:
    """Test parsing entity-to-entity hit line."""
    result = parse_combat_line("blue-7 hit red-9")
    assert result == CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")


def test_parse_combat_line_entity_deactivated() -> None:
    """Test parsing entity-to-entity deactivation line."""
    result = parse_combat_line("red-9 has been deactivated by blue-7")
    assert result == CombatEvent(event_type="entity_deactivated", attacker="blue-7", target="red-9")


def test_parse_combat_line_entity_destroyed() -> None:
    """Test parsing entity-to-entity destruction line."""
    result = parse_combat_line("green-3 has been destroyed by cyan-4")
    assert result == CombatEvent(event_type="entity_destroyed", attacker="cyan-4", target="green-3")


def test_parse_combat_line_entity_hit_not_player() -> None:
    """Test entity hit doesn't match when target is 'you'."""
    # "X hit you" should be hit_by_enemy, not entity_hit
    result = parse_combat_line("blue-7 hit you")
    assert result == CombatEvent(event_type="hit_by_enemy", attacker="blue-7", target="player")


def test_parse_entity_to_entity_deactivated_by_you_falls_through() -> None:
    """Test entity deactivated pattern with 'you' falls through to next check."""
    from tankpit_bot.combat import _parse_entity_to_entity

    # When attacker is "you", should fall through (returns None, caught by player-centric)
    result = _parse_entity_to_entity("target has been deactivated by you")
    # Falls through and hits destroyed pattern (no match), then hit pattern (no match)
    assert result is None


def test_parse_entity_to_entity_destroyed_by_you_falls_through() -> None:
    """Test entity destroyed pattern with 'you' falls through."""
    from tankpit_bot.combat import _parse_entity_to_entity

    result = _parse_entity_to_entity("target has been destroyed by you")
    assert result is None


def test_parse_entity_to_entity_you_hit_falls_through() -> None:
    """Test entity hit with 'You' as attacker falls through."""
    from tankpit_bot.combat import _parse_entity_to_entity

    result = _parse_entity_to_entity("You hit target")
    assert result is None


# =============================================================================
# make_entity_pair_key Tests
# =============================================================================


def test_make_entity_pair_key() -> None:
    """Test make_entity_pair_key creates correct key."""
    key = make_entity_pair_key("blue-7", "red-9")
    assert key == "blue-7->red-9"


def test_make_entity_pair_key_different_order() -> None:
    """Test make_entity_pair_key preserves order."""
    key1 = make_entity_pair_key("blue-7", "red-9")
    key2 = make_entity_pair_key("red-9", "blue-7")
    assert key1 != key2
    assert key1 == "blue-7->red-9"
    assert key2 == "red-9->blue-7"


# =============================================================================
# CombatTracker Tests - Player-centric
# =============================================================================


def test_combat_tracker_init() -> None:
    """Test CombatTracker initialization."""
    tracker = CombatTracker()
    assert tracker.get_all_stats() == []
    assert tracker.get_all_entity_pair_stats() == []
    assert tracker.get_events() == []
    assert tracker.get_unknown_hits_received() == 0


def test_combat_tracker_record_hit_by_player() -> None:
    """Test recording player hits."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    tracker.record_event(event)

    stats = tracker.get_stats("blue-7")
    assert stats == CombatStats(
        name="blue-7",
        hits_given=1,
        hits_received=0,
        deactivated=False,
        destroyed=False,
    )


def test_combat_tracker_record_hit_by_enemy() -> None:
    """Test recording enemy hits."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_enemy", attacker="red-5", target="player")
    tracker.record_event(event)

    stats = tracker.get_stats("red-5")
    assert stats == CombatStats(
        name="red-5",
        hits_given=0,
        hits_received=1,
        deactivated=False,
        destroyed=False,
    )


def test_combat_tracker_record_hit_by_unknown() -> None:
    """Test recording off-screen hits."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")
    tracker.record_event(event)

    assert tracker.get_unknown_hits_received() == 1


def test_combat_tracker_record_deactivated() -> None:
    """Test recording deactivation."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="deactivated", attacker="player", target="green-1")
    tracker.record_event(event)

    stats = tracker.get_stats("green-1")
    assert stats == CombatStats(
        name="green-1",
        hits_given=0,
        hits_received=0,
        deactivated=True,
        destroyed=False,
    )


def test_combat_tracker_record_destroyed() -> None:
    """Test recording destruction."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="destroyed", attacker="player", target="cyan-2")
    tracker.record_event(event)

    stats = tracker.get_stats("cyan-2")
    assert stats == CombatStats(
        name="cyan-2",
        hits_given=0,
        hits_received=0,
        deactivated=False,
        destroyed=True,
    )


def test_combat_tracker_multiple_hits_same_target() -> None:
    """Test multiple hits on same target accumulate."""
    tracker = CombatTracker()

    hit_player = CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    for _ in range(5):
        tracker.record_event(hit_player)

    hit_enemy = CombatEvent(event_type="hit_by_enemy", attacker="blue-7", target="player")
    for _ in range(3):
        tracker.record_event(hit_enemy)

    stats = tracker.get_stats("blue-7")
    assert stats == CombatStats(
        name="blue-7",
        hits_given=5,
        hits_received=3,
        deactivated=False,
        destroyed=False,
    )


def test_combat_tracker_multiple_unknown_hits() -> None:
    """Test multiple off-screen hits accumulate."""
    tracker = CombatTracker()

    hit_unknown = CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")
    for _ in range(4):
        tracker.record_event(hit_unknown)

    assert tracker.get_unknown_hits_received() == 4


# =============================================================================
# CombatTracker Tests - Entity-to-Entity
# =============================================================================


def test_combat_tracker_record_entity_hit() -> None:
    """Test recording entity-to-entity hit."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    tracker.record_event(event)

    stats = tracker.get_entity_pair_stats("blue-7", "red-9")
    assert stats == EntityPairStats(
        attacker="blue-7",
        target="red-9",
        hits=1,
        deactivated=False,
        destroyed=False,
    )


def test_combat_tracker_record_entity_deactivated() -> None:
    """Test recording entity-to-entity deactivation."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_deactivated", attacker="blue-7", target="red-9")
    tracker.record_event(event)

    stats = tracker.get_entity_pair_stats("blue-7", "red-9")
    assert stats == EntityPairStats(
        attacker="blue-7",
        target="red-9",
        hits=0,
        deactivated=True,
        destroyed=False,
    )


def test_combat_tracker_record_entity_destroyed() -> None:
    """Test recording entity-to-entity destruction."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_destroyed", attacker="cyan-4", target="green-3")
    tracker.record_event(event)

    stats = tracker.get_entity_pair_stats("cyan-4", "green-3")
    assert stats == EntityPairStats(
        attacker="cyan-4",
        target="green-3",
        hits=0,
        deactivated=False,
        destroyed=True,
    )


def test_combat_tracker_multiple_entity_hits_same_pair() -> None:
    """Test multiple entity hits on same pair accumulate."""
    tracker = CombatTracker()

    event = CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    for _ in range(5):
        tracker.record_event(event)

    stats = tracker.get_entity_pair_stats("blue-7", "red-9")
    assert stats == EntityPairStats(
        attacker="blue-7",
        target="red-9",
        hits=5,
        deactivated=False,
        destroyed=False,
    )


def test_combat_tracker_entity_pairs_direction_matters() -> None:
    """Test that entity pair direction matters (A->B != B->A)."""
    tracker = CombatTracker()

    # blue-7 hits red-9
    tracker.record_event(CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9"))
    # red-9 hits blue-7
    tracker.record_event(CombatEvent(event_type="entity_hit", attacker="red-9", target="blue-7"))

    stats_ab = tracker.get_entity_pair_stats("blue-7", "red-9")
    stats_ba = tracker.get_entity_pair_stats("red-9", "blue-7")

    assert stats_ab == EntityPairStats(
        attacker="blue-7", target="red-9", hits=1, deactivated=False, destroyed=False
    )
    assert stats_ba == EntityPairStats(
        attacker="red-9", target="blue-7", hits=1, deactivated=False, destroyed=False
    )


def test_combat_tracker_get_all_entity_pair_stats() -> None:
    """Test get_all_entity_pair_stats returns all pairs."""
    tracker = CombatTracker()
    tracker.record_event(CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9"))
    tracker.record_event(CombatEvent(event_type="entity_hit", attacker="green-1", target="cyan-4"))

    all_stats = tracker.get_all_entity_pair_stats()
    assert len(all_stats) == 2


def test_combat_tracker_get_entity_pair_stats_unknown() -> None:
    """Test get_entity_pair_stats returns None for unknown pair."""
    tracker = CombatTracker()
    assert tracker.get_entity_pair_stats("nonexistent", "pair") is None


# =============================================================================
# CombatTracker Tests - process_log_line
# =============================================================================


def test_combat_tracker_process_log_line() -> None:
    """Test processing log line creates event."""
    tracker = CombatTracker()
    event = tracker.process_log_line("You hit blue-7")

    assert event == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    assert len(tracker.get_events()) == 1


def test_combat_tracker_process_log_line_non_combat() -> None:
    """Test processing non-combat log line returns None."""
    tracker = CombatTracker()
    event = tracker.process_log_line("LOCATION: 10,20")

    assert event is None
    assert len(tracker.get_events()) == 0


def test_combat_tracker_process_log_line_entity_hit() -> None:
    """Test processing entity-to-entity hit log line."""
    tracker = CombatTracker()
    event = tracker.process_log_line("blue-7 hit red-9")

    assert event == CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    assert len(tracker.get_all_entity_pair_stats()) == 1


def test_combat_tracker_get_events() -> None:
    """Test get_events returns all recorded events."""
    tracker = CombatTracker()
    tracker.process_log_line("You hit blue-7")
    tracker.process_log_line("red-5 hit you")

    events = tracker.get_events()
    assert len(events) == 2
    assert events[0]["target"] == "blue-7"
    assert events[1]["attacker"] == "red-5"


def test_combat_tracker_get_all_stats() -> None:
    """Test get_all_stats returns stats for all targets."""
    tracker = CombatTracker()
    tracker.process_log_line("You hit blue-7")
    tracker.process_log_line("red-5 hit you")
    tracker.process_log_line("You hit green-1")

    all_stats = tracker.get_all_stats()
    assert len(all_stats) == 3
    names = [s["name"] for s in all_stats]
    assert "blue-7" in names
    assert "red-5" in names
    assert "green-1" in names


def test_combat_tracker_get_stats_unknown_target() -> None:
    """Test get_stats returns None for unknown target."""
    tracker = CombatTracker()
    assert tracker.get_stats("nonexistent") is None


# =============================================================================
# CombatTracker Tests - log_event
# =============================================================================


def test_combat_tracker_log_event_hit_by_player() -> None:
    """Test log_event logs hit by player correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_hit_by_enemy() -> None:
    """Test log_event logs hit by enemy correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_enemy", attacker="red-5", target="player")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_hit_by_unknown() -> None:
    """Test log_event logs off-screen hit correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_deactivated() -> None:
    """Test log_event logs deactivation correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="deactivated", attacker="player", target="green-1")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_destroyed() -> None:
    """Test log_event logs destruction correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="destroyed", attacker="player", target="cyan-2")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_entity_hit() -> None:
    """Test log_event logs entity-to-entity hit correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_entity_deactivated() -> None:
    """Test log_event logs entity-to-entity deactivation correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_deactivated", attacker="blue-7", target="red-9")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_entity_destroyed() -> None:
    """Test log_event logs entity-to-entity destruction correctly."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_destroyed", attacker="cyan-4", target="green-3")
    tracker.record_event(event)
    # Should not raise
    tracker.log_event(event)


def test_combat_tracker_log_event_without_recording() -> None:
    """Test log_event when stats not yet recorded (edge case)."""
    tracker = CombatTracker()
    # Create event but don't record it
    event = CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    # Should not raise even without recording
    tracker.log_event(event)


def test_combat_tracker_log_event_hit_by_enemy_without_recording() -> None:
    """Test log_event for enemy hit when stats not yet recorded."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="hit_by_enemy", attacker="red-5", target="player")
    # Should not raise even without recording
    tracker.log_event(event)


def test_combat_tracker_log_event_entity_hit_without_recording() -> None:
    """Test log_event for entity hit when stats not yet recorded."""
    tracker = CombatTracker()
    event = CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    # Should not raise even without recording
    tracker.log_event(event)


# =============================================================================
# Encode/Decode Tests - CombatEvent
# =============================================================================


def test_encode_combat_event() -> None:
    """Test encoding CombatEvent to JSON."""
    event = CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    encoded = encode_combat_event(event)

    assert encoded["event_type"] == "hit_by_player"
    assert encoded["attacker"] == "player"
    assert encoded["target"] == "blue-7"


def test_decode_combat_event() -> None:
    """Test decoding JSON to CombatEvent."""
    obj: JSONObject = {"event_type": "hit_by_enemy", "attacker": "red-5", "target": "player"}
    decoded = decode_combat_event(obj)

    assert decoded["event_type"] == "hit_by_enemy"
    assert decoded["attacker"] == "red-5"
    assert decoded["target"] == "player"


def test_encode_decode_combat_event_roundtrip() -> None:
    """Test encode/decode roundtrip for CombatEvent."""
    original = CombatEvent(event_type="deactivated", attacker="player", target="green-1")
    encoded = encode_combat_event(original)
    decoded = decode_combat_event(encoded)

    assert decoded == original


def test_encode_decode_combat_event_entity_hit() -> None:
    """Test encode/decode roundtrip for entity_hit event."""
    original = CombatEvent(event_type="entity_hit", attacker="blue-7", target="red-9")
    encoded = encode_combat_event(original)
    decoded = decode_combat_event(encoded)

    assert decoded == original


# =============================================================================
# Encode/Decode Tests - CombatStats
# =============================================================================


def test_encode_combat_stats() -> None:
    """Test encoding CombatStats to JSON."""
    stats = CombatStats(
        name="blue-7",
        hits_given=5,
        hits_received=3,
        deactivated=True,
        destroyed=False,
    )
    encoded = encode_combat_stats(stats)

    assert encoded["name"] == "blue-7"
    assert encoded["hits_given"] == 5
    assert encoded["hits_received"] == 3
    assert encoded["deactivated"] is True
    assert encoded["destroyed"] is False


def test_decode_combat_stats() -> None:
    """Test decoding JSON to CombatStats."""
    obj: JSONObject = {
        "name": "red-5",
        "hits_given": 10,
        "hits_received": 2,
        "deactivated": False,
        "destroyed": True,
    }
    decoded = decode_combat_stats(obj)

    assert decoded["name"] == "red-5"
    assert decoded["hits_given"] == 10
    assert decoded["hits_received"] == 2
    assert decoded["deactivated"] is False
    assert decoded["destroyed"] is True


def test_encode_decode_combat_stats_roundtrip() -> None:
    """Test encode/decode roundtrip for CombatStats."""
    original = CombatStats(
        name="cyan-2",
        hits_given=7,
        hits_received=1,
        deactivated=True,
        destroyed=True,
    )
    encoded = encode_combat_stats(original)
    decoded = decode_combat_stats(encoded)

    assert decoded == original


# =============================================================================
# Encode/Decode Tests - EntityPairStats
# =============================================================================


def test_encode_entity_pair_stats() -> None:
    """Test encoding EntityPairStats to JSON."""
    stats = EntityPairStats(
        attacker="blue-7",
        target="red-9",
        hits=5,
        deactivated=False,
        destroyed=True,
    )
    encoded = encode_entity_pair_stats(stats)

    assert encoded["attacker"] == "blue-7"
    assert encoded["target"] == "red-9"
    assert encoded["hits"] == 5
    assert encoded["deactivated"] is False
    assert encoded["destroyed"] is True


def test_decode_entity_pair_stats() -> None:
    """Test decoding JSON to EntityPairStats."""
    obj: JSONObject = {
        "attacker": "cyan-4",
        "target": "green-3",
        "hits": 10,
        "deactivated": True,
        "destroyed": False,
    }
    decoded = decode_entity_pair_stats(obj)

    assert decoded["attacker"] == "cyan-4"
    assert decoded["target"] == "green-3"
    assert decoded["hits"] == 10
    assert decoded["deactivated"] is True
    assert decoded["destroyed"] is False


def test_encode_decode_entity_pair_stats_roundtrip() -> None:
    """Test encode/decode roundtrip for EntityPairStats."""
    original = EntityPairStats(
        attacker="blue-7",
        target="red-9",
        hits=3,
        deactivated=True,
        destroyed=True,
    )
    encoded = encode_entity_pair_stats(original)
    decoded = decode_entity_pair_stats(encoded)

    assert decoded == original


def test_decode_entity_pair_stats_missing_field() -> None:
    """Test decode_entity_pair_stats raises on missing field."""
    from platform_core.json_utils import JSONTypeError

    obj: JSONObject = {"attacker": "blue-7", "target": "red-9"}
    with pytest.raises(JSONTypeError):
        decode_entity_pair_stats(obj)


# =============================================================================
# Validation Tests
# =============================================================================


def test_valid_combat_event_types_constant() -> None:
    """Test VALID_COMBAT_EVENT_TYPES contains all types."""
    assert "hit_by_player" in VALID_COMBAT_EVENT_TYPES
    assert "hit_by_enemy" in VALID_COMBAT_EVENT_TYPES
    assert "hit_by_unknown" in VALID_COMBAT_EVENT_TYPES
    assert "deactivated" in VALID_COMBAT_EVENT_TYPES
    assert "destroyed" in VALID_COMBAT_EVENT_TYPES
    assert "entity_hit" in VALID_COMBAT_EVENT_TYPES
    assert "entity_deactivated" in VALID_COMBAT_EVENT_TYPES
    assert "entity_destroyed" in VALID_COMBAT_EVENT_TYPES
    assert len(VALID_COMBAT_EVENT_TYPES) == 8


def test_validate_combat_event_type_all_valid() -> None:
    """Test validate_combat_event_type accepts all valid types."""
    assert validate_combat_event_type("hit_by_player") == "hit_by_player"
    assert validate_combat_event_type("hit_by_enemy") == "hit_by_enemy"
    assert validate_combat_event_type("hit_by_unknown") == "hit_by_unknown"
    assert validate_combat_event_type("deactivated") == "deactivated"
    assert validate_combat_event_type("destroyed") == "destroyed"
    assert validate_combat_event_type("entity_hit") == "entity_hit"
    assert validate_combat_event_type("entity_deactivated") == "entity_deactivated"
    assert validate_combat_event_type("entity_destroyed") == "entity_destroyed"


def test_validate_combat_event_type_invalid() -> None:
    """Test validate_combat_event_type raises on invalid type."""
    with pytest.raises(ValueError, match="Invalid combat event type"):
        validate_combat_event_type("invalid_type")


def test_decode_combat_event_invalid_type() -> None:
    """Test decode_combat_event raises on invalid event type."""
    obj: JSONObject = {"event_type": "bad_type", "attacker": "player", "target": "enemy"}
    with pytest.raises(ValueError, match="Invalid combat event type"):
        decode_combat_event(obj)


def test_decode_combat_event_missing_field() -> None:
    """Test decode_combat_event raises on missing field."""
    from platform_core.json_utils import JSONTypeError

    obj: JSONObject = {"event_type": "hit_by_player", "attacker": "player"}
    with pytest.raises(JSONTypeError):
        decode_combat_event(obj)


def test_decode_combat_stats_missing_field() -> None:
    """Test decode_combat_stats raises on missing field."""
    from platform_core.json_utils import JSONTypeError

    obj: JSONObject = {"name": "blue-7", "hits_given": 5}
    with pytest.raises(JSONTypeError):
        decode_combat_stats(obj)
