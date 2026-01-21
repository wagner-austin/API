"""Tests for CombatEvent, MessageStats, and SessionSummary TypedDicts."""

from __future__ import annotations

from tankpit_bot.types import (
    CombatEvent,
    GameLogEntryWithTimestamp,
    MessageStats,
    SessionSummary,
    encode_combat_event,
    encode_message_stats,
    encode_session_summary,
)

# =============================================================================
# CombatEvent Tests
# =============================================================================


def test_encode_combat_event() -> None:
    """Test encoding CombatEvent to JSON."""
    event = CombatEvent(
        timestamp_ms=1234567890,
        event_type="hit",
        target="enemy-tank",
        tank_id=42,
    )
    result = encode_combat_event(event)
    assert result["timestamp_ms"] == 1234567890
    assert result["event_type"] == "hit"
    assert result["target"] == "enemy-tank"
    assert result["tank_id"] == 42


def test_encode_combat_event_with_none_tank_id() -> None:
    """Test encoding CombatEvent with None tank_id."""
    event = CombatEvent(
        timestamp_ms=1000,
        event_type="kill",
        target="victim",
        tank_id=None,
    )
    result = encode_combat_event(event)
    assert result["tank_id"] is None


# =============================================================================
# MessageStats Tests
# =============================================================================


def test_encode_message_stats() -> None:
    """Test encoding MessageStats to JSON."""
    stats = MessageStats(
        decoded={"len=11 combat_hit": 5, "len=13 position": 10},
        unknown={
            "len=7": {"count": 3, "samples": ["abc123", "def456"]},
        },
        total_received=100,
        decode_coverage="85% understood",
    )
    result = encode_message_stats(stats)
    # Verify decoded dict was converted (exercises _int_dict_to_json)
    decoded = result["decoded"]
    assert type(decoded) is dict
    assert decoded["len=11 combat_hit"] == 5
    # Verify unknown dict was converted with nested list (exercises lines 398-408)
    unknown = result["unknown"]
    assert type(unknown) is dict
    len7_entry = unknown["len=7"]
    assert type(len7_entry) is dict
    assert len7_entry["count"] == 3
    samples = len7_entry["samples"]
    assert type(samples) is list
    assert len(samples) == 2


def test_encode_message_stats_with_int_in_unknown() -> None:
    """Test encoding MessageStats handles count as int."""
    stats = MessageStats(
        decoded={},
        unknown={
            "len=5": {"count": 7, "samples": []},
        },
        total_received=10,
        decode_coverage="70%",
    )
    result = encode_message_stats(stats)
    unknown = result["unknown"]
    assert type(unknown) is dict
    len5_entry = unknown["len=5"]
    assert type(len5_entry) is dict
    assert len5_entry["count"] == 7
    assert len5_entry["samples"] == []


# =============================================================================
# SessionSummary Tests
# =============================================================================


def test_encode_session_summary() -> None:
    """Test encoding SessionSummary to JSON."""
    combat_event = CombatEvent(
        timestamp_ms=1000,
        event_type="hit",
        target="enemy",
        tank_id=10,
    )
    log_entry = GameLogEntryWithTimestamp(
        timestamp_ms=1000,
        text="You hit enemy",
        category="combat",
    )
    stats = MessageStats(
        decoded={"combat": 5},
        unknown={},
        total_received=50,
        decode_coverage="100%",
    )
    summary = SessionSummary(
        session_id="summary-123",
        start_timestamp_ms=0,
        end_timestamp_ms=10000,
        magic="xor_key",
        tanks={"1": "Player", "2": "Enemy"},
        combat=[combat_event],
        equipment_gains=[{"type": "armor", "value": 10}],
        game_log=[log_entry],
        message_stats=stats,
    )
    result = encode_session_summary(summary)
    assert result["session_id"] == "summary-123"
    # Verify tanks dict conversion (exercises _str_dict_to_json)
    tanks = result["tanks"]
    assert type(tanks) is dict
    assert tanks["1"] == "Player"
    # Verify equipment_gains conversion (exercises _mixed_dict_to_json, lines 452-453)
    equipment = result["equipment_gains"]
    assert type(equipment) is list
    assert len(equipment) == 1
    equipment_entry = equipment[0]
    assert type(equipment_entry) is dict
    assert equipment_entry["type"] == "armor"
    assert equipment_entry["value"] == 10


def test_encode_session_summary_empty_equipment() -> None:
    """Test encoding SessionSummary with empty equipment_gains."""
    stats = MessageStats(
        decoded={},
        unknown={},
        total_received=0,
        decode_coverage="0%",
    )
    summary = SessionSummary(
        session_id="empty-summary",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        magic=None,
        tanks={},
        combat=[],
        equipment_gains=[],
        game_log=[],
        message_stats=stats,
    )
    result = encode_session_summary(summary)
    equipment = result["equipment_gains"]
    assert type(equipment) is list
    assert len(equipment) == 0
