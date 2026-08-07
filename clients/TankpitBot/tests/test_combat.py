"""Tests for combat tracking: shots, hits, and damage accounting.

``test_combat.py`` was 776 lines; the outcome and kill-attribution
tests are now a sibling.
"""

from __future__ import annotations

from tankpit_bot.combat import (
    CombatEvent,
    CombatStats,
    EntityPairStats,
    make_entity_pair_key,
    parse_combat_line,
)
from tankpit_bot.combat_tracker import CombatTracker


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
