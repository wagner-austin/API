"""Shared test data factories for game state testing."""

from __future__ import annotations

from tankpit_bot.combat import CombatStats, EntityPairStats
from tankpit_bot.game_state import (
    GameStateSnapshot,
    LocationState,
    NearbyEntity,
    SessionInfo,
)
from tankpit_bot.inventory import InventoryItem, InventoryState


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
