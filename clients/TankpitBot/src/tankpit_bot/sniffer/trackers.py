"""Tracker instances and initialization for message processing.

This module manages all the tracker instances used to process different
types of WebSocket messages and extract game state information.
"""

from __future__ import annotations

from tankpit_bot.capture.trackers import (
    ContainerTracker,
    DeactivationTracker,
    EquipmentGainTracker,
    EquipmentToggleTracker,
    FuelDepositTracker,
    ItemPickupTracker,
    MineTracker,
    PositionTracker,
    RadarAckTracker,
    RadarTracker,
    TankExitTracker,
    TankTracker,
)
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol.codec import extract_magic_from_auth_payload

# Global tracker instances
position_tracker = PositionTracker()
deactivation_tracker = DeactivationTracker()
item_tracker = ItemPickupTracker()
radar_tracker = RadarTracker()
tank_tracker = TankTracker()
mine_tracker = MineTracker()
equip_tracker = EquipmentToggleTracker()
container_tracker = ContainerTracker()
exit_tracker = TankExitTracker()
equip_gain_tracker = EquipmentGainTracker()
deposit_tracker = FuelDepositTracker()
radar_ack_tracker = RadarAckTracker()

# All trackers for bulk initialization (all have set_magic and _xor_table)
ALL_TRACKERS = (
    position_tracker,
    deactivation_tracker,
    item_tracker,
    radar_tracker,
    tank_tracker,
    mine_tracker,
    equip_tracker,
    container_tracker,
    exit_tracker,
    equip_gain_tracker,
    deposit_tracker,
    radar_ack_tracker,
)

# Trackers for received messages (all except mine_tracker which needs direction)
RECEIVED_TRACKERS = (
    position_tracker,
    deactivation_tracker,
    item_tracker,
    radar_tracker,
    tank_tracker,
    equip_tracker,
    container_tracker,
    exit_tracker,
    equip_gain_tracker,
    deposit_tracker,
    radar_ack_tracker,
)


def init_trackers_with_magic(magic: str) -> None:
    """Initialize all trackers with magic key if not already set.

    Args:
        magic: The session magic string for XOR encoding.
    """
    for tracker in ALL_TRACKERS:
        if tracker._xor_table is None:
            tracker.set_magic(magic)


def extract_magic_from_auth(payload: str) -> str | None:
    """Extract magic key from AUTH message payload.

    AUTH message format: %AUTH !be <session_id>|<hash>|<magic>
    The magic is the last space-separated token.

    Args:
        payload: Base64-encoded AUTH message payload.

    Returns:
        Magic key string, or None if not an AUTH message or extraction fails.
    """
    data = decode_base64_safe(payload)
    if data is None:
        return None
    return extract_magic_from_auth_payload(data)


def reset_all_trackers() -> None:
    """Reset all tracker XOR tables for testing."""
    for tracker in ALL_TRACKERS:
        tracker._xor_table = None
        tracker._static_key = None


__all__ = [
    "ALL_TRACKERS",
    "RECEIVED_TRACKERS",
    "container_tracker",
    "deactivation_tracker",
    "deposit_tracker",
    "equip_gain_tracker",
    "equip_tracker",
    "exit_tracker",
    "extract_magic_from_auth",
    "init_trackers_with_magic",
    "item_tracker",
    "mine_tracker",
    "position_tracker",
    "radar_ack_tracker",
    "radar_tracker",
    "reset_all_trackers",
    "tank_tracker",
]
