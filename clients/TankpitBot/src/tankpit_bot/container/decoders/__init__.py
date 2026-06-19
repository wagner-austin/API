"""Container message decoders.

Length-based fallback dispatcher for 0x2E container bodies that have
no unique subtype byte (TipNotification, ChunkData, WorldState,
TeleportLanded, TankStatusShort, TankLeave, TankUpdate*, TankRegistry)
and for subtypes that are container-only with length-based variants
(PositionUpdate, ContainerPickup / DeactivationDeath, MineDetonation,
MinePlacement, PlayerListShort / PlayerListExtended).

This module is callable for tests that target container-only behavior.
In production, the unified entrypoint is
`tankpit_bot.protocol.decoders.tank.decode_0x2e_message`, which
dispatches subtype-first (covering protocol-tunneled types) and falls
through here for length-based container types.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.combat import (
    decode_deactivation_death,
    decode_mine_detonation,
    decode_mine_placement,
    is_deactivation_death_structure,
    is_mine_detonation_structure,
    is_mine_placement_structure,
)
from tankpit_bot.container.decoders.misc import (
    decode_chunk_data,
    decode_container_pickup,
    decode_player_list_extended,
    decode_player_list_short,
    decode_teleport_landed,
    decode_tip_notification,
    decode_unknown_container,
    decode_world_state,
    is_chunk_data_structure,
    is_container_pickup_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_teleport_landed_structure,
    is_tip_notification_structure,
    is_world_state_structure,
)
from tankpit_bot.container.decoders.position import (
    decode_position_update,
    is_position_update_structure,
)
from tankpit_bot.container.decoders.tank import (
    DIRECTION_BYTES,
    DIRECTION_EAST,
    DIRECTION_NORTH,
    DIRECTION_SOUTH,
    DIRECTION_WEST,
    SUBTYPE_MOVEMENT,
    SUBTYPE_TANK_REGISTRY,
    decode_tank_leave,
    decode_tank_registry,
    decode_tank_status_short,
    decode_tank_update_compact,
    decode_tank_update_extended,
    decode_tank_update_full,
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_tank_update_compact_structure,
    is_tank_update_extended_structure,
    is_tank_update_full_structure,
)
from tankpit_bot.container.identification import identify_container_type
from tankpit_bot.container.types import ContainerMessage


def _dispatch_container_subtype(data: bytes) -> ContainerMessage | None:
    """Subtypes valid only inside a 0x2E envelope (no protocol counterpart)."""
    subtype = data[0]
    if subtype == 0x24 and is_position_update_structure(data):
        return decode_position_update(data)
    if subtype == 0x43:
        if is_container_pickup_structure(data):
            return decode_container_pickup(data)
        if is_deactivation_death_structure(data):
            return decode_deactivation_death(data)
    if subtype == 0x45 and is_mine_detonation_structure(data):
        return decode_mine_detonation(data)
    if subtype == 0x4B and is_mine_placement_structure(data):
        return decode_mine_placement(data)
    if subtype == 0x79:
        if is_player_list_short_structure(data):
            return decode_player_list_short(data)
        if is_player_list_extended_structure(data):
            return decode_player_list_extended(data)
    return None


def _dispatch_length_tank(data: bytes) -> ContainerMessage | None:
    """Length-based dispatch for tank-shaped container messages."""
    if is_tank_leave_structure(data):
        return decode_tank_leave(data)
    if is_tank_status_short_structure(data):
        return decode_tank_status_short(data)
    if is_tank_update_compact_structure(data):
        return decode_tank_update_compact(data)
    if is_tank_registry_structure(data):
        return decode_tank_registry(data)
    if is_tank_update_extended_structure(data):
        return decode_tank_update_extended(data)
    if is_tank_update_full_structure(data):
        return decode_tank_update_full(data)
    return None


def _dispatch_length(data: bytes) -> ContainerMessage | None:
    """Length-based fallback for container types without a unique subtype byte."""
    if is_teleport_landed_structure(data):
        return decode_teleport_landed(data)
    tank = _dispatch_length_tank(data)
    if tank is not None:
        return tank
    if is_tip_notification_structure(data):
        return decode_tip_notification(data)
    if is_chunk_data_structure(data):
        return decode_chunk_data(data)
    if is_world_state_structure(data):
        return decode_world_state(data)
    return None


def decode_container_message(data: bytes) -> ContainerMessage:
    """Decode a 0x2E body using container-only logic.

    For the full subtype-first dispatch that also covers protocol-tunneled
    types (TankInfo, MovementResponse, Inventory, ShootEvent, etc.),
    call `decode_0x2e_message` from the protocol module instead.

    Args:
        data: XOR-decoded 0x2E body (subtype byte at data[0]).

    Returns:
        Decoded ContainerMessage, or `UnknownContainerDict` if nothing
        matches.

    Raises:
        ContainerDecodeError: If `data` is empty (no subtype byte).
    """
    if len(data) < 1:
        return decode_unknown_container(data)
    by_subtype = _dispatch_container_subtype(data)
    if by_subtype is not None:
        return by_subtype

    by_length = _dispatch_length(data)
    if by_length is not None:
        return by_length

    return decode_unknown_container(data)


__all__ = [
    "DIRECTION_BYTES",
    "DIRECTION_EAST",
    "DIRECTION_NORTH",
    "DIRECTION_SOUTH",
    "DIRECTION_WEST",
    "SUBTYPE_MOVEMENT",
    "SUBTYPE_TANK_REGISTRY",
    "decode_chunk_data",
    "decode_container_message",
    "decode_container_pickup",
    "decode_deactivation_death",
    "decode_mine_detonation",
    "decode_mine_placement",
    "decode_player_list_extended",
    "decode_player_list_short",
    "decode_position_update",
    "decode_tank_leave",
    "decode_tank_registry",
    "decode_tank_status_short",
    "decode_tank_update_compact",
    "decode_tank_update_extended",
    "decode_tank_update_full",
    "decode_teleport_landed",
    "decode_tip_notification",
    "decode_unknown_container",
    "decode_world_state",
    "identify_container_type",
    "is_chunk_data_structure",
    "is_container_pickup_structure",
    "is_deactivation_death_structure",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
    "is_player_list_extended_structure",
    "is_player_list_short_structure",
    "is_position_update_structure",
    "is_tank_leave_structure",
    "is_tank_registry_structure",
    "is_tank_status_short_structure",
    "is_tank_update_compact_structure",
    "is_tank_update_extended_structure",
    "is_tank_update_full_structure",
    "is_teleport_landed_structure",
    "is_tip_notification_structure",
    "is_world_state_structure",
]
