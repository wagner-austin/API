"""Container message decoders.

Length-based dispatcher for 0x2E container bodies that have no unique
subtype byte (TeleportLanded) and for subtypes that are container-only
with length-based variants (ContainerPickup, MineDetonation,
MinePlacement).

This module is callable for tests that target container-only behavior.
In production, the unified entrypoint is
`tankpit_bot.protocol.decoders.tank.decode_0x2e_message`, which
dispatches subtype-first (covering protocol-tunneled types) and falls
through here for length-based container types.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.events import (
    decode_container_pickup,
    decode_teleport_landed,
    decode_unknown_container,
    is_container_pickup_structure,
    is_teleport_landed_structure,
)
from tankpit_bot.container.decoders.mines import (
    decode_mine_detonation,
    decode_mine_placement,
    is_mine_detonation_structure,
    is_mine_placement_structure,
)
from tankpit_bot.container.identification import identify_container_type
from tankpit_bot.container.types import ContainerMessage


def _dispatch_container_subtype(data: bytes) -> ContainerMessage | None:
    """Subtypes valid only inside a 0x2E envelope (no protocol counterpart)."""
    subtype = data[0]
    if subtype == 0x43 and is_container_pickup_structure(data):
        return decode_container_pickup(data)
    if subtype == 0x45 and is_mine_detonation_structure(data):
        return decode_mine_detonation(data)
    if subtype == 0x4B and is_mine_placement_structure(data):
        return decode_mine_placement(data)
    return None


def _dispatch_length(data: bytes) -> ContainerMessage | None:
    """Length-based dispatch for container types without a unique subtype byte."""
    if is_teleport_landed_structure(data):
        return decode_teleport_landed(data)
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
    "decode_container_message",
    "decode_container_pickup",
    "decode_mine_detonation",
    "decode_mine_placement",
    "decode_teleport_landed",
    "decode_unknown_container",
    "identify_container_type",
    "is_container_pickup_structure",
    "is_mine_detonation_structure",
    "is_mine_placement_structure",
    "is_teleport_landed_structure",
]
