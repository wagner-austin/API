"""Container message type identification.

This module provides functions to identify container message types
by their structure (length, byte patterns) rather than XOR-dependent
subtype values.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.events import (
    is_container_pickup_structure,
    is_teleport_landed_structure,
)
from tankpit_bot.container.decoders.mines import (
    is_mine_detonation_structure,
    is_mine_placement_structure,
)
from tankpit_bot.container.types import ContainerMessageType


def _identify_subtype_specific(data: bytes) -> ContainerMessageType:
    """Identify message types by first-byte subtype.

    Checks for types that require a specific subtype byte:
    - Mine detonation: 0x45 subtype with coordinate pairs.
    - Container pickup: 0x43 subtype, 5 bytes.
    - Mine placement: 0x4B subtype, 5 + count*2 bytes.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified type, or UNKNOWN if not matched.
    """
    if is_mine_detonation_structure(data):
        return ContainerMessageType.MINE_DETONATION
    if is_container_pickup_structure(data):
        return ContainerMessageType.CONTAINER_PICKUP
    if is_mine_placement_structure(data):
        return ContainerMessageType.MINE_PLACEMENT
    return ContainerMessageType.UNKNOWN


def identify_container_type(data: bytes) -> ContainerMessageType:
    """Identify the type of container message by structure.

    Order of checks matters - more specific patterns first.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified message type.
    """
    if len(data) < 1:
        return ContainerMessageType.UNKNOWN

    subtype_type = _identify_subtype_specific(data)
    if subtype_type != ContainerMessageType.UNKNOWN:
        return subtype_type
    if is_teleport_landed_structure(data):
        return ContainerMessageType.TELEPORT_LANDED
    return ContainerMessageType.UNKNOWN


__all__ = [
    "identify_container_type",
]
