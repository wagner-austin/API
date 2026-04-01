"""World message decoders.

This module handles decoding of world/environment messages:
viewport updates, terrain updates, sync, containers.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    SUPERVISOR_STATUS_PROMO_ELIGIBLE,
    SUPERVISOR_STATUS_PROMO_KILL,
)
from tankpit_bot.protocol.helpers import require_min_length, x16
from tankpit_bot.protocol.types import (
    ContainerDict,
    SupervisorDict,
    SyncDict,
    TerrainUpdateDict,
    ViewportEntityDict,
    ViewportUpdateDict,
)


def decode_sync(data: bytes) -> SyncDict:
    """Decode sync message.

    Args:
        data: XOR-decoded message body.

    Returns:
        Empty sync dict.
    """
    return SyncDict(msg_type=0x3F)


def decode_container(data: bytes) -> ContainerDict:
    """Decode container from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x43 prefix).

    Returns:
        Decoded container.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 4, "Container")
    return ContainerDict(
        msg_type=0x43,
        container_id=x16(data[0], data[1]),
        fuel=x16(data[2], data[3]),
    )


def decode_terrain_update(data: bytes) -> TerrainUpdateDict:
    """Decode terrain update from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x4A prefix).

    Returns:
        Decoded terrain update with (x, y, terrain_type) tuples.
    """
    updates: list[tuple[int, int, int]] = []
    for i in range(0, len(data) - 2, 3):
        x = data[i]
        y = data[i + 1]
        terrain_type = data[i + 2]
        updates.append((x, y, terrain_type))
    return TerrainUpdateDict(msg_type=0x4A, updates=updates)


def viewport_entity_is_equipment(entity: ViewportEntityDict) -> bool:
    """Check if a viewport row marks equipment.

    Args:
        entity: Viewport entity.

    Returns:
        True if the row marks equipment on that tile.
    """
    return entity["entity_id"] == -1


def viewport_entity_is_fuel(entity: ViewportEntityDict) -> bool:
    """Check if a viewport row marks fuel.

    Args:
        entity: Viewport entity.

    Returns:
        True if the row marks fuel on that tile.
    """
    return entity["entity_id"] > 0


def viewport_entity_is_empty(entity: ViewportEntityDict) -> bool:
    """Check if tile is empty.

    Args:
        entity: Viewport entity.

    Returns:
        True if tile is empty.
    """
    return entity["entity_id"] == 0


def decode_viewport_update(data: bytes) -> ViewportUpdateDict:
    """Decode viewport update from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x5A prefix).

    Returns:
        Decoded viewport update.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "ViewportUpdate")

    viewport_left = data[0]
    viewport_top = data[1]
    entities: list[ViewportEntityDict] = []
    col, row, t = 0, 0, 2

    while t < len(data):
        v = data[t]
        t += 1

        col += v % 18
        row += v // 18
        while col >= 18:
            row += 1
            col -= 18

        if v != 255:
            if t + 3 > len(data):
                break

            b1, b2, b3 = data[t], data[t + 1], data[t + 2]
            t += 3

            z = 256 * (256 * b1 + b2) + b3
            z &= 0xFFFFFF

            terrain_type = z & 0xF
            z >>= 4
            value = z & 0xF
            if value >= 8:
                value = 255
            z >>= 4
            entity_id = z if z != 65535 else -1

            entities.append(
                ViewportEntityDict(
                    col=col,
                    row=row,
                    entity_id=entity_id,
                    value=value,
                    terrain_type=terrain_type,
                )
            )

    return ViewportUpdateDict(
        msg_type=0x5A,
        viewport_left=viewport_left,
        viewport_top=viewport_top,
        entities=entities,
    )


def decode_supervisor(data: bytes) -> SupervisorDict:
    """Decode supervisor message from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x52 prefix).

    Returns:
        Decoded supervisor message.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 3, "Supervisor")
    return SupervisorDict(
        msg_type=0x52,
        status=data[0],
        reserved=data[1],
        data=data[2],
    )


def supervisor_is_promo_eligible(supervisor: SupervisorDict) -> bool:
    """Check if player is eligible for promotion.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if eligible for promotion.
    """
    return supervisor["status"] == SUPERVISOR_STATUS_PROMO_ELIGIBLE


def supervisor_has_promo_kill(supervisor: SupervisorDict) -> bool:
    """Check if player got a promotion kill.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if got promotion kill.
    """
    return supervisor["status"] == SUPERVISOR_STATUS_PROMO_KILL


__all__ = [
    "decode_container",
    "decode_supervisor",
    "decode_sync",
    "decode_terrain_update",
    "decode_viewport_update",
    "supervisor_has_promo_kill",
    "supervisor_is_promo_eligible",
    "viewport_entity_is_empty",
    "viewport_entity_is_equipment",
    "viewport_entity_is_fuel",
]
