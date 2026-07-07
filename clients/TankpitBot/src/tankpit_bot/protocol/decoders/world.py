"""World message decoders.

This module handles decoding of world/environment messages:
viewport updates, terrain updates, sync, containers.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.protocol.helpers import DecodeError, require_min_length, x16
from tankpit_bot.protocol.types import (
    CacheUpdateDict,
    OverlayUpdateDict,
    SupervisorDict,
    SupervisorTextDict,
    SyncDict,
    TerrainUpdateDict,
    ViewportEntityDict,
    ViewportUpdateDict,
)
from tankpit_bot.state.viewport_geometry import VIEWPORT_PATCH_WIDTH


def decode_sync(data: bytes) -> SyncDict:
    """Decode sync message.

    Args:
        data: XOR-decoded message body.

    Returns:
        Empty sync dict.
    """
    return SyncDict(msg_type=0x3F)


def _decode_cache_value(low: int, high: int) -> int:
    """Decode a cache value from two bytes.

    Args:
        low: Low byte.
        high: High byte.

    Returns:
        Cache value with ``0xFFFF`` mapped to ``-1``.
    """
    raw_value = x16(low, high)
    if raw_value == 0xFFFF:
        return -1
    return raw_value


def decode_cache_update(data: bytes) -> CacheUpdateDict:
    """Decode cache-only tile patch from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x43 prefix).

    Returns:
        Decoded cache update entries.

    Raises:
        DecodeError: If payload length is invalid.
    """
    if len(data) % 4 != 0:
        raise DecodeError(f"CacheUpdate: expected 4-byte entries, got {len(data)} bytes")
    updates: list[tuple[int, int, int]] = []
    for offset in range(0, len(data), 4):
        updates.append(
            (
                data[offset],
                data[offset + 1],
                _decode_cache_value(data[offset + 2], data[offset + 3]),
            )
        )
    return CacheUpdateDict(
        msg_type=0x43,
        updates=updates,
    )


def decode_overlay_update(data: bytes) -> OverlayUpdateDict:
    """Decode overlay-only tile patch from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x40 prefix).

    Returns:
        Decoded overlay update entries.

    Raises:
        DecodeError: If payload length is invalid.
    """
    if len(data) % 3 != 0:
        raise DecodeError(f"OverlayUpdate: expected 3-byte entries, got {len(data)} bytes")
    updates: list[tuple[int, int, int]] = []
    for offset in range(0, len(data), 3):
        updates.append((data[offset], data[offset + 1], data[offset + 2]))
    return OverlayUpdateDict(msg_type=0x40, updates=updates)


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


def viewport_entity_has_equipment_cache(entity: ViewportEntityDict) -> bool:
    """Check if a viewport row marks equipment.

    Args:
        entity: Viewport entity.

    Returns:
        True if the row marks equipment on that tile.
    """
    return entity["cache_value"] == -1


def viewport_entity_has_fuel_cache(entity: ViewportEntityDict) -> bool:
    """Check if a viewport row marks fuel.

    Args:
        entity: Viewport entity.

    Returns:
        True if the row marks fuel on that tile.
    """
    return entity["cache_value"] > 0


def viewport_entity_has_no_cache(entity: ViewportEntityDict) -> bool:
    """Check if viewport row carries no container cache value.

    Args:
        entity: Viewport entity.

    Returns:
        True if tile is empty.
    """
    return entity["cache_value"] == 0


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

        col += v % VIEWPORT_PATCH_WIDTH
        row += v // VIEWPORT_PATCH_WIDTH
        while col >= VIEWPORT_PATCH_WIDTH:
            row += 1
            col -= VIEWPORT_PATCH_WIDTH

        if v != 255:
            if t + 3 > len(data):
                break

            b1, b2, b3 = data[t], data[t + 1], data[t + 2]
            t += 3

            z = 256 * (256 * b1 + b2) + b3
            z &= 0xFFFFFF

            terrain_type = z & 0xF
            z >>= 4
            overlay_value = z & 0xF
            if overlay_value >= 8:
                overlay_value = 255
            z >>= 4
            cache_value = z if z != 65535 else -1

            entities.append(
                ViewportEntityDict(
                    col=col,
                    row=row,
                    cache_value=cache_value,
                    overlay_value=overlay_value,
                    terrain_type=terrain_type,
                )
            )

    return ViewportUpdateDict(
        msg_type=0x5A,
        viewport_left=viewport_left,
        viewport_top=viewport_top,
        entities=entities,
    )


def decode_supervisor_text(data: bytes) -> SupervisorTextDict:
    """Decode the 0x3C '<' SupervisorText (free-form server message).

    Trace-verified from tpclient.js wg.h (V['<']):
      ``wg.h(a) = new wg(p(a))`` -- ``p()`` is ``String.fromCharCode``
      over every byte of the XOR-decoded body. The renderer prints
      "Message from the Supervisor:\\n<message>".

    The body is decoded as Latin-1 / char-code per the JS p() helper.

    Args:
        data: XOR-decoded message body (without the 0x3C prefix).

    Returns:
        Decoded SupervisorText message.
    """
    return SupervisorTextDict(
        msg_type=0x3C,
        message=data.decode("latin-1"),
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
        reset_action=data[0],
        close_map=data[1],
        error_code=data[2],
    )


def supervisor_error_code(supervisor: SupervisorDict) -> int:
    """Return the error code from a supervisor command-failure message.

    The ``data`` field is an index into the game client's Gb[] error
    string array (tpclient.js ``xg.h``). See ``SUPERVISOR_ERROR_*``
    constants for the full mapping.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        Error code (0-10, or 128+ for custom text).
    """
    return supervisor["error_code"]


def supervisor_is_cant_go(supervisor: SupervisorDict) -> bool:
    """Check if the server rejected a move command.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if error is "You can't go there!".
    """
    return supervisor["error_code"] == SUPERVISOR_ERROR_CANT_GO


def supervisor_is_insufficient_fuel(supervisor: SupervisorDict) -> bool:
    """Check if the server rejected a command for insufficient fuel.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if error is "Insufficient fuel".
    """
    return supervisor["error_code"] == SUPERVISOR_ERROR_INSUFFICIENT_FUEL


__all__ = [
    "decode_cache_update",
    "decode_overlay_update",
    "decode_supervisor",
    "decode_supervisor_text",
    "decode_sync",
    "decode_terrain_update",
    "decode_viewport_update",
    "supervisor_error_code",
    "supervisor_is_cant_go",
    "supervisor_is_insufficient_fuel",
    "viewport_entity_has_equipment_cache",
    "viewport_entity_has_fuel_cache",
    "viewport_entity_has_no_cache",
]
