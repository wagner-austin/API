"""World message decoders.

This module handles decoding of world/environment messages:
viewport updates, terrain updates, sync, containers.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    SUPERVISOR_STATUS_PROMO_ELIGIBLE,
    SUPERVISOR_STATUS_PROMO_KILL,
)
from tankpit_bot.protocol.helpers import DecodeError, require_min_length, x16
from tankpit_bot.protocol.types import (
    CacheUpdateDict,
    CombinedTileUpdateDict,
    OverlayUpdateDict,
    SupervisorDict,
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


def decode_combined_tile_update(data: bytes) -> CombinedTileUpdateDict:
    """Decode combined cache+overlay tile patch from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x4F prefix).

    Returns:
        Decoded combined cache and overlay updates.

    Raises:
        DecodeError: If payload length is invalid.
    """
    require_min_length(data, 2, "CombinedTileUpdate")
    cache_count = x16(data[0], data[1])
    cache_data_len = cache_count * 4
    cache_data_start = 2
    cache_data_end = cache_data_start + cache_data_len
    if cache_data_end > len(data):
        raise DecodeError("CombinedTileUpdate: cache section exceeds payload length")
    remaining_overlay_len = len(data) - cache_data_end
    if remaining_overlay_len % 3 != 0:
        raise DecodeError("CombinedTileUpdate: overlay section must be 3-byte entries")

    cache_updates: list[tuple[int, int, int]] = []
    for offset in range(cache_data_start, cache_data_end, 4):
        cache_updates.append(
            (
                data[offset],
                data[offset + 1],
                _decode_cache_value(data[offset + 2], data[offset + 3]),
            )
        )

    overlay_updates: list[tuple[int, int, int]] = []
    for offset in range(cache_data_end, len(data), 3):
        overlay_updates.append((data[offset], data[offset + 1], data[offset + 2]))

    return CombinedTileUpdateDict(
        msg_type=0x4F,
        cache_updates=cache_updates,
        overlay_updates=overlay_updates,
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
    "decode_cache_update",
    "decode_combined_tile_update",
    "decode_overlay_update",
    "decode_supervisor",
    "decode_sync",
    "decode_terrain_update",
    "decode_viewport_update",
    "supervisor_has_promo_kill",
    "supervisor_is_promo_eligible",
    "viewport_entity_has_equipment_cache",
    "viewport_entity_has_fuel_cache",
    "viewport_entity_has_no_cache",
]
