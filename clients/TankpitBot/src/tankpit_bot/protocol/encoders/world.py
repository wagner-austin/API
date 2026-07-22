"""World message encoders — exact byte inverses of ``decoders.world``."""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16
from tankpit_bot.protocol.types import (
    CacheUpdateDict,
    OverlayUpdateDict,
    SupervisorDict,
    SupervisorTextDict,
    SyncDict,
    TerrainUpdateDict,
    ViewportUpdateDict,
)
from tankpit_bot.state.viewport_geometry import VIEWPORT_PATCH_WIDTH

# The overlay nibble the wire uses for "no mine here". decode maps every
# nibble >= 8 to the 255 sentinel; the corpus (2026-07-21, 3,724 bodies)
# shows the server only ever emits 8.
_OVERLAY_EMPTY_NIBBLE = 8

# 0x3F Sync bodies carry a single constant byte (1,166/1,166 corpus
# bodies); the decoder discards it, so the encoder re-emits the constant.
_SYNC_BODY = bytes([1])


def encode_sync(message: SyncDict) -> bytes:
    """Encode a 0x3F Sync payload (inverse of ``decode_sync``).

    Args:
        message: Decoded sync heartbeat (carries nothing).

    Returns:
        The 1-byte wire body.
    """
    del message
    return _SYNC_BODY


def encode_cache_update(message: CacheUpdateDict) -> bytes:
    """Encode a 0x43 CacheUpdate payload (inverse of ``decode_cache_update``).

    Args:
        message: Decoded cache update.

    Returns:
        Payload bytes without the 0x43 prefix: 4-byte entries with the
        -1 equipment sentinel restored to 0xFFFF.
    """
    out = bytearray()
    for x, y, value in message["updates"]:
        out += bytes([x, y]) + pack16(0xFFFF if value == -1 else value)
    return bytes(out)


def encode_overlay_update(message: OverlayUpdateDict) -> bytes:
    """Encode a 0x40 OverlayUpdate payload (inverse of ``decode_overlay_update``).

    Args:
        message: Decoded overlay update.

    Returns:
        Payload bytes without the 0x40 prefix.
    """
    out = bytearray()
    for x, y, value in message["updates"]:
        out += bytes([x, y, value])
    return bytes(out)


def encode_terrain_update(message: TerrainUpdateDict) -> bytes:
    """Encode a 0x4A TerrainUpdate payload (inverse of ``decode_terrain_update``).

    Args:
        message: Decoded terrain update.

    Returns:
        Payload bytes without the 0x4A prefix.
    """
    out = bytearray()
    for x, y, terrain_type in message["updates"]:
        out += bytes([x, y, terrain_type])
    return bytes(out)


def encode_viewport_update(message: ViewportUpdateDict) -> bytes:
    """Encode a 0x5A ViewportUpdate payload (inverse of ``decode_viewport_update``).

    Entities are emitted with the same skip-RLE cursor the decoder
    walks: each step byte advances a linear cursor over the
    ``VIEWPORT_PATCH_WIDTH``-wide patch, 255 is a pure skip, any other
    byte introduces a 3-byte packed tile record. The greedy encoding
    (as many 255-skips as needed, then the remainder) is
    byte-identical to the server's on the full corpus.

    Args:
        message: Decoded viewport update.

    Returns:
        Payload bytes without the 0x5A prefix.
    """
    out = bytearray([message["viewport_left"], message["viewport_top"]])
    cursor = 0
    for entity in message["entities"]:
        target = entity["row"] * VIEWPORT_PATCH_WIDTH + entity["col"]
        delta = target - cursor
        while delta >= 255:
            out.append(255)
            delta -= 255
        out.append(delta)
        cursor = target
        overlay = entity["overlay_value"]
        raw_overlay = _OVERLAY_EMPTY_NIBBLE if overlay == 255 else overlay
        cache = 0xFFFF if entity["cache_value"] == -1 else entity["cache_value"]
        packed = (cache << 8) | ((raw_overlay & 0xF) << 4) | (entity["terrain_type"] & 0xF)
        out += bytes([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF])
    return bytes(out)


def encode_supervisor(message: SupervisorDict) -> bytes:
    """Encode a 0x52 Supervisor payload (inverse of ``decode_supervisor``).

    Args:
        message: Decoded supervisor command result.

    Returns:
        Payload bytes without the 0x52 prefix.
    """
    return bytes([message["reset_action"], message["close_map"], message["error_code"]])


def encode_supervisor_text(message: SupervisorTextDict) -> bytes:
    """Encode a 0x3C SupervisorText payload (inverse of ``decode_supervisor_text``).

    Args:
        message: Decoded supervisor free-form text.

    Returns:
        Payload bytes without the 0x3C prefix (latin-1, per the JS
        ``p()`` char-code helper).
    """
    return message["message"].encode("latin-1")


__all__ = [
    "encode_cache_update",
    "encode_overlay_update",
    "encode_supervisor",
    "encode_supervisor_text",
    "encode_sync",
    "encode_terrain_update",
    "encode_viewport_update",
]
