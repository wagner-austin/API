"""Radar message encoders — exact byte inverses of ``decoders.radar``.

The container/mine/scan-result encoders moved here from
``decoders.radar`` when the encoder package was created (Phase 4
step a, 2026-07-21) so every server-message encoder has one home.
"""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16
from tankpit_bot.protocol.types import (
    EnemyDetectionDict,
    RadarContainerDict,
    RadarMineClearDict,
    RadarMineDict,
    RadarResultDict,
    RadarScanResultDict,
)


def encode_radar_result(message: RadarResultDict) -> bytes:
    """Encode a 0x46 RadarResult payload (inverse of ``decode_radar_result``).

    Args:
        message: Decoded radar acknowledgement.

    Returns:
        Payload bytes without the 0x46 prefix.
    """
    return bytes([message["detection_type"], 1 if message["found"] else 0])


def encode_enemy_detection(message: EnemyDetectionDict) -> bytes:
    """Encode a 0x48 EnemyDetection payload (inverse of ``decode_enemy_detection``).

    Args:
        message: Decoded enemy detection.

    Returns:
        Payload bytes without the 0x48 prefix.
    """
    return bytes([message["x"], message["y"], message["team"], message["rank"]]) + pack16(
        message["tank_id"]
    )


def encode_radar_container(container: RadarContainerDict) -> bytes:
    """Encode radar container to bytes.

    Args:
        container: Container entry to encode.

    Returns:
        4-byte encoding: x, y, val_lo, val_hi.
    """
    x = container["x"]
    y = container["y"]
    volume = container["volume"]
    val = 0xFFFF if volume == -1 else volume & 0xFFFF
    return bytes([x, y, val & 0xFF, (val >> 8) & 0xFF])


def encode_radar_mine(mine: RadarMineDict) -> bytes:
    """Encode radar mine to bytes.

    Args:
        mine: Mine entry to encode.

    Returns:
        3-byte encoding: x, y, team.
    """
    return bytes([mine["x"], mine["y"], mine["team"]])


def encode_radar_mine_clear(clear: RadarMineClearDict) -> bytes:
    """Encode a mine-clear overlay entry to bytes.

    Args:
        clear: Mine-clear entry to encode.

    Returns:
        3-byte encoding: x, y, 255 (the JS no-mine sentinel).
    """
    return bytes([clear["x"], clear["y"], 255])


def encode_radar_scan_result(result: RadarScanResultDict) -> bytes:
    """Encode radar scan result to bytes.

    Args:
        result: Radar scan result to encode.

    Returns:
        Encoded bytes: count (LE u16), containers, overlay entries.
    """
    container_count = len(result["containers"])
    parts: list[bytes] = [pack16(container_count)]
    for container in result["containers"]:
        parts.append(encode_radar_container(container))
    for mine in result["mines"]:
        parts.append(encode_radar_mine(mine))
    for clear in result["mine_clears"]:
        parts.append(encode_radar_mine_clear(clear))
    return b"".join(parts)


__all__ = [
    "encode_enemy_detection",
    "encode_radar_container",
    "encode_radar_mine",
    "encode_radar_mine_clear",
    "encode_radar_result",
    "encode_radar_scan_result",
]
