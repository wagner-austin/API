"""Radar message decoders.

This module handles decoding of radar-related messages:
radar results, enemy detection, radar scan results with containers and mines.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from tankpit_bot.protocol.helpers import DecodeError, require_min_length, x16
from tankpit_bot.protocol.types import (
    EnemyDetectionDict,
    RadarContainerDict,
    RadarMineDict,
    RadarResultDict,
    RadarScanResultDict,
)


def decode_radar_result(data: bytes) -> RadarResultDict:
    """Decode radar result from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded radar result.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "RadarResult")
    return RadarResultDict(
        msg_type=0x46,
        detection_type=data[0],
        found=data[1] == 1,
    )


def decode_enemy_detection(data: bytes) -> EnemyDetectionDict:
    """Decode enemy detection from XOR-decoded data.

    Layout from tpclient.js Tg.h (V.H), trace-verified 2026-06-19:
      a[0]   = x (world coordinate of detected enemy)
      a[1]   = y (world coordinate of detected enemy)
      a[2]   = team (this.j, used as color index)
      a[3]   = rank (this.m, used with ec[] rank names)
      a[4:6] = tank_id (LE u16)

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded enemy detection.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 6, "EnemyDetection")
    return EnemyDetectionDict(
        msg_type=0x48,
        x=data[0],
        y=data[1],
        team=data[2],
        rank=data[3],
        tank_id=x16(data[4], data[5]),
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


def decode_radar_container(data: bytes, offset: int) -> RadarContainerDict:
    """Decode radar container from bytes at offset.

    Args:
        data: Raw bytes.
        offset: Offset into data.

    Returns:
        Decoded container entry.

    Raises:
        DecodeError: If not enough bytes.
    """
    if offset + 4 > len(data):
        raise DecodeError("RadarContainer: not enough bytes")
    x = data[offset]
    y = data[offset + 1]
    val = data[offset + 2] | (data[offset + 3] << 8)
    volume = -1 if val == 0xFFFF else val
    return RadarContainerDict(x=x, y=y, volume=volume)


def require_radar_container(value: JSONObject) -> RadarContainerDict:
    """Validate and convert JSON object to radar container dict.

    Args:
        value: JSON object to validate.

    Returns:
        Validated RadarContainerDict.

    Raises:
        ValueError: If validation fails.
    """
    x = value.get("x")
    if not isinstance(x, int):
        raise ValueError("RadarContainer: x must be int")
    y = value.get("y")
    if not isinstance(y, int):
        raise ValueError("RadarContainer: y must be int")
    volume = value.get("volume")
    if not isinstance(volume, int):
        raise ValueError("RadarContainer: volume must be int")
    if not 0 <= x <= 255:
        raise ValueError(f"RadarContainer: x out of range: {x}")
    if not 0 <= y <= 255:
        raise ValueError(f"RadarContainer: y out of range: {y}")
    if not (-1 <= volume <= 32767):
        raise ValueError(f"RadarContainer: volume out of range: {volume}")
    return RadarContainerDict(x=x, y=y, volume=volume)


def encode_radar_mine(mine: RadarMineDict) -> bytes:
    """Encode radar mine to bytes.

    Args:
        mine: Mine entry to encode.

    Returns:
        3-byte encoding: x, y, team.
    """
    return bytes([mine["x"], mine["y"], mine["team"]])


def decode_radar_mine(data: bytes, offset: int) -> RadarMineDict:
    """Decode radar mine from bytes at offset.

    Args:
        data: Raw bytes.
        offset: Offset into data.

    Returns:
        Decoded mine entry.

    Raises:
        DecodeError: If not enough bytes.
    """
    if offset + 3 > len(data):
        raise DecodeError("RadarMine: not enough bytes")
    return RadarMineDict(x=data[offset], y=data[offset + 1], team=data[offset + 2])


def require_radar_mine(value: JSONObject) -> RadarMineDict:
    """Validate and convert JSON object to radar mine dict.

    Args:
        value: JSON object to validate.

    Returns:
        Validated RadarMineDict.

    Raises:
        ValueError: If validation fails.
    """
    x = value.get("x")
    if not isinstance(x, int):
        raise ValueError("RadarMine: x must be int")
    y = value.get("y")
    if not isinstance(y, int):
        raise ValueError("RadarMine: y must be int")
    team = value.get("team")
    if not isinstance(team, int):
        raise ValueError("RadarMine: team must be int")
    if not 0 <= x <= 255:
        raise ValueError(f"RadarMine: x out of range: {x}")
    if not 0 <= y <= 255:
        raise ValueError(f"RadarMine: y out of range: {y}")
    if not 0 <= team <= 3:
        raise ValueError(f"RadarMine: team out of range: {team}")
    return RadarMineDict(x=x, y=y, team=team)


def encode_radar_scan_result(result: RadarScanResultDict) -> bytes:
    """Encode radar scan result to bytes.

    Args:
        result: Radar scan result to encode.

    Returns:
        Encoded bytes: container_count, flags(0), containers, mines.
    """
    container_count = len(result["containers"])
    parts: list[bytes] = [bytes([container_count, 0])]
    for container in result["containers"]:
        parts.append(encode_radar_container(container))
    for mine in result["mines"]:
        parts.append(encode_radar_mine(mine))
    return b"".join(parts)


def decode_radar_scan_result(data: bytes) -> RadarScanResultDict:
    """Decode radar scan result from XOR-decoded data.

    Format:
        - Byte 0: container count
        - Byte 1: flags (unused, always 0)
        - Containers: 4 bytes each (x, y, val_lo, val_hi)
        - Remaining bytes: mines as 3 bytes each (x, y, team)

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded radar scan result with containers and mines.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "RadarScanResult")
    container_count = data[0]
    containers: list[RadarContainerDict] = []
    idx = 2

    for _ in range(container_count):
        if idx + 4 > len(data):
            raise DecodeError(
                f"RadarScanResult: expected {container_count} containers, "
                f"ran out of data at container {len(containers)}"
            )
        containers.append(decode_radar_container(data, idx))
        idx += 4

    mines: list[RadarMineDict] = []
    remaining = len(data) - idx
    if remaining % 3 != 0:
        raise DecodeError(f"RadarScanResult: remaining bytes ({remaining}) not divisible by 3")

    while idx + 3 <= len(data):
        mines.append(decode_radar_mine(data, idx))
        idx += 3

    return RadarScanResultDict(msg_type=0x4F, containers=containers, mines=mines)


def require_radar_scan_result(value: JSONObject) -> RadarScanResultDict:
    """Validate and convert JSON object to radar scan result dict.

    Args:
        value: JSON object to validate.

    Returns:
        Validated RadarScanResultDict.

    Raises:
        ValueError: If validation fails.
    """
    if value.get("msg_type") != 0x4F:
        raise ValueError(f"RadarScanResult: msg_type must be 0x4F, got {value.get('msg_type')}")
    raw_containers = value.get("containers")
    if not isinstance(raw_containers, list):
        raise ValueError("RadarScanResult: containers must be list")
    containers: list[RadarContainerDict] = []
    for i, c in enumerate(raw_containers):
        if not isinstance(c, dict):
            raise ValueError(f"RadarScanResult: container[{i}] must be dict")
        try:
            containers.append(require_radar_container(c))
        except ValueError as e:
            raise ValueError(f"RadarScanResult: container[{i}]: {e}") from e
    raw_mines = value.get("mines")
    if not isinstance(raw_mines, list):
        raise ValueError("RadarScanResult: mines must be list")
    mines: list[RadarMineDict] = []
    for i, m in enumerate(raw_mines):
        if not isinstance(m, dict):
            raise ValueError(f"RadarScanResult: mine[{i}] must be dict")
        try:
            mines.append(require_radar_mine(m))
        except ValueError as e:
            raise ValueError(f"RadarScanResult: mine[{i}]: {e}") from e
    return RadarScanResultDict(msg_type=0x4F, containers=containers, mines=mines)


__all__ = [
    "decode_enemy_detection",
    "decode_radar_container",
    "decode_radar_mine",
    "decode_radar_result",
    "decode_radar_scan_result",
    "encode_radar_container",
    "encode_radar_mine",
    "encode_radar_scan_result",
    "require_radar_container",
    "require_radar_mine",
    "require_radar_scan_result",
]
