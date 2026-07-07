"""Radar message decoders.

This module handles decoding of radar-related messages:
radar results, enemy detection, radar scan results with containers and mines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from platform_core.json_utils import JSONObject

from tankpit_bot.protocol.helpers import DecodeError, require_min_length, x16
from tankpit_bot.protocol.types import (
    EnemyDetectionDict,
    RadarContainerDict,
    RadarMineClearDict,
    RadarMineDict,
    RadarResultDict,
    RadarScanResultDict,
)

# JS ch handler (tpclient.pretty.js:4811-4813) writes the overlay byte
# into tile.m raw; the dh detonation handler uses 255 as the canonical
# "no mine" sentinel and the 0x5A parse maps 8..15 -> 255. So overlay
# 0-7 = mine present (team in the low 2 bits), >= 8 = tile has no mine.
_OVERLAY_NO_MINE_THRESHOLD = 8

_ScanEntryT = TypeVar("_ScanEntryT", RadarContainerDict, RadarMineDict, RadarMineClearDict)


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


def encode_radar_mine_clear(clear: RadarMineClearDict) -> bytes:
    """Encode a mine-clear overlay entry to bytes.

    Args:
        clear: Mine-clear entry to encode.

    Returns:
        3-byte encoding: x, y, 255 (the JS no-mine sentinel).
    """
    return bytes([clear["x"], clear["y"], 255])


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
        Encoded bytes: count (LE u16), containers, overlay entries.
    """
    container_count = len(result["containers"])
    parts: list[bytes] = [bytes([container_count & 0xFF, (container_count >> 8) & 0xFF])]
    for container in result["containers"]:
        parts.append(encode_radar_container(container))
    for mine in result["mines"]:
        parts.append(encode_radar_mine(mine))
    for clear in result["mine_clears"]:
        parts.append(encode_radar_mine_clear(clear))
    return b"".join(parts)


def decode_radar_scan_result(data: bytes) -> RadarScanResultDict:
    """Decode radar scan result from XOR-decoded data.

    Format per JS ``ch.h`` (tpclient.pretty.js:4800-4809):
        - Bytes 0-1: cache entry count (LE u16)
        - Cache entries: 4 bytes each (x, y, val_lo, val_hi);
          value 0 = tile now empty, 65535 -> -1 = equipment, else fuel
        - Remaining bytes: overlay entries as 3 bytes each (x, y, value);
          value 0-7 = mine with ``team = value & 3``, >= 8 = no mine

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded radar scan result with containers, mines, and mine clears.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "RadarScanResult")
    container_count = x16(data[0], data[1])
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
    mine_clears: list[RadarMineClearDict] = []
    remaining = len(data) - idx
    if remaining % 3 != 0:
        raise DecodeError(f"RadarScanResult: remaining bytes ({remaining}) not divisible by 3")

    while idx + 3 <= len(data):
        x, y, overlay = data[idx], data[idx + 1], data[idx + 2]
        if overlay < _OVERLAY_NO_MINE_THRESHOLD:
            mines.append(RadarMineDict(x=x, y=y, team=overlay & 3))
        else:
            mine_clears.append(RadarMineClearDict(x=x, y=y))
        idx += 3

    return RadarScanResultDict(
        msg_type=0x4F,
        containers=containers,
        mines=mines,
        mine_clears=mine_clears,
    )


def require_radar_mine_clear(value: JSONObject) -> RadarMineClearDict:
    """Validate and convert JSON object to radar mine-clear dict.

    Args:
        value: JSON object to validate.

    Returns:
        Validated RadarMineClearDict.

    Raises:
        ValueError: If validation fails.
    """
    x = value.get("x")
    if not isinstance(x, int):
        raise ValueError("RadarMineClear: x must be int")
    y = value.get("y")
    if not isinstance(y, int):
        raise ValueError("RadarMineClear: y must be int")
    if not 0 <= x <= 255:
        raise ValueError(f"RadarMineClear: x out of range: {x}")
    if not 0 <= y <= 255:
        raise ValueError(f"RadarMineClear: y out of range: {y}")
    return RadarMineClearDict(x=x, y=y)


def _require_scan_entry_list(
    value: JSONObject,
    key: str,
    label: str,
    item_require: Callable[[JSONObject], _ScanEntryT],
) -> list[_ScanEntryT]:
    """Validate one entry list of a JSON radar scan result.

    Args:
        value: Enclosing JSON object.
        key: List field name to extract.
        label: Singular entry label used in error messages.
        item_require: Per-entry validator.

    Returns:
        Validated entry list.

    Raises:
        ValueError: If the field is not a list or any entry fails.
    """
    raw = value.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"RadarScanResult: {key} must be list")
    entries: list[_ScanEntryT] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"RadarScanResult: {label}[{i}] must be dict")
        try:
            entries.append(item_require(item))
        except ValueError as e:
            raise ValueError(f"RadarScanResult: {label}[{i}]: {e}") from e
    return entries


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
    return RadarScanResultDict(
        msg_type=0x4F,
        containers=_require_scan_entry_list(
            value, "containers", "container", require_radar_container
        ),
        mines=_require_scan_entry_list(value, "mines", "mine", require_radar_mine),
        mine_clears=_require_scan_entry_list(
            value, "mine_clears", "mine_clear", require_radar_mine_clear
        ),
    )


__all__ = [
    "decode_enemy_detection",
    "decode_radar_container",
    "decode_radar_result",
    "decode_radar_scan_result",
    "encode_radar_container",
    "encode_radar_mine",
    "encode_radar_mine_clear",
    "encode_radar_scan_result",
    "require_radar_container",
    "require_radar_mine",
    "require_radar_mine_clear",
    "require_radar_scan_result",
]
