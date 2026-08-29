"""Encode/decode for the fleet knowledge report.

The report crosses a process boundary as JSON on disk, so the decode
side validates every field with ``require_*`` — a malformed report is
a bug (writes are atomic, so readers never see partial files) and
raises with the offending key named.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.fleetshare.types import (
    FLEET_ROLES,
    FleetContainerRemovalDict,
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetReportDict,
    FleetRole,
    FleetScannedTileDict,
)


def require_fleet_role(data: JSONObject, key: str) -> FleetRole:
    """Validate and extract a fleet role from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        The validated role.

    Raises:
        JSONTypeError: If the value is not a known fleet role.
    """
    raw = require_str(data, key)
    for role in FLEET_ROLES:
        if raw == role:
            return role
    raise JSONTypeError(f"{key} must be one of {FLEET_ROLES}, got {raw!r}")


def encode_fleet_enemy_sighting(sighting: FleetEnemySightingDict) -> JSONObject:
    """Encode one enemy sighting for the report payload.

    Args:
        sighting: Enemy sighting to encode.

    Returns:
        JSON object mirroring the sighting's fields.
    """
    return {
        "tank_id": sighting["tank_id"],
        "name": sighting["name"],
        "team": sighting["team"],
        "rank": sighting["rank"],
        "x": sighting["x"],
        "y": sighting["y"],
        "damage_state": sighting["damage_state"],
        "observed_ms": sighting["observed_ms"],
    }


def decode_fleet_enemy_sighting(data: JSONValue) -> FleetEnemySightingDict:
    """Decode and validate one enemy sighting.

    Args:
        data: JSON value holding one encoded sighting.

    Returns:
        The validated sighting.

    Raises:
        JSONTypeError: If the value is not an object or a field fails
            validation.
    """
    if not isinstance(data, dict):
        raise JSONTypeError(f"enemy sighting must be an object, got {type(data).__name__}")
    return FleetEnemySightingDict(
        tank_id=require_int(data, "tank_id"),
        name=require_str(data, "name"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        damage_state=require_int(data, "damage_state"),
        observed_ms=require_int(data, "observed_ms"),
    )


def encode_fleet_container_sighting(sighting: FleetContainerSightingDict) -> JSONObject:
    """Encode one container sighting for the report payload.

    Args:
        sighting: Container sighting to encode.

    Returns:
        JSON object mirroring the sighting's fields.
    """
    return {
        "x": sighting["x"],
        "y": sighting["y"],
        "is_fuel": sighting["is_fuel"],
        "volume": sighting["volume"],
        "observed_ms": sighting["observed_ms"],
    }


def decode_fleet_container_sighting(data: JSONValue) -> FleetContainerSightingDict:
    """Decode and validate one container sighting.

    Args:
        data: JSON value holding one encoded sighting.

    Returns:
        The validated sighting.

    Raises:
        JSONTypeError: If the value is not an object or a field fails
            validation.
    """
    if not isinstance(data, dict):
        raise JSONTypeError(f"container sighting must be an object, got {type(data).__name__}")
    return FleetContainerSightingDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        is_fuel=require_bool(data, "is_fuel"),
        volume=require_int(data, "volume"),
        observed_ms=require_int(data, "observed_ms"),
    )


def encode_fleet_container_removal(removal: FleetContainerRemovalDict) -> JSONObject:
    """Encode one container removal for the report payload.

    Args:
        removal: Removal to encode.

    Returns:
        JSON object mirroring the removal's fields.
    """
    return {
        "x": removal["x"],
        "y": removal["y"],
        "removed_ms": removal["removed_ms"],
    }


def decode_fleet_container_removal(data: JSONValue) -> FleetContainerRemovalDict:
    """Decode and validate one container removal.

    Args:
        data: JSON value holding one encoded removal.

    Returns:
        The validated removal.

    Raises:
        JSONTypeError: If the value is not an object or a field fails
            validation.
    """
    if not isinstance(data, dict):
        raise JSONTypeError(f"container removal must be an object, got {type(data).__name__}")
    return FleetContainerRemovalDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        removed_ms=require_int(data, "removed_ms"),
    )


def encode_fleet_scanned_tile(tile: FleetScannedTileDict) -> JSONObject:
    """Encode one covered tile for the report payload.

    Args:
        tile: Covered tile to encode.

    Returns:
        JSON object mirroring the tile's fields.
    """
    return {
        "x": tile["x"],
        "y": tile["y"],
        "observed_ms": tile["observed_ms"],
    }


def decode_fleet_scanned_tile(data: JSONValue) -> FleetScannedTileDict:
    """Decode and validate one covered tile.

    Args:
        data: JSON value holding one encoded tile.

    Returns:
        The validated tile.

    Raises:
        JSONTypeError: If the value is not an object or a field fails
            validation.
    """
    if not isinstance(data, dict):
        raise JSONTypeError(f"scanned tile must be an object, got {type(data).__name__}")
    return FleetScannedTileDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        observed_ms=require_int(data, "observed_ms"),
    )


def encode_fleet_report(report: FleetReportDict) -> JSONObject:
    """Encode a fleet report for its on-disk JSON form.

    Args:
        report: Report to encode.

    Returns:
        JSON object mirroring the report's fields.
    """
    return {
        "instance": report["instance"],
        "team": report["team"],
        "room": report["room"],
        "tank_id": report["tank_id"],
        "role": report["role"],
        "x": report["x"],
        "y": report["y"],
        "engaged_target_id": report["engaged_target_id"],
        "forage_goal_x": report["forage_goal_x"],
        "forage_goal_y": report["forage_goal_y"],
        "collect_claim_x": report["collect_claim_x"],
        "collect_claim_y": report["collect_claim_y"],
        "combat_consent_ids": list(report["combat_consent_ids"]),
        "written_ms": report["written_ms"],
        "enemies": [encode_fleet_enemy_sighting(sighting) for sighting in report["enemies"]],
        "containers": [
            encode_fleet_container_sighting(sighting) for sighting in report["containers"]
        ],
        "removed": [encode_fleet_container_removal(removal) for removal in report["removed"]],
        "scanned": [encode_fleet_scanned_tile(tile) for tile in report["scanned"]],
    }


def _require_int_list(data: JSONObject, key: str) -> list[int]:
    """Decode a required list-of-ints field.

    Args:
        data: JSON object holding the field.
        key: Field name.

    Returns:
        The validated int list.

    Raises:
        JSONTypeError: If the field is absent, not a list, or holds a
            non-int entry.
    """
    raw = require_list(data, key)
    out: list[int] = []
    for i, value in enumerate(raw):
        if not isinstance(value, int) or isinstance(value, bool):
            raise JSONTypeError(f"{key}[{i}] must be an int, got {type(value).__name__}")
        out.append(value)
    return out


def decode_fleet_report(data: JSONValue) -> FleetReportDict:
    """Decode and validate a fleet report read from disk.

    Args:
        data: Parsed JSON value of one report file.

    Returns:
        The validated report.

    Raises:
        JSONTypeError: If the value is not an object or any field
            fails validation.
    """
    if not isinstance(data, dict):
        raise JSONTypeError(f"fleet report must be an object, got {type(data).__name__}")
    return FleetReportDict(
        instance=require_str(data, "instance"),
        team=require_int(data, "team"),
        room=require_str(data, "room"),
        tank_id=require_int(data, "tank_id"),
        role=require_fleet_role(data, "role"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        engaged_target_id=require_int(data, "engaged_target_id"),
        forage_goal_x=require_int(data, "forage_goal_x"),
        forage_goal_y=require_int(data, "forage_goal_y"),
        collect_claim_x=require_int(data, "collect_claim_x"),
        collect_claim_y=require_int(data, "collect_claim_y"),
        combat_consent_ids=_require_int_list(data, "combat_consent_ids"),
        written_ms=require_int(data, "written_ms"),
        enemies=[decode_fleet_enemy_sighting(entry) for entry in require_list(data, "enemies")],
        containers=[
            decode_fleet_container_sighting(entry) for entry in require_list(data, "containers")
        ],
        removed=[decode_fleet_container_removal(entry) for entry in require_list(data, "removed")],
        scanned=[decode_fleet_scanned_tile(entry) for entry in require_list(data, "scanned")],
    )


__all__ = [
    "decode_fleet_container_removal",
    "decode_fleet_container_sighting",
    "decode_fleet_enemy_sighting",
    "decode_fleet_report",
    "decode_fleet_scanned_tile",
    "encode_fleet_container_removal",
    "encode_fleet_container_sighting",
    "encode_fleet_enemy_sighting",
    "encode_fleet_report",
    "encode_fleet_scanned_tile",
    "require_fleet_role",
]
