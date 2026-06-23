"""Message formatting functions for human-readable output.

This module provides functions to format decoded protocol messages
into human-readable strings for logging and display.
"""

from __future__ import annotations

from tankpit_bot import protocol
from tankpit_bot.container.types import ContainerPickupRecordDict
from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
from tankpit_bot.sniffer.constants import (
    COMBAT_MSG_TYPES,
    DAMAGE_NAMES,
    MISC_MSG_TYPES,
    MSG_TYPE_NAMES,
    POSITION_MSG_TYPES,
    RADAR_MSG_TYPES,
    RANK_NAMES,
    RESOURCE_MSG_TYPES,
    TANK_MSG_TYPES,
    TEAM_NAMES,
)


def rank_name(rank: int) -> str:
    """Get rank name from rank number.

    Args:
        rank: Rank number (0-7).

    Returns:
        Rank name string.
    """
    return RANK_NAMES[rank] if 0 <= rank < len(RANK_NAMES) else f"r{rank}"


def damage_name(damage: int) -> str:
    """Get damage description from damage_state.

    Args:
        damage: Damage state (0-3).

    Returns:
        Damage description string.
    """
    return DAMAGE_NAMES[damage] if 0 <= damage < len(DAMAGE_NAMES) else f"d{damage}"


def team_name(team: int) -> str:
    """Get team name from team number.

    Args:
        team: Team number (0-3).

    Returns:
        Team name string.
    """
    return TEAM_NAMES[team] if 0 <= team < len(TEAM_NAMES) else f"t{team}"


def format_decoded_message(msg_type: int, decoded: protocol.BinaryMessage) -> str:
    """Format a decoded protocol message as readable string.

    Args:
        msg_type: Message type byte.
        decoded: Decoded binary protocol message.

    Returns:
        Formatted string for logging.
    """
    # For container messages, use the string msg_type from container_decoder
    actual_type = decoded["msg_type"]
    if isinstance(actual_type, str):
        # Container message - use specific type name
        type_name = actual_type.replace("_", " ").title().replace(" ", "")
    else:
        # Protocol message - use decoded msg_type for lookup (handles
        # 0x2E-tunneled messages where wire type differs from actual type)
        type_name = MSG_TYPE_NAMES.get(actual_type, f"Msg0x{actual_type:02X}")
    details = format_message_details(decoded)
    if details:
        return f"[{type_name}] {details}"
    return f"[{type_name}]"


def format_combat_details(d: protocol.BinaryMessage) -> str:
    """Format combat-related message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted combat details string.
    """
    if d["msg_type"] == 0x53:
        src = f"src=({d['source_x']},{d['source_y']})"
        tgt = f"tgt=({d['target_x']},{d['target_y']})"
        wpn_names = {0: "single", 1: "dual", 2: "missile", 3: "homing"}
        wpn = wpn_names.get(d["weapon"], f"wpn{d['weapon']}")
        return f"shooter={d['shooter_id']} team={d['team']} {src} {tgt} {wpn}"
    if d["msg_type"] == 0x41:
        return f"victim={d['victim_id']} killer={d['killer_id']}"
    return ""


def _format_tank_lifecycle(d: protocol.BinaryMessage) -> str:
    """Format tank join / info / status / exit / removal lines."""
    if d["msg_type"] == 0x28:
        return f"tank={d['tank_id']} team={d['team']} rank={d['rank']} at ({d['x']},{d['y']})"
    if d["msg_type"] == 0x58:
        return f"tank={d['tank_id']} removed"
    if d["msg_type"] == 0x29:
        outcome = "eliminated" if d["was_eliminated"] else "left"
        silent = " silent" if d["was_silent"] else ""
        return f"tank={d['tank_id']} team={d['team']} {outcome}{silent}"
    if d["msg_type"] == 0x21:
        team = team_name(d["team"])
        return f"tank={d['tank_id']} {team} name={d['name']}"
    if d["msg_type"] == 0x3E:
        rank = rank_name(d["rank"])
        team = team_name(d["team"])
        return f"tank={d['tank_id']} {team} {rank} lb={d['leaderboard_score']}"
    if d["msg_type"] == 0x2E:
        rank = rank_name(d["rank"])
        dmg = damage_name(d["damage_state"])
        return f"tank={d['tank_id']} {rank} hp={dmg} lb={d['lb_score']}"
    return ""


def _format_tank_motion(d: protocol.BinaryMessage) -> str:
    """Format movement / detection / build-pickup lines."""
    if d["msg_type"] == 0x47:
        x, y, dr = d["start_x"], d["start_y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} rank={d['rank']} lb={d['lb_score']}"
    if d["msg_type"] == 0x3D:
        rank = rank_name(d["rank"])
        x, y, dr = d["x"], d["y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} {rank} lb={d['lb_score']}"
    if d["msg_type"] == 0x48:
        rank = rank_name(d["rank"])
        return f"tank={d['tank_id']} at ({d['x']},{d['y']}) {rank}"
    if d["msg_type"] == 0x42:
        action = "bridge built" if d["obstacle_type"] == 1 else "obstacle drop/pickup"
        src = f"({d['source_x']},{d['source_y']})"
        drop = f"({d['drop_x']},{d['drop_y']})"
        return f"tank={d['tank_id']} {action} from {src} at {drop}"
    return ""


def format_tank_details(d: protocol.BinaryMessage) -> str:
    """Format tank status message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted tank details string.
    """
    lifecycle = _format_tank_lifecycle(d)
    if lifecycle:
        return lifecycle
    return _format_tank_motion(d)


def format_resource_details(d: protocol.BinaryMessage) -> str:
    """Format resource-related message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted resource details string.
    """
    if d["msg_type"] == 0x44:
        return f"fuel={d['fuel_total']} free={d['is_free']}"
    if d["msg_type"] == 0x64:
        return f"fuel={d['fuel_total']}"
    if d["msg_type"] == 0x49:
        return f"counts={d['counts']}"
    if d["msg_type"] == 0x43:
        return f"updates={len(d['updates'])}"
    return ""


def format_position_details(d: protocol.BinaryMessage) -> str:
    """Format position update message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted position details string.
    """
    if d["msg_type"] == 0x4B:
        return f"tank={d['tank_id']} count={len(d['positions'])}"
    if d["msg_type"] == 0x45:
        return f"count={len(d['positions'])}"
    return ""


def format_radar_details(d: protocol.BinaryMessage) -> str:
    """Format radar-related message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted radar details string.
    """
    if d["msg_type"] == 0x46:
        return f"type={d['detection_type']} found={d['found']}"
    match d:
        case {"msg_type": 0x4F, "containers": list(containers), "mines": list(mines)}:
            return f"containers={len(containers)} mines={len(mines)}"
        case {
            "msg_type": 0x4F,
            "cache_updates": list(cache_updates),
            "overlay_updates": list(overlay_updates),
        }:
            return f"cache_updates={len(cache_updates)} overlay_updates={len(overlay_updates)}"
    if d["msg_type"] == 0x5A:
        return f"viewport=({d['viewport_left']},{d['viewport_top']}) entities={len(d['entities'])}"
    return ""


def format_misc_details(d: protocol.BinaryMessage) -> str:
    """Format miscellaneous message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted misc details string.
    """
    if d["msg_type"] == 0x67:
        return f"gained={d['gained']}"
    if d["msg_type"] == 0x74:
        return f"enabled={d['enabled']}"
    if d["msg_type"] == 0x56:
        return f"time={d['playtime_hours']}h{d['playtime_minutes']}m"
    if d["msg_type"] == 0x52:
        return f"reset={d['reset_action']} err={d['error_code']}"
    if d["msg_type"] == 0x4D:
        return f"sender={d['sender_id']} type={d['message_type']}"
    if d["msg_type"] == 0x2B:
        banner = " (banner)" if d["was_promoted"] else ""
        return f"new_rank={rank_name(d['new_rank'])}{banner}"
    if d["msg_type"] == 0x4E:
        return f"tank={d['tank_id']} slot={d['slot']} level={d['level']}"
    if d["msg_type"] == 0x4C:
        return f"tanks={len(d['tanks'])}"
    if d["msg_type"] == 0x3C:
        preview = d["message"].replace("\n", " ")[:60]
        return f"message={preview!r}"
    return ""


def format_radar_response(containers: list[RadarContainerDict], mines: list[RadarMineDict]) -> str:
    """Format radar response container and mine list.

    Args:
        containers: List of container entries.
        mines: List of mine entries.

    Returns:
        Formatted radar response string.
    """
    parts: list[str] = []
    for c in containers:
        cx, cy = c["x"], c["y"]
        if c["volume"] >= 0:
            parts.append(f"({cx},{cy}):fuel={c['volume']}")
        else:
            parts.append(f"({cx},{cy}):equip")
    for m in mines:
        mx, my = m["x"], m["y"]
        team = TEAM_NAMES[m["team"]] if 0 <= m["team"] < len(TEAM_NAMES) else f"team{m['team']}"
        parts.append(f"({mx},{my}):mine[{team}]")
    return " ".join(parts)


def format_container_pickup(
    pickups: tuple[ContainerPickupRecordDict, ...] | list[ContainerPickupRecordDict],
) -> str:
    """Format container pickup details (one or more records).

    Each record's ``remaining_volume`` is the container's leftover fuel
    AFTER pickup, not the fuel transferred. ``remaining_volume == 0``
    means either an equipment container or a fuel container fully
    consumed by this pickup; the discriminator is the paired
    ``0x67 EquipmentGain`` (fired in the same tick for equipment) or
    the ``0x2E TankStatusSync`` fuel delta (positive for fuel).

    Args:
        pickups: Tuple/list of pickup records, each carrying x, y, and
            remaining_volume.

    Returns:
        Formatted pickup string -- single ``pos=(x,y) ...`` line for the
        common single-record case, comma-joined for multi-record bodies.
    """

    def _one(record: ContainerPickupRecordDict) -> str:
        x = record["x"]
        y = record["y"]
        remaining = record["remaining_volume"]
        if remaining > 0:
            return f"pos=({x},{y}) FUEL partial remaining={remaining}"
        return f"pos=({x},{y}) container emptied"

    if len(pickups) == 1:
        return _one(pickups[0])
    return f"pickups={len(pickups)}: " + ", ".join(_one(record) for record in pickups)


def format_container_simple(d: protocol.BinaryMessage) -> str | None:
    """Format simple container messages.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted string, or None if not handled.
    """
    match d:
        case {"msg_type": "unknown_container", "length": int(length), "data": bytes(data)}:
            return f"len={length} data={data.hex()[:40]}"
        case {"msg_type": "container_pickup", "pickups": tuple(pickups)}:
            return format_container_pickup(pickups)
        case {
            "msg_type": 0x4F,
            "containers": list(containers),
            "mines": list(mines),
        }:
            details = format_radar_response(containers, mines)
            return f"{len(containers)} containers, {len(mines)} mines: {details}"
    return None


def format_container_details(d: protocol.BinaryMessage) -> str:
    """Format container message details (string msg_type from container_decoder).

    Args:
        d: Decoded binary message.

    Returns:
        Formatted container details string.
    """
    simple = format_container_simple(d)
    if simple is not None:
        return simple
    return ""


def format_message_details(d: protocol.BinaryMessage) -> str:
    """Get formatted details for a decoded message using msg_type discriminant.

    Args:
        d: Decoded binary protocol message.

    Returns:
        Formatted details string, or empty string for simple types.
    """
    # Handle container messages (string msg_type from container_decoder)
    if isinstance(d["msg_type"], str):
        return format_container_details(d)
    mt = d["msg_type"]
    # Handle int msg_types from protocol module
    if mt in COMBAT_MSG_TYPES:
        return format_combat_details(d)
    if mt in TANK_MSG_TYPES:
        return format_tank_details(d)
    if mt in RESOURCE_MSG_TYPES:
        return format_resource_details(d)
    if mt in POSITION_MSG_TYPES:
        return format_position_details(d)
    if mt in RADAR_MSG_TYPES:
        return format_radar_details(d)
    if mt in MISC_MSG_TYPES:
        return format_misc_details(d)
    return ""


__all__ = [
    "damage_name",
    "format_combat_details",
    "format_container_details",
    "format_container_pickup",
    "format_container_simple",
    "format_decoded_message",
    "format_message_details",
    "format_misc_details",
    "format_position_details",
    "format_radar_details",
    "format_radar_response",
    "format_resource_details",
    "format_tank_details",
    "rank_name",
    "team_name",
]
