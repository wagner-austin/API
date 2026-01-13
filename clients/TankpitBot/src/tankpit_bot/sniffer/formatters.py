"""Message formatting functions for human-readable output.

This module provides functions to format decoded protocol messages
into human-readable strings for logging and display.
"""

from __future__ import annotations

from tankpit_bot import protocol
from tankpit_bot.container import RadarContainerDict, RadarMineDict
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
from tankpit_bot.sniffer.player_tracking import (
    record_movement_response,
    register_tank_name,
    resolve_movement_tank,
)
from tankpit_bot.sniffer.viewport import get_viewport_left, update_viewport_from_position_update


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
        # Protocol message - use int-based lookup
        type_name = MSG_TYPE_NAMES.get(msg_type, f"Msg0x{msg_type:02X}")
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
        return f"shooter={d['shooter_id']} tgt=({d['target_x']},{d['target_y']})"
    if d["msg_type"] == 0x41:
        return f"victim={d['victim_id']} killer={d['killer_id']}"
    return ""


def format_tank_details(d: protocol.BinaryMessage) -> str:
    """Format tank status message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted tank details string.
    """
    if d["msg_type"] == 0x28:
        # TankEntryDict: tank_id, x, y, name (no rank/team/damage)
        return f"tank={d['tank_id']} at ({d['x']},{d['y']}) name={d['name']}"
    if d["msg_type"] == 0x58:
        return f"tank={d['tank_id']} left"
    if d["msg_type"] == 0x2E:
        # TankStatusSyncDict: has damage_state and rank
        rank = rank_name(d["rank"])
        dmg = damage_name(d["damage_state"])
        return f"tank={d['tank_id']} {rank} hp={dmg} lb={d['leaderboard_position']}"
    if d["msg_type"] == 0x3E:
        # TankStatusDict: has leaderboard_score not score
        rank = rank_name(d["rank"])
        team = team_name(d["team"])
        return f"tank={d['tank_id']} {team} {rank} score={d['leaderboard_score']}"
    if d["msg_type"] == 0x21:
        # TankInfoDict: has team
        team = team_name(d["team"])
        return f"tank={d['tank_id']} {team} name={d['name']}"
    if d["msg_type"] == 0x47:
        # MovementDict: no rank, has fuel
        x, y, dr = d["start_x"], d["start_y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} fuel={d['fuel']}"
    if d["msg_type"] == 0x3D:
        # MovementResponseDict: has rank and leaderboard_position
        rank = rank_name(d["rank"])
        x, y, dr = d["x"], d["y"], d["direction"]
        return f"tank={d['tank_id']} at ({x},{y}) dir={dr} {rank} lb={d['leaderboard_position']}"
    if d["msg_type"] == 0x48:
        rank = rank_name(d["rank"])
        return f"tank={d['tank_id']} at ({d['x']},{d['y']}) {rank}"
    return ""


def format_resource_details(d: protocol.BinaryMessage) -> str:
    """Format resource-related message details.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted resource details string.
    """
    if d["msg_type"] == 0x44:
        return f"amount={d['amount']} free={d['is_free']}"
    if d["msg_type"] == 0x64:
        return f"amount={d['amount']}"
    if d["msg_type"] == 0x49:
        return f"counts={d['counts']}"
    if d["msg_type"] == 0x43:
        return f"id={d['container_id']} fuel={d['fuel']}"
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
    if d["msg_type"] == 0x4F:
        containers = len(d["containers"])
        mines = len(d["mines"])
        return f"containers={containers} mines={mines}"
    if d["msg_type"] == 0x5A:
        return f"dir={d['direction']} entities={len(d['entities'])}"
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
        return f"status={d['status']} data={d['data']}"
    if d["msg_type"] == 0x4D:
        return f"sender={d['sender_id']} type={d['message_type']}"
    return ""


def format_tank_registry_details(
    tid: int,
    name: str,
    team: str,
    rank: int,
    badges: int,
    is_bot: bool,
    is_container: bool,
    container_y: int | None,
    container_viewport_x: int | None,
) -> str:
    """Format tank_registry message details.

    Args:
        tid: Tank ID.
        name: Tank name.
        team: Team name.
        rank: Military rank.
        badges: Badge count.
        is_bot: Whether tank is a bot.
        is_container: Whether entry is a container.
        container_y: Container Y coordinate (absolute).
        container_viewport_x: Container X relative to viewport left edge.

    Returns:
        Formatted details string.
    """
    if is_container:
        # Calculate absolute x if viewport_left is known
        viewport_left = get_viewport_left()
        if container_y is not None and container_viewport_x is not None:
            if viewport_left is not None:
                container_x = viewport_left + container_viewport_x
                return f"container id={tid} pos=({container_x},{container_y})"
            return f"container id={tid} y={container_y} vx={container_viewport_x}"
        return f"container id={tid}"
    rank_str = rank_name(rank)
    bot_str = " [BOT]" if is_bot else ""
    badge_str = f" badges={badges}" if badges > 0 else ""
    return f'tank={tid} "{name}" {team} {rank_str}{badge_str}{bot_str}'


def format_tank_update_details(tid: int, flags: int, status_data: bytes) -> str:
    """Format tank_update_* message details.

    Args:
        tid: Tank ID.
        flags: Message flags.
        status_data: Status data bytes.

    Returns:
        Formatted details string.
    """
    return f"tank={tid} flags=0x{flags:02X} data={status_data.hex()}"


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
    team_names_list = ["red", "purple", "blue", "orange"]
    for m in mines:
        mx, my = m["x"], m["y"]
        team = team_names_list[m["team"]] if 0 <= m["team"] < 4 else f"team{m['team']}"
        parts.append(f"({mx},{my}):mine[{team}]")
    return " ".join(parts)


def format_container_pickup(x: int, y: int, vol: int, is_fuel: bool) -> str:
    """Format container pickup details.

    Args:
        x: X coordinate.
        y: Y coordinate.
        vol: Volume.
        is_fuel: Whether it's a fuel container.

    Returns:
        Formatted pickup string.
    """
    ctype = f"FUEL vol={vol}" if is_fuel else "EQUIPMENT"
    return f"pos=({x},{y}) {ctype}"


def format_position_update(tid: int, x: int, y: int, f: int, ed: bytes) -> str:
    """Format position update details.

    Args:
        tid: Tank ID.
        x: X coordinate.
        y: Y coordinate.
        f: Flags.
        ed: Extra data bytes.

    Returns:
        Formatted position update string.
    """
    record_movement_response(tank_id=tid, x=x, y=y)
    update_viewport_from_position_update(tid, x, y, ed)
    return f"tank={tid} pos=({x},{y}) flags=0x{f:02X} data={ed.hex()}"


def format_movement(sx: int, sy: int, pid: int, waypoints: str, is_self: bool) -> str:
    """Format movement details.

    Args:
        sx: Start X coordinate.
        sy: Start Y coordinate.
        pid: Player ID.
        waypoints: Waypoint string.
        is_self: Whether this is self movement.

    Returns:
        Formatted movement string.
    """
    tiles = len(waypoints)
    who = "self" if is_self else "enemy"
    tid_str = resolve_movement_tank(pid, sx, sy)
    return f'{who} from=({sx},{sy}) {tid_str} path="{waypoints}" ({tiles} tiles)'


def format_combat_hit(direction: int, aid: int) -> str:
    """Format combat hit details.

    Args:
        direction: Hit direction.
        aid: Attacker ID.

    Returns:
        Formatted combat hit string.
    """
    dir_str = "out" if direction == 0x09 else "in"
    return f"attacker={aid} dir={dir_str}"


def format_tank_status_short(tid: int, dmg: int, rank: int, lb: int) -> str:
    """Format tank status short details.

    Args:
        tid: Tank ID.
        dmg: Damage state.
        rank: Rank number.
        lb: Leaderboard position.

    Returns:
        Formatted tank status string.
    """
    rank_str = rank_name(rank)
    dmg_str = damage_name(dmg)
    return f"tank={tid} {rank_str} hp={dmg_str} lb={lb}"


def handle_tank_registry(
    tid: int,
    name: str,
    team: str,
    rank: int,
    badges: int,
    is_bot: bool,
    is_container: bool,
    container_y: int | None,
    container_viewport_x: int | None,
) -> str:
    """Handle tank registry: store name and format details.

    Args:
        tid: Tank ID.
        name: Tank name.
        team: Team name.
        rank: Military rank.
        badges: Badge count.
        is_bot: Whether tank is a bot.
        is_container: Whether entry is a container.
        container_y: Container Y coordinate.
        container_viewport_x: Container viewport X.

    Returns:
        Formatted details string.
    """
    if name and not is_container:
        register_tank_name(tid, name)
    return format_tank_registry_details(
        tid, name, team, rank, badges, is_bot, is_container, container_y, container_viewport_x
    )


def format_container_simple(d: protocol.BinaryMessage) -> str | None:
    """Format simple container messages.

    Args:
        d: Decoded binary message.

    Returns:
        Formatted string, or None if not handled.
    """
    match d:
        case {"msg_type": "tank_status_sync", "sync_data": bytes(sd)}:
            return f"data={sd.hex()}"
        case {
            "msg_type": "tank_status_short",
            "tank_id": int(tid),
            "damage_state": int(dmg),
            "rank": int(rank),
            "leaderboard_position": int(lb),
        }:
            return format_tank_status_short(tid, dmg, rank, lb)
        case {
            "msg_type": "tank_update_compact" | "tank_update_extended" | "tank_update_full",
            "tank_id": int(tid),
            "flags": int(f),
            "status_data": bytes(sd),
        }:
            return format_tank_update_details(tid, f, sd)
        case {"msg_type": "unknown_container", "length": int(length), "data": bytes(data)}:
            return f"len={length} data={data.hex()[:40]}"
        case {
            "msg_type": "container_pickup",
            "x": int(x),
            "y": int(y),
            "volume": int(vol),
            "is_fuel": bool(is_fuel),
        }:
            return format_container_pickup(x, y, vol, is_fuel)
        case {
            "msg_type": "radar_response",
            "container_count": int(count),
            "containers": list(containers),
            "mines": list(mines),
        }:
            details = format_radar_response(containers, mines)
            return f"{count} containers, {len(mines)} mines: {details}"
    return None


def format_container_details(d: protocol.BinaryMessage) -> str:
    """Format container message details (string msg_type from container_decoder).

    Args:
        d: Decoded binary message.

    Returns:
        Formatted container details string.
    """
    # Try simple message types first
    simple = format_container_simple(d)
    if simple is not None:
        return simple

    match d:
        case {"msg_type": "combat_hit", "direction": int(direction), "attacker_id": int(aid)}:
            return format_combat_hit(direction, aid)
        case {
            "msg_type": "tank_registry",
            "tank_id": int(tid),
            "tank_name": str(name),
            "team": str(team),
            "military_rank": int(rank),
            "badge_count": int(badges),
            "is_bot": bool(is_bot),
            "is_container": bool(is_container),
            "container_y": int() | None as container_y,
            "container_viewport_x": int() | None as container_viewport_x,
        }:
            return handle_tank_registry(
                tid,
                name,
                team,
                rank,
                badges,
                is_bot,
                is_container,
                container_y,
                container_viewport_x,
            )
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "player_id": int(pid),
            "waypoints": str(waypoints),
            "is_self": bool(is_self),
        }:
            return format_movement(sx, sy, pid, waypoints, is_self)
        case {
            "msg_type": "position_update",
            "tank_id": int(tid),
            "flags": int(f),
            "x": int(x),
            "y": int(y),
            "extra_data": bytes(ed),
        }:
            return format_position_update(tid, x, y, f, ed)
        case _:
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
    "format_combat_hit",
    "format_container_details",
    "format_container_pickup",
    "format_container_simple",
    "format_decoded_message",
    "format_message_details",
    "format_misc_details",
    "format_movement",
    "format_position_details",
    "format_position_update",
    "format_radar_details",
    "format_radar_response",
    "format_resource_details",
    "format_tank_details",
    "format_tank_registry_details",
    "format_tank_status_short",
    "format_tank_update_details",
    "handle_tank_registry",
    "rank_name",
    "team_name",
]
