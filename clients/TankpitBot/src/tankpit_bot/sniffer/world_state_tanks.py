"""Tank state updates for world state.

Handles tank entry, info, status, registry, damage, exit, position tracking,
and enemy detection updates.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    remove_tank,
    update_tank_damage,
    update_tank_from_registry,
)
from tankpit_bot.state.types import WorldStateDict

log = get_logger(__name__)


def update_world_state_from_tank_entry(
    ws: WorldService,
    tank_id: int,
    team: int,
    rank: int,
    x: int,
    y: int,
) -> None:
    """Add or update tank from TankEntry (0x28).

    JS-verified 2026-06-19: this message carries team, rank, position,
    rank_category, and score. Name comes from TankInfo (0x21) separately.
    """
    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        rank,
        False,
        x,
        y,
        "viewport",
        ts,
        wire_present=True,
    )


def update_world_state_from_tank_info(
    ws: WorldService,
    tank_id: int,
    team: int,
    name: str,
) -> None:
    """Store/update tank from TankInfo (0x21)."""

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        existing["rank"] if existing else 0,
        existing["is_bot"] if existing else False,
        existing["x"] if existing else 0,
        existing["y"] if existing else 0,
        existing["source"] if existing else "viewport",
        ts,
        wire_present=True,
    )


def update_world_state_from_tank_status(
    ws: WorldService,
    tank_id: int,
    team: int,
    rank: int,
    name: str,
) -> None:
    """Store/update tank from TankStatus (0x3E)."""

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        rank,
        existing["is_bot"] if existing else False,
        existing["x"] if existing else 0,
        existing["y"] if existing else 0,
        existing["source"] if existing else "viewport",
        ts,
        wire_present=True,
    )


def update_world_state_from_tank_registry(
    ws: WorldService,
    tank_id: int,
    name: str,
    team_str: str,
    rank: int,
    is_bot: bool,
    tank_y: int,
    tank_viewport_x: int,
) -> None:
    """Store tank with position from tank_registry message.

    Computes absolute X from viewport_left + viewport_x.

    Args:
        ws: World service instance.
        tank_id: Tank ID.
        name: Tank name.
        team_str: Team name string ("red", "purple", "blue", "orange").
        rank: Military rank (0-7).
        is_bot: Whether tank is a bot.
        tank_y: Absolute Y coordinate.
        tank_viewport_x: Viewport-relative X coordinate.
    """

    from tankpit_bot.protocol.constants import TEAM_NAMES

    team = TEAM_NAMES.index(team_str) if team_str in TEAM_NAMES else 0

    viewport_left = get_viewport_left()
    if viewport_left is None:
        log.info(
            "Cannot add tank_registry tank: viewport_left not yet known (tank=%d, y=%d, vx=%d)",
            tank_id,
            tank_y,
            tank_viewport_x,
        )
        return
    tank_x = viewport_left + tank_viewport_x

    ts = browser.get_current_time_ms()
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        tank_x,
        tank_y,
        "viewport",
        ts,
        wire_present=True,
    )


def update_world_state_from_move_response_full(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update self_state and tank position from MovementResponse (0x3D).

    Args:
        ws: World service instance.
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team ID (0-3).
        rank: Military rank.
    """

    ts = browser.get_current_time_ms()

    self_state = ws.world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
        from tankpit_bot.state.types import SelfStateDict

        ws.world_state = WorldStateDict(
            self_state=SelfStateDict(
                tank_id=tank_id,
                x=x,
                y=y,
                team=team,
                rank=rank,
                fuel=self_state["fuel"] if self_state else 0,
                leaderboard_position=0,
            ),
            tanks=ws.world_state["tanks"],
            containers=ws.world_state["containers"],
            mines=ws.world_state["mines"],
            terrain=ws.world_state["terrain"],
            viewport=ws.world_state["viewport"],
            scanned_viewports=ws.world_state["scanned_viewports"],
            map_fuel_dots=ws.world_state["map_fuel_dots"],
            timestamp_ms=ts,
        )
    elif self_state["tank_id"] == tank_id:
        ws.update_world_state_from_position(x, y)

    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        x,
        y,
        "viewport",
        ts,
        wire_present=True,
    )


def update_world_state_from_client_registry(
    ws: WorldService,
    tank_id: int,
    name: str,
    team: int,
    x: int,
    y: int,
) -> bool:
    """Refine a WIRE-KNOWN tank's position from the client registry.

    Args:
        ws: World service instance.
        tank_id: Tank ID (shared with the wire ID space).
        name: Tank name from the registry entry.
        team: Team ID from the verified ``h`` field.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.

    Returns:
        True when a wire-known tank was refined; False for unknown ids.
    """
    existing = ws.world_state["tanks"].get(str(tank_id))
    if existing is None:
        return False
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        existing["rank"],
        existing["is_bot"],
        x,
        y,
        "viewport",
        existing["timestamp_ms"],
        wire_present=False,
    )
    return True


def update_world_state_from_tank_damage(
    ws: WorldService,
    tank_id: int,
    damage_state: int,
    *,
    refresh_wire_timestamp: bool = True,
) -> None:
    """Update tank damage from TankStatusSync (0x2E) or registry truth.

    Args:
        ws: World service instance.
        tank_id: Tank whose damage tier is being synced.
        damage_state: New damage tier.
        refresh_wire_timestamp: True for wire-sourced updates.
    """
    previous = ws.world_state["tanks"].get(str(tank_id))
    if previous is None:
        return
    ts = browser.get_current_time_ms() if refresh_wire_timestamp else previous["timestamp_ms"]
    ws.world_state = update_tank_damage(ws.world_state, tank_id, damage_state, ts)
    if previous["damage_state"] != damage_state:
        emit_diagnostic(
            diagnostic_kind="tank_damage_changed",
            tank_id=tank_id,
            tank_name=previous["name"],
            previous_damage_state=previous["damage_state"],
            damage_state=damage_state,
        )


def update_world_state_from_tank_exit(ws: WorldService, tank_id: int) -> None:
    """Remove tank from world state on TankExit (0x58)."""

    ws.world_state = remove_tank(ws.world_state, tank_id, browser.get_current_time_ms())


def _update_tank_position(ws: WorldService, tank_id: int, x: int, y: int) -> None:
    """Update any tank's position from a position-carrying message.

    Args:
        ws: World service instance.
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        existing["team"] if existing else 0,
        existing["name"] if existing else "",
        existing["rank"] if existing else 0,
        existing["is_bot"] if existing else False,
        x,
        y,
        "viewport",
        ts,
        wire_present=True,
    )


def _update_enemy_from_detection(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update enemy tank position from EnemyDetection (0x48) response.

    Args:
        ws: World service instance.
        tank_id: Enemy tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Enemy team ID.
        rank: Enemy military rank.
    """

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = ws.world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    ws.world_state = update_tank_from_registry(
        ws.world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        x,
        y,
        "world_state",
        ts,
        wire_present=False,
    )
    log.info(
        "ENEMY_DETECT: tank=%d at (%d,%d) team=%d rank=%d name=%s",
        tank_id,
        x,
        y,
        team,
        rank,
        name,
    )
