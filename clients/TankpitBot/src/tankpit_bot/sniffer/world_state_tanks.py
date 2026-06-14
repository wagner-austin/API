"""Tank state updates for world state.

Handles tank entry, info, status, registry, damage, exit, position tracking,
and enemy detection updates.
"""

from __future__ import annotations

from platform_core.logging import get_logger

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot import browser
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.state import (
    remove_tank,
    update_tank_damage,
    update_tank_from_registry,
)
from tankpit_bot.state.types import WorldStateDict

log = get_logger(__name__)


def update_world_state_from_tank_entry(tank_id: int, x: int, y: int, name: str) -> None:
    """Add or update tank from TankEntry (0x28) — has position but no team."""

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    team = existing["team"] if existing else 0
    rank = existing["rank"] if existing else 0
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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


def update_world_state_from_tank_info(tank_id: int, team: int, name: str) -> None:
    """Store/update tank from TankInfo (0x21)."""

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
    tank_id: int,
    team: int,
    rank: int,
    name: str,
) -> None:
    """Store/update tank from TankStatus (0x3E)."""

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
        tank_id: Tank ID.
        name: Tank name.
        team_str: Team name string ("red", "purple", "blue", "orange").
        rank: Military rank (0-7).
        is_bot: Whether tank is a bot.
        tank_y: Absolute Y coordinate.
        tank_viewport_x: Viewport-relative X coordinate.
    """

    from tankpit_bot.protocol.constants import TEAM_NAMES

    # Convert team string to int
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
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update self_state and tank position from MovementResponse (0x3D).

    The first 0x3D received establishes the bot's identity (tank_id, team, rank).
    All 0x3D messages update the corresponding tank's position.

    Args:
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team ID (0-3).
        rank: Military rank.
    """

    ts = browser.get_current_time_ms()

    # Update self_state with real identity data
    self_state = _ws._world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
        from tankpit_bot.state.types import SelfStateDict

        _ws._world_state = WorldStateDict(
            self_state=SelfStateDict(
                tank_id=tank_id,
                x=x,
                y=y,
                team=team,
                rank=rank,
                fuel=self_state["fuel"] if self_state else 0,
                leaderboard_position=0,
            ),
            tanks=_ws._world_state["tanks"],
            containers=_ws._world_state["containers"],
            mines=_ws._world_state["mines"],
            terrain=_ws._world_state["terrain"],
            viewport=_ws._world_state["viewport"],
            scanned_viewports=_ws._world_state["scanned_viewports"],
            map_fuel_dots=_ws._world_state["map_fuel_dots"],
            timestamp_ms=ts,
        )
    elif self_state["tank_id"] == tank_id:
        # Update self position
        _ws.update_world_state_from_position(x, y)

    # Update the tank in the tank list
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
    tank_id: int,
    name: str,
    team: int,
    x: int,
    y: int,
) -> bool:
    """Refine a WIRE-KNOWN tank's position from the client registry.

    The wire vouches for presence; the registry only refines. The tank
    timestamp is wire provenance and is deliberately PRESERVED here --
    when registry refreshes bumped it, stale afterimage entries looked
    eternally fresh and run 20260611-103309 shot one 52 times. Unknown
    tank ids are skipped for the same reason: a tank the wire never
    announced is not in the viewport (raw-capture measurement: in-fight
    tanks emit ~one wire message per 3.3s; afterimages are silent).

    Args:
        tank_id: Tank ID (shared with the wire ID space).
        name: Tank name from the registry entry.
        team: Team ID from the verified ``h`` field.
        x: Absolute X coordinate (already mapped from render coords).
        y: Absolute Y coordinate (already mapped from render coords).

    Returns:
        True when a wire-known tank was refined; False for unknown ids.
    """
    existing = _ws._world_state["tanks"].get(str(tank_id))
    if existing is None:
        return False
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
    tank_id: int,
    damage_state: int,
    *,
    refresh_wire_timestamp: bool = True,
) -> None:
    """Update tank damage from TankStatusSync (0x2E) or registry truth.

    A damage-tier transition is the only wire evidence that an outgoing
    homing shot connected -- the game log prints just the launch marker
    (``You fire``) and the combat message carries only the weapon type
    -- so the transition is emitted as a DIAGNOSTIC instead of being
    consumed silently (run 20260610-223x had zero artifact evidence of
    27 homing shots' effect).

    Args:
        tank_id: Tank whose damage tier is being synced.
        damage_state: New damage tier.
        refresh_wire_timestamp: True for wire-sourced updates. The
            registry tier push passes False: the tank timestamp is
            wire-presence provenance, and registry refreshes bumping it
            made stale afterimage entries look eternally fresh (run
            20260611-103309 shot one 52 times).
    """
    previous = _ws._world_state["tanks"].get(str(tank_id))
    if previous is None:
        return
    ts = browser.get_current_time_ms() if refresh_wire_timestamp else previous["timestamp_ms"]
    _ws._world_state = update_tank_damage(_ws._world_state, tank_id, damage_state, ts)
    if previous["damage_state"] != damage_state:
        emit_diagnostic(
            diagnostic_kind="tank_damage_changed",
            tank_id=tank_id,
            tank_name=previous["name"],
            previous_damage_state=previous["damage_state"],
            damage_state=damage_state,
        )


def update_world_state_from_tank_exit(tank_id: int) -> None:
    """Remove tank from world state on TankExit (0x58)."""

    _ws._world_state = remove_tank(_ws._world_state, tank_id, browser.get_current_time_ms())


def _update_tank_position(tank_id: int, x: int, y: int) -> None:
    """Update any tank's position from a position-carrying message.

    Creates the tank if it doesn't exist yet, preserving existing metadata.

    Args:
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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


def _update_enemy_from_detection(tank_id: int, x: int, y: int, team: int, rank: int) -> None:
    """Update enemy tank position from EnemyDetection (0x48) response.

    Sent by server in response to CMD_NEAREST_ENEMY ('e' key).
    Contains absolute x,y for the nearest enemy. The query reports the
    server's nearest known enemy anywhere on the map -- which may be
    outside the viewport -- so it refines position for acquisition but is
    NOT in-view wire-presence evidence: ``wire_present=False`` preserves
    ``last_wire_seen_ms`` so a detection-only enemy never becomes
    shoot-eligible until a genuine in-view source vouches for it.

    Args:
        tank_id: Enemy tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Enemy team ID.
        rank: Enemy military rank.
    """

    ts = browser.get_current_time_ms()
    key = str(tank_id)
    existing = _ws._world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    _ws._world_state = update_tank_from_registry(
        _ws._world_state,
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
