"""Tank state updates for world state.

Every tank-mutating helper here builds a strongly-typed
:class:`TankObservation` and routes it through
:func:`apply_tank_observation`. The mutator enforces the three-timestamp
freshness model (see :class:`TankStateDict`); these helpers exist only
to translate wire-message fields into observations and to thread the
``WorldService`` boundary.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import remove_tank
from tankpit_bot.state.mutations import apply_tank_observation
from tankpit_bot.state.types import SelfStateDict, WorldStateDict, make_tank_observation

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

    JS-verified 2026-06-19 against ``Uf.h``: this message carries team,
    rank, position, rank_category, and score. Name comes from TankInfo
    (0x21) separately.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        rank: Military rank.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_tank_info(
    ws: WorldService,
    tank_id: int,
    team: int,
    name: str,
) -> None:
    """Store/update tank from TankInfo (0x21).

    Wire message: carries team and name. Position is preserved from the
    existing registry entry (TankInfo does not refresh position).

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        name: Player name.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        team=team,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_tank_status(
    ws: WorldService,
    tank_id: int,
    team: int,
    rank: int,
    name: str,
) -> None:
    """Store/update tank from TankStatus (0x3E).

    Wire message: carries team, rank, and name. No position.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        team: Team id (0-3).
        rank: Military rank.
        name: Player name.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        team=team,
        rank=rank,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


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
    """Store tank with position from a container ``tank_registry`` message.

    Computes the absolute X coordinate from
    ``viewport_left + viewport_x``. If the viewport origin is not yet
    known the call is a no-op (the registry entry will be re-emitted on
    the next viewport).

    Args:
        ws: World service instance.
        tank_id: Tank id.
        name: Tank name.
        team_str: Team name string ("red", "purple", "blue", "orange").
        rank: Military rank (0-7).
        is_bot: Whether the tank is a bot.
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
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        position=(tank_x, tank_y),
        team=team,
        rank=rank,
        name=name,
        is_bot=is_bot,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_move_response_full(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update self_state and tank position from MovementResponse (0x3D).

    For the bot's own tank id this both promotes the tank into
    ``self_state`` (creating a minimal record on first sight) and
    updates the registry entry. For non-self tank ids it only updates
    the registry entry.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team id (0-3).
        rank: Military rank.
    """
    ts = browser.get_current_time_ms()

    self_state = ws.world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
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

    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def update_world_state_from_client_registry(
    ws: WorldService,
    tank_id: int,
    name: str,
    team: int,
    x: int,
    y: int,
) -> bool:
    """Refine a WIRE-KNOWN tank's position from the client-side registry.

    Client registry refinement is NOT a wire observation -- it is a
    DOM-scraped value used to nudge position when the wire has gone
    quiet. The observation therefore sets ``is_wire_sourced=False`` so
    neither ``last_wire_seen_ms`` nor ``last_position_update_ms``
    advance; only the position value is updated. Tanks that the wire
    has never confirmed are not refined (this method returns False).

    Args:
        ws: World service instance.
        tank_id: Tank id (shared with the wire ID space).
        name: Tank name from the registry entry.
        team: Team id from the verified ``h`` field.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.

    Returns:
        True when a wire-known tank was refined; False for unknown ids.
    """
    existing = ws.world_state["tanks"].get(str(tank_id))
    if existing is None:
        return False
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=existing["timestamp_ms"],
        is_wire_sourced=False,
        storage_source="viewport",
        position=(x, y),
        team=team,
        name=name,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    return True


def update_world_state_from_tank_damage(
    ws: WorldService,
    tank_id: int,
    damage_state: int,
    *,
    refresh_wire_timestamp: bool = True,
) -> None:
    """Update tank damage from TankStatusSync (0x2E) or registry truth.

    Damage-only wire messages refresh ``last_wire_seen_ms`` (presence
    proof) but MUST NOT refresh ``last_position_update_ms`` -- the
    historical conflation of those two timestamps is the bug this
    refactor was built to prevent. The mutator enforces that invariant.

    Args:
        ws: World service instance.
        tank_id: Tank whose damage tier is being synced.
        damage_state: New damage tier (0-3).
        refresh_wire_timestamp: True for wire-sourced updates; False
            when the caller is a non-wire diagnostic (e.g.
            registry-truth recomputation) and must preserve the
            wire-seen timestamp.
    """
    previous = ws.world_state["tanks"].get(str(tank_id))
    if previous is None:
        return
    ts = browser.get_current_time_ms() if refresh_wire_timestamp else previous["timestamp_ms"]
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=refresh_wire_timestamp,
        storage_source=previous["source"],
        damage_state=damage_state,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    if previous["damage_state"] != damage_state:
        emit_diagnostic(
            diagnostic_kind="tank_damage_changed",
            tank_id=tank_id,
            tank_name=previous["name"],
            previous_damage_state=previous["damage_state"],
            damage_state=damage_state,
        )


def update_world_state_from_tank_remove(ws: WorldService, tank_id: int) -> None:
    """Remove tank from world state on TankRemove (0x58) or container TankLeave.

    Server-driven removal — clears the tile entry and drops the rendered
    tank. Distinct from 0x29 TankExit, which is an announcement-only
    message (the player-visible "left the game" / "eliminated" log line)
    and does not mutate world state.

    Removal is a deletion, not an observation, so it does not flow
    through the observation pipeline; it goes straight to
    :func:`remove_tank`.

    Args:
        ws: World service instance.
        tank_id: Departing tank id.
    """
    ws.world_state = remove_tank(ws.world_state, tank_id, browser.get_current_time_ms())


def _update_tank_position(ws: WorldService, tank_id: int, x: int, y: int) -> None:
    """Update a tank's position from a position-bearing wire message.

    Used by container TankUpdate (compact/extended/full) where bytes
    0-1 carry the new position, by the 0x47 Movement waypoint
    destination resolution, and by the 0x53 ShootEvent enemy-source
    position update. The observation declares wire-sourced + position
    so the position-freshness timestamp advances.

    Args:
        ws: World service instance.
        tank_id: Tank identifier.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=True,
        storage_source="viewport",
        position=(x, y),
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)


def _update_enemy_from_detection(
    ws: WorldService,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update enemy tank position from EnemyDetection (0x48).

    EnemyDetect is the server's response to a radar action. Position
    semantics deliberately route through the non-wire observation path
    so neither ``last_wire_seen_ms`` nor ``last_position_update_ms``
    advance -- a radar-derived position is a map-style estimate, not
    structural wire-presence proof. The kill-shot gate must continue
    to require fresh wire-bearing position; radar alone does not
    suffice.

    Args:
        ws: World service instance.
        tank_id: Enemy tank id.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Enemy team id.
        rank: Enemy military rank.
    """
    ts = browser.get_current_time_ms()
    obs = make_tank_observation(
        tank_id=tank_id,
        timestamp_ms=ts,
        is_wire_sourced=False,
        storage_source="world_state",
        position=(x, y),
        team=team,
        rank=rank,
    )
    ws.world_state = apply_tank_observation(ws.world_state, obs)
    existing = ws.world_state["tanks"].get(str(tank_id))
    name = existing["name"] if existing is not None else ""
    log.info(
        "ENEMY_DETECT: tank=%d at (%d,%d) team=%d rank=%d name=%s",
        tank_id,
        x,
        y,
        team,
        rank,
        name,
    )
