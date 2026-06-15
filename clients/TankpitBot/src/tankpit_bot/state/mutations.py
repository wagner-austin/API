"""World state mutation functions.

This module provides pure functions that update WorldStateDict by creating
new state objects (immutable update pattern).
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.state.types import (
    DAMAGE_FULL,
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    coord_key,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
    viewport_scan_key,
)
from tankpit_bot.state.viewport_geometry import (
    make_visible_viewport_state,
    viewport_patch_world_coords,
)

# =============================================================================
# World State Update Functions (from protocol messages)
# =============================================================================


def update_self_from_movement_response(
    state: WorldStateDict,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    leaderboard_position: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self state from MovementResponse message.

    Args:
        state: Current world state.
        tank_id: Player's tank ID.
        x: New X coordinate.
        y: New Y coordinate.
        team: Team ID.
        rank: Military rank.
        leaderboard_position: Leaderboard position.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated self state.
    """
    new_self = make_self_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=state["self_state"]["fuel"] if state["self_state"] else 0,
        leaderboard_position=leaderboard_position,
    )
    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_self_position(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self position without changing viewport bounds.

    Updates the self_state x,y coordinates and preserves the current viewport.
    Creates a minimal self_state if none exists.

    Args:
        state: Current world state.
        x: New X coordinate.
        y: New Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated position.
    """
    # Get or create self_state with updated position
    if state["self_state"] is not None:
        new_self = SelfStateDict(
            tank_id=state["self_state"]["tank_id"],
            x=x,
            y=y,
            team=state["self_state"]["team"],
            rank=state["self_state"]["rank"],
            fuel=state["self_state"]["fuel"],
            leaderboard_position=state["self_state"]["leaderboard_position"],
        )
    else:
        # Create minimal self_state when we first learn position
        new_self = SelfStateDict(
            tank_id=0,
            x=x,
            y=y,
            team=0,
            rank=0,
            fuel=0,
            leaderboard_position=0,
        )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_tank_from_registry(
    state: WorldStateDict,
    tank_id: int,
    team: int,
    name: str,
    rank: int,
    is_bot: bool,
    x: int,
    y: int,
    source: Literal["viewport", "radar", "world_state"],
    timestamp_ms: int,
    *,
    wire_present: bool = True,
) -> WorldStateDict:
    """Update tank state from TankRegistry message.

    Args:
        state: Current world state.
        tank_id: Tank ID.
        team: Team ID.
        name: Player name.
        rank: Military rank.
        is_bot: Whether this is a bot.
        x: X coordinate.
        y: Y coordinate.
        source: Source that confirmed this tank.
        timestamp_ms: Message timestamp.
        wire_present: Whether this update came from a live wire message.
            When True the tank's ``last_wire_seen_ms`` is stamped to
            *timestamp_ms*; when False the existing value is preserved
            (or 0 for a first-seen tank).

    Returns:
        New WorldStateDict with updated tank.
    """
    is_self = state["self_state"] is not None and state["self_state"]["tank_id"] == tank_id
    key = str(tank_id)
    existing = state["tanks"].get(key)
    damage_state = existing["damage_state"] if existing else DAMAGE_FULL
    if wire_present:
        last_wire_seen_ms = timestamp_ms
    else:
        last_wire_seen_ms = existing["last_wire_seen_ms"] if existing else 0

    new_tank = make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        name=name,
        is_bot=is_bot,
        is_self=is_self,
        source=source,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
    )

    new_tanks = dict(state["tanks"])
    new_tanks[key] = new_tank

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_tank_damage(
    state: WorldStateDict,
    tank_id: int,
    damage_state: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update tank damage state from TankStatusShort message.

    Args:
        state: Current world state.
        tank_id: Tank ID.
        damage_state: New damage state (0-3).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated tank damage.
    """
    key = str(tank_id)
    existing = state["tanks"].get(key)
    if existing is None:
        return state

    new_tank = TankStateDict(
        tank_id=existing["tank_id"],
        x=existing["x"],
        y=existing["y"],
        team=existing["team"],
        rank=existing["rank"],
        damage_state=damage_state,
        name=existing["name"],
        is_bot=existing["is_bot"],
        is_self=existing["is_self"],
        source=existing["source"],
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=existing["last_wire_seen_ms"],
    )

    new_tanks = dict(state["tanks"])
    new_tanks[key] = new_tank

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_terrain_from_viewport(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    entities: list[tuple[int, int, int, int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Update terrain from a visible viewport update.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        entities: List of ``0x5A`` patch-grid
            ``(col, row, terrain_type, cache_value, overlay_value)`` tuples.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated terrain, viewport, and confirmed
        local-resource coverage for that viewport origin.
    """
    new_terrain = dict(state["terrain"])
    new_viewport = make_visible_viewport_state(viewport_left, viewport_top)
    key = viewport_scan_key(viewport_left, viewport_top)
    new_scanned_viewports = dict(state["scanned_viewports"])
    new_scanned_viewports[key] = timestamp_ms

    for col, row, terrain_type, cache_value, overlay_value in entities:
        x, y = viewport_patch_world_coords(viewport_left, viewport_top, col, row)
        key = coord_key(x, y)
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=new_terrain,
        viewport=new_viewport,
        scanned_viewports=new_scanned_viewports,
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_self_fuel(
    state: WorldStateDict,
    fuel_delta: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self fuel by adding a delta.

    Args:
        state: Current world state.
        fuel_delta: Fuel amount to add (can be negative for damage).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated fuel, or unchanged if no self_state.
    """
    if state["self_state"] is None:
        return state

    new_fuel = max(0, state["self_state"]["fuel"] + fuel_delta)
    new_self = SelfStateDict(
        tank_id=state["self_state"]["tank_id"],
        x=state["self_state"]["x"],
        y=state["self_state"]["y"],
        team=state["self_state"]["team"],
        rank=state["self_state"]["rank"],
        fuel=new_fuel,
        leaderboard_position=state["self_state"]["leaderboard_position"],
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def set_self_fuel(
    state: WorldStateDict,
    fuel: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Set self fuel to absolute value (from inventory or sync messages).

    Args:
        state: Current world state.
        fuel: Absolute fuel value.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated fuel, or unchanged if no self_state.
    """
    if state["self_state"] is None:
        return state

    new_self = SelfStateDict(
        tank_id=state["self_state"]["tank_id"],
        x=state["self_state"]["x"],
        y=state["self_state"]["y"],
        team=state["self_state"]["team"],
        rank=state["self_state"]["rank"],
        fuel=max(0, fuel),
        leaderboard_position=state["self_state"]["leaderboard_position"],
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def remove_tank(
    state: WorldStateDict,
    tank_id: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Remove tank when it exits.

    Args:
        state: Current world state.
        tank_id: Tank ID to remove.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with tank removed.
    """
    key = str(tank_id)
    if key not in state["tanks"]:
        return state

    new_tanks = dict(state["tanks"])
    del new_tanks[key]

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=new_tanks,
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def mark_viewport_scanned(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Record that a viewport origin has authoritative local-resource coverage.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        timestamp_ms: Scan completion timestamp.

    Returns:
        New WorldStateDict with updated viewport confirmation metadata.
    """
    key = viewport_scan_key(viewport_left, viewport_top)
    new_scanned_viewports = dict(state["scanned_viewports"])
    new_scanned_viewports[key] = timestamp_ms
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=new_scanned_viewports,
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def replace_map_fuel_dots(
    state: WorldStateDict,
    dots: list[tuple[int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Replace the map-wide fuel-container atlas from a MAP_DATA dot layer.

    The dot layer is server-cached and arrives complete on every MAP_DATA
    response, so the previous atlas is replaced wholesale rather than
    merged -- a dot that disappeared from the layer is gone.

    Args:
        state: Current world state.
        dots: Decoded ``(x, y)`` world coordinates of every fuel dot.
        timestamp_ms: MAP_DATA processing timestamp.

    Returns:
        New WorldStateDict with the replaced fuel-dot atlas.
    """
    new_dots = {coord_key(x, y): timestamp_ms for x, y in dots}
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=new_dots,
        timestamp_ms=timestamp_ms,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "mark_viewport_scanned",
    "remove_tank",
    "replace_map_fuel_dots",
    "set_self_fuel",
    "update_self_from_movement_response",
    "update_self_fuel",
    "update_self_position",
    "update_tank_damage",
    "update_tank_from_registry",
    "update_terrain_from_viewport",
]
