"""World state mutation functions.

This module provides pure functions that update WorldStateDict by creating
new state objects (immutable update pattern).
"""

from __future__ import annotations

from tankpit_bot.state.types import (
    DAMAGE_FULL,
    SelfStateDict,
    TankStateDict,
    ViewportStateDict,
    WorldStateDict,
    coord_key,
    make_container_state,
    make_mine_state,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
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
        timestamp_ms=timestamp_ms,
    )


def update_self_position_and_viewport(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self position and center viewport around it.

    Updates the self_state x,y coordinates and recenters the viewport.
    Creates a minimal self_state if none exists.

    Args:
        state: Current world state.
        x: New X coordinate.
        y: New Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated position and viewport.
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

    # Center viewport on new position
    vp = state["viewport"]
    new_viewport = ViewportStateDict(
        left=x - vp["width"] // 2,
        top=y - vp["height"] // 2,
        width=vp["width"],
        height=vp["height"],
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=new_viewport,
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
    timestamp_ms: int,
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
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated tank.
    """
    is_self = state["self_state"] is not None and state["self_state"]["tank_id"] == tank_id
    key = str(tank_id)
    existing = state["tanks"].get(key)
    damage_state = existing["damage_state"] if existing else DAMAGE_FULL

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
        timestamp_ms=timestamp_ms,
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
        timestamp_ms=timestamp_ms,
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
        timestamp_ms=timestamp_ms,
    )


def update_container_from_radar(
    state: WorldStateDict,
    x: int,
    y: int,
    volume: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update container state from RadarResponse message.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.
        volume: Fuel volume (-1 for equipment, 0 for empty fuel, >0 for fuel with contents).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated container, or unchanged state if empty fuel.
    """
    is_fuel = volume >= 0
    key = coord_key(x, y)
    existing = state["containers"].get(key)

    # Radar explicitly reports empty fuel as volume=0. Treat that as authoritative
    # removal for any existing fuel target at this coordinate so ghost containers
    # do not linger in world state and get reselected by the planner.
    if is_fuel and volume == 0:
        if existing is None:
            return state
        new_containers = dict(state["containers"])
        del new_containers[key]
        return WorldStateDict(
            self_state=state["self_state"],
            tanks=state["tanks"],
            containers=new_containers,
            mines=state["mines"],
            terrain=state["terrain"],
            viewport=state["viewport"],
            timestamp_ms=timestamp_ms,
        )

    actual_volume = volume if is_fuel else 0
    failed_pickups = existing["failed_pickups"] if existing is not None else 0

    new_container = make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=actual_volume,
        timestamp_ms=timestamp_ms,
        failed_pickups=failed_pickups,
    )

    new_containers = dict(state["containers"])
    new_containers[key] = new_container

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=new_containers,
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        timestamp_ms=timestamp_ms,
    )


def remove_container(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Remove container after pickup.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with container removed.
    """
    key = coord_key(x, y)
    if key not in state["containers"]:
        return state

    new_containers = dict(state["containers"])
    del new_containers[key]

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=new_containers,
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        timestamp_ms=timestamp_ms,
    )


def add_mine(
    state: WorldStateDict,
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Add mine from MinePlacement message.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        mine_type: Type of mine.
        tank_id: ID of placing tank.
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with mine added.
    """
    new_mine = make_mine_state(x=x, y=y, mine_type=mine_type, tank_id=tank_id, team=team)

    key = coord_key(x, y)
    new_mines = dict(state["mines"])
    new_mines[key] = new_mine

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        timestamp_ms=timestamp_ms,
    )


def add_mine_from_radar(
    state: WorldStateDict,
    x: int,
    y: int,
    team: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Add mine discovered via radar scan.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with mine added.
    """
    new_mine = make_mine_state(x=x, y=y, mine_type=0, tank_id=-1, team=team)

    key = coord_key(x, y)
    new_mines = dict(state["mines"])
    new_mines[key] = new_mine

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        timestamp_ms=timestamp_ms,
    )


def remove_mine(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Remove mine after detonation.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with mine removed.
    """
    key = coord_key(x, y)
    if key not in state["mines"]:
        return state

    new_mines = dict(state["mines"])
    del new_mines[key]

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        timestamp_ms=timestamp_ms,
    )


def update_terrain_from_viewport(
    state: WorldStateDict,
    viewport_left: int,
    viewport_top: int,
    entities: list[tuple[int, int, int, int]],
    timestamp_ms: int,
) -> WorldStateDict:
    """Update terrain from ViewportUpdate message.

    Args:
        state: Current world state.
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        entities: List of (col, row, terrain_type, entity_id) tuples.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated terrain and viewport.
    """
    new_terrain = dict(state["terrain"])
    new_viewport = ViewportStateDict(
        left=viewport_left,
        top=viewport_top,
        width=18,
        height=18,
    )

    for col, row, terrain_type, entity_id in entities:
        x = viewport_left + col
        y = viewport_top + row
        key = coord_key(x, y)
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            entity_id=entity_id,
        )

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=new_terrain,
        viewport=new_viewport,
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
        timestamp_ms=timestamp_ms,
    )


def pickup_container(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Pick up container and add fuel to self (if fuel container).

    Removes the container from world state and adds its volume to self fuel.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with container removed and fuel updated.
    """
    key = coord_key(x, y)
    container = state["containers"].get(key)

    # Remove container
    new_containers = dict(state["containers"])
    new_containers.pop(key, None)

    # Add fuel to self if it was a fuel container
    new_self = state["self_state"]
    if new_self is not None and container is not None and container["is_fuel"]:
        new_fuel = new_self["fuel"] + container["volume"]
        new_self = SelfStateDict(
            tank_id=new_self["tank_id"],
            x=new_self["x"],
            y=new_self["y"],
            team=new_self["team"],
            rank=new_self["rank"],
            fuel=new_fuel,
            leaderboard_position=new_self["leaderboard_position"],
        )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=new_containers,
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
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
        timestamp_ms=timestamp_ms,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "add_mine",
    "add_mine_from_radar",
    "pickup_container",
    "remove_container",
    "remove_mine",
    "remove_tank",
    "set_self_fuel",
    "update_container_from_radar",
    "update_self_from_movement_response",
    "update_self_fuel",
    "update_self_position_and_viewport",
    "update_tank_damage",
    "update_tank_from_registry",
    "update_terrain_from_viewport",
]
