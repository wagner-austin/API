"""Container and mine world-state mutations."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.state.types import (
    ContainerRefreshKind,
    SelfStateDict,
    WorldStateDict,
    coord_key,
    make_container_state,
    make_mine_state,
)

log = get_logger(__name__)


def update_container_from_radar(
    state: WorldStateDict,
    x: int,
    y: int,
    volume: int,
    timestamp_ms: int,
    *,
    refresh_kind: ContainerRefreshKind = "radar_response",
) -> WorldStateDict:
    """Update container state from RadarResponse message.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.
        volume: Fuel volume (-1 for equipment, 0 for empty fuel, >0 for fuel with contents).
        timestamp_ms: Message timestamp.
        refresh_kind: Specific radar-refresh path that confirmed the container.

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
            scanned_viewports=state["scanned_viewports"],
            map_fuel_dots=state["map_fuel_dots"],
            timestamp_ms=timestamp_ms,
        )

    actual_volume = volume if is_fuel else 0
    failed_pickups = existing["failed_pickups"] if existing is not None else 0

    new_container = make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=actual_volume,
        source="radar",
        refresh_kind=refresh_kind,
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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
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
    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=mine_type,
        tank_id=tank_id,
        team=team,
        source="viewport",
        timestamp_ms=timestamp_ms,
    )

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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
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
    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=0,
        tank_id=-1,
        team=team,
        source="radar",
        timestamp_ms=timestamp_ms,
    )

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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
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
        scanned_viewports=state["scanned_viewports"],
        map_fuel_dots=state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


__all__ = [
    "add_mine",
    "add_mine_from_radar",
    "pickup_container",
    "remove_container",
    "remove_mine",
    "update_container_from_radar",
]
