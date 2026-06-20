"""Container and mine world-state mutations."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.state.types import (
    ContainerRefreshKind,
    EntitySource,
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
    """Add or refresh a mine discovered via radar scan (0x4F-tunneled).

    Radar mine entries are 3 bytes wide -- ``x, y, team`` -- and carry
    NEITHER ``mine_type`` NOR the placer's ``tank_id``. Those fields are
    only knowable via wire MinePlacement (V.K / 0x4B, per tpclient.js
    handler ``Dg.h``). When radar refreshes a tile where a wire-placed
    mine already lives, this mutator must preserve the wire-known
    ``mine_type`` and ``tank_id`` -- they came from a richer source and
    radar cannot reproduce them.

    Merge rules:
      * New tile (no existing mine): seed with ``mine_type=0``,
        ``tank_id=-1``, ``source="radar"``.
      * Existing wire-sourced mine: preserve ``mine_type`` and
        ``tank_id``, keep ``source="viewport"`` (still wire-richer),
        advance ``timestamp_ms``, update ``team`` to the radar value
        (the wire team is authoritative on placement and the radar
        team is authoritative on refresh -- a placement followed by a
        radar sighting at the same tile is the same mine, and team
        cannot legally change for an undetonated mine, so this
        difference indicates the wire team field went stale and should
        be re-synced).
      * Existing radar-sourced mine: refresh as before with
        ``source="radar"``.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with the mine added or refreshed.
    """
    key = coord_key(x, y)
    existing = state["mines"].get(key)
    if existing is None:
        merged_mine_type = 0
        merged_tank_id = -1
        merged_source: EntitySource = "radar"
    elif existing["source"] == "viewport":
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]
        merged_source = "viewport"
    else:
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]
        merged_source = "radar"

    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=merged_mine_type,
        tank_id=merged_tank_id,
        team=team,
        source=merged_source,
        timestamp_ms=timestamp_ms,
    )

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


def increment_container_failed_pickups(
    state: WorldStateDict,
    x: int,
    y: int,
) -> WorldStateDict:
    """Increment the ``failed_pickups`` counter for a container.

    Used by the planner to deprioritize containers whose pickups stall.
    The container's ``timestamp_ms`` is preserved so this is not a
    freshness update; only the diagnostic counter advances. Returns
    ``state`` unchanged if no container exists at ``(x, y)``.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.

    Returns:
        New ``WorldStateDict`` with the container's
        ``failed_pickups`` advanced by one, or the original state if
        no container is at ``(x, y)``.
    """
    key = coord_key(x, y)
    container = state["containers"].get(key)
    if container is None:
        return state
    new_container = make_container_state(
        x=container["x"],
        y=container["y"],
        is_fuel=container["is_fuel"],
        volume=container["volume"],
        source=container["source"],
        refresh_kind=container["refresh_kind"],
        timestamp_ms=container["timestamp_ms"],
        failed_pickups=container["failed_pickups"] + 1,
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
        timestamp_ms=state["timestamp_ms"],
    )


__all__ = [
    "add_mine",
    "add_mine_from_radar",
    "increment_container_failed_pickups",
    "pickup_container",
    "remove_container",
    "remove_mine",
    "update_container_from_radar",
]
