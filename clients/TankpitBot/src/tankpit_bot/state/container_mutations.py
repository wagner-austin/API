"""Container and mine world-state mutations."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.state.types import (
    ContainerRefreshKind,
    EntitySource,
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
            scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
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
        provenance=make_provenance("wire_0x4B_mine_placement", []),
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
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def pickup_container(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
    remaining_volume: int = 0,
) -> WorldStateDict:
    """Apply a 0x43 ContainerPickup record to world state.

    Only the container registry is touched here:

    * ``remaining_volume == 0`` -- container is emptied; remove it so
      the planner stops targeting it.
    * ``remaining_volume > 0`` -- partial pickup (picker hit the 1100
      cap before draining). Keep the container in state with its
      volume updated to ``remaining_volume`` so the planner can come
      back for the rest.

    Containers we've never seen before are NOT created here -- a 0x43
    record on an unknown tile is treated as a no-op so dispatch and
    radar discovery stay the single sources of new-container truth.

    ``self_state["fuel"]`` is NOT updated here. The wire always emits
    a separate absolute-fuel message (``0x44 FuelGain`` for partial /
    free pickups, ``0x2E TankStatusSync`` for the regular cadence,
    ``0x64 FuelDeposit`` for depot returns) which flows through
    :func:`tankpit_bot.state.mutations.set_self_fuel`. That path is the
    single source of truth for the bot's fuel total. Adding a local
    ``+ transferred`` delta here on top of the wire's absolute value
    double-counted every fuel pickup -- ~+438 ghost on a 438-volume
    container observed live on 2026-06-23.

    Args:
        state: Current world state.
        x: Container X coordinate.
        y: Container Y coordinate.
        timestamp_ms: Message timestamp.
        remaining_volume: Fuel left in the container after this pickup;
            ``0`` (default) means emptied.

    Returns:
        New WorldStateDict with the container registry updated.
    """
    key = coord_key(x, y)
    container = state["containers"].get(key)
    new_containers = dict(state["containers"])

    if remaining_volume <= 0:
        new_containers.pop(key, None)
    elif container is not None:
        new_containers[key] = make_container_state(
            x=container["x"],
            y=container["y"],
            is_fuel=container["is_fuel"],
            volume=remaining_volume,
            source=container["source"],
            refresh_kind=container["refresh_kind"],
            timestamp_ms=timestamp_ms,
            failed_pickups=container["failed_pickups"],
        )

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=new_containers,
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
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
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=state["timestamp_ms"],
    )


def apply_tile_overlay_update(
    state: WorldStateDict,
    x: int,
    y: int,
    overlay_value: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Reconcile ``world.mines`` from one tile's wire-decoded overlay byte.

    The 0x5A ``ViewportUpdate`` and 0x40 ``OverlayUpdate`` messages both
    carry the same per-tile mine-layer byte:

    * ``overlay_value`` in ``0..7`` -> mine present; team encoded in the
      low 2 bits (``team = overlay_value & 3``).
    * ``overlay_value >= 8`` (the decoder maps 8..15 to ``255``) -> tile
      has no mine: drop any tracked mine.

    The 0x5A / 0x40 path does NOT carry the placer's ``tank_id`` or the
    mine ``mine_type`` -- only 0x4B ``MinePlacement`` provides those. If
    a wire-rich mine already lives at the tile, preserve those fields
    while refreshing ``team`` and ``timestamp_ms``; otherwise seed with
    ``mine_type=0``, ``tank_id=-1``. Mirrors the merge policy in
    :func:`add_mine_from_radar`.

    Args:
        state: Current world state.
        x: Tile X coordinate.
        y: Tile Y coordinate.
        overlay_value: Decoded overlay byte (``0..7`` = mine, else clear).
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with ``world.mines`` reconciled for this tile.
    """
    if not 0 <= overlay_value <= 7:
        return remove_mine(state, x, y, timestamp_ms)

    team = overlay_value & 3
    key = coord_key(x, y)
    existing = state["mines"].get(key)
    if existing is None:
        merged_mine_type = 0
        merged_tank_id = -1
    else:
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]

    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=merged_mine_type,
        tank_id=merged_tank_id,
        team=team,
        source="viewport",
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
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def apply_tile_cache_update(
    state: WorldStateDict,
    x: int,
    y: int,
    cache_value: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Reconcile ``world.containers`` from one tile's wire-decoded cache byte.

    The 0x5A ``ViewportUpdate`` and 0x43 ``CacheUpdate`` messages both
    carry the same per-tile container-layer byte:

    * ``cache_value == 0``  -> tile is empty: drop any tracked container.
    * ``cache_value == -1`` -> equipment container at this tile.
    * ``cache_value > 0``   -> fuel container with that volume.

    The wire is per-tile authoritative for the tiles it enumerates.
    Tiles not enumerated by the message are not touched. The radar
    response remains envelope-authoritative via
    :func:`reconcile_radar_viewport_resources`.

    Args:
        state: Current world state.
        x: Tile X coordinate.
        y: Tile Y coordinate.
        cache_value: Decoded cache byte (``-1`` / ``0`` / fuel volume).
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with ``world.containers`` reconciled for
        this tile.
    """
    if cache_value == 0:
        return remove_container(state, x, y, timestamp_ms)

    is_fuel = cache_value > 0
    volume = cache_value if is_fuel else 0
    key = coord_key(x, y)
    existing = state["containers"].get(key)
    failed_pickups = existing["failed_pickups"] if existing is not None else 0
    new_container = make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        source="viewport",
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
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


__all__ = [
    "add_mine",
    "add_mine_from_radar",
    "apply_tile_cache_update",
    "apply_tile_overlay_update",
    "increment_container_failed_pickups",
    "pickup_container",
    "remove_container",
    "remove_mine",
    "update_container_from_radar",
]
