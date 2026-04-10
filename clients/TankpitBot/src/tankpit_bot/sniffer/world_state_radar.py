"""Radar scan handling and resource reconciliation for world state.

Handles radar scan results, differential cache promotion, viewport scan
confirmation, and stale resource reconciliation within the radar envelope.
"""

from __future__ import annotations

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.container import RadarContainerDict, RadarMineDict
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.state import (
    ContainerStateDict,
    MineStateDict,
    WorldStateDict,
    add_mine_from_radar,
    coord_key,
    make_terrain_tile,
    mark_viewport_scanned,
    update_container_from_radar,
)


def update_world_state_from_radar(
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict],
) -> None:
    """Update world state with radar scan results.

    Args:
        containers: List of containers from radar.
        mines: List of mines from radar.
    """
    ts = get_current_time_ms()
    _ws._pending_radar_cache_refresh_ms = 0
    _ws._pending_radar_empty_delta_ms = 0
    _ws.mark_radar_scan_complete()
    _ws.clear_failed_move_targets()
    reconcile_radar_viewport_resources(containers, mines)
    viewport = _ws._world_state["viewport"]
    if _ws.current_radar_uses_extra():
        _ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        _ws._world_state = mark_viewport_scanned(
            _ws._world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )

    for c in containers:
        _ws._world_state = update_container_from_radar(
            _ws._world_state,
            c["x"],
            c["y"],
            c["volume"],
            ts,
        )
    for m in mines:
        _ws._world_state = add_mine_from_radar(
            _ws._world_state,
            m["x"],
            m["y"],
            m["team"],
            ts,
        )


def containers_from_current_radar_cache() -> list[RadarContainerDict]:
    """Synthesize authoritative radar containers from current terrain cache.

    Returns:
        List of containers derived from terrain tile cache values within
        the current radar envelope.
    """
    left, top, right, bottom = _radar_bounds(_ws._world_state)
    containers: list[RadarContainerDict] = []
    for tile in _ws._world_state["terrain"].values():
        x = tile["x"]
        y = tile["y"]
        if not (left <= x <= right and top <= y <= bottom):
            continue
        cache_value = tile["cache_value"]
        if cache_value == -1:
            containers.append(RadarContainerDict(x=x, y=y, volume=-1))
        elif cache_value > 0:
            containers.append(RadarContainerDict(x=x, y=y, volume=cache_value))
    return containers


def update_world_state_from_radar_cache() -> None:
    """Promote a differential radar cache refresh to authoritative containers.

    Some radar scans refresh the current viewport through a combined tile/cache
    update plus ``RadarAck`` instead of an explicit ``radar_response`` packet.
    """
    ts = get_current_time_ms()
    containers = containers_from_current_radar_cache()
    _ws.mark_radar_scan_complete()
    _ws.clear_failed_move_targets()
    reconcile_radar_viewport_resources(containers, None)
    viewport = _ws._world_state["viewport"]
    if _ws.current_radar_uses_extra():
        _ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        _ws._world_state = mark_viewport_scanned(
            _ws._world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )
    for container in containers:
        _ws._world_state = update_container_from_radar(
            _ws._world_state,
            container["x"],
            container["y"],
            container["volume"],
            ts,
        )
    emit_world(
        "Radar cache refresh: promoted %d containers from combined tile update",
        len(containers),
    )


def update_world_state_from_radar_known_resources() -> None:
    """Confirm a zero-delta differential radar without discarding known resources.

    Preserves existing authoritative containers inside the radar bounds.
    """
    ts = get_current_time_ms()
    left, top, right, bottom = _radar_bounds(_ws._world_state)
    containers_by_key: dict[str, RadarContainerDict] = {}

    for container in _ws._world_state["containers"].values():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom:
            containers_by_key[coord_key(x, y)] = RadarContainerDict(
                x=x,
                y=y,
                volume=container["volume"],
            )

    radar_containers = list(containers_by_key.values())
    _ws.mark_radar_scan_complete()
    _ws.clear_failed_move_targets()
    viewport = _ws._world_state["viewport"]
    if _ws.current_radar_uses_extra():
        _ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        _ws._world_state = mark_viewport_scanned(
            _ws._world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )
    for rc in radar_containers:
        _ws._world_state = update_container_from_radar(
            _ws._world_state,
            rc["x"],
            rc["y"],
            rc["volume"],
            ts,
        )
    emit_world(
        "Radar differential refresh: preserved %d known containers in viewport",
        len(radar_containers),
    )


def clear_container_tile_cache(x: int, y: int) -> None:
    """Clear cached resource data for a tile without changing terrain.

    Args:
        x: Tile X coordinate.
        y: Tile Y coordinate.
    """
    key = coord_key(x, y)
    existing = _ws._world_state["terrain"].get(key)
    if existing is None:
        return
    new_terrain = dict(_ws._world_state["terrain"])
    new_terrain[key] = make_terrain_tile(
        x=x,
        y=y,
        terrain_type=existing["terrain_type"],
        cache_value=0,
        overlay_value=existing["overlay_value"],
    )
    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=new_terrain,
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        timestamp_ms=_ws._world_state["timestamp_ms"],
    )


def reconcile_radar_viewport_resources(
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict] | None,
) -> None:
    """Make current viewport resources match an authoritative radar scan.

    Any tracked container or mine inside the current viewport that is absent
    from the radar response is stale and must be removed.

    Args:
        containers: Containers returned by radar.
        mines: Mines returned by radar. ``None`` skips mine reconciliation.
    """
    left, top, right, bottom = _radar_bounds(_ws._world_state)
    radar_container_keys = {coord_key(item["x"], item["y"]) for item in containers}
    radar_mine_keys = (
        {coord_key(item["x"], item["y"]) for item in mines} if mines is not None else None
    )

    new_containers: dict[str, ContainerStateDict] | None = None
    for key, container in _ws._world_state["containers"].items():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom and key not in radar_container_keys:
            if new_containers is None:
                new_containers = dict(_ws._world_state["containers"])
            del new_containers[key]

    new_mines: dict[str, MineStateDict] | None = None
    if radar_mine_keys is not None:
        for key, mine in _ws._world_state["mines"].items():
            x = mine["x"]
            y = mine["y"]
            if left <= x <= right and top <= y <= bottom and key not in radar_mine_keys:
                if new_mines is None:
                    new_mines = dict(_ws._world_state["mines"])
                del new_mines[key]

    if new_containers is None and new_mines is None:
        return

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"] if new_containers is None else new_containers,
        mines=_ws._world_state["mines"] if new_mines is None else new_mines,
        terrain=_ws._world_state["terrain"],
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        timestamp_ms=_ws._world_state["timestamp_ms"],
    )


def _radar_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive radar coverage bounds for the current viewport.

    Args:
        world: Current world state.

    Returns:
        Inclusive ``(left, top, right, bottom)`` radar bounds.
    """
    return _ws._radar_bounds(world)


def handle_radar_ack(found: bool) -> None:
    """Process a RadarAck by promoting any pending differential refresh.

    A RadarAck can follow three kinds of radar result:

    1. A CombinedTileUpdate that refreshed the tile cache — promote
       the cached tiles to authoritative container truth.
    2. A zero-delta tunneled 0x4F that confirmed nothing changed — if
       ``found`` is True, preserve existing known resources; otherwise
       treat the viewport as empty.
    3. An explicit RadarResponse already processed — just mark
       the scan as complete.

    Args:
        found: Whether the RadarAck reports resources exist.
    """
    if _ws._consume_pending_radar_cache_refresh():
        update_world_state_from_radar_cache()
    elif _ws._consume_pending_radar_empty_delta():
        if found:
            if containers_from_current_radar_cache():
                update_world_state_from_radar_cache()
            else:
                update_world_state_from_radar_known_resources()
        else:
            update_world_state_from_radar([], [])
    else:
        _ws.mark_radar_scan_complete()


__all__ = [
    "clear_container_tile_cache",
    "containers_from_current_radar_cache",
    "handle_radar_ack",
    "reconcile_radar_viewport_resources",
    "update_world_state_from_radar",
    "update_world_state_from_radar_cache",
    "update_world_state_from_radar_known_resources",
]
