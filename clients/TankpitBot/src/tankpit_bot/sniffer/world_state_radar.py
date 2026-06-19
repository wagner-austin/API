"""Radar scan handling and resource reconciliation for world state.

Handles radar scan results, differential cache promotion, viewport scan
confirmation, and stale resource reconciliation within the radar envelope.
"""

from __future__ import annotations

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.world_service import WorldService
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
    ws: WorldService,
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict],
) -> None:
    """Update world state with radar scan results.

    Args:
        ws: World service instance.
        containers: List of containers from radar.
        mines: List of mines from radar.
    """
    ts = get_current_time_ms()
    ws.pending_radar_cache_refresh_ms = 0
    ws.pending_radar_empty_delta_ms = 0
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    reconcile_radar_viewport_resources(ws, containers, mines)
    viewport = ws.world_state["viewport"]
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        ws.world_state = mark_viewport_scanned(
            ws.world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )

    for c in containers:
        ws.world_state = update_container_from_radar(
            ws.world_state,
            c["x"],
            c["y"],
            c["volume"],
            ts,
            refresh_kind="radar_response",
        )
    for m in mines:
        ws.world_state = add_mine_from_radar(
            ws.world_state,
            m["x"],
            m["y"],
            m["team"],
            ts,
        )


def containers_from_current_radar_cache(ws: WorldService) -> list[RadarContainerDict]:
    """Synthesize authoritative radar containers from current terrain cache.

    Args:
        ws: World service instance.

    Returns:
        List of containers derived from terrain tile cache values within
        the current radar envelope.
    """
    left, top, right, bottom = ws.radar_bounds()
    containers: list[RadarContainerDict] = []
    for tile in ws.world_state["terrain"].values():
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


def update_world_state_from_radar_cache(ws: WorldService) -> None:
    """Promote a differential radar cache refresh to authoritative containers.

    Args:
        ws: World service instance.
    """
    ts = get_current_time_ms()
    containers = containers_from_current_radar_cache(ws)
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    reconcile_radar_viewport_resources(ws, containers, None)
    viewport = ws.world_state["viewport"]
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        ws.world_state = mark_viewport_scanned(
            ws.world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )
    for container in containers:
        ws.world_state = update_container_from_radar(
            ws.world_state,
            container["x"],
            container["y"],
            container["volume"],
            ts,
            refresh_kind="radar_cache_refresh",
        )
    emit_world(
        "Radar cache refresh: promoted %d containers from combined tile update",
        len(containers),
    )


def update_world_state_from_radar_known_resources(ws: WorldService) -> None:
    """Confirm a zero-delta differential radar without discarding known resources.

    Preserves existing authoritative containers inside the radar bounds.

    Args:
        ws: World service instance.
    """
    ts = get_current_time_ms()
    left, top, right, bottom = ws.radar_bounds()
    containers_by_key: dict[str, RadarContainerDict] = {}

    for container in ws.world_state["containers"].values():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom:
            containers_by_key[coord_key(x, y)] = RadarContainerDict(
                x=x,
                y=y,
                volume=container["volume"],
            )

    radar_containers = list(containers_by_key.values())
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    viewport = ws.world_state["viewport"]
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
        ws.world_state = mark_viewport_scanned(
            ws.world_state,
            viewport["left"],
            viewport["top"],
            ts,
        )
    for rc in radar_containers:
        ws.world_state = update_container_from_radar(
            ws.world_state,
            rc["x"],
            rc["y"],
            rc["volume"],
            ts,
            refresh_kind="radar_known_resources",
        )
    emit_world(
        "Radar differential refresh: preserved %d known containers in viewport",
        len(radar_containers),
    )


def clear_container_tile_cache(ws: WorldService, x: int, y: int) -> None:
    """Clear cached resource data for a tile without changing terrain.

    Args:
        ws: World service instance.
        x: Tile X coordinate.
        y: Tile Y coordinate.
    """
    key = coord_key(x, y)
    existing = ws.world_state["terrain"].get(key)
    if existing is None:
        return
    new_terrain = dict(ws.world_state["terrain"])
    new_terrain[key] = make_terrain_tile(
        x=x,
        y=y,
        terrain_type=existing["terrain_type"],
        cache_value=0,
        overlay_value=existing["overlay_value"],
    )
    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=new_terrain,
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        map_fuel_dots=ws.world_state["map_fuel_dots"],
        timestamp_ms=ws.world_state["timestamp_ms"],
    )


def reconcile_radar_viewport_resources(
    ws: WorldService,
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict] | None,
) -> None:
    """Make current viewport resources match an authoritative radar scan.

    Any tracked container or mine inside the current viewport that is absent
    from the radar response is stale and must be removed.

    Args:
        ws: World service instance.
        containers: Containers returned by radar.
        mines: Mines returned by radar. ``None`` skips mine reconciliation.
    """
    left, top, right, bottom = ws.radar_bounds()
    radar_container_keys = {coord_key(item["x"], item["y"]) for item in containers}
    radar_mine_keys = (
        {coord_key(item["x"], item["y"]) for item in mines} if mines is not None else None
    )

    new_containers: dict[str, ContainerStateDict] | None = None
    for key, container in ws.world_state["containers"].items():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom and key not in radar_container_keys:
            if new_containers is None:
                new_containers = dict(ws.world_state["containers"])
            del new_containers[key]

    new_mines: dict[str, MineStateDict] | None = None
    if radar_mine_keys is not None:
        for key, mine in ws.world_state["mines"].items():
            x = mine["x"]
            y = mine["y"]
            if left <= x <= right and top <= y <= bottom and key not in radar_mine_keys:
                if new_mines is None:
                    new_mines = dict(ws.world_state["mines"])
                del new_mines[key]

    if new_containers is None and new_mines is None:
        return

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"] if new_containers is None else new_containers,
        mines=ws.world_state["mines"] if new_mines is None else new_mines,
        terrain=ws.world_state["terrain"],
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        map_fuel_dots=ws.world_state["map_fuel_dots"],
        timestamp_ms=ws.world_state["timestamp_ms"],
    )


def handle_radar_ack(ws: WorldService, found: bool) -> None:
    """Process a RadarAck by promoting any pending differential refresh.

    Args:
        ws: World service instance.
        found: Whether the RadarAck reports resources exist.
    """
    if ws.consume_pending_radar_cache_refresh():
        update_world_state_from_radar_cache(ws)
    elif ws.consume_pending_radar_empty_delta():
        if found:
            if containers_from_current_radar_cache(ws):
                update_world_state_from_radar_cache(ws)
            else:
                update_world_state_from_radar_known_resources(ws)
        else:
            update_world_state_from_radar(ws, [], [])
    else:
        ws.mark_radar_scan_complete()


__all__ = [
    "clear_container_tile_cache",
    "containers_from_current_radar_cache",
    "handle_radar_ack",
    "reconcile_radar_viewport_resources",
    "update_world_state_from_radar",
    "update_world_state_from_radar_cache",
    "update_world_state_from_radar_known_resources",
]
