"""Viewport and tile patch updates for world state.

Handles ``0x5A`` viewport entity patches, ``0x43`` cache updates, ``0x40``
overlay updates, ``0x4A`` terrain updates, and waypoint resolution.
"""

from __future__ import annotations

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.viewport import update_viewport_origin
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    WorldStateDict,
    coord_key,
    make_terrain_tile,
    render_world_ascii,
)
from tankpit_bot.state.viewport_geometry import (
    make_visible_viewport_state,
    viewport_patch_world_coords,
)


def update_viewport_entities(
    ws: WorldService,
    viewport_left: int,
    viewport_top: int,
    entities: list[dict[str, int]],
) -> None:
    """Apply a visible viewport update using explicit viewport origin from 0x5A.

    Args:
        ws: World service instance.
        viewport_left: Absolute left edge of the visible 16x16 viewport.
        viewport_top: Absolute top edge of the visible 16x16 viewport.
        entities: Viewport entity dicts with col, row, cache_value, overlay_value, terrain_type.
    """
    update_viewport_origin(viewport_left, viewport_top)

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=ws.world_state["terrain"],
        viewport=make_visible_viewport_state(viewport_left, viewport_top),
        scanned_viewports=ws.world_state["scanned_viewports"],
        timestamp_ms=ws.world_state["timestamp_ms"],
    )

    update_viewport_tiles(ws, entities, viewport_left, viewport_top)
    ws.clear_failed_scan_viewport(viewport_left, viewport_top)


def update_viewport_tiles(
    ws: WorldService,
    entities: list[dict[str, int]],
    vp_left: int,
    vp_top: int,
) -> None:
    """Apply ``0x5A`` tile patches to viewport terrain and visual cache only.

    Args:
        ws: World service instance.
        entities: Viewport entity list.
        vp_left: Viewport left offset.
        vp_top: Viewport top offset.
    """
    ts = get_current_time_ms()
    new_terrain = dict(ws.world_state["terrain"])

    for ent in entities:
        abs_x, abs_y = viewport_patch_world_coords(
            vp_left,
            vp_top,
            ent["col"],
            ent["row"],
        )
        cache_value = ent["cache_value"]
        overlay_value = ent["overlay_value"]
        terrain_type = ent["terrain_type"]
        key = coord_key(abs_x, abs_y)
        new_terrain[key] = make_terrain_tile(
            x=abs_x,
            y=abs_y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=new_terrain,
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        timestamp_ms=ts,
    )


def update_cache_tiles(ws: WorldService, updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute cache-only tile updates to terrain/visual cache only.

    Args:
        ws: World service instance.
        updates: Absolute ``(x, y, cache_value)`` triples.
    """
    new_terrain = dict(ws.world_state["terrain"])
    timestamp_ms = get_current_time_ms()
    for x, y, cache_value in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        terrain_type = existing["terrain_type"] if existing is not None else 0
        overlay_value = existing["overlay_value"] if existing is not None else 255
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=new_terrain,
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def update_overlay_tiles(ws: WorldService, updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute overlay-only tile updates to world state.

    Args:
        ws: World service instance.
        updates: Absolute ``(x, y, overlay_value)`` triples.
    """
    new_terrain = dict(ws.world_state["terrain"])
    timestamp_ms = get_current_time_ms()
    for x, y, overlay_value in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        terrain_type = existing["terrain_type"] if existing is not None else 0
        cache_value = existing["cache_value"] if existing is not None else 0
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=new_terrain,
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def update_terrain_tiles(ws: WorldService, updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute terrain/structure tile updates to world state.

    Args:
        ws: World service instance.
        updates: Absolute ``(x, y, terrain_type)`` triples from protocol 0x4A.
    """
    new_terrain = dict(ws.world_state["terrain"])
    timestamp_ms = get_current_time_ms()

    for x, y, terrain_type in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        cache_value = existing["cache_value"] if existing is not None else 0
        overlay_value = existing["overlay_value"] if existing is not None else 255
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"],
        mines=ws.world_state["mines"],
        terrain=new_terrain,
        viewport=ws.world_state["viewport"],
        scanned_viewports=ws.world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def apply_waypoints(start_x: int, start_y: int, waypoints: str) -> tuple[int, int]:
    """Apply waypoints to calculate final position.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        waypoints: Path string (e.g., "wsss" = west, south, south, south).

    Returns:
        Tuple of (final_x, final_y) after applying all waypoints.
    """
    x, y = start_x, start_y
    for wp in waypoints:
        if wp == "n":
            y -= 1
        elif wp == "s":
            y += 1
        elif wp == "e":
            x += 1
        elif wp == "w":
            x -= 1
    return x, y


def render_ascii_if_available(ws: WorldService, event: str) -> None:
    """Render ASCII viewport if terrain map is available.

    Args:
        ws: World service instance.
        event: Event name for logging (e.g., "Enter", "Teleport", "Move").
    """
    ascii_view = render_world_state_ascii(ws)
    if ascii_view is not None:
        emit_world("[WorldState %s]\n%s", event, ascii_view)


def render_world_state_ascii(ws: WorldService) -> str | None:
    """Render an ASCII viewport from current world state.

    Args:
        ws: World service instance.

    Returns:
        ASCII representation string, or None if terrain map is not loaded.
    """
    terrain = ws.get_terrain_map()
    if terrain is None:
        return None
    return render_world_ascii(ws.world_state, terrain)


__all__ = [
    "apply_waypoints",
    "render_ascii_if_available",
    "render_world_state_ascii",
    "update_cache_tiles",
    "update_overlay_tiles",
    "update_terrain_tiles",
    "update_viewport_entities",
    "update_viewport_tiles",
]
