"""Viewport and tile patch updates for world state.

Handles ``0x5A`` viewport entity patches, ``0x43`` cache updates, ``0x40``
overlay updates, ``0x4A`` terrain updates, and waypoint resolution.
"""

from __future__ import annotations

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.viewport import update_viewport_origin
from tankpit_bot.state import (
    WorldStateDict,
    coord_key,
    make_terrain_tile,
    render_world_ascii,
)
from tankpit_bot.state.viewport_geometry import (
    VIEWPORT_PATCH_WIDTH,
    make_visible_viewport_state,
    viewport_patch_world_coords,
)


def update_viewport_entities(
    viewport_left: int,
    viewport_top: int,
    entities: list[dict[str, int]],
) -> None:
    """Apply a visible viewport update using explicit viewport origin from 0x5A.

    Args:
        viewport_left: Absolute left edge of the visible 16x16 viewport.
        viewport_top: Absolute top edge of the visible 16x16 viewport.
        entities: Viewport entity dicts with col, row, cache_value, overlay_value, terrain_type.
    """
    update_viewport_origin(viewport_left, viewport_top)

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=_ws._world_state["terrain"],
        viewport=make_visible_viewport_state(viewport_left, viewport_top),
        scanned_viewports=_ws._world_state["scanned_viewports"],
        map_fuel_dots=_ws._world_state["map_fuel_dots"],
        timestamp_ms=_ws._world_state["timestamp_ms"],
    )

    update_viewport_tiles(entities, viewport_left, viewport_top)
    _ws.clear_failed_scan_viewport(viewport_left, viewport_top)


def update_viewport_tiles(
    entities: list[dict[str, int]],
    vp_left: int,
    vp_top: int,
) -> None:
    """Apply ``0x5A`` tile patches to viewport terrain and visual cache only.

    Client JS applies ``0x5A`` rows into per-tile cache, overlay, and terrain
    fields. For bot planning, however, resource truth is radar-driven: these
    passive viewport patches must not add/remove actionable fuel or equipment
    targets in ``world["containers"]``.

    Args:
        entities: Viewport entity list.
        vp_left: Viewport left offset.
        vp_top: Viewport top offset.
    """
    ts = get_current_time_ms()
    new_terrain = dict(_ws._world_state["terrain"])

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

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=new_terrain,
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        map_fuel_dots=_ws._world_state["map_fuel_dots"],
        timestamp_ms=ts,
    )


def update_cache_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute cache-only tile updates to terrain/visual cache only.

    Args:
        updates: Absolute ``(x, y, cache_value)`` triples.
    """
    new_terrain = dict(_ws._world_state["terrain"])
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

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=new_terrain,
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        map_fuel_dots=_ws._world_state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_overlay_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute overlay-only tile updates to world state.

    Args:
        updates: Absolute ``(x, y, overlay_value)`` triples.
    """
    new_terrain = dict(_ws._world_state["terrain"])
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

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=new_terrain,
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        map_fuel_dots=_ws._world_state["map_fuel_dots"],
        timestamp_ms=timestamp_ms,
    )


def update_terrain_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute terrain/structure tile updates to world state.

    Args:
        updates: Absolute ``(x, y, terrain_type)`` triples from protocol 0x4A.
    """
    new_terrain = dict(_ws._world_state["terrain"])
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

    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=_ws._world_state["containers"],
        mines=_ws._world_state["mines"],
        terrain=new_terrain,
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        map_fuel_dots=_ws._world_state["map_fuel_dots"],
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


def is_absolute_position(x: int, y: int) -> bool:
    """Check if position_update coordinates are absolute world coordinates.

    Position updates contain either:
    - Absolute world coordinates (after join/teleport): x,y >= 18
    - Viewport-relative coordinates (during movement): x,y < 18

    Args:
        x: X coordinate from position_update.
        y: Y coordinate from position_update.

    Returns:
        True if coordinates are absolute world coordinates.
    """
    return x >= VIEWPORT_PATCH_WIDTH or y >= VIEWPORT_PATCH_WIDTH


def render_ascii_if_available(event: str) -> None:
    """Render ASCII viewport if terrain map is available.

    Args:
        event: Event name for logging (e.g., "Enter", "Teleport", "Move").
    """
    ascii_view = render_world_state_ascii()
    if ascii_view is not None:
        emit_world("[WorldState %s]\n%s", event, ascii_view)


def render_world_state_ascii() -> str | None:
    """Render an ASCII viewport from current world state.

    Returns:
        ASCII representation string, or None if terrain map is not loaded.
    """
    terrain = _ws.get_terrain_map()
    if terrain is None:
        return None
    return render_world_ascii(_ws._world_state, terrain)


__all__ = [
    "apply_waypoints",
    "is_absolute_position",
    "render_ascii_if_available",
    "render_world_state_ascii",
    "update_cache_tiles",
    "update_overlay_tiles",
    "update_terrain_tiles",
    "update_viewport_entities",
    "update_viewport_tiles",
]
