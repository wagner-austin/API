"""Container and fuel state updates for world state.

Handles container pickup, fuel total updates, container removal,
failed pickup tracking, and tank registry container hints.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    pickup_container,
    replace_map_fuel_dots,
    set_self_fuel,
)

log = get_logger(__name__)


def update_world_state_from_fuel_dots(ws: WorldService, dots: list[tuple[int, int]]) -> None:
    """Replace the map-wide fuel-dot atlas from a parsed MAP_DATA dot layer.

    Args:
        ws: World service instance.
        dots: Decoded ``(x, y)`` world coordinates of every fuel dot.
    """
    ws.world_state = replace_map_fuel_dots(
        ws.world_state,
        dots,
        get_current_time_ms(),
    )


def update_world_state_from_tank_registry_container(
    container_y: int,
    container_viewport_x: int,
) -> None:
    """Ignore non-radar container registry hints for planning state.

    Tank registry container entries expose coarse location hints but do not
    provide trustworthy resource truth. Container planning is radar-driven, so
    these messages must not populate ``world["containers"]``.

    Args:
        container_y: Absolute Y coordinate.
        container_viewport_x: Viewport-relative X coordinate.
    """
    viewport_left = get_viewport_left()
    if viewport_left is None:
        log.info(
            "Ignoring tank_registry container: viewport_left not yet known (y=%d, vx=%d)",
            container_y,
            container_viewport_x,
        )
        return
    container_x = viewport_left + container_viewport_x
    log.debug(
        "Ignoring tank_registry container hint at (%d, %d); radar is authoritative",
        container_x,
        container_y,
    )


def update_world_state_from_fuel_total(ws: WorldService, fuel_total: int) -> None:
    """Update world state with new absolute fuel level.

    Args:
        ws: World service instance.
        fuel_total: New absolute fuel level.
    """
    ts = get_current_time_ms()
    old_fuel = (
        ws.world_state["self_state"]["fuel"] if ws.world_state["self_state"] is not None else 0
    )
    ws.world_state = set_self_fuel(ws.world_state, fuel_total, ts)
    delta = fuel_total - old_fuel
    emit_world("Fuel: %d -> %d (%+d)", old_fuel, fuel_total, delta)


def update_world_state_from_container_pickup(ws: WorldService, x: int, y: int) -> None:
    """Update world state when container is picked up.

    Args:
        ws: World service instance.
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    from tankpit_bot.sniffer.world_state_radar import clear_container_tile_cache

    ts = get_current_time_ms()
    ws.world_state = pickup_container(ws.world_state, x, y, ts)
    clear_container_tile_cache(ws, x, y)
    emit_world("Picked up container at (%d, %d)", x, y)


def remove_container_at(ws: WorldService, x: int, y: int) -> None:
    """Remove a container from world state at the given position.

    Used when the bot detects a container is unreachable (stuck timeout).
    Delegates the world-state edit to the central
    :func:`state.remove_container` mutator, then clears the radar tile
    cache so the planner cannot re-acquire the same tile.

    Args:
        ws: World service instance.
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    from tankpit_bot.sniffer.world_state_radar import clear_container_tile_cache
    from tankpit_bot.state import remove_container

    key = f"{x},{y}"
    if key not in ws.world_state["containers"]:
        return
    ws.world_state = remove_container(ws.world_state, x, y, ws.world_state["timestamp_ms"])
    clear_container_tile_cache(ws, x, y)
    log.info("Removed unreachable container at (%d, %d)", x, y)


def increment_container_failed_pickups(ws: WorldService, x: int, y: int) -> None:
    """Increment the failed_pickups counter on a container.

    Called when a pickup attempt stalls. The container stays in world
    state but is deprioritized by the planner. Delegates the world-state
    edit to the central
    :func:`state.container_mutations.increment_container_failed_pickups`
    mutator.

    Args:
        ws: World service instance.
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    from tankpit_bot.state.container_mutations import (
        increment_container_failed_pickups as _bump_failed_pickups,
    )

    container = ws.world_state["containers"].get(f"{x},{y}")
    if container is None:
        return
    previous_failed_pickups = container["failed_pickups"]
    ws.world_state = _bump_failed_pickups(ws.world_state, x, y)
    log.info(
        "Container (%d,%d) failed_pickups: %d -> %d",
        x,
        y,
        previous_failed_pickups,
        previous_failed_pickups + 1,
    )


__all__ = [
    "increment_container_failed_pickups",
    "remove_container_at",
    "update_world_state_from_container_pickup",
    "update_world_state_from_fuel_dots",
    "update_world_state_from_fuel_total",
    "update_world_state_from_tank_registry_container",
]
