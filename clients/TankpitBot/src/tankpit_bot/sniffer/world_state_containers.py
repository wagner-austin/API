"""Container and fuel state updates for world state.

Handles container pickup, fuel total updates, container removal,
failed pickup tracking, and tank registry container hints.
"""

from __future__ import annotations

from platform_core.logging import get_logger

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.sniffer.world_state_radar import clear_container_tile_cache
from tankpit_bot.state import (
    WorldStateDict,
    make_container_state,
    pickup_container,
    set_self_fuel,
)

log = get_logger(__name__)


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


def update_world_state_from_fuel_total(fuel_total: int) -> None:
    """Update world state with new absolute fuel level.

    Args:
        fuel_total: New absolute fuel level.
    """
    ts = get_current_time_ms()
    old_fuel = (
        _ws._world_state["self_state"]["fuel"] if _ws._world_state["self_state"] is not None else 0
    )
    _ws._world_state = set_self_fuel(_ws._world_state, fuel_total, ts)
    delta = fuel_total - old_fuel
    emit_world("Fuel: %d -> %d (%+d)", old_fuel, fuel_total, delta)


def update_world_state_from_container_pickup(x: int, y: int) -> None:
    """Update world state when container is picked up.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    ts = get_current_time_ms()
    _ws._world_state = pickup_container(_ws._world_state, x, y, ts)
    clear_container_tile_cache(x, y)
    emit_world("Picked up container at (%d, %d)", x, y)


def remove_container_at(x: int, y: int) -> None:
    """Remove a container from world state at the given position.

    Used when the bot detects a container is unreachable (stuck timeout).

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    key = f"{x},{y}"
    if key in _ws._world_state["containers"]:
        new_containers = dict(_ws._world_state["containers"])
        del new_containers[key]
        _ws._world_state = WorldStateDict(
            self_state=_ws._world_state["self_state"],
            tanks=_ws._world_state["tanks"],
            containers=new_containers,
            mines=_ws._world_state["mines"],
            terrain=_ws._world_state["terrain"],
            viewport=_ws._world_state["viewport"],
            scanned_viewports=_ws._world_state["scanned_viewports"],
            timestamp_ms=_ws._world_state["timestamp_ms"],
        )
        clear_container_tile_cache(x, y)
        log.info("Removed unreachable container at (%d, %d)", x, y)


def increment_container_failed_pickups(x: int, y: int) -> None:
    """Increment the failed_pickups counter on a container.

    Called when a pickup attempt stalls. The container stays in world
    state but is deprioritized by the planner.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    key = f"{x},{y}"
    container = _ws._world_state["containers"].get(key)
    if container is None:
        return
    new_container = make_container_state(
        x=container["x"],
        y=container["y"],
        is_fuel=container["is_fuel"],
        volume=container["volume"],
        timestamp_ms=container["timestamp_ms"],
        failed_pickups=container["failed_pickups"] + 1,
    )
    new_containers = dict(_ws._world_state["containers"])
    new_containers[key] = new_container
    _ws._world_state = WorldStateDict(
        self_state=_ws._world_state["self_state"],
        tanks=_ws._world_state["tanks"],
        containers=new_containers,
        mines=_ws._world_state["mines"],
        terrain=_ws._world_state["terrain"],
        viewport=_ws._world_state["viewport"],
        scanned_viewports=_ws._world_state["scanned_viewports"],
        timestamp_ms=_ws._world_state["timestamp_ms"],
    )
    log.info(
        "Container (%d,%d) failed_pickups: %d -> %d",
        x,
        y,
        container["failed_pickups"],
        new_container["failed_pickups"],
    )


__all__ = [
    "increment_container_failed_pickups",
    "remove_container_at",
    "update_world_state_from_container_pickup",
    "update_world_state_from_fuel_total",
    "update_world_state_from_tank_registry_container",
]
