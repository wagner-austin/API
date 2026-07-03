"""Container and fuel state updates for world state.

Handles container pickup, fuel total updates, container removal,
failed pickup tracking, and tank registry container hints.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    pickup_container,
    set_self_fuel,
)

log = get_logger(__name__)


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


def update_world_state_from_container_pickup(
    ws: WorldService,
    x: int,
    y: int,
    remaining_volume: int = 0,
) -> None:
    """Update world state when a container pickup record arrives.

    The 0x43 wire format reports ``remaining_volume`` (fuel left in
    the container after this pickup). When emptied
    (``remaining_volume == 0``) the tile is removed from
    ``world.containers``; when fuel remains the container stays in
    state with its volume updated so the planner can drain the rest.

    Args:
        ws: World service instance.
        x: Container X coordinate.
        y: Container Y coordinate.
        remaining_volume: Fuel remaining in the container AFTER this
            pickup. Default ``0`` preserves the "emptied" semantic for
            callers without the wire field.
    """
    ts = get_current_time_ms()
    ws.world_state = pickup_container(ws.world_state, x, y, ts, remaining_volume)
    if remaining_volume <= 0:
        emit_world("Picked up container at (%d, %d)", x, y)
    else:
        emit_world(
            "Partial pickup at (%d, %d) -- %d fuel remains",
            x,
            y,
            remaining_volume,
        )


def remove_container_at(ws: WorldService, x: int, y: int) -> None:
    """Remove a container from world state at the given position.

    Used when the bot detects a container is unreachable (stuck timeout).

    Args:
        ws: World service instance.
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    from tankpit_bot.state import remove_container

    key = f"{x},{y}"
    if key not in ws.world_state["containers"]:
        return
    ws.world_state = remove_container(ws.world_state, x, y, ws.world_state["timestamp_ms"])
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
    "update_world_state_from_fuel_total",
]
