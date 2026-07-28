"""Container and fuel state updates for world state.

Handles container pickup, fuel total updates, container removal,
failed pickup tracking, and tank registry container hints.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.facts.source import FactSource
from tankpit_bot.ledger.damage_book import confirm_incoming_damage
from tankpit_bot.ledger.fuel_book import record_fuel_entry, record_fuel_reading
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.runtime_logging import emit_diagnostic, emit_world
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    pickup_container,
    set_self_fuel,
)

log = get_logger(__name__)


def update_world_state_from_fuel_total(
    ws: WorldService,
    fuel_total: int,
    fact_source: FactSource = "wire_0x2E_tank_status_sync",
) -> None:
    """Update world state with new absolute fuel level.

    Args:
        ws: World service instance.
        fuel_total: New absolute fuel level.
        fact_source: Wire channel the fuel total arrived on (0x2E sync,
            0x44 fuel gain, or 0x64 fuel total).
    """
    ts = get_current_time_ms()
    old_fuel = (
        ws.world_state["self_state"]["fuel"] if ws.world_state["self_state"] is not None else 0
    )
    ws.world_state = set_self_fuel(ws.world_state, fuel_total, ts, fact_source)
    delta = fuel_total - old_fuel
    emit_world("Fuel: %d -> %d (%+d)", old_fuel, fuel_total, delta)
    if fact_source != "wire_0x2E_tank_status_sync" and ws.fuel_book["last_fuel"] is not None:
        # 0x44 gains and 0x64 deposit totals ANNOUNCE their own delta:
        # the wire message is the explanation, so the book credits it
        # exactly before folding the reading in (2026-07-21 soak 2:
        # every positive-residual divergence was an unentered gain).
        announced = fuel_total - ws.fuel_book["last_fuel"]
        record_fuel_entry(book=ws.fuel_book, kind="pickup", lo=announced, hi=announced)
    confirm_incoming_damage(ws.damage_book, delta, ts)
    verdict = record_fuel_reading(book=ws.fuel_book, fuel_total=fuel_total)
    if verdict is not None and not verdict["balanced"]:
        emit_diagnostic(
            diagnostic_kind="physics_divergence",
            residual=verdict["residual"],
            feasible_lo=verdict["lo"],
            feasible_hi=verdict["hi"],
            entry_kinds=verdict["entry_kinds"],
            fact_source=fact_source,
        )


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
    rank = ws.world_state["self_state"]["rank"] if ws.world_state["self_state"] is not None else 8
    record_fuel_entry(book=ws.fuel_book, kind="pickup", lo=0, hi=fuel_capacity(rank))
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
