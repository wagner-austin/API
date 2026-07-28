"""Tests for the larder cascade step: knowledge hops before discovery."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.state.types import ContainerStateDict, make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _remembered_fuel(x: int, y: int, volume: int) -> ContainerStateDict:
    return make_container_state(
        x=x,
        y=y,
        is_fuel=True,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _collect_ctx(
    *,
    fuel: int,
    containers: dict[str, ContainerStateDict],
    inventory: InventoryState,
    scanned: bool = True,
    ai_state: AIStateDict | None = None,
) -> DecideCtx:
    world, self_state = make_world(fuel=fuel, scanned=scanned, containers=containers)
    state = (
        ai_state
        if ai_state is not None
        else AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
    )
    return DecideCtx(
        world,
        self_state,
        state,
        inventory,
        100000,
        InMemoryTerrainMap(),
        "",
    )


def test_fuel_larder_hop_beats_discovery_and_holds_the_lock() -> None:
    """A remembered profitable fuel container wins the tick over forage.

    The viewport is UNSCANNED, so before the larder step existed this
    tick belonged to forage_radar -- the plan's ruling is that radar
    is spent only when knowledge is exhausted. The hop locks the
    container and suppresses the landing scan.
    """
    ctx = _collect_ctx(
        fuel=700,
        scanned=False,
        containers={"140,100": _remembered_fuel(140, 100, 700)},
        inventory=make_inventory(),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["behavior"]["reason_context"]["volume"] == 700
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140
    assert decision["command"]["target_y"] == 100
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "fuel"
    assert updated["resource_target_x"] == 140
    assert updated["resource_target_y"] == 100
    assert updated["suppress_landing_scan"] is True


def test_equipment_hop_holds_the_lock_and_suppresses_the_scan() -> None:
    """The equipment larder hop locks its container for the landing pickup."""
    equipment = make_container_state(
        x=140,
        y=100,
        is_fuel=False,
        volume=0,
        timestamp_ms=100000,
        failed_pickups=0,
    )
    ctx = _collect_ctx(
        fuel=1000,
        containers={"140,100": equipment},
        inventory=make_inventory(dual_count=3),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "equipment"
    assert updated["resource_target_x"] == 140
    assert updated["resource_target_y"] == 100
    assert updated["suppress_landing_scan"] is True


def test_larder_landing_latches_the_viewport_without_a_radar() -> None:
    """The suppress flag consumes the landing scan and is cleared.

    The tank lands in a fresh viewport (latch differs) with the larder
    flag set: instead of the unconditional scan_on_landing radar, the
    origin is latched silently and the cascade proceeds -- here to
    forage (discovery is permitted AFTER the larder is exhausted).
    """
    state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport="0,0"),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "suppress_landing_scan": True,
        }
    )
    ctx = _collect_ctx(
        fuel=400,
        scanned=False,
        containers={},
        inventory=make_inventory(),
        ai_state=state,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "scan_on_landing"
    updated = decision["updated_ai_state"]
    assert updated["suppress_landing_scan"] is False
    assert updated["last_landing_scan_viewport"] == "92,92"


def test_fresh_viewport_without_the_flag_still_radars_on_landing() -> None:
    """Non-larder landings keep the unconditional 2026-07-03 scan."""
    state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport="0,0"),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    ctx = _collect_ctx(
        fuel=400,
        scanned=False,
        containers={},
        inventory=make_inventory(),
        ai_state=state,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_unprofitable_larder_hands_the_tick_to_discovery() -> None:
    """A sliver container far away does not stop forage from scanning."""
    ctx = _collect_ctx(
        fuel=400,
        scanned=False,
        containers={"140,100": _remembered_fuel(140, 100, 40)},
        inventory=make_inventory(),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"
