"""Tests for the larder cascade step: knowledge hops before discovery."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_service import WorldService
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
    ws = WorldService()
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
        ws=ws,
    )


def test_fuel_larder_hop_beats_discovery_and_holds_the_lock() -> None:
    """A remembered profitable fuel container wins the tick over forage.

    The viewport is UNSCANNED, so before the larder step existed this
    tick belonged to forage_radar -- the plan's ruling is that radar
    is spent only when knowledge is exhausted. The hop locks the
    container and suppresses the landing scan.

    Re-scoped 2026-08-06 ([[quad-sweep-doctrine]]): with extras
    stocked and the block virgin the quad sweep now leads (squeeze
    the current block dry before paying for a hop), so this fixture
    empties the extras -- the larder-beats-forage ordering it pins is
    unchanged below the sweep.
    """
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = _collect_ctx(
        fuel=700,
        scanned=False,
        containers={"140,100": _remembered_fuel(140, 100, 700)},
        inventory=inventory,
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


def test_virgin_block_sweeps_before_the_larder_hop() -> None:
    """With extras stocked and the block unscanned, the sweep leads.

    The [[quad-sweep-doctrine]] cycle (2026-08-06): recon the 31x31
    around the tank for four free-standing extras BEFORE paying
    teleport fuel for remembered stock -- the block may hold closer
    containers, and the exit hop comes only when it is squeezed dry.
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
    # The sweep opens by scanning the still-fresh CURRENT window, then
    # steers to the quadrants on later ticks.
    assert decision["behavior"]["reason_kind"] == "quad_sweep_radar"
    assert decision["command"]["cmd_type"] == "radar"
    updated = decision["updated_ai_state"]
    assert updated["sweep_anchor_x"] == 100
    assert updated["sweep_anchor_y"] == 100


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
    """A sliver container far away does not stop the tick from scanning.

    Re-pinned 2026-08-06: with extras stocked and the block virgin the
    scanning tick is now the quad sweep's first shift rather than a
    bare forage radar ([[quad-sweep-doctrine]]) -- the contract this
    test guards (the sliver does NOT win the tick) is unchanged.
    """
    ctx = _collect_ctx(
        fuel=400,
        scanned=False,
        containers={"140,100": _remembered_fuel(140, 100, 40)},
        inventory=make_inventory(),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_displaced_harvest_landing_unsuppresses_the_radar() -> None:
    """A shoved landing fires the scan and KEEPS the lock (flag s4-3).

    The harvest hop expected to stand within auto-pick reach of its
    locked container; landing 7 tiles away means the server displaced
    it off unobserved mines ([[flag-triage-20260729]] s4-3: three
    straight cant_go walks at 01:28). The radar reveals the field so
    the mine-composed passability can veto the doomed walks; the lock
    survives for the informed re-approach.
    """
    state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport="0,0"),
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "suppress_landing_scan": True,
            "resource_target_kind": "fuel",
            "resource_target_x": 95,
            "resource_target_y": 105,
        }
    )
    containers = {
        "95,105": make_container_state(
            x=95, y=105, is_fuel=True, volume=600, timestamp_ms=100000, failed_pickups=0
        ),
    }
    ctx = _collect_ctx(
        fuel=700,
        scanned=False,
        containers=containers,
        inventory=make_inventory(),
        ai_state=state,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"
    updated = decision["updated_ai_state"]
    assert updated["suppress_landing_scan"] is False
    assert updated["resource_target_kind"] == "fuel"
    assert updated["resource_target_x"] == 95
    assert updated["resource_target_y"] == 105
