"""Tests for the durable fuel recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_fuel_mode import (
    can_use_fuel_radar,
    decide_recover_fuel_mode,
    minimum_recovery_fuel_volume,
    try_collect_critical_fuel,
    try_collect_fuel,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.fakes import FakeTerrainMap


def test_recover_fuel_mode_continues_locked_fuel_target() -> None:
    """The durable owner keeps an executable locked fuel target."""
    world, self_state = make_world(
        fuel=400,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["command"]["cmd_type"] == "pickup_fuel"


def test_recover_fuel_mode_clears_combat_lock_when_fuel_mode_owns_tick() -> None:
    """Fuel recovery drops stale combat ownership when taking control."""
    world, self_state = make_world(
        fuel=400,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["updated_ai_state"]["combat_target_id"] == -1
    assert decision["updated_ai_state"]["combat_target_x"] == 0
    assert decision["updated_ai_state"]["combat_target_y"] == 0


def test_recover_fuel_mode_uses_radar_when_viewport_needs_authoritative_scan() -> None:
    """The durable owner senses before repositioning in an unscanned viewport."""
    world, self_state = make_world(fuel=400, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["behavior"]["reason"] == "radar_for_fuel"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_fuel_mode_uses_regular_radar_when_extra_charges_are_empty() -> None:
    """Fuel recovery still scans when extra radar stock is depleted."""
    world, self_state = make_world(fuel=400, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "radar_for_fuel"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_fuel_mode_raises_when_no_recovery_action_is_legal() -> None:
    """The durable owner fails explicitly when every recovery route is blocked."""
    world, self_state = make_world(fuel=140, scanned=True)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = FakeTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    with pytest.raises(ValueError, match="expected executable recovery action"):
        decide_recover_fuel_mode(ctx)


def test_try_collect_fuel_returns_none_when_fuel_is_healthy() -> None:
    """The non-owner helper does nothing when fuel is already healthy."""
    world, self_state = make_world(fuel=800)
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    decision = try_collect_fuel(ctx)

    assert decision is None


def test_try_collect_critical_fuel_returns_none_when_fuel_is_not_critical() -> None:
    """Critical fuel helper is a no-op outside the critical threshold."""
    world, self_state = make_world(fuel=800)
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    assert try_collect_critical_fuel(ctx) is None


def test_try_collect_critical_fuel_returns_recovery_decision_when_critical() -> None:
    """Critical fuel helper returns a concrete recovery action below the threshold."""
    world, self_state = make_world(
        fuel=40,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=True,
                volume=57,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    decision = try_collect_critical_fuel(ctx)

    if decision is None:
        raise ValueError("Expected critical fuel helper to produce a recovery decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=57"


def test_try_collect_critical_fuel_triggers_at_exact_threshold() -> None:
    """Critical fuel helper still triggers at the exact threshold boundary."""
    world, self_state = make_world(
        fuel=500,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=True,
                volume=57,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    decision = try_collect_critical_fuel(ctx)

    if decision is None:
        raise ValueError("Expected threshold-critical fuel to produce a recovery decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=57"


def test_selects_low_volume_fuel_when_critically_low() -> None:
    """Critical fuel recovery accepts small visible fuel containers."""
    world, self_state = make_world(
        fuel=40,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=True,
                volume=57,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    decision = try_collect_fuel(ctx)

    if decision is None:
        raise ValueError("Expected critical low-volume fuel to produce a recovery decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=57"


def test_try_collect_fuel_triggers_at_exact_low_threshold() -> None:
    """Low-fuel helper still triggers at the exact low threshold boundary."""
    world, self_state = make_world(
        fuel=500,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    decision = try_collect_fuel(ctx)

    if decision is None:
        raise ValueError("Expected threshold-low fuel to produce a recovery decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["reason"] == "fuel=700"


def test_recover_fuel_mode_skips_radar_when_fuel_too_low_to_pay_cost() -> None:
    """Fuel recovery uses repositioning when radar fuel cost is unaffordable."""
    world, self_state = make_world(fuel=5, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["reason"] == "edge_for_fuel"


def test_try_collect_fuel_returns_none_when_non_owner_paths_are_blocked() -> None:
    """Non-owner fuel helper returns None instead of raising when no path exists."""
    world, self_state = make_world(fuel=140, scanned=True)
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = FakeTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain, "")

    assert try_collect_fuel(ctx) is None


def test_recover_fuel_mode_approaches_known_off_viewport_fuel_before_edge_walk() -> None:
    """Known tracked fuel is pursued before generic exploration fallback."""
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=140,
        scanned=True,
        containers={
            "120,100": make_container_state(
                x=120,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), make_inventory(), 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "known_fuel=700"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 100


def test_can_use_fuel_radar_requires_only_fuel_budget() -> None:
    """Fuel radar legality depends on fuel cost, not extra-radar inventory."""
    world, self_state = make_world(fuel=10, scanned=False)
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    assert can_use_fuel_radar(ctx) is True
    assert minimum_recovery_fuel_volume(ctx) == 1

    low_fuel_world, low_fuel_self_state = make_world(fuel=9, scanned=False)
    low_fuel_ctx = DecideCtx(
        low_fuel_world,
        low_fuel_self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert can_use_fuel_radar(low_fuel_ctx) is False
    assert minimum_recovery_fuel_volume(low_fuel_ctx) == 1

    healthy_world, healthy_self_state = make_world(fuel=800, scanned=False)
    healthy_ctx = DecideCtx(
        healthy_world,
        healthy_self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert minimum_recovery_fuel_volume(healthy_ctx) == 100

    inventory["extra_radars"]["count"] = 0
    uncharged_ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    assert can_use_fuel_radar(uncharged_ctx) is True
