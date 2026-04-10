"""Tests for the durable equipment recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_equipment_mode import (
    decide_recover_equipment_mode,
    select_equipment_target,
    try_search_critical_equipment,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.fakes import FakeTerrainMap


def test_recover_equipment_mode_raises_when_search_contract_is_impossible() -> None:
    """The durable owner raises when no recovery action can legally be produced.

    This is a defensive invariant test for the durable owner contract. The
    production path should not hit this branch under normal thresholds because
    fuel recovery would own the tick first, but a deliberately inconsistent
    config must still fail explicitly instead of returning ``None``.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    with pytest.raises(ValueError, match="expected executable recovery search action"):
        decide_recover_equipment_mode(ctx)


def test_try_search_critical_equipment_raises_when_search_contract_is_impossible() -> None:
    """The emergency search helper raises when no legal search action exists.

    This validates the helper-level contract separately from the durable owner.
    Under the deliberately inconsistent config below, emergency equipment search
    still owns the tick but cannot radar or teleport, so it must fail
    explicitly.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    with pytest.raises(ValueError, match="expected executable recovery search action"):
        try_search_critical_equipment(ctx)


def test_try_search_critical_equipment_returns_none_when_not_in_emergency() -> None:
    """Emergency search helper is a no-op when reserves are not broken."""
    world, self_state = make_world(fuel=800, scanned=True)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )

    assert try_search_critical_equipment(ctx) is None


def test_try_search_critical_equipment_returns_radar_when_scan_is_needed() -> None:
    """Emergency search helper senses the current viewport before teleport search."""
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 5
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected radar search decision")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason"] == "radar_for_equipment"


def test_try_search_critical_equipment_uses_regular_radar_when_extra_is_empty() -> None:
    """Emergency equipment search still scans with the built-in radar."""
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected regular-radar search decision")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason"] == "radar_for_equipment"


def test_select_equipment_target_returns_none_when_teleport_is_unaffordable() -> None:
    """Blocked equipment is rejected when teleport fallback exceeds current fuel."""
    world, self_state = make_world(
        fuel=0,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    terrain_data: dict[tuple[int, int], str] = {(101, y): "#" for y in range(92, 109)}
    terrain = FakeTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    assert select_equipment_target(ctx, allow_unreachable=True) is None


def test_recover_equipment_mode_approaches_known_off_viewport_equipment_before_search() -> None:
    """Known tracked equipment is pursued before generic search hops."""
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=800,
        scanned=True,
        containers={
            "120,100": make_container_state(
                x=120,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 12
    inventory["homing_shots"]["count"] = 12
    inventory["extra_radars"]["count"] = 12
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["reason"] == "known_equipment"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 100
