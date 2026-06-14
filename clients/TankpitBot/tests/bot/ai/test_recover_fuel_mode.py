"""Tests for the durable fuel recovery owner."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai import recover_fuel_mode as _rfm_module
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_fuel_mode import (
    can_use_fuel_radar,
    decide_recover_fuel_mode,
    minimum_recovery_fuel_volume,
    select_fuel_target,
    try_collect_critical_fuel,
    try_collect_fuel,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


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


def test_recover_fuel_mode_grabs_adjacent_equipment_opportunistically() -> None:
    """Fuel recovery picks up equipment it is standing next to.

    Regression guard for live run 20260610-011x: the bot walked past
    adjacent equipment containers because the mode only looked for fuel.
    """
    world, self_state = make_world(
        fuel=400,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
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

    assert decision["behavior"]["reason"] == "opportunistic_equipment"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 101
    assert decision["behavior"]["target_y"] == 100


def test_recover_fuel_mode_releases_lock_for_markedly_closer_fuel() -> None:
    """A locked far container yields to abundant nearby fuel.

    Regression guard for live run 20260610-011x: the bot walked across
    the map to a locked container while ignoring closer fuel the whole
    way.
    """
    world, self_state = make_world(
        fuel=400,
        containers={
            "107,107": make_container_state(
                x=107,
                y=107,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "160,160": make_container_state(
                x=160,
                y=160,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "RECOVER_FUEL",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "fuel",
            "resource_target_x": 160,
            "resource_target_y": 160,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel=900"
    assert decision["behavior"]["target_x"] == 107
    assert decision["behavior"]["target_y"] == 107
    assert decision["updated_ai_state"]["resource_target_x"] == 107


def test_recover_fuel_mode_keeps_lock_against_marginally_closer_fuel() -> None:
    """A candidate inside the anti-churn threshold does not break the lock."""
    world, self_state = make_world(
        fuel=400,
        containers={
            "104,104": make_container_state(
                x=104,
                y=104,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
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

    assert decision["behavior"]["reason"] == "fuel=700"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


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


def test_recover_fuel_mode_opens_map_when_no_recovery_action_is_legal() -> None:
    """The durable owner opens the map for intel when every route is blocked.

    Regression guard: raising here used to kill the bot process
    mid-game. With every viewport tile water and no radar, the owner's
    terminal action is the free map-intel decision.
    """
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
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_FUEL"
    assert decision["behavior"]["reason"] == "map_intel_for_fuel"
    assert decision["command"]["cmd_type"] == "map_open"


def test_select_fuel_target_returns_none_when_teleport_is_unaffordable() -> None:
    """Blocked fuel is rejected when the teleport fallback exceeds current fuel.

    The candidate filter accepts the container (a landing tile exists),
    but the planner cannot afford the teleport at fuel=0 -- the selector
    must surface that as "no executable target" instead of a command.
    """
    world, self_state = make_world(
        fuel=0,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    terrain_data: dict[tuple[int, int], str] = {(101, y): "#" for y in range(92, 109)}
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    assert select_fuel_target(ctx, allow_unreachable=True) is None


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
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
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


def test_can_use_fuel_radar_keeps_operating_reserve() -> None:
    """Fuel radar needs cost plus reserve; inventory never matters.

    Regression guard for live run 20260612-131003: 64 unreserved scans
    burned fuel to 7 and stranded the session for 28 minutes.
    """
    world, self_state = make_world(fuel=110, scanned=False)
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    assert can_use_fuel_radar(ctx) is True
    assert minimum_recovery_fuel_volume(ctx) == 1

    low_fuel_world, low_fuel_self_state = make_world(fuel=109, scanned=False)
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


def test_recover_fuel_mode_dot_refuel_outranks_visible_container() -> None:
    """A nearer fuel-dot teleport outranks a farther visible fuel container."""
    world, self_state = make_world(
        fuel=600,
        scanned=False,
        containers={
            "107,100": make_container_state(
                x=107,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    world["map_fuel_dots"] = {"104,100": 1}
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

    assert decision["behavior"]["reason"] == "fuel_dot_refuel"
    assert decision["command"]["cmd_type"] == "teleport"


def test_recover_fuel_mode_dot_walk_skips_blocked_dot() -> None:
    """Fuel dot walk skips dots whose movement is blocked by an enemy."""
    world, self_state = make_world(
        fuel=10,
        scanned=True,
        tanks={
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                damage_state=0,
                name="Blocker",
                is_bot=False,
                is_self=False,
                timestamp_ms=100000,
            ),
        },
    )
    world["map_fuel_dots"] = {"104,100": 1}
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

    assert decision["behavior"]["reason"] != "fuel_dot_walk"


def test_recover_fuel_mode_plans_fuel_dot_escape() -> None:
    """Marooned tank escapes via fuel dot teleport without operating reserve."""
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            if (x, y) != (100, 100):
                terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    world, self_state = make_world(fuel=50, scanned=True)
    world["map_fuel_dots"] = {"105,100": 1}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    decision = decide_recover_fuel_mode(ctx)

    assert decision["behavior"]["reason"] == "fuel_dot_escape"
    assert decision["command"]["cmd_type"] == "teleport"


def test_recover_fuel_mode_dot_walk_records_attempt_only_when_landing_on_dot() -> None:
    """Dot walk via a waypoint does NOT record an attempted-fuel-dot mark.

    When the walk planner clips a far dot to a near viewport-edge tile
    the command's target_x/y differs from the dot itself. The false
    branch at line 426 leaves ``attempted_fuel_dots`` unmodified.
    """
    # Put the dot far outside the viewport so walk_or_teleport returns
    # a waypoint closer to the player instead of the dot itself.
    world, self_state = make_world(fuel=10, scanned=True)
    world["map_fuel_dots"] = {"140,100": 1}  # far outside viewport (92..108)
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

    assert decision["behavior"]["reason"] == "fuel_dot_walk"
    # The walk lands on a waypoint, not the dot itself.
    # So attempted_fuel_dots should NOT have the dot marked.
    attempted = decision["updated_ai_state"]["attempted_fuel_dots"]
    assert "140,100" not in attempted


def test_recover_fuel_mode_dot_escape_skips_unaffordable_dot() -> None:
    """Marooned escape declines when the cheapest dot exceeds current fuel.

    Line 488: can_afford_teleport returns False for a very distant dot
    when fuel is extremely low. The escape function returns None and
    the owner falls through to the map-intel decision.
    """
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            if (x, y) != (100, 100):
                terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    world, self_state = make_world(fuel=1, scanned=True)
    # Put the dot very far away so teleport cost >> 1
    world["map_fuel_dots"] = {"250,250": 1}
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
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    decision = decide_recover_fuel_mode(ctx)

    # Cannot afford the escape teleport, falls to map_intel
    assert decision["behavior"]["reason"] == "map_intel_for_fuel"
    assert decision["command"]["cmd_type"] == "map_open"


def test_decide_recover_fuel_mode_raises_when_plan_returns_none() -> None:
    """Defensive ValueError fires when the internal planner returns None.

    This state is unreachable in normal gameplay (the map-intel fallback
    always produces a decision), but the guard exists to surface bugs in
    the planner. The test swaps ``_plan_fuel_recovery`` at module level
    with a stub that returns None, then restores it.
    """
    world, self_state = make_world(fuel=400)
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

    original = _rfm_module._plan_fuel_recovery

    def _always_none(ctx: DecideCtx, *, owner_required: bool) -> None:
        return None

    _rfm_module._plan_fuel_recovery = _always_none
    try:
        with pytest.raises(ValueError, match="RECOVER_FUEL owner failed"):
            decide_recover_fuel_mode(ctx)
    finally:
        _rfm_module._plan_fuel_recovery = original
