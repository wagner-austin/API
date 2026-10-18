"""October incident regressions through real collection and hop selection."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_hops import desperation_fuel_hop
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import select_and_pickup_fuel
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.escape_policy import escape_hop_permitted, escape_required
from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import ContainerStateDict, TankStateDict, make_container_state
from tankpit_bot.types.constants import TankLiveness
from tests.bot.ai._support import (
    make_enemy_tank,
    make_inventory,
    make_scanned_ai_state,
    make_world,
    seed_confirmed_incoming,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _fuel(x: int, y: int, volume: int = 580) -> ContainerStateDict:
    """Build a wire-fresh stock belief."""
    return make_container_state(
        x=x, y=y, is_fuel=True, volume=volume, timestamp_ms=100000, failed_pickups=0
    )


def _ctx(
    *,
    fuel: int = 687,
    hits: int = 2,
    latch: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
) -> DecideCtx:
    """Construct a real world service and latched retreat snapshot."""
    ws = WorldService()
    seed_confirmed_incoming(ws, hits, weapon=0, damage=-45)
    world, self_state = make_world(fuel=fuel, containers=containers, tanks=tanks)
    ws.world_state = world
    state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "break_escape_until_fuel": latch,
            "combat_target_id": 814,
            "combat_target_x": 100,
            "combat_target_y": 100,
        }
    )
    return DecideCtx(
        world,
        self_state,
        state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        CombatFeedback.NONE,
        ws=ws,
    )


@pytest.mark.parametrize("hits", [0, 1, 2, 3])
def test_committed_retreat_keeps_fuel_ahead_of_equipment(hits: int) -> None:
    """A break survives hit-window decay and never returns to equipment."""
    equipment = make_container_state(
        x=102, y=100, is_fuel=False, volume=0, timestamp_ms=100000, failed_pickups=0
    )
    ctx = _ctx(hits=hits, containers={"101,100": _fuel(101, 100), "102,100": equipment})
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["command"]["target_x"] == 101
    assert decision["updated_ai_state"]["combat_target_id"] == 814
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 800


def test_lone_hit_without_a_break_does_not_force_retreat() -> None:
    ctx = _ctx(hits=1, latch=0)
    assert escape_required(ctx, ctx.base) is False
    assert escape_hop_permitted(ctx, 101, 100) is True


@pytest.mark.parametrize("hits", [0, 2])
def test_mine_flip_cannot_bypass_clearance_and_safe_second_stock_wins(hits: int) -> None:
    ctx = _ctx(hits=hits, containers={"101,100": _fuel(101, 100), "140,100": _fuel(140, 100)})
    ctx.ws.last_own_mine_hit_ms = 99900
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140
    assert decision["command"]["target_y"] == 100
    assert decision["updated_ai_state"]["suppress_landing_scan"] is True
    assert decision["updated_ai_state"]["combat_target_id"] == 814


def test_validated_escape_leaves_before_scanning_abandoned_ground() -> None:
    ctx = _ctx(containers={"140,100": _fuel(140, 100)})
    ctx = ctx.derive(AIStateDict(**{**ctx.ai_state, "last_landing_scan_viewport": ""}))
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140


def test_walk_still_requires_landing_scan_on_unknown_ground() -> None:
    ctx = _ctx(containers={"101,100": _fuel(101, 100)})
    ctx.world["scanned_tiles"].clear()
    ctx = ctx.derive(AIStateDict(**{**ctx.ai_state, "last_landing_scan_viewport": ""}))
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 800


def test_actual_pickup_teleport_carries_harvest_landing_semantics() -> None:
    ctx = _ctx(containers={"101,100": _fuel(101, 100)}, latch=0, hits=0)
    ctx.ws.last_own_mine_hit_ms = 99900
    decision = select_and_pickup_fuel(ctx, ctx.base)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["updated_ai_state"]["suppress_landing_scan"] is True
    assert decision["updated_ai_state"]["resource_target_kind"] == "fuel"


@pytest.mark.parametrize("fuel,x,y", [(606, 194, 107), (85, 104, 113)])
def test_observed_low_margin_hops_cannot_spend_expected_pickup_early(
    fuel: int, x: int, y: int
) -> None:
    """Margins of 41 and 4 cannot fund a map/teleport under 45-fuel hits."""
    ctx = _ctx(fuel=fuel, containers={f"{x},{y}": _fuel(x, y, 1200)})
    assert escape_hop_permitted(ctx, x, y) is False
    assert desperation_fuel_hop(ctx, ctx.base) is None
    assert select_fuel_larder_hop(ctx)["container"] is None


def test_safe_larder_alternative_beats_higher_ratio_beside_second_attacker() -> None:
    # The damage fixture's attacker id is 60; the held enemy is 814.
    attacker = make_enemy_tank(tank_id=60, x=139, y=100)
    ctx = _ctx(
        containers={"140,100": _fuel(140, 100, 1000), "100,140": _fuel(100, 140)},
        tanks={"60": attacker},
    )
    selected = select_fuel_larder_hop(ctx)
    assert selected["landing_x"] == 100
    assert selected["landing_y"] == 140
    assert selected["container"] == _fuel(100, 140)


@pytest.mark.parametrize("state", ["dead", "silent", "unknown-position", "not-attacker"])
def test_irrelevant_tanks_do_not_bar_a_safe_landing(state: str) -> None:
    attacker = make_enemy_tank(tank_id=60, x=139, y=100)
    if state == "dead":
        attacker["liveness"] = TankLiveness.DEACTIVATED
    elif state == "silent":
        attacker["last_wire_seen_ms"] = 1
    elif state == "unknown-position":
        attacker["last_position_update_ms"] = 0
        attacker["x"] = 0
        attacker["y"] = 0
    else:
        attacker["tank_id"] = 61
    ctx = _ctx(tanks={"60": attacker}, fuel=1200, latch=1400)
    landing = (0, 0) if state == "unknown-position" else (140, 100)
    assert escape_hop_permitted(ctx, *landing) is True


def test_no_held_target_still_clears_confirmed_other_attacker() -> None:
    ctx = _ctx(tanks={"60": make_enemy_tank(tank_id=60, x=139, y=100)})
    ctx = ctx.derive(AIStateDict(**{**ctx.ai_state, "combat_target_id": -1}))
    assert escape_hop_permitted(ctx, 140, 100) is False
    assert escape_hop_permitted(ctx, 100, 140) is True


def test_diagonal_inside_the_viewport_is_not_an_escape() -> None:
    ctx = _ctx()
    assert escape_hop_permitted(ctx, 108, 108) is False
    assert escape_hop_permitted(ctx, 116, 108) is True


def test_retreat_walk_reveals_new_mines_before_reentering_the_route() -> None:
    ctx = _ctx(containers={"101,100": _fuel(101, 100)})
    ctx.ws.mark_mine_reveal_pending(99900)
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("retreat must produce the asserted command")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason_kind"] == "mine_hit_reveal_scan"


@pytest.mark.parametrize("attacker_x,expected_x", [(100, 140), (160, 101)])
def test_locked_mine_flip_uses_the_same_safety_and_landing_rules(
    attacker_x: int, expected_x: int
) -> None:
    ctx = _ctx(containers={"101,100": _fuel(101, 100), "140,100": _fuel(140, 100)})
    ctx.ws.last_own_mine_hit_ms = 99900
    state = set_resource_target(ctx.ai_state, "fuel", 101, 100)
    ctx = ctx.derive(AIStateDict(**{**state, "combat_target_x": attacker_x}))
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("expected the eligible harvest hop")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == expected_x
    assert decision["updated_ai_state"]["suppress_landing_scan"] is True


def test_completed_full_tank_plan_releases_before_search() -> None:
    ctx = _ctx(fuel=1200, hits=3, latch=0, containers={"101,100": _fuel(101, 100)})
    ctx = ctx.derive(set_resource_target(ctx.ai_state, "fuel", 101, 100))
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("expected the atlas request after releasing full-tank lock")
    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["resource_target_kind"] == ""


def test_dot_search_rejects_near_attacker_and_selects_safe_alternative() -> None:
    ctx = _ctx()
    ctx.map_fuel_dots = ((108, 108), (140, 140))
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("expected the safe dot hop")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140
    assert decision["command"]["target_y"] == 140


def test_unknown_ground_is_scanned_before_declaring_retreat_exhausted() -> None:
    ctx = _ctx(fuel=85)
    ctx.world["scanned_tiles"].clear()
    ctx = ctx.derive(
        AIStateDict(**{**ctx.ai_state, "last_map_open_ms": 99000, "last_landing_scan_viewport": ""})
    )
    decision = decide_collect_mode(ctx)
    if decision is None:
        raise AssertionError("expected sensing before an out-of-fuel conclusion")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"


def test_fuel_loss_during_map_open_invalidates_the_original_hop() -> None:
    before = _ctx(fuel=687)
    after = _ctx(fuel=606)
    assert escape_hop_permitted(before, 194, 107) is True
    assert escape_hop_permitted(after, 194, 107) is False
