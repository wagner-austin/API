"""Block harvest and anchor-law framing ([[quad-sweep-doctrine]]);
recon pins live in ``test_quad_sweep``."""

from __future__ import annotations

from tankpit_bot.bot.ai.block_harvest import (
    BLOCK_REACH_TILES,
    anchored_window_origin,
    frame_direction,
    plan_block_harvest_leg,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.protocol.commands import (
    SCOPE_EAST,
    SCOPE_NORTH,
    SCOPE_NORTHEAST,
    SCOPE_NORTHWEST,
    SCOPE_SOUTH,
    SCOPE_SOUTHEAST,
    SCOPE_SOUTHWEST,
    SCOPE_WEST,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_sweep_ctx,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000


def test_frame_direction_compass_and_no_op() -> None:
    """The anchor-law compass covers all eight ways and refuses no-ops."""
    window = (92, 92, 107, 107)
    assert frame_direction(window, 100, 100, 120, 100) == SCOPE_EAST
    assert frame_direction(window, 100, 100, 80, 100) == SCOPE_WEST
    assert frame_direction(window, 100, 100, 100, 120) == SCOPE_SOUTH
    assert frame_direction(window, 100, 100, 100, 80) == SCOPE_NORTH
    assert frame_direction(window, 100, 100, 120, 80) == SCOPE_NORTHEAST
    assert frame_direction(window, 100, 100, 80, 80) == SCOPE_NORTHWEST
    assert frame_direction(window, 100, 100, 120, 120) == SCOPE_SOUTHEAST
    assert frame_direction(window, 100, 100, 80, 120) == SCOPE_SOUTHWEST
    # Goal on the tank's own tile: no direction exists.
    assert frame_direction(window, 100, 100, 100, 100) is None
    # Window already anchored east: the same shift changes nothing.
    assert frame_direction((100, 92, 115, 107), 100, 100, 112, 100) is None
    # Map-edge clamp: a westward shift from a tank at the west edge
    # cannot move a window already parked at origin zero.
    assert frame_direction((0, 92, 15, 107), 3, 100, 1, 100) is None


def test_anchored_window_origin_center_recenters() -> None:
    """Direction 8 (Scope Center) recenters like a teleport landing."""
    assert anchored_window_origin(50, 50, 100, 100, 8) == (92, 92)
    assert anchored_window_origin(50, 50, 3, 3, 8) == (0, 0)


def test_harvest_declines_without_terrain_and_at_low_fuel() -> None:
    """No terrain view or fuel at the break: harvest never fires."""
    ws = WorldService()
    equipment = make_container_state(
        x=120, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    no_terrain = DecideCtx(
        make_sweep_ctx(now_ms=_NOW, scanned=True, containers={"120,100": equipment}).world,
        make_sweep_ctx(now_ms=_NOW, scanned=True).self_state,
        make_scanned_ai_state(),
        make_inventory(),
        _NOW,
        None,
        "",
        ws=ws,
    )
    assert plan_block_harvest_leg(no_terrain, make_scanned_ai_state()) is None

    low = make_sweep_ctx(now_ms=_NOW, fuel=100, scanned=True, containers={"120,100": equipment})
    assert plan_block_harvest_leg(low, make_scanned_ai_state()) is None


def test_harvest_frames_the_nearest_block_container() -> None:
    """An out-of-window equipment container draws the framing shift."""
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"112,100": equipment},
        inventory=make_inventory(dual_count=3),
    )

    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "harvest_frame_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_EAST
    assert decision["behavior"]["target_x"] == 112
    assert decision["behavior"]["target_y"] == 100


def test_harvest_skips_a_move_failed_candidate() -> None:
    """A candidate the server refused movement to is not block stock.

    Flag s11-5 (run arterial 2026-08-13 23:39-23:40): the walk to
    equipment at (165,161) drew cant_go, the structural move-failed
    release correctly dropped the lock -- and the harvest candidate
    search re-latched the SAME tile as nearest block stock one tick
    later. Release -> re-latch -> release, one free scope shift per
    cycle, the window ping-ponging dir 3 <-> dir 7 at full fuel until
    the session ended. The candidate filter and the release rule must
    read the same marks.
    """
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"112,100": equipment},
        inventory=make_inventory(dual_count=3),
    )
    ctx.ws.mark_move_target_failed(112, 100, _NOW)

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_harvest_declines_while_a_lock_is_held() -> None:
    """A held resource lock is never overwritten by the harvest latch.

    Flag s11-5, second violation: while the (136,145) lock was HELD
    ("holding plan", merely not executable this tick), the harvest
    branch latched a different target over it -- an un-enumerated
    re-target the committed-intent design forbids. While a lock is
    held, its continuation owns the pursuit.
    """
    from tankpit_bot.bot.ai.context import set_resource_target

    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"112,100": equipment},
        inventory=make_inventory(dual_count=3),
    )
    held = set_resource_target(make_scanned_ai_state(), "equipment", 136, 145)

    assert plan_block_harvest_leg(ctx, held) is None


def test_harvest_walks_a_leg_when_the_window_is_already_anchored() -> None:
    """A far target with the window at its anchored limit draws a walk."""
    equipment = make_container_state(
        x=125, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"125,100": equipment},
        viewport_origin=(100, 92),
        inventory=make_inventory(dual_count=3),
    )

    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "harvest_leg_walk"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["behavior"]["target_x"] == 125
    assert decision["behavior"]["target_y"] == 100


def test_harvest_decisions_latch_the_target_as_a_lock() -> None:
    """Framing and walking both commit the target ([[committed-intent]]).

    The flag-4 oscillator (2026-08-13 20:50): an uncommitted harvest
    leg re-derived its target every tick, and because a frame shift
    changes the window that feeds the next derivation, two
    out-of-window containers on opposite sides ping-ponged the scope
    forever with zero movement. Latching the lock hands the pursuit
    to the lock-continuation step, whose releases are enumerated.
    """
    framed = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"112,100": framed},
        inventory=make_inventory(dual_count=3),
    )
    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())
    if decision is None:
        raise AssertionError("expected a decision")
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "equipment"
    assert updated["resource_target_x"] == 112
    assert updated["resource_target_y"] == 100

    walked = make_container_state(
        x=125, y=100, is_fuel=True, volume=700, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        fuel=600,
        scanned=True,
        containers={"125,100": walked},
        viewport_origin=(100, 92),
    )
    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())
    if decision is None:
        raise AssertionError("expected a decision")
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "fuel"
    assert updated["resource_target_x"] == 125
    assert updated["resource_target_y"] == 100


def test_harvest_skips_unqualified_containers() -> None:
    """In-window, over-reach, drained-fuel, and failed targets all skip."""
    in_window = make_container_state(
        x=105, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    beyond_reach = make_container_state(
        x=100 + BLOCK_REACH_TILES + 1,
        y=100,
        is_fuel=False,
        volume=0,
        timestamp_ms=_NOW,
        failed_pickups=0,
    )
    drained_fuel = make_container_state(
        x=115, y=104, is_fuel=True, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    failed = make_container_state(
        x=115, y=96, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=1
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={
            "105,100": in_window,
            f"{100 + BLOCK_REACH_TILES + 1},100": beyond_reach,
            "115,104": drained_fuel,
            "115,96": failed,
        },
        inventory=make_inventory(dual_count=3),
    )

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_harvest_skips_fuel_at_cap_and_slivers_not_worth_the_walk() -> None:
    """Fuel candidates respect the cap and the worth-the-walk rate."""
    sliver = make_container_state(
        x=125, y=100, is_fuel=True, volume=10, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, containers={"125,100": sliver})
    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None

    rich = make_container_state(
        x=112, y=100, is_fuel=True, volume=700, timestamp_ms=_NOW, failed_pickups=0
    )
    at_cap = make_sweep_ctx(now_ms=_NOW, fuel=1200, scanned=True, containers={"112,100": rich})
    assert plan_block_harvest_leg(at_cap, make_scanned_ai_state()) is None

    below_cap = make_sweep_ctx(now_ms=_NOW, fuel=600, scanned=True, containers={"112,100": rich})
    decision = plan_block_harvest_leg(below_cap, make_scanned_ai_state())
    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "harvest_frame_shift"


def test_harvest_skips_equipment_when_inventory_is_full() -> None:
    """A full inventory refuses equipment framing outright."""
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    full = make_inventory(default_count=40, dual_count=40)
    ctx = make_sweep_ctx(
        now_ms=_NOW, scanned=True, containers={"112,100": equipment}, inventory=full
    )

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_harvest_skips_block_unreachable_containers() -> None:
    """A container walled off by water never draws a shift or a leg."""
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    # Water everywhere except the tank's own tile: no path, no service.
    terrain = InMemoryTerrainMap.from_passable_set({(100, 100)})
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        scanned=True,
        containers={"112,100": equipment},
        terrain=terrain,
        inventory=make_inventory(dual_count=3),
    )

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_harvest_leg_skips_a_candidate_with_no_walk_plan() -> None:
    """A wanted candidate whose leg cannot be planned moves on.

    The candidate itself stays clean (a move-failed candidate is
    filtered upstream since flag s11-5): the window is already
    anchored southeast so no shift helps, the approach edge tile is
    move-failed so the walk arm yields nothing, and the diagonal
    reach makes the teleport fallback unaffordable (cost 254 at fuel
    210, above the 200 collect gate). The loop continues past the
    candidate instead of stalling.
    """
    equipment = make_container_state(
        x=130, y=130, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = make_sweep_ctx(
        now_ms=_NOW,
        fuel=210,
        scanned=True,
        containers={"130,130": equipment},
        viewport_origin=(100, 100),
        inventory=make_inventory(dual_count=3),
    )
    # The southeast corner clamp is the approach tile a diagonal
    # target faces; fail it so the walk arm yields nothing while the
    # candidate itself stays wanted.
    ctx.ws.mark_move_target_failed(115, 115, _NOW)
    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())

    assert decision is None
