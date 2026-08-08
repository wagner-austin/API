"""Quad-sweep recon and block harvest ([[quad-sweep-doctrine]])."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.quad_sweep import (
    BLOCK_REACH_TILES,
    frame_direction,
    plan_block_harvest_leg,
    plan_quad_sweep,
    quadrant_bounds,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.inventory import InventoryState
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
from tankpit_bot.state.types import ContainerStateDict, make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000


def _sweep_ctx(
    *,
    fuel: int = 900,
    scanned: bool = False,
    block_scanned: bool | None = None,
    containers: dict[str, ContainerStateDict] | None = None,
    inventory: InventoryState | None = None,
    ai_state: AIStateDict | None = None,
    terrain: InMemoryTerrainMap | None = None,
    viewport_origin: tuple[int, int] | None = None,
    self_x: int = 100,
    self_y: int = 100,
) -> DecideCtx:
    ws = WorldService()
    world, self_state = make_world(
        self_x=self_x,
        self_y=self_y,
        fuel=fuel,
        scanned=scanned,
        block_scanned=block_scanned,
        containers=containers,
    )
    if viewport_origin is not None:
        world["viewport"]["left"] = viewport_origin[0]
        world["viewport"]["top"] = viewport_origin[1]
    return DecideCtx(
        world,
        self_state,
        ai_state if ai_state is not None else make_scanned_ai_state(),
        inventory if inventory is not None else make_inventory(),
        _NOW,
        terrain if terrain is not None else InMemoryTerrainMap(),
        "",
        ws=ws,
    )


def _anchored_ai_state(x: int, y: int) -> AIStateDict:
    return AIStateDict(**{**make_scanned_ai_state(), "sweep_anchor_x": x, "sweep_anchor_y": y})


def _cover_window(world_tiles: dict[str, int], left: int, top: int) -> None:
    for y in range(top, top + 16):
        for x in range(left, left + 16):
            world_tiles[f"{x},{y}"] = _NOW


def test_quadrant_bounds_follow_the_anchor_law_and_clamp() -> None:
    """Quadrant origins are tank-anchored and map-clamped like the server's."""
    assert quadrant_bounds(100, 100, -15, -15) == (85, 85, 100, 100)
    assert quadrant_bounds(100, 100, 0, 0) == (100, 100, 115, 115)
    # A corner anchor clamps both quadrant origins onto the same window.
    assert quadrant_bounds(3, 3, -15, -15) == (0, 0, 15, 15)
    assert quadrant_bounds(252, 252, 0, 0) == (240, 240, 255, 255)


def test_sweep_declines_without_extras_and_at_low_fuel() -> None:
    """No extras or fuel at the low break: recon never wins the tick."""
    empty = make_inventory()
    empty["extra_radars"]["count"] = 0
    assert plan_quad_sweep(_sweep_ctx(inventory=empty), make_scanned_ai_state()) is None

    low_fuel = _sweep_ctx(fuel=100)
    assert plan_quad_sweep(low_fuel, make_scanned_ai_state()) is None


def test_sweep_declines_on_a_covered_block() -> None:
    """A block below the start floor never begins a sweep."""
    ctx = _sweep_ctx(scanned=True)
    assert plan_quad_sweep(ctx, make_scanned_ai_state()) is None


def test_virgin_block_opens_with_a_radar_on_the_fresh_window() -> None:
    """The sweep starts by scanning the still-fresh current window."""
    ctx = _sweep_ctx(scanned=False)

    decision = plan_quad_sweep(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_radar"
    assert decision["command"]["cmd_type"] == "radar"
    updated = decision["updated_ai_state"]
    assert updated["sweep_anchor_x"] == 100
    assert updated["sweep_anchor_y"] == 100


def test_covered_window_steers_toward_the_first_pending_quadrant() -> None:
    """With the current window spent, the sweep shifts NW first."""
    ctx = _sweep_ctx(scanned=True, block_scanned=False)

    decision = plan_quad_sweep(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_NORTHWEST
    assert decision["behavior"]["reason_context"]["direction"] == SCOPE_NORTHWEST


def test_framed_quadrant_fires_its_radar() -> None:
    """A window parked exactly on a pending quadrant draws the radar."""
    ctx = _sweep_ctx(scanned=True, block_scanned=False, viewport_origin=(85, 85))
    ctx.world["scanned_tiles"].clear()
    _cover_window(ctx.world["scanned_tiles"], 92, 92)

    decision = plan_quad_sweep(ctx, _anchored_ai_state(100, 100))

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_radar"
    assert decision["behavior"]["reason_context"]["direction"] == SCOPE_NORTHWEST


def test_moved_tank_abandons_the_sweep_until_the_block_is_fresh() -> None:
    """A stale anchor in a mostly-covered block never resumes the sweep.

    Same coverage as the continuation test below (one quadrant still
    fresh, block under the start floor) -- but with the anchor
    latched to a tile the tank no longer stands on, the sweep is
    abandoned rather than continued.
    """
    ctx = _sweep_ctx(scanned=True, block_scanned=False)
    for y in range(85, 116):
        for x in range(85, 116):
            if not (x >= 100 and y >= 100):
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    decision = plan_quad_sweep(ctx, _anchored_ai_state(90, 100))

    assert decision is None


def test_anchored_sweep_continues_below_the_start_floor() -> None:
    """Standing on the anchor, per-quadrant economics alone continue it."""
    ctx = _sweep_ctx(scanned=True, block_scanned=False)
    # Cover most of the block so the START floor would refuse, leaving
    # one quadrant's worth of fresh ground.
    for y in range(85, 116):
        for x in range(85, 116):
            if not (x >= 100 and y >= 100):
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    decision = plan_quad_sweep(ctx, _anchored_ai_state(100, 100))

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_SOUTHEAST


def test_last_extra_needs_the_reserve_floor() -> None:
    """At the reserve, a quadrant must clear the bigger reveal floor."""
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = _sweep_ctx(scanned=True, block_scanned=False, inventory=inventory)
    # Leave under 128 uncovered tiles per quadrant: cover all but a
    # 7-row strip of the block.
    for y in range(85, 116):
        for x in range(85, 116):
            if y >= 92:
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    assert plan_quad_sweep(ctx, _anchored_ai_state(100, 100)) is None


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


def test_harvest_declines_without_terrain_and_at_low_fuel() -> None:
    """No terrain view or fuel at the break: harvest never fires."""
    ws = WorldService()
    equipment = make_container_state(
        x=120, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    no_terrain = DecideCtx(
        _sweep_ctx(scanned=True, containers={"120,100": equipment}).world,
        _sweep_ctx(scanned=True).self_state,
        make_scanned_ai_state(),
        make_inventory(),
        _NOW,
        None,
        "",
        ws=ws,
    )
    assert plan_block_harvest_leg(no_terrain, make_scanned_ai_state()) is None

    low = _sweep_ctx(fuel=100, scanned=True, containers={"120,100": equipment})
    assert plan_block_harvest_leg(low, make_scanned_ai_state()) is None


def test_harvest_frames_the_nearest_block_container() -> None:
    """An out-of-window equipment container draws the framing shift."""
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = _sweep_ctx(
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


def test_harvest_walks_a_leg_when_the_window_is_already_anchored() -> None:
    """A far target with the window at its anchored limit draws a walk."""
    equipment = make_container_state(
        x=125, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = _sweep_ctx(
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
    ctx = _sweep_ctx(
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
    ctx = _sweep_ctx(scanned=True, containers={"125,100": sliver})
    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None

    rich = make_container_state(
        x=112, y=100, is_fuel=True, volume=700, timestamp_ms=_NOW, failed_pickups=0
    )
    at_cap = _sweep_ctx(fuel=1200, scanned=True, containers={"112,100": rich})
    assert plan_block_harvest_leg(at_cap, make_scanned_ai_state()) is None

    below_cap = _sweep_ctx(fuel=600, scanned=True, containers={"112,100": rich})
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
    ctx = _sweep_ctx(scanned=True, containers={"112,100": equipment}, inventory=full)

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_harvest_skips_block_unreachable_containers() -> None:
    """A container walled off by water never draws a shift or a leg."""
    equipment = make_container_state(
        x=112, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    # Water everywhere except the tank's own tile: no path, no service.
    terrain = InMemoryTerrainMap.from_passable_set({(100, 100)})
    ctx = _sweep_ctx(scanned=True, containers={"112,100": equipment}, terrain=terrain)

    assert plan_block_harvest_leg(ctx, make_scanned_ai_state()) is None


def test_anchor_math_direct_edges() -> None:
    """Direct anchor-law coverage: center recenters, zero extras refuse."""
    from tankpit_bot.bot.ai.quad_sweep import (
        _quadrant_spend_worthwhile,
        anchored_window_origin,
    )

    # Direction 8 (Scope Center) recenters like a teleport landing.
    assert anchored_window_origin(50, 50, 100, 100, 8) == (92, 92)
    assert anchored_window_origin(50, 50, 3, 3, 8) == (0, 0)
    # The sweep is an extras strategy: zero extras never qualify, no
    # matter how much ground is uncovered.
    assert _quadrant_spend_worthwhile(961, 0) is False


def test_harvest_leg_skips_a_failed_move_target() -> None:
    """A leg whose walk plan fails moves on instead of stalling."""

    equipment = make_container_state(
        x=125, y=100, is_fuel=False, volume=0, timestamp_ms=_NOW, failed_pickups=0
    )
    ctx = _sweep_ctx(
        scanned=True,
        containers={"125,100": equipment},
        viewport_origin=(100, 92),
        inventory=make_inventory(dual_count=3),
    )
    # The approach edge tile the leg would walk to is the window's
    # east column clamp (115,100); fail it AND the real target so the
    # movement layer's approach planner yields nothing.
    ctx.ws.mark_move_target_failed(125, 100, _NOW)
    ctx.ws.mark_move_target_failed(115, 100, _NOW)
    decision = plan_block_harvest_leg(ctx, make_scanned_ai_state())

    assert decision is None
