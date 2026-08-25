"""Tests for the marooned walk-for-fuel last resort."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.maroon_walk import _maroon_pan_toward
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.protocol.commands import SCOPE_EAST
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state, make_viewport_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _marooned_ctx(
    *,
    fuel: int = 88,
    map_fuel_dots: tuple[tuple[int, int], ...] = (),
    with_blacklisted_sliver: bool = False,
) -> DecideCtx:
    """A broke tank in a scanned viewport with nothing hoppable.

    Mirrors run bot-20260728-091209: fuel below every teleport, the
    viewport fully covered so forage declines, the map recently
    opened so the dot-hop path cannot learn anything new.
    """
    ws = WorldService()
    containers = {}
    if with_blacklisted_sliver:
        sliver = make_container_state(
            x=101,
            y=100,
            is_fuel=True,
            volume=39,
            timestamp_ms=100000,
        )
        sliver["failed_pickups"] = 1
        containers["101,100"] = sliver
    world, self_state = make_world(fuel=fuel, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        map_fuel_dots,
        ws=ws,
    )


def test_marooned_tank_walks_toward_a_known_dot_instead_of_exiting() -> None:
    """The bot-20260728-091209 shape: broke, dots 20+ tiles out -> walk.

    The dot at (130,100) is 30 tiles away -- every teleport
    unaffordable at fuel 88 -- so the last resort walks the viewport
    leg toward it instead of raising ``out_of_fuel``.
    """
    decision = decide_collect_mode(_marooned_ctx(map_fuel_dots=((130, 100), (200, 200))))

    if decision is None:
        raise AssertionError("expected walk-for-fuel decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 100


def test_marooned_walk_targets_a_believed_container_too() -> None:
    """A remembered live container inside the cap also anchors the walk."""
    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=True,
            volume=400,
            timestamp_ms=100000,
        )
    }
    world, self_state = make_world(fuel=88, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected walk-for-fuel decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel"
    assert decision["command"]["cmd_type"] == "move"


def test_marooned_exit_stands_when_all_fuel_is_beyond_the_walk_cap() -> None:
    """Nothing within 48 tiles: the out_of_fuel exit is unchanged."""
    with pytest.raises(SessionExitError, match="no walkable fuel within 48 tiles"):
        decide_collect_mode(_marooned_ctx(map_fuel_dots=((200, 200),)))


def test_marooned_exit_ignores_blacklisted_containers() -> None:
    """A failed-pickup sliver never anchors the walk (run -091209)."""
    with pytest.raises(SessionExitError, match="out_of_fuel"):
        decide_collect_mode(_marooned_ctx(with_blacklisted_sliver=True))


def _edge_ctx(
    *,
    maroon_pan: tuple[int, int] | None = None,
    terrain: InMemoryTerrainMap | None = None,
    terrain_blind: bool = False,
) -> DecideCtx:
    """A broke tank ON the viewport's right edge, the only dot beyond it.

    The bot-20260825-133452 geometry: self at (100,100) on the window's
    east edge (85,92)-(100,107), the fuel dot at (130,100) outside it.
    Every walking leg clamps onto the tank's own tile, so the window is
    exhausted and only a free pan can make progress.
    """
    ws = WorldService()
    world, self_state = make_world(self_x=100, self_y=100, fuel=88, scanned=True)
    world["viewport"] = make_viewport_state(left=85, top=92, width=16, height=16)
    world["scanned_tiles"] = {f"{x},{y}": 100000 for y in range(92, 108) for x in range(85, 101)}
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(landing_scan_viewport="85,92"),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    if maroon_pan is not None:
        ai_state = AIStateDict(
            **{**ai_state, "maroon_pan_x": maroon_pan[0], "maroon_pan_y": maroon_pan[1]}
        )
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None if terrain_blind else (terrain if terrain is not None else InMemoryTerrainMap()),
        "",
        ((130, 100),),
        ws=ws,
    )


def test_marooned_edge_clamp_pans_the_window_toward_the_fuel() -> None:
    """A target clamping onto the tank's own tile buys a free pan.

    The run bot-20260825-133452 root: with autoscroll pinned OFF the
    window never moves on its own, and the pre-pan walker skipped any
    candidate whose leg clamped onto the tank — shuttling between clamp
    tiles for 331 s while fuel sat three tiles past the edge. The
    exhausted window now spends the free Rb pan instead: the anchor law
    reveals 15 fresh tiles toward the dot and the next leg walks them.
    """
    decision = decide_collect_mode(_edge_ctx())

    if decision is None:
        raise AssertionError("expected walk-for-fuel pan decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel_pan"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_EAST
    updated = decision["updated_ai_state"]
    assert updated["maroon_pan_x"] == 100
    assert updated["maroon_pan_y"] == 100


def test_maroon_pan_obeys_the_movement_law() -> None:
    """No second pan from the latched tile: the exit stands.

    A pan must pay for itself in movement — with the latch already at
    the tank's tile, the rung refuses to ping-pong the free window and
    the out_of_fuel exit takes over.
    """
    ctx = _edge_ctx(maroon_pan=(100, 100))

    with pytest.raises(SessionExitError, match="out_of_fuel"):
        decide_collect_mode(ctx)


def test_maroon_pan_allowed_after_movement_releases_the_latch() -> None:
    """A latch at a DIFFERENT tile does not bar the pan."""
    ctx = _edge_ctx(maroon_pan=(99, 100))
    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected walk-for-fuel pan decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel_pan"


def test_maroon_pan_refuses_a_known_impassable_post_pan_leg() -> None:
    """Post-pan clamp tile known impassable: no free action is wasted.

    The east pan would clamp the dot to (115,100); with that tile
    water, the terrain veto refuses the pan up front and the exit
    stands instead of proving on the wire what the map already knows.
    """
    ctx = _edge_ctx(terrain=InMemoryTerrainMap({(115, 100): "W"}))

    with pytest.raises(SessionExitError, match="out_of_fuel"):
        decide_collect_mode(ctx)


def test_maroon_pan_serves_a_terrain_blind_context() -> None:
    """Without a terrain map the veto cannot judge; the pan proceeds.

    The terrain veto is an optimization over KNOWN ground, never a
    requirement — a terrain-blind session still gets its recovery
    gait and lets the wire answer for the revealed tiles.
    """
    ctx = _edge_ctx(terrain_blind=True)
    decision = _maroon_pan_toward(ctx, ctx.ai_state, 130, 100)
    if decision is None:
        raise AssertionError("expected a pan decision")
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_EAST


def test_maroon_pan_helper_declines_the_tanks_own_tile_target() -> None:
    """The helper guards its own contract: target on self pans nowhere.

    The walker's distance filter never passes a zero-distance
    candidate, but the helper's refusal must not depend on caller
    filtering.
    """
    ctx = _edge_ctx()
    decision = _maroon_pan_toward(ctx, ctx.ai_state, 100, 100)
    assert decision is None


def test_desperation_hop_beats_a_long_walk_to_a_far_dot() -> None:
    """A shore container within teleport reach outranks a 35-tile walk.

    The water-sitting container at (110,100) cannot be walked to, but
    its shore landing (109,100) costs 54 of the tank's 88 -- one hop
    and an adjacent auto-pick beat twenty walking ticks to the dot.
    Volume 90 keeps this the DESPERATION rung's case: since the F16
    net-of-gain reserve, the ordinary larder takes rich (>= 100)
    containers even at desperation fuel; sub-floor dregs are the
    desperation hop's remaining niche.
    """
    ws = WorldService()
    containers = {
        "110,100": make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=90,
            timestamp_ms=100000,
        )
    }
    world, self_state = make_world(fuel=88, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    terrain = InMemoryTerrainMap(
        {
            (110, 100): "W",
            (111, 100): "W",
            (110, 99): "W",
            (110, 101): "W",
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        terrain,
        "",
        ((120, 115),),
        ws=ws,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected desperation hop decision")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 109
    assert decision["command"]["target_y"] == 100


def test_desperation_hop_crosses_a_water_channel_to_shore_fuel() -> None:
    """The bot-20260728-093011 shape: islet tank, fuel across the water.

    A water channel (column x=105) blocks every walk east, so the
    walk-pickup and walk rungs decline; the believed container at
    (108,100) sits ON water with a land landing at (109,100) costing
    54 of the tank's 68 -- the desperation hop crosses the channel
    and the adjacent auto-pick refuels, instead of exiting. Volume 90
    keeps this the desperation rung's case (rich containers now route
    through the larder at any fuel, F16).
    """
    ws = WorldService()
    containers = {
        "108,100": make_container_state(
            x=108,
            y=100,
            is_fuel=True,
            volume=90,
            timestamp_ms=100000,
        )
    }
    world, self_state = make_world(fuel=68, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    terrain_data: dict[tuple[int, int], str] = {(105, y): "W" for y in range(92, 108)}
    terrain_data[(108, 100)] = "W"
    terrain_data[(107, 100)] = "W"
    terrain_data[(108, 99)] = "W"
    terrain_data[(108, 101)] = "W"
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(terrain_data),
        "",
        ws=ws,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected desperation hop decision")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 109
    assert decision["command"]["target_y"] == 100
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "fuel"
    assert updated["suppress_landing_scan"] is True


def test_desperation_hop_declines_when_unaffordable_and_walk_takes_over() -> None:
    """A landing costing more than the tank holds falls through to walk."""
    ws = WorldService()
    containers = {
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=True,
            volume=300,
            timestamp_ms=100000,
        )
    }
    world, self_state = make_world(fuel=68, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ((130, 100),),
        ws=ws,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected walk-for-fuel fallback decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel"
    assert decision["command"]["cmd_type"] == "move"


def test_healthy_tank_never_reaches_the_walk_rung() -> None:
    """Above the fuel-low break the affordable dot hop wins, never the walk."""
    decision = decide_collect_mode(_marooned_ctx(fuel=1100, map_fuel_dots=((130, 100),)))
    if decision is None:
        raise AssertionError("expected a dot-hop decision at healthy fuel")
    assert decision["behavior"]["reason_kind"] == "search_collect_local"
    assert decision["command"]["cmd_type"] == "teleport"


def test_desperation_hop_picks_the_cheaper_of_two_dregs() -> None:
    """Two sub-floor candidates: the cheaper landing wins.

    Both containers are below the larder's 100 floor (desperation's
    niche since F16), so the desperation rung scores them and the
    nearer shore landing outranks the farther one.
    """
    ws = WorldService()
    containers = {
        "110,100": make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=90,
            timestamp_ms=100000,
        ),
        "100,112": make_container_state(
            x=100,
            y=112,
            is_fuel=True,
            volume=90,
            timestamp_ms=100000,
        ),
    }
    world, self_state = make_world(fuel=88, scanned=True, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
        }
    )
    terrain = InMemoryTerrainMap(
        {
            (110, 100): "W",
            (100, 112): "W",
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        terrain,
        "",
        ws=ws,
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected desperation hop decision")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 111
    assert decision["command"]["target_y"] == 100
