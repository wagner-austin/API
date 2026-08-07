"""Tests for the marooned walk-for-fuel last resort."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
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


def test_marooned_walk_declines_when_the_leg_is_the_current_tile() -> None:
    """A target clamping onto the tank's own tile produces no walk.

    The tank sits on the viewport's right edge (the frame recenters
    only at edges, so self-on-edge is a real geometry): the eastward
    dot clamps onto the tank's own tile, no leg exists, and the exit
    stands.
    """
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
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ((130, 100),),
    )

    with pytest.raises(SessionExitError, match="out_of_fuel"):
        decide_collect_mode(ctx)


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
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected desperation hop decision")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 111
    assert decision["command"]["target_y"] == 100
