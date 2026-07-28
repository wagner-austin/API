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


def test_marooned_exit_ignores_session_blacklisted_containers() -> None:
    """A session-blacklisted container never anchors the walk."""
    from tankpit_bot.bot.ai.collect_mode import (
        _blacklist_container,
        reset_container_blacklist,
    )

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
    reset_container_blacklist()
    _blacklist_container(130, 100)
    try:
        with pytest.raises(SessionExitError, match="out_of_fuel"):
            decide_collect_mode(ctx)
    finally:
        reset_container_blacklist()


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


def test_walk_skips_a_water_locked_nearer_candidate() -> None:
    """The run bot-20260728-092357 shape: nearest fuel is on water.

    The water-sitting container at (110,100) is closer but cannot be
    stood on; the walk falls through to the farther land dot instead
    of giving up.
    """
    containers = {
        "110,100": make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=200,
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
        InMemoryTerrainMap({(110, 100): "W"}),
        "",
        ((120, 115),),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected walk-for-fuel decision")
    assert decision["behavior"]["reason_kind"] == "walk_for_fuel"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 107


def test_healthy_tank_never_reaches_the_walk_rung() -> None:
    """Above the fuel-low break the affordable dot hop wins, never the walk."""
    decision = decide_collect_mode(_marooned_ctx(fuel=1100, map_fuel_dots=((130, 100),)))
    if decision is None:
        raise AssertionError("expected a dot-hop decision at healthy fuel")
    assert decision["behavior"]["reason_kind"] == "search_collect_local"
    assert decision["command"]["cmd_type"] == "teleport"
