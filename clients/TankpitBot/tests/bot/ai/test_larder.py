"""Tests for the larder fuel scorer."""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.state.types import ContainerStateDict
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_RANK = 2


def _never_blacklisted(x: int, y: int) -> bool:
    del x, y
    return False


def _ctx(
    *,
    fuel: int,
    containers: dict[str, ContainerStateDict],
    terrain: TerrainMapProtocol | None = None,
) -> DecideCtx:
    world, self_state = make_world(self_x=100, self_y=100, fuel=fuel, containers=containers)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain if terrain is not None else InMemoryTerrainMap(),
        "",
    )


def _fuel_at_deficit(deficit: int) -> int:
    return fuel_capacity(_RANK) - deficit


def test_big_deficit_prefers_the_rich_far_container() -> None:
    """Uncapped by the deficit, volume/cost picks the 600 at 20 tiles."""
    containers = {
        "120,100": make_container(120, 100, 600, is_fuel=True),
        "110,100": make_container(110, 100, 250, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    winner = selection["container"]
    assert winner == containers["120,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (120, 100)
    assert selection["cost"] == 120
    assert selection["candidates"] == 2


def test_small_deficit_prefers_the_near_container() -> None:
    """The deficit clamp flips the argmax to the closer 250."""
    containers = {
        "120,100": make_container(120, 100, 600, is_fuel=True),
        "110,100": make_container(110, 100, 250, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(150), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["110,100"]
    assert selection["cost"] == 60


def test_skips_equipment_empty_failed_blacklisted_and_adjacent() -> None:
    """Only live believed fuel beyond auto-pick reach enters scoring."""
    failed = make_container(130, 100, 400, is_fuel=True)
    failed["failed_pickups"] = 2
    containers = {
        "115,100": make_container(115, 100, 0, is_fuel=False),
        "116,100": make_container(116, 100, 0, is_fuel=True),
        "130,100": failed,
        "125,100": make_container(125, 100, 400, is_fuel=True),
        "101,101": make_container(101, 101, 400, is_fuel=True),
        "120,100": make_container(120, 100, 500, is_fuel=True),
    }

    def _blacklist_125(x: int, y: int) -> bool:
        return (x, y) == (125, 100)

    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_blacklist_125,
    )
    assert selection["container"] == containers["120,100"]
    assert selection["candidates"] == 2
    assert selection["too_close"] == 1


def test_water_locked_container_counts_no_landing() -> None:
    """A container with no passable tile in reach is skipped, tallied."""
    terrain = InMemoryTerrainMap(
        {
            (120, 100): "W",
            (121, 100): "W",
            (119, 100): "W",
            (120, 99): "W",
            (120, 101): "W",
        }
    )
    containers = {"120,100": make_container(120, 100, 500, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers, terrain=terrain),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["no_landing"] == 1


def test_shore_container_lands_on_the_cardinal_neighbor() -> None:
    """A water-sitting container is harvested from its shore tile."""
    terrain = InMemoryTerrainMap({(120, 100): "W"})
    containers = {"120,100": make_container(120, 100, 500, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers, terrain=terrain),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["120,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (121, 100)


def test_reserve_blocked_hop_is_declined() -> None:
    """A hop that would land below the fuel reserve never wins."""
    containers = {"120,100": make_container(120, 100, 900, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=250, containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["reserve_blocked"] == 1


def test_gain_below_cost_is_unprofitable() -> None:
    """A sliver container far away hands the tick to discovery."""
    containers = {"120,100": make_container(120, 100, 50, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["unprofitable"] == 1


def test_empty_registry_returns_no_candidates() -> None:
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers={}),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["candidates"] == 0
