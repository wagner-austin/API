"""The walk-corridor clearance arm and clamped-gain pricing.

The 2026-08-13 additions to the mine-clearance planner (HUD flags 3,
4, 6): corridor clearance for COLLECT walks, and fuel worth priced by
the CLAMPED transfer instead of the container's volume.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.mine_clearance import (
    find_mine_clearance_shot,
    find_walk_clearance_shot,
)
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import (
    make_container_state,
    make_mine_state,
    make_viewport_state,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world_with_self() -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world with the bot at (100,100) inside a matching viewport.

    Returns:
        World state and the bot's self state.
    """
    world = make_empty_world_state()
    world["viewport"] = make_viewport_state(left=92, top=92, width=16, height=16)
    self_state = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    world["self_state"] = self_state
    return world, self_state


def _add_corridor_blocked_equipment(world: WorldStateDict) -> None:
    """Place equipment east of the bot with a hostile mine on the corridor.

    The flag-3 geometry in miniature: equipment at (106,100), hostile
    mine at (103,100) on the straight walk line.
    """
    world["containers"]["106,100"] = make_container_state(x=106, y=100, is_fuel=False, volume=0)
    world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)


def test_corridor_mine_to_wanted_equipment_draws_the_walk_shot() -> None:
    """The flag-3 fix: shoot the path mine instead of teleporting.

    Run bot-20260813-195231 shot the mines COVERING containers but
    never the ones blocking its walk, then paid a 42-fuel equipment
    hop onto ground a free single plus a short walk served.
    """
    world, self_state = _world_with_self()
    _add_corridor_blocked_equipment(world)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=True,
        fuel_deficit=200,
        fuel_gain_per_walk_tile=3,
    )

    assert aim == (103, 100)


def test_unwanted_equipment_draws_no_walk_shot() -> None:
    """A full inventory never buys a corridor shot for equipment."""
    world, self_state = _world_with_self()
    _add_corridor_blocked_equipment(world)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=False,
        fuel_deficit=200,
        fuel_gain_per_walk_tile=3,
    )

    assert aim is None


def test_near_cap_fuel_draws_no_walk_shot() -> None:
    """The flag-4 regime: a sliver-gain drink never buys a shot.

    An 852-volume container at deficit 18 transfers 18 -- below both
    the dreg floor and the walk pricing -- so no corridor shot fires
    for it, however rich the container looks.
    """
    world, self_state = _world_with_self()
    world["containers"]["106,100"] = make_container_state(x=106, y=100, is_fuel=True, volume=852)
    world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=True,
        fuel_deficit=18,
        fuel_gain_per_walk_tile=3,
    )

    assert aim is None


def test_clear_corridor_draws_no_walk_shot() -> None:
    """Wanted stock with a mine-free straight corridor needs no shot."""
    world, self_state = _world_with_self()
    world["containers"]["106,100"] = make_container_state(x=106, y=100, is_fuel=False, volume=0)
    world["mines"]["106,104"] = make_mine_state(x=106, y=104, mine_type=0, tank_id=-1, team=1)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=True,
        fuel_deficit=200,
        fuel_gain_per_walk_tile=3,
    )

    assert aim is None


def test_own_tile_container_is_never_a_walk_target() -> None:
    """The container under the tank needs no corridor and no shot."""
    world, self_state = _world_with_self()
    world["containers"]["100,100"] = make_container_state(x=100, y=100, is_fuel=False, volume=0)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=True,
        fuel_deficit=200,
        fuel_gain_per_walk_tile=3,
    )

    assert aim is None


def test_nearest_wanted_container_owns_the_corridor_choice() -> None:
    """Two corked containers: the nearer one's corridor mine is the aim."""
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(x=104, y=100, is_fuel=False, volume=0)
    world["mines"]["102,100"] = make_mine_state(x=102, y=100, mine_type=0, tank_id=-1, team=1)
    world["containers"]["100,107"] = make_container_state(x=100, y=107, is_fuel=False, volume=0)
    world["mines"]["100,104"] = make_mine_state(x=100, y=104, mine_type=0, tank_id=-1, team=1)

    aim = find_walk_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        equipment_wanted=True,
        fuel_deficit=200,
        fuel_gain_per_walk_tile=3,
    )

    assert aim == (102, 100)


def test_covered_shot_prices_fuel_by_clamped_gain() -> None:
    """The flag-4 fix on the covered arm: gain, not volume, pays.

    The same 852-volume covered fuel container draws the shot at a
    real deficit and is skipped at deficit 18 -- the run that exposed
    this spent two shots un-covering containers it then refused every
    tick as "clamped gain 18 not worth the walk".
    """
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(x=104, y=100, is_fuel=True, volume=852)
    world["mines"]["104,100"] = make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1)

    thirsty = find_mine_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        fuel_deficit=700,
        fuel_gain_per_walk_tile=3,
    )
    assert thirsty == (104, 100)

    near_cap = find_mine_clearance_shot(
        world,
        self_state,
        InMemoryTerrainMap(),
        fuel_deficit=18,
        fuel_gain_per_walk_tile=3,
    )
    assert near_cap is None
