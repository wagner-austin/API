"""Tests for the mine-clearance shot planner."""

from __future__ import annotations

from tankpit_bot.bot.ai.mine_clearance import find_mine_clearance_shot
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


def _world_with_self(*, rank: int = 1) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world with the bot at (100,100) inside a matching viewport.

    Args:
        rank: The bot's rank (0 recruit, 1+ private and above).

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
        rank=rank,
        fuel=900,
        leaderboard_position=1,
    )
    world["self_state"] = self_state
    return world, self_state


def _add_covered_container(world: WorldStateDict, x: int, y: int, *, team: int = 1) -> None:
    """Place a container with a hostile mine on its own tile.

    Args:
        world: World to mutate.
        x: Container/mine X.
        y: Container/mine Y.
        team: Mine team (hostile to the team-2 bot by default).
    """
    world["containers"][f"{x},{y}"] = make_container_state(
        x=x,
        y=y,
        is_fuel=False,
        volume=0,
    )
    world["mines"][f"{x},{y}"] = make_mine_state(
        x=x,
        y=y,
        mine_type=0,
        tank_id=-1,
        team=team,
    )


def test_single_covered_container_with_clear_line_is_the_aim() -> None:
    """One covered container in view with open ground gets the shot."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) == (104, 100)


def test_no_covered_containers_returns_none() -> None:
    """Bare containers and bare mines are not clearance targets."""
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(x=104, y=100, is_fuel=True, volume=500)
    world["mines"]["106,100"] = make_mine_state(x=106, y=100, mine_type=0, tank_id=-1, team=1)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) is None


def test_friendly_mine_on_container_is_not_a_target() -> None:
    """Own-team mines are passable and need no clearance shot."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100, team=2)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) is None


def test_rock_in_the_shot_line_disqualifies_the_aim() -> None:
    """Mine shots never arc over mountains — the blocked aim is skipped."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100)
    terrain = InMemoryTerrainMap({(102, 100): InMemoryTerrainMap.ROCK})

    assert find_mine_clearance_shot(world, self_state, terrain) is None


def test_intermediate_mines_do_not_occlude_the_shot() -> None:
    """ "We can shoot over other mines of course" — a mined lane stays clear."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100)
    world["mines"]["102,100"] = make_mine_state(x=102, y=100, mine_type=0, tank_id=-1, team=1)
    world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) == (104, 100)


def test_out_of_viewport_covered_container_is_skipped() -> None:
    """The server rejects out-of-view aims, so off-view cover waits."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 130, 100)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) is None


def test_private_prefers_the_aim_exposing_the_most_containers() -> None:
    """A 3x3 blast that uncovers a cluster beats a nearer lone cover.

    User law (flag s3-14): "1 single shot can clear liek 9 mines.
    which may unlock multiple equipment containers."
    """
    world, self_state = _world_with_self(rank=1)
    _add_covered_container(world, 102, 100)
    _add_covered_container(world, 106, 104)
    _add_covered_container(world, 107, 104)
    _add_covered_container(world, 106, 105)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) == (106, 104)


def test_recruit_scores_single_tile_blast_and_takes_the_nearest() -> None:
    """A recruit's shot clears one mine, so cluster bonuses vanish."""
    world, self_state = _world_with_self(rank=0)
    _add_covered_container(world, 102, 100)
    _add_covered_container(world, 106, 104)
    _add_covered_container(world, 107, 104)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) == (102, 100)


def test_blast_clips_at_the_map_edge() -> None:
    """A corner aim only counts in-bounds blast tiles."""
    world, self_state = _world_with_self()
    world["viewport"] = make_viewport_state(left=0, top=0, width=16, height=16)
    self_state["x"], self_state["y"] = 4, 4
    _add_covered_container(world, 0, 0)

    assert find_mine_clearance_shot(world, self_state, InMemoryTerrainMap()) == (0, 0)
