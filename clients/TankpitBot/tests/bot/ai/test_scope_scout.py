"""The ferry scope scout — a free pan at water the larder cannot serve.

User doctrine ("we want ferries... technically we could just use a
viewport shift", [[viewport-shift-protocol]]): when a believed
container is water-locked and no fresh ferry is known, one free Rb
pan at its water beats paying for discovery.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.scope_scout import (
    SCOPE_SCOUT_COOLDOWN_MS,
    scope_direction_toward,
    scope_scout_for_ferry,
)
from tankpit_bot.bot.ai.types import AIStateDict
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
from tankpit_bot.state.types import ContainerStateDict, WorldStateDict
from tankpit_bot.state.types.terrain import make_terrain_tile
from tankpit_bot.types.constants import TERRAIN_FERRY
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_WATER = "W"
_REST_WINDOW = (92, 92, 107, 107)


def test_direction_covers_the_full_compass() -> None:
    """Each tank->goal sign pair maps to its clockwise compass byte."""
    cases = (
        ((100, 90), SCOPE_NORTH),
        ((110, 90), SCOPE_NORTHEAST),
        ((110, 100), SCOPE_EAST),
        ((110, 110), SCOPE_SOUTHEAST),
        ((100, 110), SCOPE_SOUTH),
        ((90, 110), SCOPE_SOUTHWEST),
        ((90, 100), SCOPE_WEST),
        ((90, 90), SCOPE_NORTHWEST),
    )
    for (goal_x, goal_y), expected in cases:
        assert scope_direction_toward(_REST_WINDOW, 100, 100, goal_x, goal_y) == expected


def test_direction_declines_goals_beyond_one_pan() -> None:
    """The anchor law reaches 15 tiles — 16 is out of a single pan."""
    assert scope_direction_toward(_REST_WINDOW, 100, 100, 115, 100) == SCOPE_EAST
    assert scope_direction_toward(_REST_WINDOW, 100, 100, 116, 100) is None


def test_direction_declines_the_tanks_own_tile() -> None:
    """No compass sign, no pan."""
    assert scope_direction_toward(_REST_WINDOW, 100, 100, 100, 100) is None


def test_direction_declines_a_pan_that_changes_nothing() -> None:
    """An east goal after an east pan re-derives the same window."""
    panned_east = (100, 92, 115, 107)
    assert scope_direction_toward(panned_east, 100, 100, 110, 100) is None


def _water_blob(cx: int, cy: int, radius: int = 2) -> dict[tuple[int, int], str]:
    """A water square around a tile, big enough to defeat any landing."""
    return {
        (cx + dx, cy + dy): _WATER
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    }


def _ctx(
    world: WorldStateDict,
    terrain: InMemoryTerrainMap | None,
    now_ms: int = 100000,
) -> DecideCtx:
    """Decision context at the standard (100,100) rest state."""
    self_state = world["self_state"]
    assert self_state is not None
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        now_ms,
        terrain,
        "",
    )


def _believe_ferry(world: WorldStateDict, x: int, y: int, observed_ms: int) -> None:
    """Plant a ferry terrain belief (a prior 0x5A patch's leaving)."""
    world["terrain"][f"{x},{y}"] = make_terrain_tile(
        x=x, y=y, terrain_type=TERRAIN_FERRY, observed_ms=observed_ms
    )


def _water_locked_world(
    goal_x: int = 110, goal_y: int = 100
) -> tuple[WorldStateDict, InMemoryTerrainMap]:
    """A believed fuel container floating in open water east of the tank."""
    containers: dict[str, ContainerStateDict] = {
        f"{goal_x},{goal_y}": make_container(goal_x, goal_y, 600, is_fuel=True),
    }
    world, _ = make_world(containers=containers)
    return world, InMemoryTerrainMap(terrain_data=_water_blob(goal_x, goal_y))


def test_water_locked_container_draws_a_pan_toward_its_water() -> None:
    """The scout pans east and latches the cooldown stamp."""
    world, terrain = _water_locked_world()
    base = make_scanned_ai_state()

    decision = scope_scout_for_ferry(_ctx(world, terrain), base)

    if decision is None:
        raise AssertionError("expected the scout to pan")
    assert decision["command"] == {"cmd_type": "scope_shift", "direction": SCOPE_EAST}
    assert decision["behavior"]["reason_kind"] == "ferry_scope_scout"
    assert decision["behavior"]["target_x"] == 110
    assert decision["behavior"]["target_y"] == 100
    assert decision["behavior"]["reason_context"] == {"direction": SCOPE_EAST}
    assert decision["updated_ai_state"]["last_scope_scout_ms"] == 100000


def test_nearest_water_locked_goal_wins_the_pan() -> None:
    """Two stuck goals: the pan aims at the closer water."""
    containers: dict[str, ContainerStateDict] = {
        "110,100": make_container(110, 100, 600, is_fuel=True),
        "100,113": make_container(100, 113, 600, is_fuel=True),
    }
    world, _ = make_world(containers=containers)
    terrain = InMemoryTerrainMap(terrain_data={**_water_blob(110, 100), **_water_blob(100, 113)})

    decision = scope_scout_for_ferry(_ctx(world, terrain), make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected the scout to pan")
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["behavior"]["target_x"] == 110
    assert decision["behavior"]["target_y"] == 100


def test_water_locked_equipment_also_draws_the_pan() -> None:
    """The equipment atlas feeds the scout exactly like the larder."""
    containers: dict[str, ContainerStateDict] = {
        "110,100": make_container(110, 100, -1, is_fuel=False),
    }
    world, _ = make_world(containers=containers)
    terrain = InMemoryTerrainMap(terrain_data=_water_blob(110, 100))

    decision = scope_scout_for_ferry(_ctx(world, terrain), make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected the scout to pan")
    assert decision["command"]["cmd_type"] == "scope_shift"


def test_held_combat_lock_bars_the_scout() -> None:
    """Mid-fight restocks never sightsee (the F21 discipline)."""
    world, terrain = _water_locked_world()
    base = AIStateDict(**{**make_scanned_ai_state(), "combat_target_id": 50})

    assert scope_scout_for_ferry(_ctx(world, terrain), base) is None


def test_cooldown_holds_the_scout_quiet() -> None:
    """A pan that revealed nothing must not re-fire every tick."""
    world, terrain = _water_locked_world()
    recent = 100000 - SCOPE_SCOUT_COOLDOWN_MS + 1
    base = AIStateDict(**{**make_scanned_ai_state(), "last_scope_scout_ms": recent})

    assert scope_scout_for_ferry(_ctx(world, terrain), base) is None


def test_unknown_terrain_bars_the_scout() -> None:
    """Without terrain no landing can be judged missing."""
    world, _ = _water_locked_world()

    assert scope_scout_for_ferry(_ctx(world, None), make_scanned_ai_state()) is None


def test_fresh_ferry_belief_needs_no_scout() -> None:
    """A boardable ferry means the larder already serves the goal."""
    world, terrain = _water_locked_world()
    _believe_ferry(world, 112, 100, observed_ms=90000)

    assert scope_scout_for_ferry(_ctx(world, terrain), make_scanned_ai_state()) is None


def test_old_ferry_belief_still_serves_and_needs_no_pan() -> None:
    """Ferry memory is positional, not clocked: an old sighting still serves.

    The no-drift law ([[ferry-mechanics]]) plus the three positional
    invalidation channels (0x4A move pairs, re-observation patches,
    displacement disproof) replaced the 60 s TTL on 2026-08-05 — a
    99-second-old sighting is a boarding tile, not a reason to spend
    a pan rediscovering what nothing has contradicted.
    """
    world, terrain = _water_locked_world()
    _believe_ferry(world, 112, 100, observed_ms=1000)

    assert scope_scout_for_ferry(_ctx(world, terrain), make_scanned_ai_state()) is None


def test_ground_served_container_never_draws_a_pan() -> None:
    """A container with a legal landing is the larder's, not the scout's."""
    containers: dict[str, ContainerStateDict] = {
        "110,100": make_container(110, 100, 600, is_fuel=True),
    }
    world, _ = make_world(containers=containers)

    assert scope_scout_for_ferry(_ctx(world, InMemoryTerrainMap()), make_scanned_ai_state()) is None


def test_unpannable_and_unqualified_goals_leave_no_scout() -> None:
    """Out-of-reach water, drained, and refused all decline."""
    drained = make_container(110, 102, 0, is_fuel=True)
    refused = make_container(110, 98, 500, is_fuel=True)
    refused["failed_pickups"] = 1
    containers: dict[str, ContainerStateDict] = {
        "130,100": make_container(130, 100, 600, is_fuel=True),
        "110,102": drained,
        "110,98": refused,
        "100,121": make_container(100, 121, 500, is_fuel=True),
    }
    world, _ = make_world(containers=containers)
    terrain = InMemoryTerrainMap(
        terrain_data={
            **_water_blob(130, 100),
            **_water_blob(110, 102),
            **_water_blob(110, 98),
            **_water_blob(100, 121),
        }
    )

    assert scope_scout_for_ferry(_ctx(world, terrain), make_scanned_ai_state()) is None


def test_collect_cascade_reaches_the_scout_after_the_larder_declines() -> None:
    """The pan rides the COLLECT cascade between larder and discovery.

    A water-locked believed container with no fresh ferry: pickups
    and the larder both decline, and the free pan is taken BEFORE
    any discovery teleport spends fuel. The scout beat derives the
    SEARCH substate — the tick is a look, not travel.
    """
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
    from tankpit_bot.bot.ai.mode_controller import derive_collect_mode_state

    containers: dict[str, ContainerStateDict] = {
        "110,100": make_container(110, 100, 600, is_fuel=True),
    }
    world, _ = make_world(fuel=700, containers=containers)
    terrain = InMemoryTerrainMap(terrain_data=_water_blob(110, 100))

    decision = decide_collect_mode(_ctx(world, terrain))

    if decision is None:
        raise AssertionError("expected the cascade to produce the scout decision")
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["behavior"]["reason_kind"] == "ferry_scope_scout"
    assert derive_collect_mode_state(decision) == "SEARCH"
