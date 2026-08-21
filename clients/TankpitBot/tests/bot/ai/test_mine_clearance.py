"""Tests for the mine-clearance shot planner."""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.ferry import FerryAwareTerrain
from tankpit_bot.bot.ai.mine_clearance import (
    find_mine_clearance_shot,
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


def _find_shot(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    *,
    fuel_deficit: int = 1100,
) -> tuple[int, int] | None:
    """Call the planner with a full-tank-of-headroom deficit by default.

    The wide default keeps the access-geometry tests reading about
    geometry; the gain-pricing tests pass their own deficit.

    Args:
        world: World under test.
        self_state: Bot state under test.
        terrain: Terrain view under test.
        fuel_deficit: Fuel headroom to price clamped transfers with.

    Returns:
        The planner's aim.
    """
    return find_mine_clearance_shot(
        world,
        self_state,
        terrain,
        fuel_deficit=fuel_deficit,
        fuel_gain_per_walk_tile=3,
    )


def _composed(base: InMemoryTerrainMap, hostile_keys: frozenset[str]) -> FerryAwareTerrain:
    """Compose the decision view exactly as production does.

    Args:
        base: Static terrain data.
        hostile_keys: Team-scoped hostile-mine "x,y" keys.

    Returns:
        The composed terrain view.
    """
    return FerryAwareTerrain(
        base,
        {},
        riding=False,
        hostile_mine_keys=hostile_keys,
        occupied_tank_keys=frozenset(),
        refused_landing_keys=frozenset(),
    )


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

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (104, 100)


def test_no_covered_containers_returns_none() -> None:
    """Bare containers and bare mines are not clearance targets."""
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(x=104, y=100, is_fuel=True, volume=500)
    world["mines"]["106,100"] = make_mine_state(x=106, y=100, mine_type=0, tank_id=-1, team=1)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) is None


def test_friendly_mine_on_container_is_not_a_target() -> None:
    """Own-team mines are passable and need no clearance shot."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100, team=2)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) is None


def test_rock_in_the_shot_line_disqualifies_the_aim() -> None:
    """Mine shots never arc over mountains — the blocked aim is skipped."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100)
    terrain = InMemoryTerrainMap({(102, 100): InMemoryTerrainMap.ROCK})

    assert _find_shot(world, self_state, terrain) is None


def test_intermediate_mines_do_not_occlude_the_shot() -> None:
    """ "We can shoot over other mines of course" — a mined lane stays clear.

    The winning aim is (103,100) — the mined service tile nearest the
    bot whose blast still covers the container — and its shot line
    crosses the mine at (102,100), which never occludes.
    """
    world, self_state = _world_with_self()
    _add_covered_container(world, 104, 100)
    world["mines"]["102,100"] = make_mine_state(x=102, y=100, mine_type=0, tank_id=-1, team=1)
    world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (103, 100)


def test_out_of_viewport_covered_container_is_skipped() -> None:
    """The server rejects out-of-view aims, so off-view cover waits."""
    world, self_state = _world_with_self()
    _add_covered_container(world, 130, 100)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) is None


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

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (106, 104)


def test_recruit_scores_single_tile_blast_and_takes_the_nearest() -> None:
    """A recruit's shot clears one mine, so cluster bonuses vanish."""
    world, self_state = _world_with_self(rank=0)
    _add_covered_container(world, 102, 100)
    _add_covered_container(world, 106, 104)
    _add_covered_container(world, 107, 104)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (102, 100)


def test_dreg_fuel_under_mines_is_not_worth_the_shot() -> None:
    """A covered fuel container below the value floor draws no clearance.

    Flag 8 (run bot-20260730-015x): a shot was spent un-covering a
    21-volume fuel dreg. Equipment is always worth the tick; fuel
    must hold a real drink.
    """
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(
        x=104,
        y=100,
        is_fuel=True,
        volume=21,
    )
    world["mines"]["104,100"] = make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) is None


def test_rich_fuel_under_mines_still_draws_the_shot() -> None:
    """A covered fuel container at the value floor stays a clearance aim."""
    world, self_state = _world_with_self()
    world["containers"]["104,100"] = make_container_state(
        x=104,
        y=100,
        is_fuel=True,
        volume=100,
    )
    world["mines"]["104,100"] = make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (104, 100)


def test_blast_clips_at_the_map_edge() -> None:
    """A corner aim only counts in-bounds blast tiles."""
    world, self_state = _world_with_self()
    world["viewport"] = make_viewport_state(left=0, top=0, width=16, height=16)
    self_state["x"], self_state["y"] = 4, 4
    _add_covered_container(world, 0, 0)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (0, 0)


def _session4_pocket() -> tuple[WorldStateDict, SelfStateDict, FerryAwareTerrain]:
    """Rebuild the bot-20260805-173034 geometry trap, verbatim.

    Equipment at (58,95) sits ON water; its west and south service
    tiles are water, its north (58,94) and east (59,95) service tiles
    are ground carrying hostile mines. The bot stands at (60,94). The
    live session re-aimed 1,068 hops / 534 displaced teleports at
    (59,95) over 43 minutes because no planner connected the mine to
    the free clearance single.
    """
    world = make_empty_world_state()
    world["viewport"] = make_viewport_state(left=52, top=86, width=16, height=16)
    self_state = make_self_state(
        tank_id=1,
        x=60,
        y=94,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    world["self_state"] = self_state
    world["containers"]["58,95"] = make_container_state(x=58, y=95, is_fuel=False, volume=0)
    for mine_x, mine_y in ((58, 94), (59, 95)):
        world["mines"][f"{mine_x},{mine_y}"] = make_mine_state(
            x=mine_x,
            y=mine_y,
            mine_type=0,
            tank_id=-1,
            team=1,
        )
    terrain = _composed(
        InMemoryTerrainMap(
            {
                (58, 95): InMemoryTerrainMap.WATER,
                (57, 95): InMemoryTerrainMap.WATER,
                (58, 96): InMemoryTerrainMap.WATER,
            }
        ),
        frozenset({"58,94", "59,95"}),
    )
    return world, self_state, terrain


def test_water_locked_equipment_with_mined_flanks_draws_the_unlock_shot() -> None:
    """The session-4 trap resolves to one free shot, not 1,068 hops.

    Neither old trigger fired live: the container tile carries no mine
    (it is water) and no walk was ever planned. The general trigger —
    a hostile mine denies every attainable landing — aims at (58,94),
    whose 3x3 blast destroys both flank mines and reopens the landing.
    """
    world, self_state, terrain = _session4_pocket()

    assert _find_shot(world, self_state, terrain) == (58, 94)


def test_pure_water_lock_without_mines_draws_no_shot() -> None:
    """A blocked container with mine-free service tiles is ferry business."""
    world, self_state, _terrain = _session4_pocket()
    world["mines"].clear()
    terrain_all_water = InMemoryTerrainMap(
        {
            (58, 95): InMemoryTerrainMap.WATER,
            (57, 95): InMemoryTerrainMap.WATER,
            (58, 96): InMemoryTerrainMap.WATER,
            (58, 94): InMemoryTerrainMap.WATER,
            (59, 95): InMemoryTerrainMap.WATER,
        }
    )

    assert _find_shot(world, self_state, terrain_all_water) is None


def test_blocked_arm_needs_terrain() -> None:
    """Without terrain, attainability is unanswerable — no blocked aim."""
    world, self_state, _ = _session4_pocket()

    assert _find_shot(world, self_state, None) is None


def test_los_blocked_flank_mines_defer_the_unlock_shot() -> None:
    """Rock between the bot and every service mine defers the clearance."""
    world, self_state, _ = _session4_pocket()
    self_state["x"], self_state["y"] = 63, 95
    terrain = _composed(
        InMemoryTerrainMap(
            {
                (58, 95): InMemoryTerrainMap.WATER,
                (57, 95): InMemoryTerrainMap.WATER,
                (58, 96): InMemoryTerrainMap.WATER,
                (61, 93): InMemoryTerrainMap.ROCK,
                (61, 94): InMemoryTerrainMap.ROCK,
                (61, 95): InMemoryTerrainMap.ROCK,
                (61, 96): InMemoryTerrainMap.ROCK,
            }
        ),
        frozenset({"58,94", "59,95"}),
    )

    assert _find_shot(world, self_state, terrain) is None


def test_recruit_single_tile_blast_still_opens_the_aim_tile() -> None:
    """A recruit's 1-tile blast opens the very tile it clears."""
    world, self_state, terrain = _session4_pocket()
    self_state["rank"] = 0

    aim = _find_shot(world, self_state, terrain)

    assert aim in {(58, 94), (59, 95)}


def test_recruit_covered_flank_aim_opens_nothing_and_is_skipped() -> None:
    """An aim whose blast reopens no denied container never wins.

    Recruit blast is one tile: shooting the flank mine at (105,100)
    neither exposes the covered container tile (104,100) nor opens a
    landing that is missing (the clean west cardinal already serves
    it), so that aim scores zero and the container tile itself wins.
    """
    world, self_state = _world_with_self(rank=0)
    _add_covered_container(world, 104, 100)
    world["mines"]["105,100"] = make_mine_state(x=105, y=100, mine_type=0, tank_id=-1, team=1)

    assert _find_shot(world, self_state, InMemoryTerrainMap()) == (104, 100)


def test_edge_of_viewport_container_with_out_of_view_mine_waits() -> None:
    """A denied container whose only service mine sits off-view has no aim."""
    world, self_state, _ = _session4_pocket()
    world["containers"].clear()
    world["mines"].clear()
    world["containers"]["67,94"] = make_container_state(x=67, y=94, is_fuel=False, volume=0)
    world["mines"]["68,94"] = make_mine_state(x=68, y=94, mine_type=0, tank_id=-1, team=1)
    terrain = _composed(
        InMemoryTerrainMap(
            {
                (67, 94): InMemoryTerrainMap.WATER,
                (66, 94): InMemoryTerrainMap.WATER,
                (67, 93): InMemoryTerrainMap.WATER,
                (67, 95): InMemoryTerrainMap.WATER,
            }
        ),
        frozenset({"68,94"}),
    )

    assert _find_shot(world, self_state, terrain) is None


class TestServiceClearanceAim:
    """Tests for the single-target unlock aim used by the lock verdict."""

    def test_mined_flanks_name_the_nearest_service_mine(self) -> None:
        """The session-4 locked target holds because this aim exists."""
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, terrain = _session4_pocket()

        assert find_service_clearance_aim(world, self_state, terrain, 58, 95) == (59, 95)

    def test_open_access_needs_no_aim(self) -> None:
        """A target with an attainable landing already never draws a shot."""
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, _ = _session4_pocket()
        open_terrain = InMemoryTerrainMap()

        assert find_service_clearance_aim(world, self_state, open_terrain, 58, 95) is None

    def test_terrain_none_proposes_no_aim(self) -> None:
        """Attainability is unanswerable without terrain — no aim."""
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, _ = _session4_pocket()

        assert find_service_clearance_aim(world, self_state, None, 58, 95) is None

    def test_los_blocked_service_mine_yields_no_aim(self) -> None:
        """A rock wall between the bot and every service mine defers the aim."""
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, _ = _session4_pocket()
        self_state["x"], self_state["y"] = 63, 95
        terrain = _composed(
            InMemoryTerrainMap(
                {
                    (58, 95): InMemoryTerrainMap.WATER,
                    (57, 95): InMemoryTerrainMap.WATER,
                    (58, 96): InMemoryTerrainMap.WATER,
                    (61, 93): InMemoryTerrainMap.ROCK,
                    (61, 94): InMemoryTerrainMap.ROCK,
                    (61, 95): InMemoryTerrainMap.ROCK,
                    (61, 96): InMemoryTerrainMap.ROCK,
                }
            ),
            frozenset({"58,94", "59,95"}),
        )

        assert find_service_clearance_aim(world, self_state, terrain, 58, 95) is None

    def test_recruit_shot_at_a_water_mine_opens_nothing_and_is_skipped(self) -> None:
        """A mined WATER service tile is a dead aim at recruit blast.

        The goal tile itself carries a hostile mine but sits on water:
        a recruit's 1-tile blast clears that mine yet water can never
        be landed on, and the east ground tile stays mined — so the
        water aim is skipped and the ground mine is the aim instead.
        """
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, _terrain = _session4_pocket()
        self_state["rank"] = 0
        world["mines"]["58,95"] = make_mine_state(x=58, y=95, mine_type=0, tank_id=-1, team=1)
        del world["mines"]["58,94"]
        terrain_water_north = _composed(
            InMemoryTerrainMap(
                {
                    (58, 95): InMemoryTerrainMap.WATER,
                    (57, 95): InMemoryTerrainMap.WATER,
                    (58, 96): InMemoryTerrainMap.WATER,
                    (58, 94): InMemoryTerrainMap.WATER,
                }
            ),
            frozenset({"58,95", "59,95"}),
        )

        assert find_service_clearance_aim(world, self_state, terrain_water_north, 58, 95) == (
            59,
            95,
        )

    def test_out_of_viewport_service_mine_cannot_be_aimed_at(self) -> None:
        """The server rejects off-view aims; the verdict must not hold on one."""
        from tankpit_bot.bot.ai.mine_clearance import find_service_clearance_aim

        world, self_state, terrain = _session4_pocket()
        world["viewport"] = make_viewport_state(left=70, top=86, width=16, height=16)

        assert find_service_clearance_aim(world, self_state, terrain, 58, 95) is None


class TestCorridorClearance:
    """Tests for the walk-corridor mine guard."""

    def test_first_corridor_mine_is_the_aim(self) -> None:
        """A mined walk corridor names its nearest mine for the free shot.

        Flags s6-8/9: six 45-fuel walk-ins against KNOWN mines because
        no walk consulted the mine layer before stepping.
        """
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["mines"]["102,100"] = make_mine_state(x=102, y=100, mine_type=0, tank_id=-1, team=1)
        world["mines"]["104,100"] = make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=1)

        aim = find_corridor_clearance_shot(world, self_state, InMemoryTerrainMap(), 105, 100)

        assert aim == (102, 100)

    def test_clean_corridor_needs_no_shot(self) -> None:
        """No hostile mine on the line means the walk proceeds."""
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["mines"]["102,104"] = make_mine_state(x=102, y=104, mine_type=0, tank_id=-1, team=1)

        aim = find_corridor_clearance_shot(world, self_state, InMemoryTerrainMap(), 105, 100)

        assert aim is None

    def test_destination_mine_counts_as_corridor(self) -> None:
        """A mine sitting on the walk destination itself draws the shot."""
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["mines"]["105,100"] = make_mine_state(x=105, y=100, mine_type=0, tank_id=-1, team=1)

        assert find_corridor_clearance_shot(world, self_state, InMemoryTerrainMap(), 105, 100) == (
            105,
            100,
        )

    def test_friendly_mines_do_not_block_the_walk(self) -> None:
        """Own-team mines are passable; the corridor guard ignores them."""
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["mines"]["102,100"] = make_mine_state(x=102, y=100, mine_type=0, tank_id=-1, team=2)

        aim = find_corridor_clearance_shot(world, self_state, InMemoryTerrainMap(), 105, 100)

        assert aim is None

    def test_terrain_blocked_corridor_mine_is_not_shootable(self) -> None:
        """Rock between self and the corridor mine defers the clearance."""
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)
        terrain = InMemoryTerrainMap({(101, 100): InMemoryTerrainMap.ROCK})

        assert find_corridor_clearance_shot(world, self_state, terrain, 105, 100) is None

    def test_out_of_view_corridor_mine_waits(self) -> None:
        """A corridor mine outside the visible viewport cannot be aimed at."""
        from tankpit_bot.bot.ai.mine_clearance import find_corridor_clearance_shot

        world, self_state = _world_with_self()
        world["viewport"] = make_viewport_state(left=92, top=92, width=10, height=10)
        world["mines"]["103,100"] = make_mine_state(x=103, y=100, mine_type=0, tank_id=-1, team=1)

        aim = find_corridor_clearance_shot(world, self_state, InMemoryTerrainMap(), 105, 100)

        assert aim is None
