"""Tests for corridor mine clearance and beyond-reach relay hops."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_close import (
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    TankStateDict,
    make_tank_state,
)
from tests.bot.ai._combat_fixtures import _enemy_threat
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


class TestCombatCorridorMineGuard:
    """Tests for the walk-close corridor mine clearance."""

    def test_mined_walk_corridor_draws_the_free_clearance_first(self) -> None:
        """A short close through a known mine shoots it before stepping.

        Flags s6-8/9: six 45-fuel walk-ins against mines that were in
        the mine layer the whole time. The clearance single is free
        ([[mine-mechanics]]), so the corridor is drained before the
        first step and the walk proceeds next tick.
        """
        from tankpit_bot.state.types import make_mine_state
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="NearEnemy",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        # make_world's self is team 1, so a team-2 mine is hostile.
        world["mines"]["101,100"] = make_mine_state(x=101, y=100, mine_type=0, tank_id=-1, team=2)
        terrain = InMemoryTerrainMap({(102, 100): InMemoryTerrainMap.ROCK})
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=103, y=100, name="NearEnemy"))

        if result is None:
            raise AssertionError("expected corridor clearance decision")
        assert result["command"]["cmd_type"] == "shoot"
        assert result["behavior"]["reason_kind"] == "mine_clearance_shot"
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_short_close_with_no_usable_landing_falls_through_to_teleport(self) -> None:
        """All-occupied adjacency at short range leaves the teleport close.

        The server displaces a teleport off mines and tanks, so when
        every adjacent tile is dynamically occupied the walk has no
        destination and the direct close remains the answer.
        """
        from tankpit_bot.state.types import make_mine_state
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="RingedNear",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        for mx, my in ((102, 100), (104, 100), (103, 99), (103, 101)):
            world["mines"][f"{mx},{my}"] = make_mine_state(
                x=mx, y=my, mine_type=0, tank_id=-1, team=2
            )
        terrain = InMemoryTerrainMap({(101, 100): InMemoryTerrainMap.ROCK})
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=103, y=100, name="RingedNear"))

        if result is None:
            raise AssertionError("expected teleport decision")
        assert result["command"]["cmd_type"] == "teleport"


class TestBeyondRefuelReachRelay:
    """A chase no refuel can fund takes the relay, not a top-off."""

    def test_beyond_reach_target_relays_via_dots(self) -> None:
        """Flag s10-2: at fuel 1097/1100 a 504-cost chase 'refueled' a
        3-point deficit with a 121-fuel dot teleport. Cost above
        cap-minus-reserve is a DISTANCE problem: the decision must be
        a dot_relay leg with the lock held.
        """
        ws = WorldService()
        ws.map_fuel_dots = ((60, 150),)
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=1,
                y=193,
                team=2,
                rank=1,
                name="red-50",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1195, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a relay decision")
        assert result["behavior"]["reason_kind"] == "dot_relay"
        assert result["command"]["cmd_type"] == "teleport"
        assert result["command"]["target_x"] == 60
        assert result["command"]["target_y"] == 150
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_neighbor_dot_never_beats_the_progress_dot(self) -> None:
        """The live 2026-08-04 ping-pong, pinned.

        Run bot-20260804-230342 23:24-23:29: a locked target 98 tiles
        out, and the relay branch (then wired to the COLLECT
        dot-ranker) teleported between two ADJACENT dots at one hop
        per 2 ticks forever -- the /cost denominator made the dot
        under the tank's feet unbeatable. The relay lane must pick
        the strict-progress dot even when a neighbor dot is
        thousands of times cheaper.
        """
        ws = WorldService()
        ws.map_fuel_dots = ((101, 100), (60, 150))
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=1,
                y=193,
                team=2,
                rank=1,
                name="red-50",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1195, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a relay decision")
        assert result["behavior"]["reason_kind"] == "dot_relay"
        assert result["command"]["cmd_type"] == "teleport"
        assert result["command"]["target_x"] == 60
        assert result["command"]["target_y"] == 150

    def test_no_progress_dot_at_cap_blocks_the_unreachable_target(self) -> None:
        """With no strict-progress dot and fuel at capacity, the relay
        cannot help and refueling cannot help: the target is blocked
        so the next pass finds closer prey instead of treadmilling.
        """
        ws = WorldService()
        ws.map_fuel_dots = ((101, 100),)
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=1,
                y=193,
                team=2,
                rank=1,
                name="red-50",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1195, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a block-and-replan decision")
        assert result["behavior"]["reason_kind"] != "dot_relay"
        assert "50" in result["updated_ai_state"]["blocked_combat_targets"]

    def test_beyond_reach_with_no_usable_dot_falls_back_to_refuel(self) -> None:
        """When no relay leg exists the beyond-reach chase still refuels.

        The only atlas dot is the bot's own tile (own_tile rejection)
        and the map was just opened, so the hop declines outright.
        """
        ws = WorldService()
        ws.map_fuel_dots = ((100, 100),)
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=1,
                y=193,
                team=2,
                rank=1,
                name="red-50",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1195, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "",
            ws=ws,
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a fallback decision")
        assert result["behavior"]["reason_kind"] != "dot_relay"
