"""Tests for the combat teleport guards.

Every refusal and replan path :func:`teleport_to_target` can take when
the landing tile is unusable or the hop is unaffordable.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_close import (
    close_target,
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


class TestCombatTeleportGuards:
    """Tests for combat teleport affordability and legality."""

    def test_teleport_to_target_refuels_when_unaffordable(self) -> None:
        """An unaffordable combat target delegates the tick to fuel recovery.

        Threats sort nearest-first and teleport cost is monotone in
        distance, so an unaffordable nearest target means every target
        is unaffordable. Blocking and replanning instead cascaded
        through the roster and ended in a map-reopen spin: run
        20260611-025636 spawned at fuel 620 -- above the fuel-low
        entry rule but below every engagement's cost-plus-reserve --
        and spent its entire 240s on 115 map reopens without a shot.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=190,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=520, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=190, y=100, name="FarEnemy"))

        if result is None:
            raise AssertionError("expected fuel recovery decision")
        assert result["behavior"]["mode"] == "COLLECT"
        assert result["updated_ai_state"]["combat_target_id"] == 50
        assert result["updated_ai_state"]["blocked_combat_targets"] == {}

    def test_teleport_to_target_blocks_when_collect_declines(self) -> None:
        """An unaffordable target with a fully-exhausted collect cascade
        blocks and replans instead of exiting the session.

        Live run 2026-07-06 exited ``out_of_fuel`` at fuel 1100 with a
        stocked tank because the collect cascade could not produce a
        legal action either. With the ``None`` yield from collect,
        ``_refuel_for_hunt`` now falls through to
        :func:`block_combat_target_and_replan` so the tick advances
        instead of raising.

        Setup: enemy 90 tiles east (teleport unaffordable at fuel 550
        against the ``fuel_low_threshold`` reserve of 200), viewport
        fully scanned (forage declines), no visible containers (both
        pickup branches decline), and an empty dot atlas with a recent
        map open (``last_map_open_ms=96000`` -> 4000 ms age, inside the
        default 5000 ms cooldown) so the dot-hop path also declines.
        Fuel 550 > 200, so collect returns ``None`` rather than raising
        ``out_of_fuel``.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=190,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=550, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 96000,
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

        result = teleport_to_target(ctx, _enemy_threat(x=190, y=100, name="FarEnemy"))

        assert result["behavior"]["mode"] == "HUNT"
        assert result["behavior"]["reason_kind"] == "find_enemies"
        assert result["updated_ai_state"]["combat_target_id"] == -1
        assert "50" in result["updated_ai_state"]["blocked_combat_targets"]
        assert result["updated_ai_state"]["blocked_combat_targets"]["50"] == 100000

    def test_failed_landing_with_target_in_view_fires_from_stand_off(self) -> None:
        """A dead landing next to an in-view target becomes a shot, not a block.

        Mine-ring counterplay (user ruling 2026-07-29: "fix the bot so
        it can hunt even if i put mines around me... it can still fire
        a dual shot right?"). Live proof of the old hole at 21:35:06:
        the ring displaced the landing, the re-close failed on the
        same tile, and Yuppler was BLOCKED while in plain view. Every
        in-view target now short-circuits to the shot before the
        landing check even runs, so a recorded failed landing must
        never block a visible target -- this pins the viewport corner
        with a dead landing on record.
        """

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "90": make_tank_state(
                tank_id=90,
                x=107,
                y=107,
                team=2,
                rank=1,
                name="Yuppler",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1100, tanks=tanks)
        ws.mark_move_target_failed(107, 107, 99000)
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

        result = teleport_to_target(ctx, _enemy_threat(tank_id=90, x=107, y=107, name="Yuppler"))

        assert result["command"]["cmd_type"] == "shoot"
        assert "90" not in result["updated_ai_state"]["blocked_combat_targets"]

    def test_failed_landing_with_target_off_view_still_blocks(self) -> None:
        """A dead landing at an OFF-view target keeps the block behavior.

        Firing needs the target inside the visible viewport (the
        server rejects out-of-view aims); with no legal landing and no
        legal shot, blocking and replanning is still correct.
        """

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "90": make_tank_state(
                tank_id=90,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="Yuppler",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1100, tanks=tanks)
        ws.mark_move_target_failed(120, 100, 99000)
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

        result = teleport_to_target(ctx, _enemy_threat(tank_id=90, x=120, y=100, name="Yuppler"))

        assert result["command"]["cmd_type"] != "shoot"
        assert "90" in result["updated_ai_state"]["blocked_combat_targets"]

    def test_teleport_reserve_includes_the_engagement_budget(self) -> None:
        """A chase the old reserve passed by 14 fuel now refuels first.

        Regression shape from run 20260729-105325: fuel 372, chase
        cost 158, and the ``fuel_low_threshold``-only reserve (200)
        passed the teleport -- the bot landed at 214, LOW_FUEL
        hijacked one shot later, and the session bottomed at 140.
        User ruling 2026-07-29: "we cant kill anyone if we die...
        we should fuel before chasing." The reserve now includes the
        full ``engagement_fuel_budget`` (matching the acquisition
        gate), so this teleport delegates to lock-held refueling.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=127,
                y=100,
                team=2,
                rank=1,
                name="red-7",
                is_self=False,
                is_bot=True,
                damage_state=2,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=372, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=127, y=100, name="red-7"))

        if result is None:
            raise AssertionError("expected fuel recovery decision")
        assert result["command"]["cmd_type"] != "teleport"
        assert result["behavior"]["mode"] == "COLLECT"
        # Never-drop: the lock rides through the refuel detour.
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_teleport_to_target_returns_teleport_for_affordable_close(self) -> None:
        """Combat teleport emits a teleport decision when the landing is affordable."""
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="CloseEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=120, y=100, name="CloseEnemy"))

        if result is None:
            raise AssertionError("expected teleport decision")
        assert result["command"]["cmd_type"] == "teleport"
        assert result["behavior"]["reason_kind"] == "teleport_target"

    def test_teleport_to_target_shoots_when_target_already_in_view_within_range(self) -> None:
        """An in-view target inside shot range is engaged without a teleport.

        Flag 1 of run bot-20260730-000030: purple-4 stood 2 tiles away
        inside the freshly scanned viewport and the map-acquire path
        still paid a teleport onto him, because only ``close_target``
        (which needs an existing lock) asked the shot-range question.
        The acquire teleport now short-circuits to the shot, and the
        shot latches the combat lock itself.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="InViewEnemy",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=104, y=100, name="InViewEnemy"))

        if result is None:
            raise AssertionError("expected shoot decision")
        assert result["command"]["cmd_type"] == "shoot"
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_teleport_to_target_closes_when_in_view_shot_line_is_blocked(self) -> None:
        """An in-view target behind rock gets a close teleport, not a shot.

        The firing law's clearance clause (user verbatim: "...and its a
        CLEAR dual shot"), enforced after flag s3-16 and the Artax
        death: shooting through the occluder spends half-damage
        over-terrain homings while the enemy duals back for 90. The
        close lands adjacent, and adjacency always has a clear line.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="BehindRock",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=104, y=100, name="BehindRock"))

        if result is None:
            raise AssertionError("expected close teleport decision")
        assert result["command"]["cmd_type"] == "teleport"
        assert result["behavior"]["reason_kind"] == "teleport_target"

    def test_engaged_target_behind_terrain_recloses_for_a_clear_shot(self) -> None:
        """Mid-fight, an occluded line re-closes instead of arcing homings.

        Flag s3-16 ("we're shooting over terrain when we should have
        teleported back adjacent") and the Artax death: the stay-put
        homing trade is 45 out against 90 in. The engaged branch now
        pays the re-close teleport when the dual line is blocked.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="BehindRock",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        terrain = InMemoryTerrainMap({(102, 100): InMemoryTerrainMap.ROCK})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 104,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
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

        result = close_target(ctx, _enemy_threat(x=104, y=100, name="BehindRock"))

        if result is None:
            raise AssertionError("expected re-close teleport decision")
        assert result["command"]["cmd_type"] == "teleport"

    def test_blocked_line_short_close_walks_instead_of_teleporting(self) -> None:
        """A 2-tile re-close with a blocked line is a walk, not a teleport.

        Flag 1 of run bot-20260730-011x ("i think it should ahve waked
        back instead of teleporting"): a walk tile costs ~2 s and no
        fuel; a mid-fight teleport costs ~4 s plus fuel plus the
        map-open tick the last shot closed. Within WALK_CLOSE_TILES
        the walk wins outright.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=102,
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

        result = teleport_to_target(ctx, _enemy_threat(x=102, y=100, name="NearEnemy"))

        if result is None:
            raise AssertionError("expected walk decision")
        assert result["command"]["cmd_type"] == "move"
        assert result["behavior"]["reason_kind"] == "walk_to_target"
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_blocked_line_long_close_still_teleports(self) -> None:
        """Beyond the walk break-even the blocked-line close pays the teleport."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=106,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        terrain = InMemoryTerrainMap({(103, 100): InMemoryTerrainMap.ROCK})
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

        result = teleport_to_target(ctx, _enemy_threat(x=106, y=100, name="FarEnemy"))

        if result is None:
            raise AssertionError("expected teleport decision")
        assert result["command"]["cmd_type"] == "teleport"

    def test_teleport_to_target_shoots_when_in_view_beyond_shot_range_bound(self) -> None:
        """In view beyond ``SHOT_RANGE_TILES`` is still a shot, not a teleport.

        Flag s2-13 (run bot-20260730-000030, 00:31:37): purple-9 stood
        at Manhattan 9 -- inside the viewport, beyond the old 8-tile
        short-circuit bound -- and the bot paid a teleport to close.
        User law: in-view is the firing criterion; the server serves
        any in-view range.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=107,
                y=107,
                team=2,
                rank=1,
                name="FarCornerEnemy",
                is_self=False,
                is_bot=True,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
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

        result = teleport_to_target(ctx, _enemy_threat(x=107, y=107, name="FarCornerEnemy"))

        if result is None:
            raise AssertionError("expected shoot decision")
        assert result["command"]["cmd_type"] == "shoot"
        assert result["updated_ai_state"]["combat_target_id"] == 50
