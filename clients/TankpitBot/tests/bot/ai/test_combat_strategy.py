"""Focused tests for combat route primitives."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    SHOT_RANGE_TILES,
    _combat_landing_candidates,
    close_target,
    engage_target,
    get_locked_target,
    has_combat_shot,
    is_already_engaged,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, WorldStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _enemy_threat(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Enemy",
    last_wire_seen_ms: int = 100000,
    last_position_update_ms: int = 100000,
) -> EnemyThreatDict:
    """Create a typed enemy threat for combat helper tests.

    Args:
        tank_id: Enemy tank identifier.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        last_wire_seen_ms: Last wire-presence confirmation (defaults to
            the helper's ``timestamp_ms`` so the threat is wire-present at
            the tests' 100000 clock unless overridden to model a ghost).
        last_position_update_ms: Last wire-sourced position confirmation
            (defaults to the helper's clock so the threat passes the
            kill-shot gate unless overridden to model a stale-position
            target).

    Returns:
        Enemy threat payload.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=abs(x - 100) + abs(y - 100),
        damage_state=0,
        rank=1,
        team=2,
        name=name,
        is_bot=False,
        timestamp_ms=100000,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_aim_x=-1,
        last_aim_y=-1,
        last_aim_weapon=-1,
        last_aim_ms=0,
    )


class TestCombatTargetSelection:
    """Tests for combat target selection constraints."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_select_new_combat_target_allows_non_emergency_reserve_band(self) -> None:
        """Combat still acquires targets above the break threshold."""
        world, self_state = make_world(fuel=800)
        inventory = make_inventory()
        inventory["dual_shots"]["count"] = 10
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            "",
        )

        result = select_new_combat_target(ctx, [_enemy_threat()])

        if result is None:
            raise AssertionError("expected viable combat target in non-emergency reserve band")
        assert result["tank_id"] == 50

    def test_select_new_combat_target_skips_blocked_and_killed_targets(self) -> None:
        """Combat target selection ignores blocked and killed enemies."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["blocked_combat_targets"] = {"50": 99000}
        ai_state["killed_tank_ids"] = {"60": 99500}
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, name="BlockedEnemy"),
                _enemy_threat(tank_id=60, x=118, name="KilledEnemy"),
                _enemy_threat(tank_id=70, x=116, name="ViableEnemy"),
            ],
        )

        if result is None:
            raise AssertionError("expected viable combat target")
        assert result["tank_id"] == 70
        assert result["name"] == "ViableEnemy"

    def test_select_new_combat_target_returns_none_when_no_viable_targets(self) -> None:
        """Combat target selection returns None when every threat is excluded."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["blocked_combat_targets"] = {"50": 99000}
        ai_state["killed_tank_ids"] = {"60": 99500}
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, name="BlockedEnemy"),
                _enemy_threat(tank_id=60, x=118, name="KilledEnemy"),
            ],
        )

        assert result is None

    def test_select_new_combat_target_picks_closest(self) -> None:
        """Target selection picks the closest viable enemy."""
        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, x=110, name="Nearest"),
                _enemy_threat(tank_id=60, x=114, name="Middle"),
                _enemy_threat(tank_id=70, x=130, name="Farthest"),
            ],
        )

        if result is None:
            raise AssertionError("expected nearest combat target")
        assert result["tank_id"] == 50
        assert result["name"] == "Nearest"

    def test_select_new_combat_target_skips_blocked(self) -> None:
        """Target selection skips blocked and killed targets, takes next closest."""
        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        ctx.blocked_targets["50"] = 100000

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, x=110, name="BlockedNearest"),
                _enemy_threat(tank_id=60, x=140, name="NextClosest"),
                _enemy_threat(tank_id=61, x=146, name="Farther"),
            ],
        )

        if result is None:
            raise AssertionError("expected least-clustered combat target")
        assert result["tank_id"] == 60
        assert result["name"] == "NextClosest"

    def test_select_new_combat_target_keeps_nearest_among_isolated(self) -> None:
        """Equal cluster counts fall back to the distance ordering."""
        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, x=112, name="IsolatedNear"),
                _enemy_threat(tank_id=60, x=130, name="IsolatedFar"),
            ],
        )

        if result is None:
            raise AssertionError("expected nearest isolated combat target")
        assert result["tank_id"] == 50

    def test_select_new_combat_target_skips_when_no_standoff_landing(self) -> None:
        """Targets with no passable tile inside the shot-range diamond are skipped."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        terrain = InMemoryTerrainMap.from_passable_set(set())

        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [_enemy_threat(tank_id=50, x=110, y=100, name="WaterLocked")],
        )
        assert result is None

    def test_select_new_combat_target_keeps_ringed_target_with_standoff(self) -> None:
        """A target whose ring is impassable stays viable via stand-off range.

        Mine rings and ferry riders share this shape: the tiles touching
        the target are impassable but ground exists within
        ``SHOT_RANGE_TILES``, so the stand-off engagement (teleport near,
        dual from range) still works and the target must not be skipped.
        """
        from tests.action_lab.conftest import Terrain

        rocks = {
            (111, 100): Terrain.ROCK,
            (109, 100): Terrain.ROCK,
            (110, 101): Terrain.ROCK,
            (110, 99): Terrain.ROCK,
        }
        terrain = Terrain(overrides=rocks)

        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [_enemy_threat(tank_id=50, x=110, y=100, name="Ringed")],
        )
        if result is None:
            raise AssertionError("expected the ringed target to stay viable")
        assert result["tank_id"] == 50


class TestGetLockedTargetWorldStateFallback:
    """Tests for get_locked_target world-state fallback."""

    def setup_method(self) -> None:
        reset_world_state()

    def teardown_method(self) -> None:
        reset_world_state()

    def test_returns_threat_when_in_threat_list(self) -> None:
        """Threat-list match takes priority over world-state fallback."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 50
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")
        threats = [_enemy_threat(tank_id=50, x=101, y=100)]

        result = get_locked_target(ctx, threats)

        if result is None:
            raise AssertionError("expected target from threat list")
        assert result["tank_id"] == 50
        assert result["x"] == 101

    def test_returns_none_when_locked_target_drops_off_threats(self) -> None:
        """No world-state fallback: lock release IS dropping off the threat list.

        The pre-2026-06-21 implementation synthesised a fake threat
        from ``world.tanks`` when the locked id was missing from
        ``threats``. The enemy-tracking probe proved that fallback
        was the source of the "fires one shot then hops" failure
        loop: it kept locks alive on tanks the JS client itself no
        longer listed in ``activeGame.P.j``. ``get_locked_target``
        now returns ``None`` the moment a tank leaves the threat
        list, letting ``_decide_hunt_engage`` enter confirm_kill
        and re-acquire from fresh viewport intel.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=130,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 50
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        assert get_locked_target(ctx, []) is None

    def test_returns_none_when_no_combat_target(self) -> None:
        """No locked target (combat_target_id == -1) returns None."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        assert get_locked_target(ctx, []) is None


class TestIsAlreadyEngaged:
    """Tests for the engaged-vs-fresh discriminator."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_true_when_last_shot_target_matches_combat_target(self) -> None:
        """A dispatched shoot at the current lock proves the bot is engaged."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 50
        ai_state["last_shot_target_id"] = 50
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        assert is_already_engaged(ctx) is True

    def test_false_when_last_shot_target_differs(self) -> None:
        """A fresh lock with no shot dispatched at this id is not engaged."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 50
        ai_state["last_shot_target_id"] = -1
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        assert is_already_engaged(ctx) is False

    def test_false_when_last_shot_target_is_old_kill(self) -> None:
        """Carryover ``last_shot_target_id`` from a prior kill does not count.

        After a kill, the planner picks up a new lock with a different
        id; ``last_shot_target_id`` still points at the dead enemy
        until the next shoot dispatches. The mismatch correctly says
        "fresh acquire" for the new target.
        """
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 60
        ai_state["last_shot_target_id"] = 50
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        assert is_already_engaged(ctx) is False


def test_combat_landing_candidates_delegate_to_shared_helper() -> None:
    """Combat landing candidates expose shared adjacent-tile ordering."""
    world, self_state = make_world(fuel=800)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )

    assert _combat_landing_candidates(ctx, _enemy_threat()) == [
        (119, 100),
        (121, 100),
        (120, 101),
        (120, 99),
    ]


class TestCombatTeleportGuards:
    """Tests for combat teleport affordability and legality."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

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
        from tankpit_bot.sniffer.world_state import mark_move_target_failed

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
        mark_move_target_failed(107, 107, 99000)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
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
        from tankpit_bot.sniffer.world_state import mark_move_target_failed

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
        mark_move_target_failed(120, 100, 99000)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
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
        still paid a teleport onto him, because only ``_combat_close``
        (which needs an existing lock) asked the shot-range question.
        The acquire teleport now short-circuits to the shot, and the
        shot latches the combat lock itself.
        """
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
        )

        result = teleport_to_target(ctx, _enemy_threat(x=107, y=107, name="FarCornerEnemy"))

        if result is None:
            raise AssertionError("expected shoot decision")
        assert result["command"]["cmd_type"] == "shoot"
        assert result["updated_ai_state"]["combat_target_id"] == 50


class TestKillShotWireGate:
    """Tests for the wire-presence kill gate at the shoot chokepoint."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def _adjacent_world_and_ctx(
        self,
        last_wire_seen_ms: int,
        *,
        last_position_update_ms: int | None = None,
    ) -> tuple[DecideCtx, EnemyThreatDict]:
        """Build a ctx with one adjacent enemy at the given freshness stamps.

        Args:
            last_wire_seen_ms: The enemy's last wire-presence confirmation.
            last_position_update_ms: The enemy's last wire-sourced
                position confirmation. Defaults to ``last_wire_seen_ms``
                so the enemy is both wire- and position-fresh; pass an
                older value to model the stale-position case.

        Returns:
            A decision context (tick clock 100000) and the matching
            adjacent enemy threat.
        """
        position_stamp = (
            last_position_update_ms if last_position_update_ms is not None else last_wire_seen_ms
        )
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="Adjacent",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=last_wire_seen_ms,
                last_position_update_ms=position_stamp,
                last_viewport_observation_ms=position_stamp,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        target = _enemy_threat(
            x=101,
            y=100,
            name="Adjacent",
            last_wire_seen_ms=last_wire_seen_ms,
            last_position_update_ms=position_stamp,
        )
        return ctx, target

    def test_wire_fresh_adjacent_target_is_shot(self) -> None:
        """A wire-present adjacent enemy is fired at directly."""
        ctx, target = self._adjacent_world_and_ctx(last_wire_seen_ms=100000)

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["reason_kind"] == "shoot_target"

    def test_wire_stale_adjacent_target_is_still_shot(self) -> None:
        """A wire-silent target is still engaged.

        Wire-silence is not a stop signal. A locked target that
        teleports off the bot's viewport stops broadcasting wire
        updates -- the server only emits wire events for tanks the
        local viewport can see -- which is the expected case the
        pursuit cascade exists to handle. The lock holds until an
        authoritative deactivation signal arrives (``liveness``
        flips to ``deactivated`` or the tank lands in
        ``killed_tank_ids``).

        Pre-2026-06-23 the wire-presence gate in ``_combat_shoot``
        blocked any target whose ``last_wire_seen_ms`` exceeded the
        7000ms presence TTL. That gate killed pursuit shots: live
        run 2026-06-23 19:31:43 saw purple-8 (id=516) teleport off
        viewport during an engagement, go wire-silent for 8224ms, get
        flagged as a "ghost" and blocked despite being genuinely alive
        and the bot's active combat lock. The gate was removed
        2026-06-23; this test guards against re-introduction.
        """
        ctx, target = self._adjacent_world_and_ctx(last_wire_seen_ms=100000 - 60000)

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert "50" not in decision["updated_ai_state"]["blocked_combat_targets"]


class TestMissOnMovedTarget:
    """Tests for the miss-on-moved-target re-aim path in _combat_shoot."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_miss_on_moved_target_re_aims_instead_of_blocking(self) -> None:
        """A miss against a target that moved since the shot re-aims at the new position.

        When ``combat_feedback == "miss"`` and the target's current
        coordinates differ from the stored last-shot position, the enemy
        is NOT blocked.  Instead the bot fires at the target's fresh
        registry position -- the miss was ambiguous (the enemy may have
        stepped off the tile as the shot resolved), so abandoning a live
        mover is premature.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=102,
                y=100,
                team=2,
                rank=1,
                name="Mover",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, tanks=tanks)
        ai_state = make_scanned_ai_state()
        ai_state["combat_target_id"] = 50
        ai_state["combat_target_x"] = 101
        ai_state["combat_target_y"] = 100
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "miss",
        )
        target = _enemy_threat(x=102, y=100, name="Mover", last_wire_seen_ms=100000)

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 102
        assert decision["command"]["target_y"] == 100
        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert "50" not in decision["updated_ai_state"]["blocked_combat_targets"]


class TestHasCombatShot:
    """Tests for the ``has_combat_shot`` range predicate."""

    def test_has_combat_shot_returns_true_within_range(self) -> None:
        """A target at SHOT_RANGE_TILES distance is within shot range."""
        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        target = _enemy_threat(x=108, y=100)  # distance=8

        assert SHOT_RANGE_TILES == 8
        assert has_combat_shot(ctx, target) is True

    def test_has_combat_shot_returns_false_beyond_range(self) -> None:
        """A target beyond SHOT_RANGE_TILES distance is out of shot range."""
        world, self_state = make_world(fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        target = _enemy_threat(x=109, y=100)  # distance=9

        assert has_combat_shot(ctx, target) is False


class TestFindCombatPickup:
    """Tests for _find_combat_pickup mid-combat pickup selection."""

    def setup_method(self) -> None:
        reset_world_state()

    def teardown_method(self) -> None:
        reset_world_state()

    def test_finds_adjacent_fuel_when_low(self) -> None:
        """Returns fuel pickup when fuel is below threshold and fuel is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.state import make_container_state

        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=200,
            containers={
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=True,
                    volume=300,
                    timestamp_ms=100000,
                ),
            },
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = _find_combat_pickup(ctx)
        if result is None:
            raise AssertionError("expected a pickup command")
        assert result["cmd_type"] == "pickup_fuel"

    def test_finds_adjacent_equipment(self) -> None:
        """Returns equipment pickup when equipment is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.state import make_container_state

        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=800,
            containers={
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=False,
                    volume=0,
                    timestamp_ms=100000,
                ),
            },
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = _find_combat_pickup(ctx)
        if result is None:
            raise AssertionError("expected a pickup command")
        assert result["cmd_type"] == "pickup_equipment"

    def test_returns_none_when_no_adjacent(self) -> None:
        """Returns None when nothing is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup

        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        assert _find_combat_pickup(ctx) is None

    def test_combat_shoot_includes_secondary_pickup(self) -> None:
        """_combat_shoot produces secondary pickup when container is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _combat_shoot
        from tankpit_bot.state import make_container_state

        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=200,
            tanks={
                "50": make_tank_state(
                    tank_id=50,
                    x=101,
                    y=100,
                    team=1,
                    rank=0,
                    damage_state=0,
                    name="Enemy",
                    is_bot=False,
                    is_self=False,
                    source="viewport",
                    timestamp_ms=100000,
                    last_wire_seen_ms=100000,
                    last_position_update_ms=100000,
                    last_viewport_observation_ms=100000,
                ),
            },
            containers={
                "99,100": make_container_state(
                    x=99,
                    y=100,
                    is_fuel=True,
                    volume=300,
                    timestamp_ms=100000,
                ),
            },
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        target = _enemy_threat(tank_id=50, x=101, y=100, last_wire_seen_ms=100000)

        decision = _combat_shoot(ctx, target)
        assert decision["command"]["cmd_type"] == "shoot"
        secondary = decision["secondary_command"]
        if secondary is None:
            raise AssertionError("expected secondary pickup command")
        assert secondary["cmd_type"] == "pickup_fuel"

    def test_returns_none_when_self_state_none(self) -> None:
        """Line 429: self_state is None guard."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup

        world, self_state = make_world(fuel=800)
        world_no_self = WorldStateDict(
            self_state=None,
            tanks=world["tanks"],
            containers=world["containers"],
            mines=world["mines"],
            terrain=world["terrain"],
            viewport=world["viewport"],
            scanned_tiles=world["scanned_tiles"],
            timestamp_ms=world["timestamp_ms"],
        )
        ctx = DecideCtx(
            world_no_self,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        ctx.world = world_no_self
        assert _find_combat_pickup(ctx) is None


class TestCombatCorridorMineGuard:
    """Tests for the walk-close corridor mine clearance."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_mined_walk_corridor_draws_the_free_clearance_first(self) -> None:
        """A short close through a known mine shoots it before stepping.

        Flags s6-8/9: six 45-fuel walk-ins against mines that were in
        the mine layer the whole time. The clearance single is free
        ([[mine-mechanics]]), so the corridor is drained before the
        first step and the walk proceeds next tick.
        """
        from tankpit_bot.state.types import make_mine_state
        from tests.in_memory_terrain_map import InMemoryTerrainMap

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
            ((60, 150),),
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a relay decision")
        assert result["behavior"]["reason_kind"] == "dot_relay"
        assert result["command"]["cmd_type"] == "teleport"
        assert result["command"]["target_x"] == 60
        assert result["command"]["target_y"] == 150
        assert result["updated_ai_state"]["combat_target_id"] == 50

    def test_beyond_reach_with_no_usable_dot_falls_back_to_refuel(self) -> None:
        """When no relay leg exists the beyond-reach chase still refuels.

        The only atlas dot is the bot's own tile (own_tile rejection)
        and the map was just opened, so the hop declines outright.
        """
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
            ((100, 100),),
        )

        result = teleport_to_target(ctx, _enemy_threat(x=1, y=193, name="red-50", tank_id=50))

        if result is None:
            raise AssertionError("expected a fallback decision")
        assert result["behavior"]["reason_kind"] != "dot_relay"
