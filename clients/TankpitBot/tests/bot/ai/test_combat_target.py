"""Tests for :mod:`tankpit_bot.bot.ai.combat_target`.

Target selection, the locked-target accessors and their world-state
fallback, the already-engaged predicate, and landing-candidate
delegation.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_target import (
    _combat_landing_candidates,
    get_locked_target,
    is_already_engaged,
    select_new_combat_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
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


class TestCombatTargetSelection:
    """Tests for combat target selection constraints."""

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
