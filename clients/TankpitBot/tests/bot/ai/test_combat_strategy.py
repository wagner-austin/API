"""Focused tests for combat route primitives."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    SHOT_RANGE_TILES,
    _combat_landing_candidates,
    engage_target,
    get_locked_target,
    has_combat_shot,
    is_already_engaged,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import EnemyThreatDict
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

    def test_select_new_combat_target_skips_impassable_adjacent(self) -> None:
        """Targets with no passable adjacent tile are skipped."""
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
            [_enemy_threat(tank_id=50, x=110, y=100, name="Surrounded")],
        )
        assert result is None


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
        assert result["updated_ai_state"]["combat_target_id"] == -1
        assert result["updated_ai_state"]["blocked_combat_targets"] == {}

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
        assert result["behavior"]["reason"] == "teleport CloseEnemy"


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
        assert decision["behavior"]["reason"] == "shoot Adjacent"

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
        assert decision["behavior"]["reason"] == "shoot Mover"
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
        target = _enemy_threat(x=102, y=100)  # distance=2

        assert SHOT_RANGE_TILES == 2
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
        target = _enemy_threat(x=103, y=100)  # distance=3

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
