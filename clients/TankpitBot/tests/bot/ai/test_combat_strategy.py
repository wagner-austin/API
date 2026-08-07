"""Tests for :mod:`tankpit_bot.bot.ai.combat_strategy`.

The fire decision: the kill-shot wire gate, miss handling on a moved
target, shot-range predicates, and the in-combat pickup.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import SHOT_RANGE_TILES
from tankpit_bot.bot.ai.combat_strategy import (
    engage_target,
    has_combat_shot,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import (
    TankStateDict,
    WorldStateDict,
    make_tank_state,
)
from tests.bot.ai._combat_fixtures import _enemy_threat
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


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

        Pre-2026-06-23 the wire-presence gate in ``engage_target``
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
    """Tests for the miss-on-moved-target re-aim path in engage_target."""

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
        """Returns equipment pickup when adjacent and a slot is deficient."""
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
            make_inventory(default_count=10),
            100000,
            None,
            "",
        )

        result = _find_combat_pickup(ctx)
        if result is None:
            raise AssertionError("expected a pickup command")
        assert result["cmd_type"] == "pickup_equipment"

    def test_full_tank_skips_adjacent_fuel(self) -> None:
        """At rank fuel capacity the only adjacent container is fuel:
        the sip is a predicted code-5 refusal (physics/supervisor.py)
        and the scanner declines it — the 48-refusal shape of the
        20-kill soak. One fuel of headroom flips it back to a sip."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.bot.types import BotCommand
        from tankpit_bot.physics.capacity import fuel_capacity
        from tankpit_bot.state import make_container_state

        def scan(fuel: int) -> BotCommand | None:
            world, self_state = make_world(
                self_x=100,
                self_y=100,
                fuel=fuel,
                containers={
                    "101,100": make_container_state(
                        x=101, y=100, is_fuel=True, volume=508, timestamp_ms=100000
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
            return _find_combat_pickup(ctx)

        _, self_state = make_world()
        cap = fuel_capacity(self_state["rank"])
        assert scan(cap) is None
        headroom = scan(cap - 1)
        if headroom is None:
            raise AssertionError("one fuel of headroom must allow the sip")
        assert headroom["cmd_type"] == "pickup_fuel"

    def test_full_inventory_skips_adjacent_equipment(self) -> None:
        """All five slots at the rank cap: the grab is a predicted
        code-7 refusal and the scanner declines it."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.physics.capacity import inventory_capacity
        from tankpit_bot.state import make_container_state

        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=800,
            containers={
                "101,100": make_container_state(
                    x=101, y=100, is_fuel=False, volume=0, timestamp_ms=100000
                ),
            },
        )
        cap = inventory_capacity(self_state["rank"])
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(dual_count=cap, default_count=cap),
            100000,
            None,
            "",
        )

        assert _find_combat_pickup(ctx) is None

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
        """engage_target produces secondary pickup when container is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import engage_target
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

        decision = engage_target(ctx, target)
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
