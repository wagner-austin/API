"""Tests for :mod:`tankpit_bot.bot.ai.combat_strategy`.

The fire decision: the kill-shot wire gate, miss handling on a moved
target, shot-range predicates, and the in-combat pickup.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import SHOT_RANGE_TILES
from tankpit_bot.bot.ai.combat_strategy import (
    engage_target,
    frame_target_shift,
    has_combat_shot,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.protocol.commands import SCOPE_NORTH
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    TankStateDict,
    WorldStateDict,
    make_tank_state,
    make_viewport_state,
)
from tests.bot.ai._combat_fixtures import _enemy_threat
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


class TestKillShotWireGate:
    """Tests for the wire-presence kill gate at the shoot chokepoint."""

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
        ws = WorldService()
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
            ws=ws,
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

    def test_miss_on_moved_target_re_aims_instead_of_blocking(self) -> None:
        """A miss against a target that moved since the shot re-aims at the new position.

        When ``combat_feedback == "miss"`` and the target's current
        coordinates differ from the stored last-shot position, the enemy
        is NOT blocked.  Instead the bot fires at the target's fresh
        registry position -- the miss was ambiguous (the enemy may have
        stepped off the tile as the shot resolved), so abandoning a live
        mover is premature.
        """
        ws = WorldService()
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
            ws=ws,
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
        ws = WorldService()
        world, self_state = make_world(fuel=800)
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
        target = _enemy_threat(x=108, y=100)  # distance=8

        assert SHOT_RANGE_TILES == 8
        assert has_combat_shot(ctx, target) is True

    def test_has_combat_shot_returns_false_beyond_range(self) -> None:
        """A target beyond SHOT_RANGE_TILES distance is out of shot range."""
        ws = WorldService()
        world, self_state = make_world(fuel=800)
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
        target = _enemy_threat(x=109, y=100)  # distance=9

        assert has_combat_shot(ctx, target) is False


class TestFrameTargetShift:
    """The visibility law at the shoot chokepoint (flag s11-2, 2026-08-13).

    A shot only resolves inside the visible viewport, and the server
    refuses to homing-track an enemy close enough that a viewport
    shift would reveal them -- so a target within one anchor-law
    shift but outside the window gets the free framing shift, never
    the clamped ground-fire miss (artax fired six ``weapon=0``
    singles at the window edge while Arterial sat one row above it,
    returning 90/window).
    """

    def _ctx_with_target(self, tx: int, ty: int) -> tuple[DecideCtx, EnemyThreatDict]:
        """Build a ctx (self at (100,100), window (92,92)-(107,107)) and a target.

        Args:
            tx: Target X.
            ty: Target Y.

        Returns:
            Decision context and the matching enemy threat.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=tx,
                y=ty,
                team=2,
                rank=1,
                name="EdgeSitter",
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
        target = _enemy_threat(x=tx, y=ty, name="EdgeSitter")
        return ctx, target

    def test_one_row_off_window_target_is_framed_not_shot(self) -> None:
        """Flag s11-2 exactly: the enemy one row above the window gets
        the free framing shift, not the doomed edge-clamped single."""
        ctx, target = self._ctx_with_target(100, 91)

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "scope_shift"
        assert decision["command"]["direction"] == SCOPE_NORTH
        assert decision["behavior"]["reason_kind"] == "combat_frame_shift"
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_far_target_keeps_the_clamped_homing_snipe(self) -> None:
        """Beyond one shift's reach the server's seeker genuinely
        tracks, so the clamped snipe remains the right dispatch."""
        ctx, target = self._ctx_with_target(100, 80)

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 100
        assert decision["command"]["target_y"] == 92

    def test_stale_viewport_record_declines_the_shift(self) -> None:
        """A viewport record that excludes the bot is stale or not yet
        established (the origin arrives with the landing 0x5A) --
        framing against it would aim at garbage, same guard as the
        clamp."""
        ctx, target = self._ctx_with_target(100, 91)
        ctx.world["viewport"] = make_viewport_state(left=0, top=0, width=16, height=16)

        assert frame_target_shift(ctx, target) is None

    def test_unmovable_window_declines_the_shift(self) -> None:
        """A window the anchor law cannot move toward the target
        (already pinned to the tank on the approach axis) answers
        None instead of dispatching a no-op shift forever."""
        ctx, target = self._ctx_with_target(112, 100)
        # A narrowed 10-wide record whose left edge sits ON the tank:
        # an eastward shift pins left = tank_x, which IS the current
        # origin, so the anchored window equals the current window.
        ctx.world["viewport"] = make_viewport_state(left=100, top=92, width=10, height=16)

        assert frame_target_shift(ctx, target) is None


class TestFindCombatPickup:
    """Tests for _find_combat_pickup mid-combat pickup selection."""

    def test_finds_adjacent_fuel_when_low(self) -> None:
        """Returns fuel pickup when fuel is below threshold and fuel is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.state import make_container_state

        ws = WorldService()
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
            ws=ws,
        )

        result = _find_combat_pickup(ctx)
        if result is None:
            raise AssertionError("expected a pickup command")
        assert result["cmd_type"] == "pickup_fuel"

    def test_finds_adjacent_equipment(self) -> None:
        """Returns equipment pickup when adjacent and a slot is deficient."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup
        from tankpit_bot.state import make_container_state

        ws = WorldService()
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
            ws=ws,
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
            ws = WorldService()
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
                ws=ws,
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

        ws = WorldService()
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
            ws=ws,
        )

        assert _find_combat_pickup(ctx) is None

    def test_returns_none_when_no_adjacent(self) -> None:
        """Returns None when nothing is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import _find_combat_pickup

        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
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

        assert _find_combat_pickup(ctx) is None

    def test_combat_shoot_includes_secondary_pickup(self) -> None:
        """engage_target produces secondary pickup when container is adjacent."""
        from tankpit_bot.bot.ai.combat_strategy import engage_target
        from tankpit_bot.state import make_container_state

        ws = WorldService()
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
            ws=ws,
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

        ws = WorldService()
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
            ws=ws,
        )
        ctx.world = world_no_self
        assert _find_combat_pickup(ctx) is None
