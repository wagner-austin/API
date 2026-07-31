"""Integration tests: combat kill-gate fires or blocks based on wire state.

Drives the dispatch chain to populate world state, then asks the combat
strategy to engage. Asserts the wire-presence and position-fresh gates
behave correctly:

  - When both ``last_wire_seen_ms`` and ``last_position_update_ms`` are
    within their TTLs, ``engage_target`` produces a ``shoot`` command.
  - When ``last_wire_seen_ms`` is stale (older than
    ``WIRE_PRESENCE_TTL_MS``), engagement blocks the target without
    firing and emits the ``combat_ghost_detected`` diagnostic.
  - When ``last_position_update_ms`` is stale (status-only broadcasts
    kept wire fresh but no position arrived), engagement blocks the
    target without firing.

The 2026-06-19 stale-registry combat-miss loop is the regression these
gates prevent. Do not modify combat_strategy logic in these tests --
they assert against the current behavior so refactors stay safe.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import engage_target
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.threats import (
    POSITION_FRESHNESS_TTL_MS,
    WIRE_PRESENCE_TTL_MS,
    analyze_threats,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.protocol import MovementResponseDict, TankInfoDict
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tests.bot.ai._support import make_inventory, make_scanned_ai_state


def _seed_self_and_enemy() -> int:
    """Push real wire messages into world state for self and an enemy.

    Positions mirror the practice-vs-real 2026-06-20 capture: Artax
    (tank 1301, blue team) at (131, 122) and Yuppler (tank 1229,
    purple team) at (131, 124) — adjacent shot range.

    Returns the wall-clock timestamp the dispatcher stamped on the
    enemy's freshness fields; the caller uses it to compute
    fresh/stale clock offsets relative to that real timestamp instead
    of a synthetic one. This keeps the test honest to the production
    dispatch path (which always uses the wall clock).
    """
    ws = get_world_service()

    dispatch_world_state_update(
        ws,
        MovementResponseDict(
            msg_type=0x3D,
            team=2,
            tank_id=1301,
            x=131,
            y=122,
            direction=0,
            damage_state=0,
            rank=1,
            lb_score=72,
            carrying=0,
        ),
    )
    dispatch_world_state_update(
        ws,
        TankInfoDict(
            msg_type=0x21,
            tank_id=1229,
            team=1,
            name="red-77",
            decoration_state=b"",
            persistent_tank_id=0,
        ),
    )
    dispatch_world_state_update(
        ws,
        MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=1229,
            x=131,
            y=124,
            direction=0,
            damage_state=0,
            rank=1,
            lb_score=107,
            carrying=0,
        ),
    )
    return ws.world_state["tanks"]["1229"]["last_wire_seen_ms"]


class TestCombatGates:
    """Integration tests for the kill-gate wire-presence checks."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_combat_fires_when_both_gates_pass(self) -> None:
        """Wire-fresh + position-fresh enemy at shot range -> shoot command."""
        seed_ts = _seed_self_and_enemy()
        ws = get_world_service()
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse")

        threats = analyze_threats(ws.world_state, self_state, now_ms=seed_ts)
        assert len(threats) == 1, "Wire-fresh enemy must surface as a threat"
        target = threats[0]
        assert target["tank_id"] == 1229

        ctx = DecideCtx(
            ws.world_state,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            seed_ts,
            None,
            "",
        )

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        shoot_cmd = decision["command"]
        assert shoot_cmd["target_x"] == 131
        assert shoot_cmd["target_y"] == 124

    def test_combat_fires_when_wire_presence_stale(self) -> None:
        """Wire stale beyond TTL -> still shoot (no block).

        Pre-2026-06-23 the wire-presence gate in ``_combat_shoot``
        blocked any target whose ``last_wire_seen_ms`` exceeded
        ``WIRE_PRESENCE_TTL_MS`` (7000ms). That gate killed pursuit
        shots when a locked target teleported off the bot's viewport:
        the server only emits wire events for tanks the local
        viewport can see, so an off-viewport target naturally goes
        wire-silent. Live run 2026-06-23 19:31:43 saw the bot engage
        purple-8, fire two homing pursuits, then block the target at
        wire-age 8224ms despite an active combat lock and a live
        target.

        The gate was removed 2026-06-23; the lock now holds until an
        authoritative deactivation signal arrives. This test guards
        against re-introduction.
        """
        seed_ts = _seed_self_and_enemy()
        ws = get_world_service()
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse")

        threats = analyze_threats(ws.world_state, self_state, now_ms=seed_ts)
        assert len(threats) == 1
        target = threats[0]

        stale_now_ms = seed_ts + WIRE_PRESENCE_TTL_MS + 1
        ctx = DecideCtx(
            ws.world_state,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            stale_now_ms,
            None,
            "",
        )

        decision = engage_target(ctx, target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert "1229" not in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_combat_fires_at_stationary_target_without_position_refresh(self) -> None:
        """Wire-fresh target stays fireable even when position hasn't refreshed.

        Practice-room bots don't move; their wire activity is
        status-only broadcasts (0x2E) that refresh
        ``last_wire_seen_ms`` but carry no position. The position
        freshness gate was removed 2026-06-22 because viewport
        presence (in ``analyze_threats``) already proves the target
        is at the registry position right now -- the extra gate
        was over-restricting and blocking kills on stationary bots.
        """
        seed_ts = _seed_self_and_enemy()
        ws = get_world_service()
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse")

        threats = analyze_threats(ws.world_state, self_state, now_ms=seed_ts)
        target = threats[0]

        # Build a wire-fresh target whose position update went stale
        # (status broadcasts kept the wire stamp current while the
        # tank sat still without moving).
        stationary_target = EnemyThreatDict(
            tank_id=target["tank_id"],
            x=target["x"],
            y=target["y"],
            distance=target["distance"],
            damage_state=target["damage_state"],
            rank=target["rank"],
            team=target["team"],
            name=target["name"],
            is_bot=target["is_bot"],
            timestamp_ms=target["timestamp_ms"],
            last_wire_seen_ms=seed_ts,
            last_position_update_ms=seed_ts - POSITION_FRESHNESS_TTL_MS - 1,
            last_aim_x=target["last_aim_x"],
            last_aim_y=target["last_aim_y"],
            last_aim_weapon=target["last_aim_weapon"],
            last_aim_ms=target["last_aim_ms"],
        )

        ctx = DecideCtx(
            ws.world_state,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            seed_ts,
            None,
            "",
        )

        decision = engage_target(ctx, stationary_target)

        assert decision["command"]["cmd_type"] == "shoot"
        assert "1229" not in decision["updated_ai_state"]["blocked_combat_targets"]
