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
            name="Yuppler",
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

    def test_combat_blocks_when_wire_presence_stale(self) -> None:
        """Wire stale beyond TTL -> block target and replan (no shoot)."""
        seed_ts = _seed_self_and_enemy()
        ws = get_world_service()
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse")

        threats = analyze_threats(ws.world_state, self_state, now_ms=seed_ts)
        assert len(threats) == 1
        target = threats[0]

        # Advance the decision clock past the wire-presence TTL relative
        # to the dispatcher's actual timestamp so engage_target treats
        # the target as wire-silent.
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

        assert decision["command"]["cmd_type"] != "shoot"
        assert "1229" in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_combat_blocks_when_position_stale(self) -> None:
        """Wire fresh, position stale -> block target without firing.

        Regression for the 2026-06-19 stale-registry combat-miss loop:
        status-only broadcasts (0x2E damage syncs) kept ``last_wire_seen_ms``
        fresh while no position-bearing message arrived; the bot
        otherwise fired at the stale registry position.
        """
        seed_ts = _seed_self_and_enemy()
        ws = get_world_service()
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse")

        threats = analyze_threats(ws.world_state, self_state, now_ms=seed_ts)
        target = threats[0]

        # Construct a stale-position threat: wire just refreshed (status
        # sync), position older than POSITION_FRESHNESS_TTL_MS.
        stale_position_target = EnemyThreatDict(
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

        decision = engage_target(ctx, stale_position_target)

        assert decision["command"]["cmd_type"] != "shoot"
        assert "1229" in decision["updated_ai_state"]["blocked_combat_targets"]
