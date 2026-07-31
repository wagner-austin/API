"""Integration test: HUNT acquires an enemy after wire-confirmation.

Drives the protocol dispatch chain end-to-end:
  1. 0x3D MovementResponse for the bot's own tank establishes self_state.
  2. 0x21 TankInfo for an enemy creates the tank registry entry (no
     position, no wire timestamp).
  3. 0x3D MovementResponse for the enemy attaches wire-confirmed
     position and advances ``last_wire_seen_ms``.
  4. ``analyze_threats`` returns the enemy as an acquisition candidate.

Wire bytes mirror the practice-vs-real capture 2026-06-20: Yuppler
(tank 1229, purple team) MovementResponse at (131, 124) while Artax
(tank 1301, blue team) is at (131, 122).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.protocol import MovementResponseDict, TankInfoDict
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestHuntAcquiresWireConfirmedEnemy:
    """Integration test for HUNT acquisition of a wire-confirmed enemy."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_enemy_visible_in_threat_list_after_wire_confirmation(self) -> None:
        """Wire-confirmed enemy MUST surface in analyze_threats."""
        now_ms = 100_000
        ws = get_world_service()

        # Self (Artax / blue team) at (131, 122)
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
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse for self")

        # Enemy (Yuppler / purple team) TankInfo (registry only, no wire ts).
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

        # Enemy MovementResponse: position + wire timestamp.
        ws.world_state["timestamp_ms"] = now_ms
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

        threats = analyze_threats(ws.world_state, self_state, now_ms=now_ms)

        assert len(threats) == 1, "Wire-confirmed enemy must appear in threat list"
        threat = threats[0]
        assert threat["tank_id"] == 1229
        assert threat["x"] == 131
        assert threat["y"] == 124
        assert threat["team"] == 1
        assert threat["distance"] == 2  # |131-131| + |124-122| = 2

    def test_unsynced_tank_info_alone_does_not_surface_as_threat(self) -> None:
        """TankInfo without a wire-position event must NOT acquire.

        TankInfo establishes the registry entry (name, team) but does
        not advance ``last_wire_seen_ms``. The wire-presence gate in
        ``analyze_threats`` must reject the tank until a MovementResponse
        or other wire message arrives.
        """
        now_ms = 100_000
        ws = get_world_service()

        # Self at (131, 122)
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
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after MovementResponse for self")

        # Enemy registry entry only -- no wire position.
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

        threats = analyze_threats(ws.world_state, self_state, now_ms=now_ms)
        assert threats == []
