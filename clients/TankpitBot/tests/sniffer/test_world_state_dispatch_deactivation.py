"""Tests for protocol-path 0x41 Deactivation dispatch.

Split out of ``test_world_state_dispatch_container.py`` 2026-08-26
when the wrap-dedup test pushed that file over the 600-line ceiling.
0x41 moved out of container into the protocol layer 2026-06-19;
``dispatch_world_state_update`` routes the integer msg_type 0x41.
"""

from __future__ import annotations

import logging

import pytest

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestDispatchProtocolDeactivation:
    """Tests for protocol-path 0x41 Deactivation dispatch."""

    def test_dispatch_deactivation_marks_liveness_deactivated(self) -> None:
        """Dispatch 0x41 marks the victim ``liveness="deactivated"`` and
        preserves the death tile.

        Replaces the prior ``position-set-to-(0,0)`` sentinel with the
        explicit liveness state machine introduced 2026-06-20.
        """
        from tankpit_bot.protocol import DeactivationDict, TankEntryDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=900, rank=0, damage_state=0, score=0, x=100, y=100
        )
        dispatch_world_state_update(ws, entry)

        msg = DeactivationDict(
            msg_type=0x41,
            status=0,
            victim_id=900,
            promo_eligible=True,
            killer_id=1,
            is_mine_kill=False,
        )
        dispatch_world_state_update(ws, msg)

        tank = ws.world_state["tanks"]["900"]
        assert tank["liveness"] == "deactivated"
        assert tank["x"] == 100
        assert tank["y"] == 100

    def test_own_0x41_after_the_wrap_receipt_books_one_death(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The wrap lands first in the drain batch; the 0x41 dedups.

        Arterial's 2026-08-26 18:37:41 death: the u16 fuel-wrap
        reading arrived one message BEFORE the 0x41 self-receipt.
        Both producers raise the same ``self_deactivated`` flag, so
        whichever lands second must book nothing — one death, one
        receipt, or every reporting consumer double-counts.
        """
        from tankpit_bot.protocol import DeactivationDict
        from tankpit_bot.sniffer.world_state_containers import (
            update_world_state_from_fuel_total,
        )

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(ws, 14)
        update_world_state_from_fuel_total(ws, 65460)
        assert ws.self_deactivated is True
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")

        msg = DeactivationDict(
            msg_type=0x41,
            status=1,
            victim_id=self_state["tank_id"],
            promo_eligible=False,
            killer_id=719,
            is_mine_kill=False,
        )
        with caplog.at_level(logging.INFO):
            dispatch_world_state_update(ws, msg)

        assert ws.self_deactivated is True
        assert "SELF DEACTIVATED" not in caplog.text
