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

    def test_own_0x41_halves_the_ammo_book_baseline(self) -> None:
        """A tank-kill death sets every book slot to ceil(n/2).

        Wire-verified across all six corpus deaths ([[equipment-system]]);
        without the transform every death's next 0x49 snapshot read as
        an infeasible fall and burned a false ammo divergence (three
        deaths, three divergences, desert 2026-08-26). The penalty
        applies even on the deduped path — the wrap receipt carries no
        mine sentinel, so the 0x41 owns the book transform.
        """
        from tankpit_bot.protocol import DeactivationDict
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        update_inventory_from_protocol(ws, [45, 9, 45, 37, 24], [True, True, True, True, True])
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
        dispatch_world_state_update(ws, msg)

        assert ws.ammo_book["last_counts"] == [23, 5, 23, 19, 12]

    def test_own_mine_0x41_zeroes_the_ammo_book_baseline(self) -> None:
        """The mine sentinel wipes the whole baseline, and the
        ``self_deactivated`` diagnostic carries the attribution."""
        from tankpit_bot.protocol import DeactivationDict
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol
        from tests._runtime_logging_support import capture_runtime_events

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        update_inventory_from_protocol(ws, [35, 35, 35, 35, 29], [True, True, True, True, True])
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")

        msg = DeactivationDict(
            msg_type=0x41,
            status=1,
            victim_id=self_state["tank_id"],
            promo_eligible=False,
            killer_id=3,
            is_mine_kill=True,
        )
        with capture_runtime_events() as records:
            dispatch_world_state_update(ws, msg)

        assert ws.ammo_book["last_counts"] == [0, 0, 0, 0, 0]
        deaths = [
            r for r in records if r.getMessage() == "DIAGNOSTIC: diagnostic_kind=self_deactivated"
        ]
        assert len(deaths) == 1
        record_dict: dict[str, str | int | float | bool | dict[str, str | int | float | bool]] = (
            deaths[0].__dict__
        )
        assert record_dict["runtime_fields"] == {
            "diagnostic_kind": "self_deactivated",
            "origin": "protocol_0x41",
            "killer_id": 3,
            "is_mine_kill": True,
        }
