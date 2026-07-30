"""Tests for the requested-vs-landed teleport displacement receipt."""

from __future__ import annotations

import logging

from tankpit_bot.container import TeleportLandedDict
from tankpit_bot.ledger.outcome.teleport import (
    record_teleport_dispatch,
    reset_teleport_dispatch_tracking,
)
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state import make_self_state

_DIAGNOSTIC_LINE = "DIAGNOSTIC: diagnostic_kind=teleport_displacement"


def _dispatch_landed_and_capture() -> list[logging.LogRecord]:
    """Dispatch a teleport_landed confirm and capture runtime events.

    Returns:
        All runtime-event log records emitted by the dispatch.
    """
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("tankpit_bot.runtime.events")
    handler = _Capture()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        dispatch_world_state_update(
            get_world_service(),
            TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C),
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
    return records


def _seed_self_at(x: int, y: int) -> None:
    """Seed the world's self position as the wire's SelfMovement would.

    Args:
        x: Self X at landed-confirm time.
        y: Self Y at landed-confirm time.
    """
    get_world_service().world_state["self_state"] = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )


class TestTeleportDisplacementReceipt:
    """Tests for ``_emit_teleport_displacement`` via the dispatch entry point."""

    def setup_method(self) -> None:
        """Reset world state and dispatch tracking before each test."""
        reset_world_state()
        reset_teleport_dispatch_tracking()

    def teardown_method(self) -> None:
        """Reset world state and dispatch tracking after each test."""
        reset_world_state()
        reset_teleport_dispatch_tracking()

    def test_displaced_landing_emits_the_receipt(self) -> None:
        """Landing off the requested tile emits teleport_displacement.

        Flag s2-7 (run bot-20260730-000030): the user watched teleports
        near the orange minefield get "put back to the safe location"
        with nothing in the events stream to prove it. The receipt
        carries requested, landed, and the Manhattan displacement so
        the analyzer can bucket routine combat-close displacement
        (aim at the enemy's own tile, land adjacent) apart from
        minefield ejections.
        """
        record_teleport_dispatch(target_x=235, target_y=5, message_index=0, sent_window="(none)")
        _seed_self_at(230, 10)

        records = _dispatch_landed_and_capture()

        displaced = [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]
        assert len(displaced) == 1
        record_dict: dict[str, str | int | float | bool | dict[str, str | int | float | bool]] = (
            displaced[0].__dict__
        )
        assert record_dict["runtime_fields"] == {
            "diagnostic_kind": "teleport_displacement",
            "requested_x": 235,
            "requested_y": 5,
            "landed_x": 230,
            "landed_y": 10,
            "displacement": 10,
        }
        world_lines = [
            r
            for r in records
            if r.getMessage() == "WORLD: TELEPORT_DISPLACED: requested (235,5) landed (230,10)"
        ]
        assert len(world_lines) == 1

    def test_exact_landing_stays_silent(self) -> None:
        """A landing on the requested tile emits no displacement receipt."""
        record_teleport_dispatch(target_x=230, target_y=10, message_index=0, sent_window="(none)")
        _seed_self_at(230, 10)

        records = _dispatch_landed_and_capture()

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]

    def test_no_pending_dispatch_stays_silent(self) -> None:
        """A landed confirm with no recorded dispatch emits no receipt."""
        _seed_self_at(230, 10)

        records = _dispatch_landed_and_capture()

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]

    def test_missing_self_state_stays_silent(self) -> None:
        """A landed confirm before any self sync emits no receipt."""
        record_teleport_dispatch(target_x=235, target_y=5, message_index=0, sent_window="(none)")

        records = _dispatch_landed_and_capture()

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]
