"""Tests for the requested-vs-landed teleport displacement receipt."""

from __future__ import annotations

import logging

from tankpit_bot.container import TeleportLandedDict
from tankpit_bot.ledger.outcome.teleport import record_teleport_dispatch
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state import make_self_state
from tests._runtime_logging_support import capture_runtime_events

_DIAGNOSTIC_LINE = "DIAGNOSTIC: diagnostic_kind=teleport_displacement"


def _dispatch_landed_and_capture(ws: WorldService) -> list[logging.LogRecord]:
    """Dispatch a teleport_landed confirm and capture runtime events.

    Args:
        ws: The world service holding the pending dispatch and self state.

    Returns:
        All runtime-event log records emitted by the dispatch.
    """
    with capture_runtime_events() as records:
        dispatch_world_state_update(
            ws,
            TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C),
        )
    return records


def _seed_self_at(ws: WorldService, x: int, y: int) -> None:
    """Seed the world's self position as the wire's SelfMovement would.

    Args:
        ws: The world service the self record belongs to.
        x: Self X at landed-confirm time.
        y: Self Y at landed-confirm time.
    """
    ws.world_state["self_state"] = make_self_state(
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
        ws = WorldService()
        record_teleport_dispatch(
            ws.ledger,
            target_x=235,
            target_y=5,
            message_index=0,
            sent_window="(none)",
        )
        _seed_self_at(ws, 230, 10)

        records = _dispatch_landed_and_capture(ws)

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
        ws = WorldService()
        record_teleport_dispatch(
            ws.ledger,
            target_x=230,
            target_y=10,
            message_index=0,
            sent_window="(none)",
        )
        _seed_self_at(ws, 230, 10)

        records = _dispatch_landed_and_capture(ws)

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]

    def test_no_pending_dispatch_stays_silent(self) -> None:
        """A landed confirm with no recorded dispatch emits no receipt."""
        ws = WorldService()
        _seed_self_at(ws, 230, 10)

        records = _dispatch_landed_and_capture(ws)

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]

    def test_missing_self_state_stays_silent(self) -> None:
        """A landed confirm before any self sync emits no receipt."""
        ws = WorldService()
        record_teleport_dispatch(
            ws.ledger,
            target_x=235,
            target_y=5,
            message_index=0,
            sent_window="(none)",
        )

        records = _dispatch_landed_and_capture(ws)

        assert not [r for r in records if r.getMessage() == _DIAGNOSTIC_LINE]


class TestFerryBeliefDisproof:
    """A displaced boarding landing expires the ferry belief it rode."""

    def test_displaced_landing_deletes_the_ferry_belief(self) -> None:
        """Flags s9-7/8: the loop root -- the stale belief must die.

        The hop teleported to a 60-second-old ferry tile, the server
        displaced the landing (its receipt that nothing boardable is
        there), and the surviving belief re-derived the identical
        boarding plan every lap -- a teleport plus a landing radar per
        lap, 17 extras to 0.
        """
        from tankpit_bot.ledger.outcome.teleport import record_teleport_dispatch
        from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        ws = WorldService()
        ws.update_world_state_from_position(118, 108)
        ws.world_state["terrain"]["111,104"] = make_terrain_tile(
            111, 104, TERRAIN_FERRY, observed_ms=100000
        )
        record_teleport_dispatch(
            ws.ledger, target_x=111, target_y=104, message_index=0, sent_window=""
        )

        landed = TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C)
        dispatch_world_state_update(ws, landed)

        assert "111,104" not in ws.world_state["terrain"]

    def test_exact_landing_keeps_the_ferry_belief(self) -> None:
        """Landing ON the requested ferry tile proves the belief right."""
        from tankpit_bot.ledger.outcome.teleport import record_teleport_dispatch
        from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
        from tankpit_bot.state.types import make_terrain_tile
        from tankpit_bot.types.constants import TERRAIN_FERRY

        ws = WorldService()
        ws.update_world_state_from_position(111, 104)
        ws.world_state["terrain"]["111,104"] = make_terrain_tile(
            111, 104, TERRAIN_FERRY, observed_ms=100000
        )
        record_teleport_dispatch(
            ws.ledger, target_x=111, target_y=104, message_index=0, sent_window=""
        )

        landed = TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C)
        dispatch_world_state_update(ws, landed)

        assert "111,104" in ws.world_state["terrain"]

    def test_displaced_landing_on_plain_ground_expires_nothing(self) -> None:
        """A displacement off ordinary ground touches no beliefs."""
        from tankpit_bot.ledger.outcome.teleport import record_teleport_dispatch
        from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
        from tankpit_bot.state.types import make_terrain_tile

        ws = WorldService()
        ws.update_world_state_from_position(118, 108)
        ws.world_state["terrain"]["111,104"] = make_terrain_tile(111, 104, 0, observed_ms=100000)
        record_teleport_dispatch(
            ws.ledger, target_x=111, target_y=104, message_index=0, sent_window=""
        )

        landed = TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C)
        dispatch_world_state_update(ws, landed)

        assert "111,104" in ws.world_state["terrain"]


class TestDisplacementEvidence:
    """The bounce receipt writes belief, not just a diagnostic."""

    def test_unexplained_one_tile_displacement_tombstones_the_tile(self) -> None:
        """One mystery displacement is enough: the tile is never re-aimed.

        Operator doctrine 2026-08-27 ("if we get displaced once then
        that should be enough info... why re-attempt unless we cleared
        mines"): session five re-aimed (18,123) four times because
        routine one-tile displacements fed no belief. With no known
        tank on the aimed tile, the displacement proves an invisible
        occupant — exactly one tile of evidence, no ring.
        """
        ws = WorldService()
        record_teleport_dispatch(
            ws.ledger,
            target_x=100,
            target_y=100,
            message_index=0,
            sent_window="(none)",
        )
        _seed_self_at(ws, 100, 101)

        _dispatch_landed_and_capture(ws)

        assert "100,100" in ws.displacement_tombstones
        keys = ws.hostile_landing_keys(ws.displacement_tombstones["100,100"] + 1)
        assert "100,100" in keys
        # Single-tile evidence: the neighbor is NOT blocked.
        assert "100,99" not in keys
        # And it ages out with the shared TTL, pruning the store.
        from tankpit_bot.sniffer.world_service_movement import _LANDING_REFUSAL_TTL_MS

        marked = ws.displacement_tombstones["100,100"]
        assert ws.hostile_landing_keys(marked + _LANDING_REFUSAL_TTL_MS) == frozenset()
        assert ws.displacement_tombstones == {}

    def test_one_tile_displacement_onto_a_tank_body_is_exempt(self) -> None:
        """Aiming at a tank's own tile displaces by one legitimately.

        Combat closes aim at the enemy's body and land adjacent every
        time — a known tank on the requested tile fully explains the
        displacement, so no tombstone may be written or every kill
        approach would poison its own target tile.
        """
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        ws.world_state["tanks"]["900"] = make_tank_state(
            tank_id=900,
            x=100,
            y=100,
            team=3,
            rank=1,
            name="red-1",
            is_self=False,
            is_bot=True,
            damage_state=3,
            timestamp_ms=1000,
            last_wire_seen_ms=1000,
            last_position_update_ms=1000,
            last_viewport_observation_ms=1000,
        )
        record_teleport_dispatch(
            ws.ledger,
            target_x=100,
            target_y=100,
            message_index=0,
            sent_window="(none)",
        )
        _seed_self_at(ws, 100, 101)

        _dispatch_landed_and_capture(ws)

        assert ws.displacement_tombstones == {}

    def test_displaced_landing_writes_landing_hostility_evidence(self) -> None:
        """A meaningful bounce records the requested zone as hostile.

        For four months the receipt fed nothing, so the identical hop
        could re-certify against mine-blind beliefs forever (the
        08-05 534-bounce session; the 2026-08-21 marooning). The
        evidence radius is the chebyshev the server proved it
        controls: (235,5) landed (230,10) is chebyshev 5.
        """
        ws = WorldService()
        record_teleport_dispatch(
            ws.ledger,
            target_x=235,
            target_y=5,
            message_index=0,
            sent_window="(none)",
        )
        _seed_self_at(ws, 230, 10)

        _dispatch_landed_and_capture(ws)

        assert "235,5" in ws.landing_refusals
        assert ws.landing_refusals["235,5"] > 0
