"""Kill attribution and channel exclusivity for the scorecard's kill count.

Split from ``test_session_scorecard_accumulator.py`` (2026-08-20, at
the file-size bar) when killer attribution landed: routing, fuel,
teleport-spend and equipment tests stay there; everything about WHOSE
kill a ``tank_deactivated`` is lives here.
"""

from __future__ import annotations

from tests.diagnostics._scorecard_fixtures import _record, _routed


def test_a_kill_diagnostic_on_another_channel_is_not_counted() -> None:
    """A record's CHANNEL decides its router, not the fields it carries.

    The channel cascade ends with an explicit ``!= "DIAGNOSTIC"`` arm,
    and it is the only thing enforcing that. Without it every record
    that is not STATE, WORLD, or a WIRE shot is handed to the diagnostic
    router -- 925,744 AI, SYNC, WIRE and WIRE_COMPLETE records across the
    427 archived runs -- and stays harmless only because all ~20 routing
    branches happen to be gated on a ``diagnostic_kind`` literal that
    those records do not carry.

    Routing all 1,403,706 archived records with the arm removed produces
    a byte-identical accumulator, so no current emitter puts a
    ``diagnostic_kind`` on another channel and the record below is
    constructed rather than observed. The arm is what keeps that true:
    one ``emit_ai(diagnostic_kind=...)`` would otherwise inflate the kill
    count of every scorecard, silently and in the bot's favour.
    """
    accumulator = _routed([_record(channel="AI", fields={"diagnostic_kind": "tank_deactivated"})])

    assert accumulator["kills"] == 0


def test_control_the_same_kill_diagnostic_on_its_own_channel_counts() -> None:
    """Control: on the DIAGNOSTIC channel that exact record IS a kill."""
    accumulator = _routed(
        [
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_identity", "tank_id": 7},
            ),
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_deactivated", "victim_id": 529, "killer_id": 7},
            ),
        ]
    )

    assert accumulator["kills"] == 1


def test_kills_count_only_deactivations_our_own_tank_killed() -> None:
    """The 0x41 stream is killer-attributed; only OUR kills count.

    The pre-fleet counter took the raw ``tank_deactivated`` count --
    correct while every 0x41 in view was our own kill, falsified the
    day two bots shared a room (2026-08-14 live registry split) and
    caught in this counter on the first gatherer run (2026-08-20:
    arterial's issue report read kills=2 with shots=0, both 0x41s
    naming artax as killer). Sibling kills, pre-identity events, and
    pre-fleet artifacts without a ``killer_id`` field never count.
    """
    accumulator = _routed(
        [
            # Before any identity: unattributable, never counted.
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_deactivated", "victim_id": 11, "killer_id": 7},
            ),
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_identity", "tank_id": 7},
            ),
            # Our own kill: counts.
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_deactivated", "victim_id": 529, "killer_id": 7},
            ),
            # A fleet sibling's kill in the same room: never ours.
            _record(
                channel="DIAGNOSTIC",
                fields={
                    "diagnostic_kind": "tank_deactivated",
                    "victim_id": 530,
                    "killer_id": 1301,
                },
            ),
            # Pre-fleet artifact shape (no killer_id field): unattributable.
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_deactivated", "victim_id": 531},
            ),
        ]
    )

    assert accumulator["self_tank_id"] == 7
    assert accumulator["kills"] == 1


def test_first_tank_identity_wins_over_later_ones() -> None:
    """A later ``tank_identity`` never rewrites the session's self id.

    Mirrors the run digest's first-wins law: the wire names this
    session's own tank at entry; anything later claiming otherwise
    must not silently re-attribute every kill counted so far.
    """
    accumulator = _routed(
        [
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_identity", "tank_id": 7},
            ),
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_identity", "tank_id": 1301},
            ),
            _record(
                channel="DIAGNOSTIC",
                fields={"diagnostic_kind": "tank_deactivated", "victim_id": 529, "killer_id": 7},
            ),
        ]
    )

    assert accumulator["self_tank_id"] == 7
    assert accumulator["kills"] == 1
