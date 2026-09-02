"""The per-record fold behind the run digest, made resumable.

Split out of :mod:`tankpit_bot.diagnostics.run_digest` 2026-09-01.
The reduction was always a fold -- one pass over records mutating a
digest plus four pieces of carry state -- but it was welded to
"read the whole file first", which made every consumer pay for the
entire run's history on every read.

That cost was invisible to the CLI (``make digest`` runs once on a
finished artifact) and crushing to the fleet control page, which polls
LIVE runs every second: a bot six minutes into a session has a 13 MB
events file, and re-reading plus re-decoding all of it twice per
second, per bot, is what made the dashboard take forever to reconnect.

:class:`RunDigestAccumulator` is the same arithmetic with the carry
state made explicit and durable, so a caller that has already folded
the first N records can fold record N+1 alone. Nothing here reads a
file; the accumulator never learns where its records came from beyond
the ``source`` label it is handed.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from datetime import datetime

from tankpit_bot.diagnostics.run_digest_types import (
    ClearanceShotRowDict,
    DisplacementRowDict,
    RunDigestDict,
    TimelineRowDict,
)
from tankpit_bot.protocol import RANK_NAMES
from tankpit_bot.runtime_records import RuntimeEventRecordDict

_TIMELINE_BUCKET_S = 300
_CLEARANCE_CONVERT_WINDOW_S = 10
#: A live tick loop dispatches every ~2 s; a silence this long between
#: consecutive WIRE dispatches is a stall the five-minute timeline
#: buckets smooth over (the 2026-08-20 arterial run idled 193 s inside
#: a 261 s session and the digest showed nothing).
_WIRE_GAP_STALL_S = 30

_SHOOT_WIRE = re.compile(r"^shoot\(")
_TELEPORT_WIRE = re.compile(r"^teleport\(")
_PICKUP_WIRE = re.compile(r"^pickup_(fuel|equipment)")


def _ts_seconds(timestamp: str) -> float:
    """Convert an event timestamp string to epoch seconds.

    Args:
        timestamp: ISO-format local timestamp from an event record.

    Returns:
        Epoch seconds.
    """
    return datetime.fromisoformat(timestamp).timestamp()


def _field_int(record: RuntimeEventRecordDict, key: str) -> int:
    """Read one structured field as an int.

    Args:
        record: Event record.
        key: Field name.

    Returns:
        The value as an int.

    Raises:
        ValueError: If the field is missing or not an int.
    """
    value = record["fields"].get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"event field {key!r} is not an int: {value!r}")
    return value


def _inventory_row(record: RuntimeEventRecordDict) -> list[int]:
    """Extract the five inventory counts from an ``inventory_sample``.

    Args:
        record: The sample event.

    Returns:
        ``[armor, dual, missile, homing, radar]``.
    """
    return [
        _field_int(record, "armor"),
        _field_int(record, "dual"),
        _field_int(record, "missile"),
        _field_int(record, "homing"),
        _field_int(record, "radar"),
    ]


def _bucket(rows: list[TimelineRowDict], start_s: float, t_s: float) -> TimelineRowDict:
    """Return the timeline bucket covering a timestamp, growing the list.

    Args:
        rows: Timeline accumulated so far.
        start_s: Session start epoch seconds.
        t_s: Event epoch seconds.

    Returns:
        The bucket row for the event's five-minute window.
    """
    index = max(0, int((t_s - start_s) // _TIMELINE_BUCKET_S))
    while len(rows) <= index:
        rows.append(
            TimelineRowDict(
                minute=len(rows) * _TIMELINE_BUCKET_S // 60,
                kills=0,
                shots=0,
                teleports=0,
                pickups=0,
            )
        )
    return rows[index]


def _track_radar_yield(kind: str, digest: RunDigestDict, radar_pending: bool) -> bool:
    """Advance the zero-yield radar window across one event.

    A radar dispatch opens a window; a container pickup before the
    next radar closes it as productive; a new radar while a window is
    open counts the superseded scan as zero-yield.

    Args:
        kind: The event's diagnostic kind (``"None"`` for none).
        digest: Digest under construction.
        radar_pending: Whether a radar window is currently open.

    Returns:
        The new window state.
    """
    if kind == "radar_dispatch":
        if radar_pending:
            digest["zero_yield_radars"] += 1
        return True
    if kind == "container_pickup_dispatched":
        return False
    return radar_pending


def _apply_combat_diagnostic(
    record: RuntimeEventRecordDict,
    digest: RunDigestDict,
    kind: str,
) -> None:
    """Fold one combat-accounting DIAGNOSTIC event into the digest.

    Args:
        record: The event record.
        digest: Digest under construction.
        kind: ``"action_outcome"`` or ``"damage_ledger"``.
    """
    if kind == "action_outcome":
        outcome = str(record["fields"].get("outcome", ""))
        if outcome == "hit":
            digest["hits"] += 1
        elif outcome == "miss":
            digest["misses"] += 1
        elif outcome == "superseded":
            # The wasted-tick split the live livelock detector streaks
            # on: an undispatched supersede is planner churn (the
            # decision never reached the wire); a dispatched one is a
            # re-aim on top of real output.
            if record["fields"].get("dispatched") is True:
                digest["superseded_dispatched"] += 1
            else:
                digest["superseded_undispatched"] += 1
        return
    # Teardown damage-ledger emission with fuel-confirmed totals
    # (2026-08-06); pre-extension archives lack the numeric fields
    # and stay at zero.
    dealt_value = record["fields"].get("dealt_fuel")
    taken_value = record["fields"].get("taken_fuel")
    if isinstance(dealt_value, int):
        digest["damage_dealt"] = dealt_value
    if isinstance(taken_value, int):
        digest["damage_taken"] = taken_value


def _apply_session_diagnostic(
    record: RuntimeEventRecordDict,
    digest: RunDigestDict,
    kind: str,
) -> None:
    """Fold one session-lifecycle DIAGNOSTIC into the digest.

    Args:
        record: The event record (carries ``diagnostic_kind``).
        digest: Digest under construction.
        kind: The record's ``diagnostic_kind``.
    """
    if kind == "session_room_joined":
        digest["room_id"] = str(record["fields"].get("room_id", ""))
    elif kind == "tank_identity" and digest["self_tank_id"] == -1:
        digest["self_tank_id"] = _field_int(record, "tank_id")
    elif kind == "session_account_stats":
        digest["rank_name"] = str(record["fields"].get("rank_name", ""))
        digest["leaderboard_position"] = _field_int(record, "leaderboard_position")
        digest["promotion_points"] = _field_int(record, "promotion_points")
    elif kind == "inventory_sample":
        inventory = _inventory_row(record)
        if not digest["inventory_first"]:
            digest["inventory_first"] = inventory
        digest["inventory_last"] = inventory
    elif kind == "session_scorecard":
        digest["clean_exit"] = True
        digest["exit_reason"] = str(record["fields"].get("exit_reason", ""))


def _apply_diagnostic(
    record: RuntimeEventRecordDict,
    digest: RunDigestDict,
    displacement_counts: Counter[tuple[int, int]],
    release_counts: Counter[str],
) -> None:
    """Fold one DIAGNOSTIC event into the digest.

    Args:
        record: The event record (carries ``diagnostic_kind``).
        digest: Digest under construction.
        displacement_counts: Per-request-tile displacement tallies.
        release_counts: Per-reason ``plan_released`` tallies.
    """
    kind = str(record["fields"].get("diagnostic_kind", ""))
    if kind == "teleport_displacement":
        digest["displacements"] += 1
        displacement_counts[
            (_field_int(record, "requested_x"), _field_int(record, "requested_y"))
        ] += 1
    elif kind == "plan_released":
        release_counts[str(record["fields"].get("reason", ""))] += 1
    elif kind == "self_promotion":
        new_rank = _field_int(record, "new_rank")
        was_promoted = record["fields"].get("was_promoted") is True
        rank_name = RANK_NAMES[new_rank] if 0 <= new_rank < len(RANK_NAMES) else str(new_rank)
        verb = "promoted to" if was_promoted else "demoted to"
        digest["rank_changes"].append(f"{verb} {rank_name} (rank {new_rank})")
    elif kind == "self_deactivated":
        # The one canonical death receipt. The 0x41 and fuel-wrap
        # producers dedup on ``ws.self_deactivated``, so exactly one
        # record lands per death. The old ``DEACTIVATED: tank=N
        # killed by M`` regex was dead code: no production path ever
        # logged that line with the self id, so deaths read 0 through
        # arterial's three 2026-08-26 main-map deaths.
        digest["deaths"] += 1
    elif kind == "liveness_stall":
        digest["liveness_stalls"] += 1
    elif kind in ("action_outcome", "damage_ledger"):
        _apply_combat_diagnostic(record, digest, kind)
    else:
        _apply_session_diagnostic(record, digest, kind)


def _apply_wire(
    digest: RunDigestDict,
    message: str,
    start_s: float,
    t_s: float,
    pending_clearance: list[tuple[float, ClearanceShotRowDict]],
) -> None:
    """Fold one WIRE dispatch into the digest.

    Args:
        digest: Digest under construction.
        message: The WIRE message text.
        start_s: Session start epoch seconds.
        t_s: Event epoch seconds.
        pending_clearance: Clearance shots awaiting a converting pickup.
    """
    if _SHOOT_WIRE.match(message):
        digest["shots"] += 1
        _bucket(digest["timeline"], start_s, t_s)["shots"] += 1
    elif _TELEPORT_WIRE.match(message):
        digest["teleports"] += 1
        _bucket(digest["timeline"], start_s, t_s)["teleports"] += 1
    elif _PICKUP_WIRE.match(message):
        digest["pickups"] += 1
        _bucket(digest["timeline"], start_s, t_s)["pickups"] += 1
        for shot_s, row in pending_clearance:
            if t_s - shot_s <= _CLEARANCE_CONVERT_WINDOW_S:
                row["pickup_followed"] = True
        pending_clearance.clear()


def _apply_kill_receipt(
    record: RuntimeEventRecordDict,
    digest: RunDigestDict,
    start_s: float,
    t_s: float,
) -> None:
    """Count one ``tank_deactivated`` receipt when the killer is us.

    OUR kill only when the 0x41 names this session's tank as the
    killer -- the scorecard's attribution rule. The old free-text
    "kill registered" count missed coordinate-aimed kills (44 wire
    kills vs 43 lines, arterial 2026-08-26). The ``-1`` unidentified
    sentinel never matches a ``-1`` killer_id from a pre-fleet
    artifact that lacked the field.

    Args:
        record: The ``tank_deactivated`` event record.
        digest: Digest under construction.
        start_s: Session start epoch seconds.
        t_s: Event epoch seconds.
    """
    killer = record["fields"].get("killer_id")
    if (
        digest["self_tank_id"] != -1
        and isinstance(killer, int)
        and killer == digest["self_tank_id"]
    ):
        digest["kills"] += 1
        _bucket(digest["timeline"], start_s, t_s)["kills"] += 1


def _note_wire_gap(digest: RunDigestDict, last_wire_s: float | None, t_s: float) -> None:
    """Fold one inter-dispatch silence into the wire-gap census.

    Args:
        digest: Digest under construction.
        last_wire_s: Previous WIRE dispatch epoch seconds, or None
            before the first dispatch.
        t_s: This dispatch's epoch seconds.
    """
    if last_wire_s is None:
        return
    gap = int(t_s - last_wire_s)
    digest["max_wire_gap_s"] = max(digest["max_wire_gap_s"], gap)
    if gap > _WIRE_GAP_STALL_S:
        digest["wire_gaps_over_30s"] += 1


def _empty_digest(source: str) -> RunDigestDict:
    """Return the zero state every fold starts from.

    Args:
        source: Label naming where the records came from.

    Returns:
        A digest with every counter at its unset value.
    """
    return RunDigestDict(
        source=source,
        started_at="",
        ended_at="",
        duration_s=0,
        clean_exit=False,
        exit_reason="",
        room_id="",
        self_tank_id=-1,
        kills=0,
        deaths=0,
        rank_changes=[],
        shots=0,
        hits=0,
        misses=0,
        zero_yield_radars=0,
        damage_dealt=0,
        damage_taken=0,
        teleports=0,
        pickups=0,
        displacements=0,
        displacement_top=[],
        clearance_shots=[],
        releases_by_reason={},
        liveness_stalls=0,
        superseded_undispatched=0,
        superseded_dispatched=0,
        max_wire_gap_s=0,
        wire_gaps_over_30s=0,
        rank_name="",
        leaderboard_position=-1,
        promotion_points=-1,
        inventory_first=[],
        inventory_last=[],
        timeline=[],
    )


class RunDigestAccumulator:
    """A run digest built one record at a time, resumable across reads.

    Absorb records in file order; snapshot whenever a caller wants the
    table as it stands. Absorbing more records afterwards continues
    from where the fold left off, so a live run costs one pass over
    the bytes that arrived since the last read rather than one pass
    over the whole run.

    Snapshots are independent: :meth:`snapshot` deep-copies, because
    the fold keeps mutating rows a previous snapshot would otherwise
    share (a clearance shot's ``pickup_followed`` flips when its
    converting pickup arrives, which can be several records later).
    """

    def __init__(self, source: str) -> None:
        """Start an empty fold.

        Args:
            source: Label naming where the records came from; copied
                verbatim into the digest's ``source`` field.
        """
        self._digest = _empty_digest(source)
        self._start_s: float | None = None
        self._displacement_counts: Counter[tuple[int, int]] = Counter()
        self._release_counts: Counter[str] = Counter()
        self._pending_clearance: list[tuple[float, ClearanceShotRowDict]] = []
        self._radar_pending = False
        self._last_wire_s: float | None = None

    def absorb(self, records: list[RuntimeEventRecordDict]) -> None:
        """Fold more records into the digest, in file order.

        Args:
            records: The next records of the same run, oldest first.
                Records from a DIFFERENT run must go to a different
                accumulator; nothing here detects a mixed stream.

        Returns:
            None.
        """
        for record in records:
            self._absorb_one(record)

    def _absorb_one(self, record: RuntimeEventRecordDict) -> None:
        """Fold exactly one record.

        Args:
            record: The next event record.

        Returns:
            None.
        """
        t_s = _ts_seconds(record["timestamp"])
        if self._start_s is None:
            self._start_s = t_s
            self._digest["started_at"] = record["timestamp"]
        start_s = self._start_s
        self._digest["ended_at"] = record["timestamp"]
        self._digest["duration_s"] = int(t_s - start_s)

        message = record["message"]
        kind_field = record["fields"].get("diagnostic_kind")
        self._radar_pending = _track_radar_yield(str(kind_field), self._digest, self._radar_pending)
        if kind_field == "tank_deactivated":
            _apply_kill_receipt(record, self._digest, start_s, t_s)
        elif kind_field is not None:
            _apply_diagnostic(record, self._digest, self._displacement_counts, self._release_counts)
        elif record["fields"].get("behavior_reason") == "mine_clearance_shot":
            shot_row = ClearanceShotRowDict(
                timestamp=record["timestamp"],
                x=_field_int(record, "combat_target_x"),
                y=_field_int(record, "combat_target_y"),
                pickup_followed=False,
            )
            self._digest["clearance_shots"].append(shot_row)
            self._pending_clearance.append((t_s, shot_row))
        elif record["channel"] == "WIRE":
            _note_wire_gap(self._digest, self._last_wire_s, t_s)
            self._last_wire_s = t_s
            _apply_wire(self._digest, message, start_s, t_s, self._pending_clearance)

    def snapshot(self) -> RunDigestDict:
        """Return the digest as it stands, without ending the fold.

        The three closing steps -- charging a still-open radar window,
        ranking the displacement histogram, and freezing the release
        tallies -- are applied to the COPY, so absorbing more records
        afterwards neither double-counts them nor sees them at all.

        Returns:
            An independent digest describing every record absorbed so
            far.
        """
        digest = deepcopy(self._digest)
        if self._radar_pending:
            digest["zero_yield_radars"] += 1
        digest["displacement_top"] = [
            DisplacementRowDict(requested_x=x, requested_y=y, count=count)
            for (x, y), count in self._displacement_counts.most_common(5)
        ]
        digest["releases_by_reason"] = dict(self._release_counts)
        return digest


__all__ = [
    "RunDigestAccumulator",
]
