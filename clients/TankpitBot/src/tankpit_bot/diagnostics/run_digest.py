"""Compact per-run digest: the 40-line truth table for one session.

Born 2026-08-05 after a night of misread logs (kills double-counted by
tailing two mirrored files; a crashed run diagnosed by freestyle grep
over 193 MB of JSONL). The digest distills one events artifact into a
small pre-computed table — kills, deaths, shots, displacement
histogram, clearance-shot conversions, release reasons, account rank,
inventory arc, activity timeline — so a reader consumes computed
counts instead of re-deriving them ad hoc. It works from the events
stream alone, so a crashed run with no teardown scorecard still gets a
digest.

CLI: ``tankpit-run-digest [events.jsonl]`` (defaults to the latest bot
artifact) prints the table and writes ``<stem>.digest.json`` beside
the source.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot.diagnostics.event_stream import load_event_records, run_analyzer_cli
from tankpit_bot.diagnostics.run_digest_render import render_run_digest
from tankpit_bot.diagnostics.run_digest_types import (
    ClearanceShotRowDict,
    DisplacementRowDict,
    RunDigestDict,
    TimelineRowDict,
)
from tankpit_bot.protocol import RANK_NAMES
from tankpit_bot.runtime_records import RuntimeEventRecordDict

log = get_logger(__name__)

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
        # Archives before 2026-08-05 spell the countdown rank
        # "rank_points" (the mislabel the rename fixed); the artifact
        # is immutable so the reader takes either.
        rank_key = "rank_number" if "rank_number" in record["fields"] else "rank_points"
        digest["rank_number"] = _field_int(record, rank_key)
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


def build_run_digest(source_path: Path) -> RunDigestDict:
    """Distill one events artifact into the digest table.

    Args:
        source_path: JSONL events path.

    Returns:
        The computed digest.

    Raises:
        ValueError: If the artifact holds no events.
    """
    records = load_event_records(source_path)
    if not records:
        raise ValueError(f"no events in {source_path}")
    start_s = _ts_seconds(records[0]["timestamp"])

    digest = RunDigestDict(
        source=str(source_path),
        started_at=records[0]["timestamp"],
        ended_at=records[-1]["timestamp"],
        duration_s=int(_ts_seconds(records[-1]["timestamp"]) - start_s),
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
        rank_number=-1,
        promotion_points=-1,
        inventory_first=[],
        inventory_last=[],
        timeline=[],
    )
    displacement_counts: Counter[tuple[int, int]] = Counter()
    release_counts: Counter[str] = Counter()
    pending_clearance: list[tuple[float, ClearanceShotRowDict]] = []
    # Zero-yield radar tracking: a radar dispatch opens a window; a
    # container pickup dispatched before the next radar closes it as
    # productive. A window still open when the next radar fires (or
    # the session ends) was a scan that bought nothing collectible.
    radar_pending = False
    last_wire_s: float | None = None

    for record in records:
        t_s = _ts_seconds(record["timestamp"])
        message = record["message"]
        kind_field = record["fields"].get("diagnostic_kind")
        radar_pending = _track_radar_yield(str(kind_field), digest, radar_pending)
        if kind_field == "tank_deactivated":
            _apply_kill_receipt(record, digest, start_s, t_s)
        elif kind_field is not None:
            _apply_diagnostic(record, digest, displacement_counts, release_counts)
        elif record["fields"].get("behavior_reason") == "mine_clearance_shot":
            shot_row = ClearanceShotRowDict(
                timestamp=record["timestamp"],
                x=_field_int(record, "combat_target_x"),
                y=_field_int(record, "combat_target_y"),
                pickup_followed=False,
            )
            digest["clearance_shots"].append(shot_row)
            pending_clearance.append((t_s, shot_row))
        elif record["channel"] == "WIRE":
            _note_wire_gap(digest, last_wire_s, t_s)
            last_wire_s = t_s
            _apply_wire(digest, message, start_s, t_s, pending_clearance)

    if radar_pending:
        digest["zero_yield_radars"] += 1
    digest["displacement_top"] = [
        DisplacementRowDict(requested_x=x, requested_y=y, count=count)
        for (x, y), count in displacement_counts.most_common(5)
    ]
    digest["releases_by_reason"] = dict(release_counts)
    return digest


def build_and_persist_run_digest(source_path: Path) -> RunDigestDict:
    """Build the digest and persist its JSON beside the source artifact.

    The persisted ``<stem>.digest.json`` is the machine-readable twin
    of the rendered table, so later sessions read computed counts
    instead of re-grepping the raw events.

    Args:
        source_path: JSONL events path.

    Returns:
        The computed digest.
    """
    digest = build_run_digest(source_path)
    out_path = source_path.with_suffix("").with_suffix(".digest.json")
    out_path.write_text(dump_json_str(dict(digest), indent=1), encoding="utf-8")
    log.info("digest written: %s", out_path)
    return digest


def main() -> int:
    """Run the ``tankpit-run-digest`` CLI entrypoint.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_and_persist_run_digest, render_run_digest, log)


__all__ = [
    "build_and_persist_run_digest",
    "build_run_digest",
    "main",
]
