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
from typing_extensions import TypedDict

from tankpit_bot.diagnostics.event_stream import load_event_records, run_analyzer_cli
from tankpit_bot.runtime_logging import RuntimeEventRecordDict

log = get_logger(__name__)

_TIMELINE_BUCKET_S = 300
_CLEARANCE_CONVERT_WINDOW_S = 10

_SHOOT_WIRE = re.compile(r"^shoot\(")
_TELEPORT_WIRE = re.compile(r"^teleport\(")
_PICKUP_WIRE = re.compile(r"^pickup_(fuel|equipment)")
_DEACTIVATED = re.compile(r"^DEACTIVATED: tank=(\d+) killed by (\d+)")


class DisplacementRowDict(TypedDict):
    """One repeated-displacement histogram row.

    Attributes:
        requested_x: Aimed landing X.
        requested_y: Aimed landing Y.
        count: How many teleports at this tile displaced.
    """

    requested_x: int
    requested_y: int
    count: int


class ClearanceShotRowDict(TypedDict):
    """One mine-clearance shot and whether it converted.

    Attributes:
        timestamp: Shot decision timestamp.
        x: Aim tile X.
        y: Aim tile Y.
        pickup_followed: A pickup dispatched within the convert window.
    """

    timestamp: str
    x: int
    y: int
    pickup_followed: bool


class TimelineRowDict(TypedDict):
    """Activity counts for one five-minute bucket.

    Attributes:
        minute: Bucket start offset from session start, in minutes.
        kills: Kills registered in the bucket.
        shots: Shoot dispatches in the bucket.
        teleports: Teleport dispatches in the bucket.
        pickups: Pickup dispatches in the bucket.
    """

    minute: int
    kills: int
    shots: int
    teleports: int
    pickups: int


class RunDigestDict(TypedDict):
    """The whole-run digest table.

    Attributes:
        source: Events artifact the digest was computed from.
        started_at: First event timestamp.
        ended_at: Last event timestamp.
        duration_s: Wall seconds between first and last event.
        clean_exit: A session scorecard event was present (teardown ran).
        exit_reason: Scorecard exit reason, empty when the run crashed.
        room_id: Last joined room.
        self_tank_id: Wire id of our own tank (-1 when never identified).
        kills: Kill-registered count.
        deaths: Own deactivations observed.
        shots: Shoot dispatches.
        teleports: Teleport dispatches.
        pickups: Pickup dispatches.
        displacements: Total displaced teleports.
        displacement_top: Most-displaced request tiles, descending.
        clearance_shots: Every mine-clearance shot with conversion.
        releases_by_reason: ``plan_released`` reason counts.
        rank_name: Account rank label at startup, empty when unscraped.
        rank_number: Countdown rank number at startup (-1 unscraped).
        promotion_points: Account promotion points (-1 unscraped).
        inventory_first: First sampled (armor,dual,missile,homing,radar).
        inventory_last: Last sampled (armor,dual,missile,homing,radar).
        timeline: Five-minute activity buckets.
    """

    source: str
    started_at: str
    ended_at: str
    duration_s: int
    clean_exit: bool
    exit_reason: str
    room_id: str
    self_tank_id: int
    kills: int
    deaths: int
    shots: int
    teleports: int
    pickups: int
    displacements: int
    displacement_top: list[DisplacementRowDict]
    clearance_shots: list[ClearanceShotRowDict]
    releases_by_reason: dict[str, int]
    rank_name: str
    rank_number: int
    promotion_points: int
    inventory_first: list[int]
    inventory_last: list[int]
    timeline: list[TimelineRowDict]


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
    kind = record["fields"].get("diagnostic_kind")
    if kind == "session_room_joined":
        digest["room_id"] = str(record["fields"].get("room_id", ""))
    elif kind == "tank_identity" and digest["self_tank_id"] == -1:
        digest["self_tank_id"] = _field_int(record, "tank_id")
    elif kind == "teleport_displacement":
        digest["displacements"] += 1
        displacement_counts[
            (_field_int(record, "requested_x"), _field_int(record, "requested_y"))
        ] += 1
    elif kind == "plan_released":
        release_counts[str(record["fields"].get("reason", ""))] += 1
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
        shots=0,
        teleports=0,
        pickups=0,
        displacements=0,
        displacement_top=[],
        clearance_shots=[],
        releases_by_reason={},
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

    for record in records:
        t_s = _ts_seconds(record["timestamp"])
        message = record["message"]
        if record["fields"].get("diagnostic_kind") is not None:
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
            _apply_wire(digest, message, start_s, t_s, pending_clearance)
        elif "kill registered" in message:
            digest["kills"] += 1
            _bucket(digest["timeline"], start_s, t_s)["kills"] += 1
        else:
            deactivated = _DEACTIVATED.match(message)
            if deactivated and int(deactivated.group(1)) == digest["self_tank_id"]:
                digest["deaths"] += 1

    digest["displacement_top"] = [
        DisplacementRowDict(requested_x=x, requested_y=y, count=count)
        for (x, y), count in displacement_counts.most_common(5)
    ]
    digest["releases_by_reason"] = dict(release_counts)
    return digest


def render_run_digest(digest: RunDigestDict) -> str:
    """Render the digest as the aligned human table.

    Args:
        digest: Computed digest.

    Returns:
        Multi-line table text.
    """
    exit_line = (
        f"CLEAN {digest['exit_reason']}"
        if digest["clean_exit"]
        else "CRASHED (no teardown scorecard)"
    )
    lines = [
        "=== RUN DIGEST ===",
        f"source     {digest['source']}",
        f"window     {digest['started_at']} .. {digest['ended_at']}"
        f"  ({digest['duration_s'] // 60}m{digest['duration_s'] % 60:02d}s)",
        f"exit       {exit_line}",
        f"room       {digest['room_id']}   self tank id {digest['self_tank_id']}",
        f"combat     kills={digest['kills']} deaths={digest['deaths']} shots={digest['shots']}",
        f"movement   teleports={digest['teleports']} displaced={digest['displacements']}"
        f" pickups={digest['pickups']}",
    ]
    if digest["rank_number"] != -1:
        lines.append(
            f"account    rank={digest['rank_name']} ({digest['rank_number']})"
            f" promo={digest['promotion_points']}"
        )
    if digest["inventory_first"]:
        lines.append(
            f"inventory  first={digest['inventory_first']} last={digest['inventory_last']}"
            " (armor,dual,missile,homing,radar)"
        )
    for row in digest["displacement_top"]:
        lines.append(f"displaced  ({row['requested_x']},{row['requested_y']}) x{row['count']}")
    for shot in digest["clearance_shots"]:
        outcome = "converted" if shot["pickup_followed"] else "no pickup followed"
        lines.append(f"clearance  {shot['timestamp']} ({shot['x']},{shot['y']}) {outcome}")
    for reason, count in sorted(digest["releases_by_reason"].items()):
        lines.append(f"release    {reason} x{count}")
    lines.append("timeline   min: kills/shots/teleports/pickups")
    for bucket in digest["timeline"]:
        lines.append(
            f"           {bucket['minute']:>4}: {bucket['kills']}/{bucket['shots']}"
            f"/{bucket['teleports']}/{bucket['pickups']}"
        )
    return "\n".join(lines)


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
    "ClearanceShotRowDict",
    "DisplacementRowDict",
    "RunDigestDict",
    "TimelineRowDict",
    "build_and_persist_run_digest",
    "build_run_digest",
    "main",
    "render_run_digest",
]
