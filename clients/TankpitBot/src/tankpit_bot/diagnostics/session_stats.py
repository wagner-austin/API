"""Cross-session bot statistics over every events artifact in a runs dir.

Every bot run writes a timestamped ``bot-YYYYMMDD-HHMMSS.events.jsonl``
artifact that is never overwritten, so the full session history is on
disk -- but every other analyzer reads exactly one artifact. This CLI
(``tankpit-stats``) sweeps the whole directory and renders one row per
run (kills, teleport outcomes, shots, pickups, stalls) plus an
aggregate total, turning the per-run artifacts into longitudinal
data: regressions show up as a row that breaks the trend.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.session_stats_types import (
    SessionStatsReportDict,
    SessionStatsRowDict,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict

log = get_logger(__name__)

_EVENTS_SUFFIX = ".events.jsonl"
_LANDED_STATUSES = ("landed_exact", "landed_inexact")


def _elapsed_seconds(first_timestamp: str, last_timestamp: str) -> int:
    """Return whole seconds between two artifact timestamps.

    Args:
        first_timestamp: ISO timestamp of the first event.
        last_timestamp: ISO timestamp of the last event.

    Returns:
        Non-negative whole seconds between the two timestamps.
    """
    first = datetime.fromisoformat(first_timestamp)
    last = datetime.fromisoformat(last_timestamp)
    return max(0, int((last - first).total_seconds()))


def _is_own_kill(record: RuntimeEventRecordDict) -> bool:
    """Return True for a wire-decoded 0x41 own-kill diagnostic."""
    fields = record["fields"]
    return (
        fields.get("diagnostic_kind") == "tank_deactivated"
        and fields.get("origin") == "protocol_0x41"
    )


def _teleport_landed_verdict(record: RuntimeEventRecordDict) -> bool | None:
    """Classify a record as a landed/failed teleport resolution, if any.

    Covers both sources: action-lab ``teleport_attempt`` diagnostics
    (``status`` field) and the bot ledger's ``action_outcome`` events
    with ``action_kind == "teleport"`` (``outcome`` field).

    Args:
        record: Decoded event record.

    Returns:
        True for a landed teleport, False for a failed one, or None
        when the record is not a teleport resolution at all.
    """
    fields = record["fields"]
    kind = fields.get("diagnostic_kind")
    if kind == "teleport_attempt":
        return fields.get("status") in _LANDED_STATUSES
    if kind == "action_outcome" and fields.get("action_kind") == "teleport":
        return fields.get("outcome") in _LANDED_STATUSES
    return None


def _build_row(run_id: str, records: list[RuntimeEventRecordDict]) -> SessionStatsRowDict:
    """Build one stats row from a run's decoded event records.

    Args:
        run_id: Artifact identifier derived from the filename.
        records: Decoded event records in file order.

    Returns:
        Per-run statistics row.
    """
    kills = 0
    teleports_ok = 0
    teleports_failed = 0
    shots = 0
    pickups = 0
    stalls = 0
    for record in records:
        fields = record["fields"]
        kind = fields.get("diagnostic_kind")
        if _is_own_kill(record):
            kills += 1
        elif (landed := _teleport_landed_verdict(record)) is not None:
            if landed:
                teleports_ok += 1
            else:
                teleports_failed += 1
        if record["channel"] == "WIRE":
            if record["message"].startswith("shoot"):
                shots += 1
            elif record["message"].startswith("pickup"):
                pickups += 1
        if kind == "action_outcome" and fields.get("outcome") == "stall_timeout":
            stalls += 1
    started = records[0]["timestamp"] if records else ""
    ended = records[-1]["timestamp"] if records else ""
    duration_s = _elapsed_seconds(started, ended) if records else 0
    return SessionStatsRowDict(
        run_id=run_id,
        started=started,
        duration_s=duration_s,
        events=len(records),
        kills=kills,
        teleports_ok=teleports_ok,
        teleports_failed=teleports_failed,
        shots=shots,
        pickups=pickups,
        stalls=stalls,
    )


def _totals_row(rows: list[SessionStatsRowDict]) -> SessionStatsRowDict:
    """Return the aggregate row over every per-run row.

    Args:
        rows: Per-run statistics rows.

    Returns:
        Aggregate row labeled ``TOTAL``.
    """
    return SessionStatsRowDict(
        run_id="TOTAL",
        started=rows[0]["started"] if rows else "",
        duration_s=sum(row["duration_s"] for row in rows),
        events=sum(row["events"] for row in rows),
        kills=sum(row["kills"] for row in rows),
        teleports_ok=sum(row["teleports_ok"] for row in rows),
        teleports_failed=sum(row["teleports_failed"] for row in rows),
        shots=sum(row["shots"] for row in rows),
        pickups=sum(row["pickups"] for row in rows),
        stalls=sum(row["stalls"] for row in rows),
    )


def build_session_stats(runs_dir: Path) -> SessionStatsReportDict:
    """Sweep a runs directory and build the cross-session stats report.

    Args:
        runs_dir: Directory holding ``bot-*.events.jsonl`` artifacts.

    Returns:
        Per-run rows in chronological filename order plus totals.

    Raises:
        FileNotFoundError: When no run artifact matches in the
            directory; a typo'd path and an empty directory both mean
            there is nothing to report, and failing fast keeps the
            report from rendering an empty table that hides the typo.
    """
    artifacts = _test_hooks.glob_paths(runs_dir, f"bot-*{_EVENTS_SUFFIX}")
    if not artifacts:
        raise FileNotFoundError(f"No bot-*{_EVENTS_SUFFIX} artifacts found in: {runs_dir}")
    rows: list[SessionStatsRowDict] = []
    for artifact in artifacts:
        run_id = artifact.name[: -len(_EVENTS_SUFFIX)]
        rows.append(_build_row(run_id, load_event_records(artifact)))
    return SessionStatsReportDict(
        runs_dir=str(runs_dir),
        rows=rows,
        totals=_totals_row(rows),
    )


def render_session_stats(report: SessionStatsReportDict) -> str:
    """Render the stats report as an aligned text table.

    Args:
        report: Stats report to render.

    Returns:
        Multi-line table with one row per run plus the totals row.
    """
    header = (
        f"{'run':<22} {'started':<19} {'dur_s':>6} {'events':>6} {'kills':>5} "
        f"{'tp_ok':>5} {'tp_fail':>7} {'shots':>5} {'pickups':>7} {'stalls':>6}"
    )
    lines = [
        "TANKPIT CROSS-SESSION STATS",
        f"Runs dir: {report['runs_dir']}  ({len(report['rows'])} runs)",
        header,
        "-" * len(header),
    ]
    for row in [*report["rows"], report["totals"]]:
        lines.append(
            f"{row['run_id']:<22} {row['started']:<19} {row['duration_s']:>6} "
            f"{row['events']:>6} {row['kills']:>5} {row['teleports_ok']:>5} "
            f"{row['teleports_failed']:>7} {row['shots']:>5} {row['pickups']:>7} "
            f"{row['stalls']:>6}"
        )
    return "\n".join(lines)


def main() -> int:
    """Entry point for the ``tankpit-stats`` CLI.

    Accepts an optional runs-directory argument, defaulting to
    ``runs/bot``. Mirrors the shared analyzer CLI flow; the directory
    argument (rather than a single artifact path) is why this CLI does
    not reuse :func:`tankpit_bot.diagnostics.event_stream.run_analyzer_cli`.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.

    Raises:
        FileNotFoundError: When no run artifacts match in the directory.
    """
    setup_rich_logging(level="INFO")
    full_argv = list(_test_hooks.get_argv())
    user_args = full_argv[1:] if full_argv else []
    runs_dir = Path(user_args[0]) if user_args else Path("runs") / "bot"
    report = build_session_stats(runs_dir)
    log.info("%s", render_session_stats(report))
    return 0


__all__ = [
    "build_session_stats",
    "main",
    "render_session_stats",
]
