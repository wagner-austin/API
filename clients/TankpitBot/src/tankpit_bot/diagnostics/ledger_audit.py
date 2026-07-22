"""Deterministic checks over one run's events ledger.

Every check here encodes an interpretation a session previously had to
make by hand against the raw JSONL, converted into an
expected-vs-actual verdict (the ratchet rule):

* ``kill_double_registration`` -- the 2026-07-19 double-counted kill
  (victim 511 registered on ticks 122 and 123 by two racing channels).
  With the DOM kill channel deleted, any repeat registration inside
  the window is a channel regression.
* ``unresolved_decision`` -- the shutdown sweep's pending decisions,
  surfaced per action kind instead of buried in one diagnostic.
* ``rejection_retry_loop`` -- the executor-rejection silent-loop class
  (wiki [[executor-rejection-loops]]): the same target rejected or
  churning repeatedly means replanning is not learning.
* ``tick_cadence_gap`` -- stalls the scorecard's stall counter cannot
  see (waits below the stall timeout, MAP_DATA latency spikes).
* ``session_exit`` -- how the run ended, always surfaced.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from tankpit_bot.diagnostics.run_audit_types import FindingDict, make_finding
from tankpit_bot.ledger.events import ACTION_KINDS
from tankpit_bot.runtime_logging import RuntimeEventRecordDict

_KILL_WINDOW_S = 30
"""Repeat registrations of one victim inside this window are duplicates.

A respawned tank cannot be re-killed this fast (death -> corpse window
-> respawn -> re-engage takes well over 30 s); repeats inside the
window mean two channels registered one death.
"""

_CADENCE_GAP_S = 8
"""Tick-to-tick wall-clock gaps above this are worth surfacing.

The tick rate is 2 s and server MAP_DATA latency has produced clean
6 s gaps (run 2026-07-19); 8 s means something waited longer than any
observed healthy cause.
"""

_SUPERSEDED_CHURN_THRESHOLD = 5
"""Superseded outcomes above this per kind suggest re-dispatch churn."""


def _timestamp_s(timestamp: str) -> float:
    """Return epoch seconds for an artifact ISO timestamp.

    Args:
        timestamp: ISO timestamp string from an event record.

    Returns:
        Seconds since the epoch.
    """
    return datetime.fromisoformat(timestamp).timestamp()


def _int_field(record: RuntimeEventRecordDict, key: str) -> int | None:
    """Return an int field from a record, or None when absent/mistyped.

    Args:
        record: Event record to read.
        key: Field name.

    Returns:
        The integer value, or None. ``bool`` is excluded even though it
        is an ``int`` subtype -- a flag is not a count or id.
    """
    value = record["fields"].get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _diagnostics(records: list[RuntimeEventRecordDict], kind: str) -> list[RuntimeEventRecordDict]:
    """Return every DIAGNOSTIC record of one diagnostic kind, in order.

    Args:
        records: Decoded event records in file order.
        kind: ``diagnostic_kind`` value to collect.

    Returns:
        Matching records.
    """
    return [
        record
        for record in records
        if record["channel"] == "DIAGNOSTIC" and record["fields"].get("diagnostic_kind") == kind
    ]


def _check_kill_double_registration(
    records: list[RuntimeEventRecordDict],
) -> list[FindingDict]:
    """Flag repeat kill registrations of one victim inside the window."""
    findings: list[FindingDict] = []
    last_seen: dict[int, tuple[str, float]] = {}
    for record in _diagnostics(records, "tank_deactivated"):
        victim_id = _int_field(record, "victim_id")
        if victim_id is None:
            continue
        now_s = _timestamp_s(record["timestamp"])
        previous = last_seen.get(victim_id)
        if previous is not None and now_s - previous[1] <= _KILL_WINDOW_S:
            findings.append(
                make_finding(
                    "kill_double_registration",
                    "critical",
                    f"victim {victim_id} registered twice within "
                    f"{_KILL_WINDOW_S}s -- two channels counted one death",
                    victim_id=victim_id,
                    first=previous[0],
                    second=record["timestamp"],
                )
            )
        last_seen[victim_id] = (record["timestamp"], now_s)
    return findings


def _check_unresolved_decisions(
    records: list[RuntimeEventRecordDict],
) -> list[FindingDict]:
    """Surface the shutdown sweep's pending decisions per action kind."""
    findings: list[FindingDict] = []
    for record in _diagnostics(records, "session_unresolved_decisions"):
        for kind in ACTION_KINDS:
            event_id = _int_field(record, kind)
            if event_id is None:
                continue
            findings.append(
                make_finding(
                    "unresolved_decision",
                    "warning",
                    f"{kind} decision {event_id} never got an outcome before shutdown",
                    action_kind=kind,
                    decision_event_id=event_id,
                )
            )
    return findings


def _outcome_rows(
    records: list[RuntimeEventRecordDict],
) -> list[tuple[RuntimeEventRecordDict, str, str]]:
    """Return ``(record, action_kind, outcome)`` for every action outcome."""
    rows: list[tuple[RuntimeEventRecordDict, str, str]] = []
    for record in _diagnostics(records, "action_outcome"):
        action_kind = record["fields"].get("action_kind")
        outcome = record["fields"].get("outcome")
        if isinstance(action_kind, str) and isinstance(outcome, str):
            rows.append((record, action_kind, outcome))
    return rows


def _classify_attempt_outcome(
    record: RuntimeEventRecordDict,
    action_kind: str,
    outcome: str,
) -> FindingDict | None:
    """Return the immediate finding for one outcome row, if any.

    Args:
        record: The outcome's event record.
        action_kind: Ledger action kind of the outcome.
        outcome: Outcome label.

    Returns:
        A stall or rejection finding, or None for aggregate-only rows.
    """
    if outcome == "stall_timeout":
        return make_finding(
            "stall_timeout",
            "critical",
            f"{action_kind} hit the stall timeout -- the wire "
            "never answered and the bot burned the full wait",
            action_kind=action_kind,
            timestamp=record["timestamp"],
        )
    if outcome == "command_rejected":
        error_code = _int_field(record, "error_code")
        return make_finding(
            "command_rejection",
            "info",
            f"server rejected a {action_kind} with error code "
            f"{-1 if error_code is None else error_code}",
            action_kind=action_kind,
            error_code=-1 if error_code is None else error_code,
            timestamp=record["timestamp"],
        )
    if outcome == "pickup_empty":
        return make_finding(
            "command_rejection",
            "info",
            "pickup found the container drained -- consumed by someone "
            "else between scan and pickup",
            action_kind=action_kind,
            timestamp=record["timestamp"],
        )
    if outcome == "inventory_full":
        return make_finding(
            "command_rejection",
            "info",
            "equipment pickup refused: all inventory slots full "
            "(beliefs reconciled) -- the fullness gate should have "
            "prevented this dispatch",
            action_kind=action_kind,
            timestamp=record["timestamp"],
        )
    return None


def _aggregate_findings(
    superseded_counts: dict[str, int],
    failures_by_target: dict[tuple[str, int, int], list[str]],
) -> list[FindingDict]:
    """Build the aggregate verdicts from the one-pass tallies.

    Args:
        superseded_counts: Superseded outcome counts per kind.
        failures_by_target: Failure outcome labels per (kind, x, y).

    Returns:
        Churn and retry-loop findings.
    """
    findings: list[FindingDict] = []
    for kind, count in sorted(superseded_counts.items()):
        if count > _SUPERSEDED_CHURN_THRESHOLD:
            findings.append(
                make_finding(
                    "superseded_churn",
                    "warning",
                    f"{count} {kind} decisions were superseded mid-action "
                    "-- heavy re-dispatch churn",
                    action_kind=kind,
                    count=count,
                )
            )
    for (action_kind, target_x, target_y), outcomes in sorted(failures_by_target.items()):
        if len(outcomes) >= 2:
            findings.append(
                make_finding(
                    "rejection_retry_loop",
                    "critical",
                    f"{action_kind} at ({target_x},{target_y}) failed "
                    f"{len(outcomes)} times -- replanning is not learning "
                    "from the failure",
                    action_kind=action_kind,
                    target_x=target_x,
                    target_y=target_y,
                    failures=",".join(outcomes),
                )
            )
    return findings


def _check_failed_attempts(
    records: list[RuntimeEventRecordDict],
) -> list[FindingDict]:
    """Audit rejections, churn, stalls, and retry loops on one pass."""
    findings: list[FindingDict] = []
    failures_by_target: dict[tuple[str, int, int], list[str]] = {}
    superseded_counts: dict[str, int] = {}
    for record, action_kind, outcome in _outcome_rows(records):
        immediate = _classify_attempt_outcome(record, action_kind, outcome)
        if immediate is not None:
            findings.append(immediate)
        elif outcome == "superseded":
            superseded_counts[action_kind] = superseded_counts.get(action_kind, 0) + 1
        # ``clamped_transfer`` is deliberately absent: a cap-clamped
        # fuel pickup is a success (the fuel arrived), never a
        # failure signal for the retry-loop detector.
        is_failure = outcome in (
            "command_rejected",
            "stall_timeout",
            "pickup_empty",
            "inventory_full",
        )
        target_x = _int_field(record, "target_x")
        target_y = _int_field(record, "target_y")
        if is_failure and target_x is not None and target_y is not None:
            target_key = (action_kind, target_x, target_y)
            failures_by_target.setdefault(target_key, []).append(outcome)
    findings.extend(_aggregate_findings(superseded_counts, failures_by_target))
    return findings


def _check_tick_cadence(records: list[RuntimeEventRecordDict]) -> list[FindingDict]:
    """Flag wall-clock gaps between consecutive ticks above the threshold."""
    first_seen: dict[int, str] = {}
    for record in records:
        tick_n = _int_field(record, "tick_n")
        if tick_n is not None and tick_n not in first_seen:
            first_seen[tick_n] = record["timestamp"]
    findings: list[FindingDict] = []
    ordered = sorted(first_seen.items())
    for (prev_tick, prev_ts), (next_tick, next_ts) in pairwise(ordered):
        gap_s = int(_timestamp_s(next_ts) - _timestamp_s(prev_ts))
        if gap_s > _CADENCE_GAP_S:
            findings.append(
                make_finding(
                    "tick_cadence_gap",
                    "warning",
                    f"{gap_s}s of wall clock between ticks {prev_tick} and "
                    f"{next_tick} -- something waited longer than any "
                    "healthy cause explains",
                    prev_tick=prev_tick,
                    next_tick=next_tick,
                    gap_s=gap_s,
                    at=next_ts,
                )
            )
    return findings


def _check_session_exit(records: list[RuntimeEventRecordDict]) -> list[FindingDict]:
    """Surface how the session ended; its absence is itself a finding."""
    scorecards = _diagnostics(records, "session_scorecard")
    if not scorecards:
        return [
            make_finding(
                "session_exit",
                "warning",
                "no session scorecard in the artifact -- the run died before the shutdown path ran",
            )
        ]
    record = scorecards[-1]
    exit_reason = record["fields"].get("exit_reason")
    ticks = _int_field(record, "ticks")
    kills = _int_field(record, "kills")
    return [
        make_finding(
            "session_exit",
            "info",
            f"session ended: {exit_reason if isinstance(exit_reason, str) else 'unknown'}",
            exit_reason=exit_reason if isinstance(exit_reason, str) else "unknown",
            ticks=-1 if ticks is None else ticks,
            kills=-1 if kills is None else kills,
        )
    ]


def audit_ledger(records: list[RuntimeEventRecordDict]) -> list[FindingDict]:
    """Run every ledger check over one run's decoded event records.

    Args:
        records: Decoded event records in file order.

    Returns:
        Findings from all checks, in production order (the report
        assembler sorts them).
    """
    if not records:
        return [
            make_finding(
                "empty_run",
                "critical",
                "the events artifact contains no records -- the session "
                "died before the game loop produced anything",
            )
        ]
    findings: list[FindingDict] = []
    findings.extend(_check_kill_double_registration(records))
    findings.extend(_check_unresolved_decisions(records))
    findings.extend(_check_failed_attempts(records))
    findings.extend(_check_tick_cadence(records))
    findings.extend(_check_session_exit(records))
    return findings


__all__ = [
    "audit_ledger",
]
