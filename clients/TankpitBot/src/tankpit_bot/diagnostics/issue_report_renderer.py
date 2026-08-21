"""Issue report text rendering."""

from __future__ import annotations

from collections import Counter

from platform_core.logging import get_logger

from tankpit_bot.diagnostics.event_stream import run_analyzer_cli
from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_types import (
    IssueReportDict,
    SessionScorecardDict,
)
from tankpit_bot.diagnostics.session_scorecard_render import (
    render_fuel_low_water_lines,
    render_shot_billing_lines,
    render_state_budget_lines,
    render_teleport_spend_lines,
)
from tankpit_bot.ledger.outcomes import LIVENESS_STALL_STREAK

log = get_logger(__name__)

_LANDED_STATUSES: frozenset[str] = frozenset({"landed_exact", "landed_inexact"})


def _render_header(report: IssueReportDict) -> list[str]:
    """Return the header lines of the rendered report."""
    lines = ["=" * 72, "TANKPIT ISSUE REPORT", "=" * 72]
    lines.append(f"Source: {report['source_path']}")
    lines.append(f"Mode:   {report['mode']}")
    lines.append(f"Events: {report['event_count']}")
    room = report["session_room"]
    if room is None:
        lines.append("Session room: UNKNOWN (no session_room_joined diagnostic in stream)")
    else:
        lines.append(f"Session room: id={room['room_id']} field_image={room['field_image']}")
    lines.append("")
    return lines


def _render_teleport_section(report: IssueReportDict) -> list[str]:
    """Return the teleport-attempt section lines."""
    superseded = report["scorecard"]["action_outcome_counts"].get("teleport:superseded", 0)
    lines = [
        f"=== TELEPORTS (success={report['teleport_success_count']} "
        f"failure={report['teleport_failure_count']} "
        f"superseded={superseded}; "
        f"lab_attempt_rows={len(report['teleport_attempts'])}) ==="
    ]
    for attempt in report["teleport_attempts"]:
        marker = "[OK]" if attempt["status"] in _LANDED_STATUSES else "[FAIL]"
        lines.append(
            f"  {marker} cycle={attempt['teleport_cycle_id']} "
            f"target=({attempt['target_x']},{attempt['target_y']}) "
            f"status={attempt['status']}"
        )
        if attempt["status"] not in _LANDED_STATUSES:
            lines.append(f"        sent     = {attempt['sent_window']}")
            lines.append(f"        received = {attempt['received_window']}")
    lines.append("")
    return lines


def _render_map_open_section(report: IssueReportDict) -> list[str]:
    """Return the map_open section lines, tallied per origin.

    The already-open case is the NORMAL path, not an anomaly: a teleport
    needs the map overlay open, the hop itself closes it, and the
    executor checks ``map_visible`` before sending a second open that the
    server would treat as a no-op. It therefore fires on essentially
    every successful teleport -- 10,569 times across the 427 archived
    runs.

    One line per event listed those as ``[SKIP]``, which read as a list
    of failures and buried the case worth seeing: 24 of the archived
    events come from ``executor.dispatch_command.map_open`` rather than
    the teleport precondition, meaning the planner asked to open a map
    that was already open. A per-origin tally puts that on its own line
    instead of 24 rows inside 10,545.
    """
    lines = [
        f"=== MAP_OPENS (dispatched={report['map_open_dispatches']}, "
        f"completed_via_map_data={report['map_open_completions']}, "
        f"already_open={len(report['map_open_skipped'])}) ==="
    ]
    per_origin = Counter(skipped["origin"] for skipped in report["map_open_skipped"])
    for origin, count in sorted(per_origin.items()):
        lines.append(f"  already open, no open sent: {count}x from {origin}")
    lines.append("")
    return lines


def _render_fuel_section(report: IssueReportDict) -> list[str]:
    """Return the fuel target selection section lines."""
    lines = [
        f"=== FUEL TARGET SELECTION ({len(report['fuel_target_selections'])} cycles; "
        f"selected={report['fuel_selected_count']} "
        f"rejected={report['fuel_rejected_count']}) ==="
    ]
    for selection in report["fuel_target_selections"]:
        marker = "[PICK]" if selection["target_present"] else "[SKIP]"
        target = (
            "none"
            if not selection["target_present"]
            else f"({selection['target_x']},{selection['target_y']})"
        )
        lines.append(f"  {marker} cycle={selection['radar_cycle_id']} target={target}")
        lines.append(f"        summary  = {selection['summary']}")
    lines.append("")
    return lines


def _render_action_outcome_section(report: IssueReportDict) -> list[str]:
    """Return the ACTION OUTCOMES section lines."""
    lines = ["=== ACTION OUTCOMES ==="]
    if not report["action_outcomes"]:
        lines.append("  (none)")
    else:
        for row in report["action_outcomes"]:
            lines.append(
                f"  {row['action_kind']}#{row['attempt_id']} "
                f"outcome={row['outcome']} "
                f"duration_ms={row['duration_ms']}"
            )
    lines.append("")
    return lines


def _render_suppressed_section(report: IssueReportDict) -> list[str]:
    """Return the SUPPRESSED DISPATCHES section lines (empty when none).

    Every row is a target the executor's belief-veto refused at least
    once without sending anything; the tally is how many times the
    planner re-selected it anyway.
    """
    if not report["suppressed_dispatches"]:
        return []
    lines = ["=== SUPPRESSED DISPATCHES (belief-refuted, nothing sent) ==="]
    for row in report["suppressed_dispatches"]:
        lines.append(
            f"  {row['command_name']} to ({row['target_x']},{row['target_y']}) "
            f"x{row['count']} (predicted 0x52 code {row['predicted_error_code']})"
        )
    lines.append("")
    return lines


def _render_scorecard_section(report: IssueReportDict) -> list[str]:
    """Return the session scorecard section lines."""
    scorecard = report["scorecard"]
    fuel_text = (
        "no samples"
        if scorecard["fuel_sample_count"] == 0
        else f"min={scorecard['fuel_min']} last={scorecard['fuel_last']} "
        f"samples={scorecard['fuel_sample_count']}"
    )
    lines = [
        "=== SESSION SCORECARD ===",
        f"  duration={scorecard['duration_seconds']}s "
        f"kills={scorecard['kills']} shots={scorecard['shots']}",
        f"  combat gates: misses={scorecard['combat_misses']} "
        f"ghosts_blocked={scorecard['combat_ghosts_blocked']} "
        f"stale_pos_blocked={scorecard['combat_stale_positions_blocked']} "
        f"damage_changes={scorecard['tank_damage_changes']}",
        "  outcomes: "
        + (
            " ".join(f"{key}={count}" for key, count in scorecard["action_outcome_counts"].items())
            or "(none)"
        ),
        f"  fuel: {fuel_text}",
    ]
    lines.extend(render_shot_billing_lines(scorecard))
    lines.extend(render_fuel_low_water_lines(scorecard))
    lines.extend(render_teleport_spend_lines(scorecard))
    lines.extend(render_state_budget_lines(scorecard))
    lines.append("")
    return lines


# One suppression is the executor's refusal prediction working (a
# spared server call); two can be one belief refreshing mid-window.
# THREE same-target suppressions mean the planner was told "this cannot
# transfer" twice and selected the identical action anyway -- the
# planner/veto feedback gap that produced the 2026-08-20 gatherer
# livelock (93 consecutive suppressions on one tile while this report
# read "no top-level issues detected").
_SUPPRESSED_STREAK_ISSUE_THRESHOLD = 3


def _zero_dispatch_streaks(report: IssueReportDict) -> dict[str, int]:
    """Return the longest zero-dispatch replan streak per action kind.

    The generalized liveness scan (the suppressed-dispatch rule's
    veto-agnostic sibling): a zero-duration ``superseded`` outcome is a
    decision replaced before anything dispatched, and a long
    same-kind run of them is a planner producing plans that never
    reach the wire — whatever the veto. Mirrors the ledger's live
    counter so post-run analysis catches the class even on artifacts
    from builds without the ``liveness_stall`` diagnostic.

    Args:
        report: Report whose ``action_outcomes`` rows are scanned in
            stream order.

    Returns:
        Per-kind maximum consecutive zero-duration superseded count.
    """
    best: dict[str, int] = {}
    current_kind = ""
    current = 0
    for row in report["action_outcomes"]:
        if row["outcome"] == "superseded" and row["duration_ms"] == 0:
            current = current + 1 if row["action_kind"] == current_kind else 1
            current_kind = row["action_kind"]
            if current > best.get(current_kind, 0):
                best[current_kind] = current
        else:
            current_kind = ""
            current = 0
    return best


# A displaced teleport is a SUCCESS to the ledger (landed_inexact), so
# destination repetition hides from every failure counter — the third
# liveness flavor. Empirical (459-run archive sweep 2026-08-21): the 11
# pathological runs all repeat a destination >= 3 times (worst: 534 at
# one tile, the 08-05 ancestor); healthy runs repeat at most twice
# (combat re-aims at a stationary enemy).
_DISPLACEMENT_ORBIT_THRESHOLD = 3

# Sessions that shoot this much without a single observed deactivation
# are chasing unkillable or repairing targets.
_COMBAT_FUTILITY_SHOT_THRESHOLD = 20

# The fuel-critical band: combat needs ~10 fuel per shot and teleports
# cost 6 per tile, so dipping below this means the session nearly
# stranded itself.
_FUEL_FLOOR_THRESHOLD = 100


def _collect_scorecard_issues(scorecard: SessionScorecardDict) -> list[str]:
    """Return top-level issue lines derived from the session scorecard.

    Args:
        scorecard: Session scorecard to inspect.

    Returns:
        Human-readable issue lines (possibly empty).
    """
    issues: list[str] = []
    stalls = {
        key.removesuffix(":stall_timeout"): count
        for key, count in scorecard["action_outcome_counts"].items()
        if key.endswith(":stall_timeout") and count > 0
    }
    if stalls:
        # Recovered anomalies must be VISIBLE (2026-08-20 lesson: the
        # scope-pending radar drop hid in stall counts for 19 days
        # because every stall self-healed — the report surfaces what
        # breaks, so it must also surface what limps). Post-July the
        # archive baseline is under one stall per run; any stall is
        # worth a line.
        breakdown = " ".join(f"{kind}={count}" for kind, count in sorted(stalls.items()))
        issues.append(
            f"{sum(stalls.values())} action(s) stalled to timeout and replanned "
            f"({breakdown}) -- self-healed, but each stall is ~10 s of session "
            "time with a cause worth naming"
        )
    if 0 <= scorecard["fuel_min"] < _FUEL_FLOOR_THRESHOLD:
        issues.append(
            f"fuel floor critical: belief fuel dipped to {scorecard['fuel_min']} "
            f"(below {_FUEL_FLOOR_THRESHOLD})"
        )
    if scorecard["shots"] >= _COMBAT_FUTILITY_SHOT_THRESHOLD and scorecard["kills"] == 0:
        issues.append(f"combat futility: {scorecard['shots']} shots produced 0 observed kills")
    return issues


def _collect_top_level_issues(report: IssueReportDict) -> list[str]:
    """Return one human-readable issue line per top-level problem.

    The ``map_open`` dispatch/completion mismatch check is intentionally
    gated to ``mode == "bot"``: only the live HFSM runtime emits
    ``action_outcome`` events on map_open completion via
    :func:`tankpit_bot.bot.tick_loop._clear_completed_map_open`. The
    action_lab probe paths have their own per-attempt phase machinery
    (``run_tracked_acquisition_phase`` / ``run_tracked_teleport_command``)
    and never reach the HFSM completion gate, so for a ``probe:<name>``
    mode any dispatch/completion delta is expected -- not an issue --
    and surfacing it would clutter every probe report with a false
    positive.
    """
    issues: list[str] = []
    if report["teleport_failure_count"] > 0:
        denom = max(1, len(report["teleport_attempts"]))
        pct = 100.0 * report["teleport_failure_count"] / denom
        issues.append(
            f"{report['teleport_failure_count']}/{len(report['teleport_attempts'])} "
            f"teleports failed ({pct:.0f}%)"
        )
    if report["fuel_rejected_count"] > 0:
        denom = max(1, len(report["fuel_target_selections"]))
        pct = 100.0 * report["fuel_rejected_count"] / denom
        issues.append(
            f"{report['fuel_rejected_count']}/{len(report['fuel_target_selections'])} "
            f"fuel cycles had no actionable target ({pct:.0f}%)"
        )
    if report["mode"] == "bot" and report["map_open_dispatches"] != report["map_open_completions"]:
        delta = report["map_open_dispatches"] - report["map_open_completions"]
        issues.append(
            f"map_open dispatch/completion mismatch: dispatched="
            f"{report['map_open_dispatches']} "
            f"vs completed={report['map_open_completions']} (delta={delta})"
        )
    if report["session_room"] is None:
        issues.append("session room unknown -- analysis terrain is unverifiable")
    issues.extend(_collect_repetition_issues(report))
    issues.extend(_collect_scorecard_issues(report["scorecard"]))
    return issues


def _collect_repetition_issues(report: IssueReportDict) -> list[str]:
    """Return the liveness-flavor issue lines, one rule per flavor.

    Each flavor is a way a run can burn time without any failure
    counter noticing: vetoed re-selection (suppressed dispatches),
    zero-dispatch replans (liveness stalls), and
    successful-but-bounced landings (displacement orbits).

    Args:
        report: Report whose tallies are inspected.

    Returns:
        Human-readable issue lines (possibly empty).
    """
    issues: list[str] = []
    for suppressed in report["suppressed_dispatches"]:
        if suppressed["count"] >= _SUPPRESSED_STREAK_ISSUE_THRESHOLD:
            issues.append(
                f"planner re-selected a belief-refuted {suppressed['command_name']} to "
                f"({suppressed['target_x']},{suppressed['target_y']}) {suppressed['count']}x "
                f"(predicted 0x52 code {suppressed['predicted_error_code']}) -- "
                "the executor's veto is not feeding back into selection"
            )
    for displaced in report["displaced_teleports"]:
        if displaced["count"] >= _DISPLACEMENT_ORBIT_THRESHOLD:
            issues.append(
                f"displacement orbit: {displaced['count']} teleports at "
                f"({displaced['requested_x']},{displaced['requested_y']}) all refused "
                f"(max displacement {displaced['max_displacement']}) -- landings that "
                "resolve as successes while the tank never left its origin"
            )
    for kind, streak in sorted(_zero_dispatch_streaks(report).items()):
        if streak >= LIVENESS_STALL_STREAK:
            issues.append(
                f"liveness stall: {streak} consecutive {kind} decisions replaced with "
                f"zero dispatches (healthy archive ceiling is 7) -- the planner is "
                "spinning without reaching the wire"
            )
    return issues


def _render_summary_section(report: IssueReportDict) -> list[str]:
    """Return the trailing top-level summary section lines."""
    lines = ["=== TOP-LEVEL ISSUE SUMMARY ==="]
    issues = _collect_top_level_issues(report)
    if not issues:
        lines.append("  (no top-level issues detected)")
    else:
        lines.extend(f"  - {issue}" for issue in issues)
    lines.append("=" * 72)
    return lines


def render_issue_report(report: IssueReportDict) -> str:
    """Render an :class:`IssueReportDict` to a human-readable string.

    Args:
        report: Report to render.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    return "\n".join(
        _render_header(report)
        + _render_scorecard_section(report)
        + _render_teleport_section(report)
        + _render_map_open_section(report)
        + _render_fuel_section(report)
        + _render_suppressed_section(report)
        + _render_action_outcome_section(report)
        + _render_summary_section(report)
    )


def main() -> int:
    """Run the ``tankpit-issue-report`` CLI entrypoint.

    Reads a JSONL events artifact (path resolved from the user-supplied
    args -- ``sys.argv`` with the script name stripped -- defaulting to
    ``runs/bot/latest.events.jsonl``), builds an :class:`IssueReportDict`,
    and prints it to the rich console logger.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_issue_report, render_issue_report, log)


__all__ = [
    "main",
    "render_issue_report",
]
