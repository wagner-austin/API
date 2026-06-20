"""Issue report text rendering."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.diagnostics.event_stream import run_analyzer_cli
from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_types import (
    IssueReportDict,
    SessionScorecardDict,
)

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
    lines = [
        f"=== TELEPORTS ({len(report['teleport_attempts'])} attempts; "
        f"success={report['teleport_success_count']} "
        f"failure={report['teleport_failure_count']}) ==="
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
    """Return the map_open section lines."""
    lines = [
        f"=== MAP_OPENS (dispatched={report['map_open_dispatches']}, "
        f"completed_via_map_data={report['map_open_completions']}, "
        f"skipped_already_open={len(report['map_open_skipped'])}) ==="
    ]
    for skipped in report["map_open_skipped"]:
        lines.append(f"  [SKIP] origin={skipped['origin']} at {skipped['timestamp']}")
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


def _render_wire_complete_section(report: IssueReportDict) -> list[str]:
    """Return the WIRE_COMPLETE section lines."""
    lines = ["=== WIRE_COMPLETE EVENTS ==="]
    if not report["wire_completes"]:
        lines.append("  (none)")
    else:
        for wc in report["wire_completes"]:
            lines.append(
                f"  action_kind={wc['action_kind']} "
                f"signal={wc['signal']} "
                f"duration_ms={wc['duration_ms']}"
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
        f"  fuel: {fuel_text}",
        f"  dot hops: events={len(scorecard['dot_hops'])} "
        f"distinct={scorecard['dot_hop_distinct_targets']} "
        f"max_repeats={scorecard['dot_hop_max_repeats']}",
    ]
    if not scorecard["state_budget"]:
        lines.append("  state budget: (no transitions)")
    else:
        for record in scorecard["state_budget"]:
            lines.append(f"  {record['state']:>22}: {record['seconds']}s")
    lines.append("")
    return lines


# A dot teleported to this many times in one session was never revealed
# or refuted by a scan -- the orbit class of bug from live run
# 20260612-062453 (fuel bled 151->119 around one in-viewport dot).
_DOT_ORBIT_REPEAT_THRESHOLD = 3

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
    if scorecard["dot_hop_max_repeats"] >= _DOT_ORBIT_REPEAT_THRESHOLD:
        issues.append(
            f"fuel-dot orbit: one dot targeted {scorecard['dot_hop_max_repeats']} times "
            "without being revealed or refuted"
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
    ``WIRE_COMPLETE`` events on map_open completion via
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
    if report["recovery_boxed_in_count"] > 0:
        issues.append(
            f"recovery owner hit its boxed-in terminal action "
            f"{report['recovery_boxed_in_count']} time(s)"
        )
    if report["session_room"] is None:
        issues.append("session room unknown -- analysis terrain is unverifiable")
    issues.extend(_collect_scorecard_issues(report["scorecard"]))
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
        + _render_wire_complete_section(report)
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
