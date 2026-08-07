"""Scorecard rendering: report section lines and top-level issues.

The output half of the scorecard concern. Consumes a finished
:class:`SessionScorecardDict` and produces the rendered report
section plus the scorecard-derived issue list. The issue report
composes these functions; it does not reimplement them.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.issue_report_types import (
    SessionScorecardDict,
)
from tankpit_bot.diagnostics.session_scorecard import _FUEL_FLOOR_THRESHOLD

# An equipment container teleport-approached this many times never
# became collectable -- the unreachable-pocket orbit from live run
# 20260612-071918 ((128,126)/(129,127) re-approached 7x each).
_EQUIPMENT_ORBIT_REPEAT_THRESHOLD = 3

# Sessions that shoot this much without a single observed deactivation
# are chasing unkillable or repairing targets.
_COMBAT_FUTILITY_SHOT_THRESHOLD = 20


# Low-water episodes listed in full in the rendered report; beyond this
# many, the tail is summarized as a count to keep the section readable.
_LOW_WATER_RENDER_CAP = 10


def render_state_budget_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the per-state time budget with stretch statistics.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        One line per state (or the no-transitions placeholder).
    """
    if not scorecard["state_budget"]:
        return ["  state budget: (no transitions)"]
    return [
        f"  {record['state']:>22}: {record['seconds']}s "
        f"({record['stretches']}x, max {record['max_seconds']}s)"
        for record in scorecard["state_budget"]
    ]


def render_shot_billing_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the ledger's shot billing with the singles reconciliation.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        A single billing line, or no lines when the run ended without
        a ``damage_ledger`` event.
    """
    if scorecard["ledger_shot_singles"] < 0:
        return []
    return [
        f"  shot billing (ledger): dual={scorecard['ledger_shot_duals']} "
        f"homing={scorecard['ledger_shot_homings']} "
        f"single={scorecard['ledger_shot_singles']} "
        "-- singles are server-billed non-connects (weapon=0 misses/clips), "
        "not loadout drift"
    ]


def render_fuel_low_water_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the fuel low-water episode narrative.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Header plus one line per episode (capped), or the all-clear
        line when fuel never dipped below the threshold.
    """
    threshold = scorecard["fuel_low_water_threshold"]
    episodes = scorecard["fuel_low_water_episodes"]
    if not episodes:
        return [f"  fuel low-water: none (never below {threshold})"]
    lines = [f"  fuel low-water (below {threshold}): {len(episodes)} episode(s)"]
    for episode in episodes[:_LOW_WATER_RENDER_CAP]:
        entry = "start" if episode["entry_fuel"] < 0 else str(episode["entry_fuel"])
        recovery = (
            "session end"
            if episode["recovery_fuel"] < 0
            else f"{episode['recovery_fuel']} via {episode['recovery_kind']}"
        )
        lines.append(
            f"    {episode['start_timestamp']} ({episode['duration_seconds']}s) "
            f"entry={entry} min={episode['min_fuel']} "
            f"cause={episode['cause_kind']} -{episode['cause_drop']} "
            f"in {episode['cause_state']} recovery={recovery}"
        )
    hidden = len(episodes) - _LOW_WATER_RENDER_CAP
    if hidden > 0:
        lines.append(f"    ... and {hidden} more episode(s)")
    return lines


def render_teleport_spend_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the teleport fuel spend grouped by paying bot state.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Header plus one line per bot-state group, or the no-spend
        line when no in-flight teleport drops were observed.
    """
    spend_min = scorecard["ledger_teleport_spend_min"]
    spend_max = scorecard["ledger_teleport_spend_max"]
    ledger_text = f" (ledger bound {spend_min}..{spend_max})" if spend_max >= 0 else ""
    if not scorecard["teleport_spend"]:
        return [f"  teleport spend: none observed{ledger_text}"]
    lines = [f"  teleport spend: {scorecard['teleport_spend_total']} fuel{ledger_text}"]
    lines.extend(
        f"    {record['bot_state'] or '(no context)'}: {record['fuel_spent']} "
        f"over {record['drops']} drop(s)"
        for record in scorecard["teleport_spend"]
    )
    return lines


def render_scorecard_section(scorecard: SessionScorecardDict) -> list[str]:
    """Return the session scorecard section lines for the report.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Section lines including the trailing blank separator.
    """
    fuel_text = (
        "no samples"
        if scorecard["fuel_sample_count"] == 0
        else f"min={scorecard['fuel_min']} last={scorecard['fuel_last']} "
        f"samples={scorecard['fuel_sample_count']}"
    )
    first = scorecard["inventory_first"]
    last = scorecard["inventory_last"]
    inventory_text = (
        "no samples"
        if scorecard["inventory_sample_count"] == 0
        else f"dual {first['dual']}->{last['dual']} "
        f"homing {first['homing']}->{last['homing']} "
        f"radar {first['radar']}->{last['radar']} "
        f"samples={scorecard['inventory_sample_count']}"
    )
    gained = scorecard["equipment_gained"]
    lines = [
        "=== SESSION SCORECARD ===",
        f"  duration={scorecard['duration_seconds']}s "
        f"kills={scorecard['kills']} shots={scorecard['shots']}",
        f"  fuel: {fuel_text}",
        f"  inventory: {inventory_text}",
        f"  equipment gains: events={scorecard['equipment_gain_events']} "
        f"armor={gained['armor']} dual={gained['dual']} missile={gained['missile']} "
        f"homing={gained['homing']} radar={gained['radar']}",
        f"  scans: extra={scorecard['scans_extra']} builtin={scorecard['scans_builtin']}",
        f"  physics divergences: {scorecard['physics_divergences']}",
        f"  equipment approaches: events={len(scorecard['equipment_approaches'])} "
        f"distinct={scorecard['equipment_approach_distinct_targets']} "
        f"max_repeats={scorecard['equipment_approach_max_repeats']}",
    ]
    lines.extend(render_shot_billing_lines(scorecard))
    lines.extend(render_fuel_low_water_lines(scorecard))
    lines.extend(render_teleport_spend_lines(scorecard))
    lines.extend(render_state_budget_lines(scorecard))
    lines.append("")
    return lines


def collect_scorecard_issues(scorecard: SessionScorecardDict) -> list[str]:
    """Return top-level issue lines derived from the session scorecard.

    Args:
        scorecard: Session scorecard to inspect.

    Returns:
        Human-readable issue lines (possibly empty).
    """
    issues: list[str] = []
    if scorecard["physics_divergences"] > 0:
        issues.append(
            f"physics divergences: {scorecard['physics_divergences']} fuel window(s) "
            "outside the physics-predicted feasibility interval -- each is a candidate "
            "wiki claim (new mechanic or drifted constant); query "
            "diagnostic_kind=physics_divergence in the events log"
        )
    if scorecard["equipment_approach_max_repeats"] >= _EQUIPMENT_ORBIT_REPEAT_THRESHOLD:
        issues.append(
            "equipment-approach orbit: one container teleport-approached "
            f"{scorecard['equipment_approach_max_repeats']} times without completing a pickup"
        )
    if 0 <= scorecard["fuel_min"] < _FUEL_FLOOR_THRESHOLD:
        issues.append(
            f"fuel floor critical: belief fuel dipped to {scorecard['fuel_min']} "
            f"(below {_FUEL_FLOOR_THRESHOLD})"
        )
    if scorecard["shots"] >= _COMBAT_FUTILITY_SHOT_THRESHOLD and scorecard["kills"] == 0:
        issues.append(f"combat futility: {scorecard['shots']} shots produced 0 observed kills")
    if scorecard["inventory_sample_count"] > 0 and scorecard["inventory_last"]["radar"] == 0:
        issues.append(
            "extra radars exhausted: run ended with 0 extra radars "
            "(scans degrade to the 5x5 built-in and equipment discovery stalls)"
        )
    return issues


__all__ = [
    "collect_scorecard_issues",
    "render_fuel_low_water_lines",
    "render_scorecard_section",
    "render_shot_billing_lines",
    "render_state_budget_lines",
    "render_teleport_spend_lines",
]
