"""Forage-economy analyzer: WHY a run was fast or slow, from its events.

The issue report answers "did anything break"; this module answers
"where did the time go". It reduces a run's JSONL events artifact to
the resource-economy numbers that decide session length: how the
wall-clock split between hunting and foraging, how many forage
viewports funded each kill, what each equipment pickup paid in
weapons, and how the hop cascade spent its candidates.

Origin: the 2026-07-26 10-kill pair — 803 s vs 1,187 s with identical
map density (~580 dots) and an issue report reading "no top-level
issues" for both. The whole delta was COLLECT time driven by
weapons-per-equipment-pickup (3.34 vs 2.14); every number this module
prints was first derived by hand in that investigation.

CLI: ``tankpit-forage-economy [events.jsonl [baseline.jsonl]]`` — one
path analyzes that run (default ``runs/bot/latest.events.jsonl``);
two paths render both runs side by side for exactly the fast-vs-slow
comparison that surfaced the law.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import RuntimeEventRecordDict

log = get_logger(__name__)

_DEFAULT_SOURCE = Path("runs/bot/latest.events.jsonl")

#: 0x52 code 5 — the tank-full clamp receipt of a successful clamped
#: fill ([[fuel-system]]); every other code is a genuine rejection.
_CLAMP_RECEIPT_CODE = 5

_PICKUP_OUTCOMES: frozenset[str] = frozenset({"container_consumed", "clamped_transfer"})


class ForageEconomyDict(TypedDict):
    """Resource-economy aggregate for one run's events artifact.

    Attributes:
        source_path: Events artifact the report was built from.
        span_seconds: First-to-last event wall-clock span.
        hunt_seconds: Time attributed to ``HUNT/*`` states.
        collect_seconds: Time attributed to ``COLLECT/*`` states.
        other_seconds: Time attributed to any other / absent state.
        kills: Final ``session_scorecard`` kill count, or ``None``
            when the run ended without emitting a scorecard.
        forage_scans: ``radar_complete`` scans in COLLECT states —
            one per forage viewport visited.
        pickups_consumed: ``container_consumed`` collect outcomes.
        pickups_clamped: ``clamped_transfer`` collect outcomes.
        equipment_pickups: ``equipment_gain`` events.
        weapons_gained: Summed ``dual`` + ``homing`` across gains.
        radars_gained: Summed ``radar`` across gains.
        hops_dot: ``hop_selected`` diagnostics with ``hop_kind=dot``.
        hops_equipment: ``hop_selected`` diagnostics with
            ``hop_kind=equipment``.
        hops_declined: ``hop_declined`` diagnostics.
        no_landing_rejections: Summed ``no_landing`` counts across
            declines. Each decline re-evaluates the whole tracked
            atlas, so this counts candidate-evaluations, not distinct
            containers (dominated by water-locked containers,
            [[terrain-system]]).
        shots_hit: ``shoot`` outcomes equal to ``hit``.
        shots_missed: ``shoot`` outcomes other than ``hit``.
        clamp_receipts: ``command_error`` diagnostics with the code-5
            clamp receipt.
        other_command_errors: ``command_error`` diagnostics with any
            other code — the ones worth reading.
    """

    source_path: str
    span_seconds: float
    hunt_seconds: float
    collect_seconds: float
    other_seconds: float
    kills: int | None
    forage_scans: int
    pickups_consumed: int
    pickups_clamped: int
    equipment_pickups: int
    weapons_gained: int
    radars_gained: int
    hops_dot: int
    hops_equipment: int
    hops_declined: int
    no_landing_rejections: int
    shots_hit: int
    shots_missed: int
    clamp_receipts: int
    other_command_errors: int


def _opt_str(fields: dict[str, str | int | float | bool], key: str) -> str | None:
    """Return a string field when present and string-typed, else None.

    Args:
        fields: Structured payload of a runtime event record.
        key: Field name to read.

    Returns:
        The string value, or ``None`` when absent or non-string.
    """
    value = fields.get(key)
    return value if isinstance(value, str) else None


def _opt_int(fields: dict[str, str | int | float | bool], key: str) -> int | None:
    """Return an int field when present and int-typed, else None.

    Booleans are rejected — they are ints to ``isinstance`` but never
    valid counts.

    Args:
        fields: Structured payload of a runtime event record.
        key: Field name to read.

    Returns:
        The int value, or ``None`` when absent or non-int.
    """
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _timestamp_seconds(record: RuntimeEventRecordDict) -> float:
    """Return a record's timestamp as epoch seconds.

    Args:
        record: Decoded runtime event record.

    Returns:
        POSIX seconds parsed from the record's ISO timestamp.
    """
    return datetime.fromisoformat(record["timestamp"]).timestamp()


def _mode_bucket(record: RuntimeEventRecordDict) -> str:
    """Return the time-attribution bucket for a record.

    Args:
        record: Decoded runtime event record.

    Returns:
        ``"HUNT"``, ``"COLLECT"``, or ``"OTHER"`` from the record's
        ``bot_state`` prefix.
    """
    state = _opt_str(record["fields"], "bot_state")
    if state is not None:
        prefix = state.split("/", 1)[0]
        if prefix in ("HUNT", "COLLECT"):
            return prefix
    return "OTHER"


def _route_action_outcome(
    record: RuntimeEventRecordDict,
    report: ForageEconomyDict,
) -> None:
    """Fold one ``action_outcome`` record into the report.

    Args:
        record: Decoded runtime event record.
        report: Report under construction (mutated in place).
    """
    fields = record["fields"]
    action = _opt_str(fields, "action_kind")
    outcome = _opt_str(fields, "outcome")
    if action == "scan" and outcome == "radar_complete" and _mode_bucket(record) == "COLLECT":
        report["forage_scans"] += 1
    elif action == "collect" and outcome == "container_consumed":
        report["pickups_consumed"] += 1
    elif action == "collect" and outcome == "clamped_transfer":
        report["pickups_clamped"] += 1
    elif action == "shoot":
        if outcome == "hit":
            report["shots_hit"] += 1
        else:
            report["shots_missed"] += 1


def _route_equipment_gain(
    fields: dict[str, str | int | float | bool],
    report: ForageEconomyDict,
) -> None:
    """Fold one ``equipment_gain`` record into the report.

    Args:
        fields: Structured payload of the record.
        report: Report under construction (mutated in place).
    """
    report["equipment_pickups"] += 1
    for weapon_key in ("dual", "homing"):
        gained = _opt_int(fields, weapon_key)
        if gained is not None:
            report["weapons_gained"] += gained
    radar = _opt_int(fields, "radar")
    if radar is not None:
        report["radars_gained"] += radar


def _route_hop_or_error(
    kind: str,
    fields: dict[str, str | int | float | bool],
    report: ForageEconomyDict,
) -> None:
    """Fold a hop or command-error record into the report.

    Args:
        kind: The record's ``diagnostic_kind``.
        fields: Structured payload of the record.
        report: Report under construction (mutated in place).
    """
    if kind == "hop_selected":
        if _opt_str(fields, "hop_kind") == "equipment":
            report["hops_equipment"] += 1
        else:
            report["hops_dot"] += 1
    elif kind == "hop_declined":
        report["hops_declined"] += 1
        no_landing = _opt_int(fields, "no_landing")
        if no_landing is not None:
            report["no_landing_rejections"] += no_landing
    else:
        if _opt_int(fields, "error_code") == _CLAMP_RECEIPT_CODE:
            report["clamp_receipts"] += 1
        else:
            report["other_command_errors"] += 1


def _route_economy_record(
    record: RuntimeEventRecordDict,
    report: ForageEconomyDict,
) -> None:
    """Fold one record's economy contribution into the report.

    Args:
        record: Decoded runtime event record.
        report: Report under construction (mutated in place).
    """
    fields = record["fields"]
    kind = _opt_str(fields, "diagnostic_kind")
    if kind == "action_outcome":
        _route_action_outcome(record, report)
    elif kind == "equipment_gain":
        _route_equipment_gain(fields, report)
    elif kind in ("hop_selected", "hop_declined", "command_error"):
        _route_hop_or_error(kind, fields, report)
    elif kind == "session_scorecard":
        report["kills"] = _opt_int(fields, "kills")


def build_forage_economy(source_path: Path) -> ForageEconomyDict:
    """Build a :class:`ForageEconomyDict` from a JSONL events artifact.

    Args:
        source_path: Path to a runtime events JSONL artifact.

    Returns:
        Aggregated forage-economy report.
    """
    records = load_event_records(source_path)
    report = ForageEconomyDict(
        source_path=str(source_path),
        span_seconds=0.0,
        hunt_seconds=0.0,
        collect_seconds=0.0,
        other_seconds=0.0,
        kills=None,
        forage_scans=0,
        pickups_consumed=0,
        pickups_clamped=0,
        equipment_pickups=0,
        weapons_gained=0,
        radars_gained=0,
        hops_dot=0,
        hops_equipment=0,
        hops_declined=0,
        no_landing_rejections=0,
        shots_hit=0,
        shots_missed=0,
        clamp_receipts=0,
        other_command_errors=0,
    )
    for record in records:
        _route_economy_record(record, report)
    if len(records) >= 2:
        report["span_seconds"] = _timestamp_seconds(records[-1]) - _timestamp_seconds(records[0])
        for earlier, later in pairwise(records):
            gap = _timestamp_seconds(later) - _timestamp_seconds(earlier)
            bucket = _mode_bucket(earlier)
            if bucket == "HUNT":
                report["hunt_seconds"] += gap
            elif bucket == "COLLECT":
                report["collect_seconds"] += gap
            else:
                report["other_seconds"] += gap
    return report


def _per(numerator: int, denominator: int) -> str:
    """Render a ratio, or ``-`` when the denominator is zero.

    Args:
        numerator: Ratio numerator.
        denominator: Ratio denominator.

    Returns:
        ``numerator/denominator`` to two decimals, or ``"-"``.
    """
    if denominator == 0:
        return "-"
    return f"{numerator / denominator:.2f}"


def render_forage_economy(report: ForageEconomyDict) -> str:
    """Render one run's forage economy as report text.

    Args:
        report: Report to render.

    Returns:
        Human-readable multi-line report.
    """
    kills = report["kills"]
    kills_text = str(kills) if kills is not None else "unknown (no scorecard)"
    pickups = report["pickups_consumed"] + report["pickups_clamped"]
    shots = report["shots_hit"] + report["shots_missed"]
    lines = [
        "=== FORAGE ECONOMY ===",
        f"source: {report['source_path']}",
        (
            f"span {report['span_seconds']:.0f} s | hunt {report['hunt_seconds']:.0f} s"
            f" | collect {report['collect_seconds']:.0f} s"
            f" | other {report['other_seconds']:.0f} s"
        ),
        f"kills: {kills_text}",
        (
            f"forage viewports: {report['forage_scans']}"
            f" ({_per(report['forage_scans'], kills) if kills is not None else '-'}/kill)"
        ),
        (
            f"pickups: {pickups} ({report['pickups_consumed']} consumed"
            f" + {report['pickups_clamped']} clamped,"
            f" {_per(pickups, report['forage_scans'])}/viewport)"
        ),
        (
            f"equipment pickups: {report['equipment_pickups']}"
            f" -> weapons {report['weapons_gained']}"
            f" ({_per(report['weapons_gained'], report['equipment_pickups'])}/pickup),"
            f" radars {report['radars_gained']}"
        ),
        (
            f"hops: dot {report['hops_dot']}, equipment {report['hops_equipment']},"
            f" declined {report['hops_declined']}"
            f" (no_landing candidate-evals {report['no_landing_rejections']})"
        ),
        f"shots: {shots} ({report['shots_hit']} hit, {report['shots_missed']} missed)",
        (
            f"command errors: {report['clamp_receipts']} clamp receipts,"
            f" {report['other_command_errors']} other"
        ),
    ]
    return "\n".join(lines)


def render_forage_comparison(
    current: ForageEconomyDict,
    baseline: ForageEconomyDict,
) -> str:
    """Render two runs side by side with the deciding ratios.

    Args:
        current: The run under analysis.
        baseline: The run to compare against.

    Returns:
        Both single-run reports followed by a delta section.
    """
    lines = [
        render_forage_economy(current),
        "",
        render_forage_economy(baseline),
        "",
        "=== DELTA (current vs baseline) ===",
        f"span: {current['span_seconds']:.0f} s vs {baseline['span_seconds']:.0f} s",
        f"collect: {current['collect_seconds']:.0f} s vs {baseline['collect_seconds']:.0f} s",
        f"forage viewports: {current['forage_scans']} vs {baseline['forage_scans']}",
        (
            "weapons/pickup: "
            f"{_per(current['weapons_gained'], current['equipment_pickups'])}"
            f" vs {_per(baseline['weapons_gained'], baseline['equipment_pickups'])}"
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the ``tankpit-forage-economy`` CLI entrypoint.

    One path argument (default ``runs/bot/latest.events.jsonl``)
    analyzes that run; a second path renders a side-by-side
    comparison against it.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    setup_rich_logging(level="INFO")
    args = list(_test_hooks.get_argv())[1:]
    paths = [Path(arg) for arg in args] if args else [_DEFAULT_SOURCE]
    current = build_forage_economy(paths[0])
    if len(paths) >= 2:
        baseline = build_forage_economy(paths[1])
        log.info("%s", render_forage_comparison(current, baseline))
    else:
        log.info("%s", render_forage_economy(current))
    return 0


__all__ = [
    "ForageEconomyDict",
    "build_forage_economy",
    "main",
    "render_forage_comparison",
    "render_forage_economy",
]
