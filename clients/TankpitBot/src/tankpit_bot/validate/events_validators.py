"""Teleport-cost validator over the bot event logs.

Re-runs the 2026-07-20 pairing method from ``wiki/log.md``
("Teleport cost floor(6*euclid) systematically validated"): every
landed teleport ``action_outcome`` is paired with the nearest
``self_alignment_sample`` fuel/position fixes before and after it;
windows containing any foreign action are excluded; the predicted
cost uses the ACTUAL landing tile. (Walk cost is validated from
capture fuel-window EPISODES in ``archive`` — bot walks end in
pickups, so events-fix pairing never finds a clean walk segment.)

Only post-fuel-fix runs (2026-06-24 and later) are eligible: earlier
runs carry the known pickup double-count corruption, which is an
artifact of the OLD bot, not of the physics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import InvalidJsonError, JSONObject, load_json_str
from platform_core.logging import get_logger

from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.validate.types import ClaimEvidenceDict

_LOGGER = get_logger(__name__)

POST_FUEL_FIX_DATE = "20260624"
"""First run date with trustworthy fuel accounting (the 2026-06-23
pickup double-count fix landed the evening before)."""

_TELEPORT_FREE_KINDS = frozenset({"teleport", "map_open"})


class _AlignmentFix(TypedDict):
    """One self_alignment_sample: position + fuel belief."""

    line: int
    x: int
    y: int
    fuel: int


class _TeleportOutcome(TypedDict):
    """One landed teleport action_outcome."""

    line: int
    landed_x: int
    landed_y: int


class _ActionLine(TypedDict):
    """One action-bearing event line with its action kind."""

    line: int
    kind: str


class _EventScan(TypedDict):
    """Everything one events file contributes to the pairing."""

    fixes: list[_AlignmentFix]
    teleport_outcomes: list[_TeleportOutcome]
    action_lines: list[_ActionLine]
    fuel_moves: list[int]
    skipped_lines: int


_FUEL_MOVE = re.compile(r"Fuel: \d+ -> \d+ \(([+-]\d+)\)")


def _fuel_move_delta(record: JSONObject) -> int | None:
    """Return a WORLD fuel line's signed delta, or None when not one.

    Every absolute-fuel wire message logs one of these, and
    ``set_self_fuel`` is the only writer of the belief they report, so a
    line here IS a wire-observed fuel movement rather than a prediction.

    Args:
        record: Parsed JSON record.

    Returns:
        The signed delta, or None when the record is not a fuel line.
    """
    message = record.get("message")
    if not isinstance(message, str):
        return None
    match = _FUEL_MOVE.search(message)
    if match is None:
        return None
    return int(match.group(1))


def _get_int(record: JSONObject, key: str) -> int | None:
    """Read an int field from a parsed JSON record.

    Args:
        record: Parsed JSON object.
        key: Field name.

    Returns:
        The int value, or None when absent or not an int.
    """
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _scan_record(scan: _EventScan, parsed: JSONObject, line_no: int) -> None:
    """Classify one parsed event record into the scan.

    Args:
        scan: Scan being accumulated.
        parsed: Parsed JSON record.
        line_no: Record's line number.
    """
    delta = _fuel_move_delta(parsed)
    if delta is not None:
        if delta != 0:
            scan["fuel_moves"].append(line_no)
        return
    diagnostic_kind = parsed.get("diagnostic_kind")
    action_kind = parsed.get("action_kind")
    if diagnostic_kind == "self_alignment_sample":
        x = _get_int(parsed, "belief_x")
        y = _get_int(parsed, "belief_y")
        fuel = _get_int(parsed, "belief_fuel")
        if x is not None and y is not None and fuel is not None:
            scan["fixes"].append(_AlignmentFix(line=line_no, x=x, y=y, fuel=fuel))
        return
    if diagnostic_kind == "container_pickup_dispatched":
        scan["action_lines"].append(_ActionLine(line=line_no, kind="pickup"))
        return
    if isinstance(action_kind, str):
        scan["action_lines"].append(_ActionLine(line=line_no, kind=action_kind))
        outcome = parsed.get("outcome")
        if diagnostic_kind != "action_outcome" or not isinstance(outcome, str):
            return
        if action_kind == "teleport" and outcome.startswith("landed"):
            landed_x = _get_int(parsed, "landed_x")
            landed_y = _get_int(parsed, "landed_y")
            if landed_x is not None and landed_y is not None:
                scan["teleport_outcomes"].append(
                    _TeleportOutcome(line=line_no, landed_x=landed_x, landed_y=landed_y)
                )


def _scan_events_file(path: Path) -> _EventScan:
    """Scan one events.jsonl file for pairing inputs.

    Args:
        path: Path to a ``bot-*.events.jsonl`` file.

    Returns:
        The file's alignment fixes, action outcomes, and every
        action-bearing line with its kind.
    """
    scan = _EventScan(
        fixes=[],
        teleport_outcomes=[],
        action_lines=[],
        fuel_moves=[],
        skipped_lines=0,
    )
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        try:
            parsed = load_json_str(raw)
        except InvalidJsonError as error:
            _LOGGER.warning(
                "events_line_unparseable file=%s line=%d error=%s", path, line_no, error
            )
            scan["skipped_lines"] += 1
            continue
        if not isinstance(parsed, dict):
            scan["skipped_lines"] += 1
            continue
        _scan_record(scan, parsed, line_no)
    return scan


def _window_fixes(
    scan: _EventScan, event_line: int
) -> tuple[_AlignmentFix | None, _AlignmentFix | None]:
    """Find the nearest fixes before and after one event line.

    Args:
        scan: One file's scanned events.
        event_line: The event's line number.

    Returns:
        Pair of (pre-fix, post-fix); either may be None at file edges.
    """
    pre: _AlignmentFix | None = None
    post: _AlignmentFix | None = None
    for fix in scan["fixes"]:
        if fix["line"] < event_line:
            pre = fix
        elif post is None:
            post = fix
    return pre, post


def _window_is_clean(
    scan: _EventScan,
    pre_line: int,
    post_line: int,
    free_kinds: frozenset[str],
) -> bool:
    """Report whether a fix window contains only permitted action kinds.

    Args:
        scan: One file's scanned events.
        pre_line: Opening fix line.
        post_line: Closing fix line.
        free_kinds: Action kinds that do not contaminate the window.

    Returns:
        True when no foreign action sits inside the window.
    """
    return not any(
        pre_line < action["line"] < post_line and action["kind"] not in free_kinds
        for action in scan["action_lines"]
    )


def _window_holds_one_fuel_move(scan: _EventScan, pre_line: int, post_line: int) -> bool:
    """Report whether at most ONE fuel movement sits in a fix window.

    The measured cost is ``pre.fuel - post.fuel``, so it is the
    teleport's cost only when the teleport is the only thing that moved
    fuel between the two fixes. A second movement -- damage taken, a
    container draining -- is folded into the difference and reported as
    a physics mismatch that never happened.

    That is not hypothetical: all 7 mismatches in the archive were this,
    and the teleport's own debit was exact in every one of them (4 single
    hits at 45, one dual at 90, two pickups at +91 and +484). Counting
    movements rather than recognising their SIZE is deliberate --
    45 is shared by the single, missile and homing victim costs, armor
    absorption and a mine detonation, so any test keyed on the magnitude
    mis-attributes the cause.

    Every clean window in the archive holds exactly one movement and
    every contaminated one holds two, so this separates them completely.

    Args:
        scan: One file's scanned events.
        pre_line: Opening fix line.
        post_line: Closing fix line.

    Returns:
        False when a second fuel movement contaminates the window.
    """
    moves = sum(1 for line in scan["fuel_moves"] if pre_line <= line <= post_line)
    return moves <= 1


def _pair_teleports(scan: _EventScan) -> tuple[int, int, int]:
    """Pair each teleport outcome with clean fuel fixes around it.

    Args:
        scan: One file's scanned events.

    Returns:
        Tuple of (samples, exact, mismatches).
    """
    samples = 0
    exact = 0
    mismatches = 0
    for outcome in scan["teleport_outcomes"]:
        pre, post = _window_fixes(scan, outcome["line"])
        if pre is None or post is None:
            continue
        other_teleport = any(
            pre["line"] < other["line"] < post["line"] and other is not outcome
            for other in scan["teleport_outcomes"]
        )
        if (
            other_teleport
            or not _window_is_clean(scan, pre["line"], post["line"], _TELEPORT_FREE_KINDS)
            or not _window_holds_one_fuel_move(scan, pre["line"], post["line"])
        ):
            continue
        predicted = teleport_cost(pre["x"], pre["y"], outcome["landed_x"], outcome["landed_y"])
        measured = pre["fuel"] - post["fuel"]
        samples += 1
        if measured == predicted:
            exact += 1
        else:
            mismatches += 1
    return samples, exact, mismatches


def _is_eligible_run(path: Path) -> bool:
    """Report whether an events file belongs to the trustworthy era.

    Args:
        path: Candidate ``*.events.jsonl`` path.

    Returns:
        True for ``bot-YYYYMMDD-*`` files dated on or after the
        post-fuel-fix cutoff. ``latest.events.jsonl`` (a copy of the
        newest run) is excluded to avoid double counting.
    """
    name = path.name
    if not name.startswith("bot-"):
        return False
    date_token = name[4:12]
    return date_token.isdigit() and date_token >= POST_FUEL_FIX_DATE


def validate_teleport_cost(events_dir: Path) -> ClaimEvidenceDict:
    """Re-derive the teleport cost from every eligible events log.

    Args:
        events_dir: Directory holding ``bot-*.events.jsonl`` files.

    Returns:
        Evidence for the teleport-cost claim.
    """
    samples = 0
    exact = 0
    mismatches = 0
    runs = 0
    for path in sorted(events_dir.glob("*.events.jsonl")):
        if not _is_eligible_run(path):
            continue
        runs += 1
        file_samples, file_exact, file_mismatches = _pair_teleports(_scan_events_file(path))
        samples += file_samples
        exact += file_exact
        mismatches += file_mismatches
    return ClaimEvidenceDict(
        claim_id="teleport-cost",
        samples=samples,
        exact=exact,
        mismatches=mismatches,
        detail=(
            f"clean dispatch/fuel-fix pairs across {runs} post-fix runs, "
            "predicted on the actual landing tile"
        ),
    )


__all__ = [
    "POST_FUEL_FIX_DATE",
    "validate_teleport_cost",
]
