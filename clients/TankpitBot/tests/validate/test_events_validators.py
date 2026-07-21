"""Tests for the events-log teleport and walk validators."""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.validate.events_validators import validate_teleport_cost

ELIGIBLE = "bot-20260701-000000.events.jsonl"


def _fix(x: int, y: int, fuel: int) -> str:
    """Build one self_alignment_sample line.

    Args:
        x: Belief X.
        y: Belief Y.
        fuel: Belief fuel.

    Returns:
        JSON line text.
    """
    return (
        '{"diagnostic_kind": "self_alignment_sample", '
        f'"belief_x": {x}, "belief_y": {y}, "belief_fuel": {fuel}}}'
    )


def _teleport(landed_x: int, landed_y: int, outcome: str = "landed_exact") -> str:
    """Build one teleport action_outcome line.

    Args:
        landed_x: Landing X.
        landed_y: Landing Y.
        outcome: Outcome string.

    Returns:
        JSON line text.
    """
    return (
        '{"diagnostic_kind": "action_outcome", "action_kind": "teleport", '
        f'"outcome": "{outcome}", "landed_x": {landed_x}, "landed_y": {landed_y}}}'
    )


def _write_events(events_dir: Path, name: str, lines: list[str]) -> None:
    """Write one events file.

    Args:
        events_dir: Target directory.
        name: File name.
        lines: JSON lines.
    """
    events_dir.mkdir(parents=True, exist_ok=True)
    (events_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_clean_pair_is_exact(tmp_path: Path) -> None:
    """A 3-4-5 hop debiting exactly 30 re-derives the formula.

    The free map_open dispatch inside the window and the extra
    trailing fix must not disturb the pairing.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            '{"action_kind": "map_open"}',
            _fix(3, 4, 470),
            _fix(3, 4, 470),
        ],
    )
    evidence = validate_teleport_cost(tmp_path)
    assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 1, 0)
    assert evidence["claim_id"] == "teleport-cost"


def test_wrong_debit_is_a_mismatch(tmp_path: Path) -> None:
    """A 3-4-5 hop debiting 29 contradicts the claim."""
    _write_events(tmp_path, ELIGIBLE, [_fix(0, 0, 500), _teleport(3, 4), _fix(3, 4, 471)])
    evidence = validate_teleport_cost(tmp_path)
    assert (evidence["samples"], evidence["exact"], evidence["mismatches"]) == (1, 0, 1)


def test_contaminated_window_is_excluded(tmp_path: Path) -> None:
    """A radar between the fuel fixes makes the pair unusable."""
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            '{"diagnostic_kind": "action_outcome", "action_kind": "scan",'
            ' "outcome": "radar_complete"}',
            _fix(3, 4, 460),
        ],
    )
    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_pickup_dispatch_contaminates(tmp_path: Path) -> None:
    """A container pickup between the fixes makes the pair unusable."""
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            '{"diagnostic_kind": "container_pickup_dispatched"}',
            _fix(3, 4, 300),
        ],
    )
    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_two_teleports_in_one_window_are_excluded(tmp_path: Path) -> None:
    """Two hops between the same fixes cannot be attributed."""
    _write_events(
        tmp_path,
        ELIGIBLE,
        [_fix(0, 0, 500), _teleport(3, 4), _teleport(6, 8), _fix(6, 8, 440)],
    )
    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_missing_fixes_are_excluded(tmp_path: Path) -> None:
    """A teleport with no pre- or no post-fix cannot be paired."""
    _write_events(tmp_path, ELIGIBLE, [_teleport(3, 4), _fix(3, 4, 470)])
    _write_events(tmp_path, "bot-20260702-000000.events.jsonl", [_fix(0, 0, 500), _teleport(3, 4)])
    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_pre_fix_and_latest_files_are_ineligible(tmp_path: Path) -> None:
    """Runs before the fuel-fix cutoff and latest.* copies are skipped."""
    clean = [_fix(0, 0, 500), _teleport(3, 4), _fix(3, 4, 470)]
    _write_events(tmp_path, "bot-20260610-000000.events.jsonl", clean)
    _write_events(tmp_path, "latest.events.jsonl", clean)
    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_malformed_and_incomplete_lines_are_skipped(tmp_path: Path) -> None:
    """Bad JSON, non-object lines, and landed-less outcomes leave no pairs."""
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            "{broken json",
            "[1, 2]",
            "",
            '{"tick_n": 5}',
            _fix(0, 0, 500),
            '{"diagnostic_kind": "action_outcome", "action_kind": "teleport",'
            ' "outcome": "landed_exact"}',
            '{"diagnostic_kind": "action_outcome", "action_kind": "teleport",'
            ' "outcome": "rejected", "landed_x": 3, "landed_y": 4}',
            '{"diagnostic_kind": "self_alignment_sample", "belief_x": 1}',
            '{"diagnostic_kind": "self_alignment_sample", "belief_x": true,'
            ' "belief_y": 2, "belief_fuel": 3}',
            _fix(3, 4, 470),
        ],
    )
    assert validate_teleport_cost(tmp_path)["samples"] == 0
