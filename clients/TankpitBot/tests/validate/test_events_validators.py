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


def test_boolean_belief_field_is_not_accepted_as_a_coordinate(tmp_path: Path) -> None:
    """A JSON ``true`` in a coordinate field does not become the number 1.

    ``bool`` is a subclass of ``int`` in Python, so a plain
    ``isinstance(value, int)`` check accepts ``True`` and silently
    yields the coordinate 1. The sample must be rejected instead, which
    leaves the teleport with no preceding fix and therefore no pair.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            '{"diagnostic_kind": "self_alignment_sample", '
            '"belief_x": true, "belief_y": 0, "belief_fuel": 500}',
            _teleport(3, 4),
            _fix(3, 4, 470),
        ],
    )

    evidence = validate_teleport_cost(tmp_path)

    assert evidence["samples"] == 0


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


def test_a_teleport_dispatch_line_is_not_a_landing(tmp_path: Path) -> None:
    """A teleport line that is not an ``action_outcome`` is never paired.

    Every dispatched teleport logs an action-bearing line carrying no
    ``outcome`` field at all: 11,133 lines across the 427 archived runs
    are exactly that shape. The record-kind check has to run before the
    outcome string is touched, because ``outcome.startswith`` on the
    absent field raises ``AttributeError`` -- without it the validator
    dies on the first real log it opens instead of returning a wrong
    cost.

    The dispatch line here carries landing coordinates too, so a mutant
    that somehow got past the attribute access would still be caught
    pairing a dispatch as a landing.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            '{"action_kind": "teleport", "landed_x": 3, "landed_y": 4}',
            _fix(3, 4, 470),
        ],
    )

    assert validate_teleport_cost(tmp_path)["samples"] == 0


def test_control_a_dispatch_line_beside_a_real_landing_still_pairs(tmp_path: Path) -> None:
    """Control: the dispatch line is ignored as a landing, not as an action.

    Its kind is teleport-free, so it does not contaminate the window it
    sits in -- the real outcome two lines later still pairs.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            '{"action_kind": "teleport", "landed_x": 3, "landed_y": 4}',
            _teleport(3, 4),
            _fix(3, 4, 470),
        ],
    )

    evidence = validate_teleport_cost(tmp_path)

    assert evidence["samples"] == 1
    assert evidence["exact"] == 1


def test_a_dated_non_bot_file_is_not_an_eligible_run(tmp_path: Path) -> None:
    """Eligibility needs the ``bot-`` prefix, not just digits at offset 4.

    The date token is read positionally (``name[4:12]``), so any name
    whose fifth through twelfth characters are digits parses as a run
    date -- ``sim-20260701-000000.events.jsonl`` included. Without the
    prefix check that file counts as a second post-fix run and its
    samples are added to the evidence a second time. The archive's own
    ``latest.events.jsonl`` escapes only by the accident that
    ``"st.event"`` is not numeric, so the prefix check is the only thing
    actually enforcing the documented exclusion.

    ``samples == 1`` is its own control: had the check wrongly excluded
    the ``bot-`` file as well, no pair would survive at all.
    """
    clean = [_fix(0, 0, 500), _teleport(3, 4), _fix(3, 4, 470)]
    _write_events(tmp_path, ELIGIBLE, clean)
    _write_events(tmp_path, "sim-20260701-000000.events.jsonl", clean)

    evidence = validate_teleport_cost(tmp_path)

    assert evidence["samples"] == 1
    assert "across 1 post-fix runs" in evidence["detail"]


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
