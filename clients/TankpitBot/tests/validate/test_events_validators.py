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


def _fuel(old: int, new: int) -> str:
    """Build one WORLD fuel-movement line.

    Args:
        old: Fuel before the movement.
        new: Fuel after it.

    Returns:
        JSON line text.
    """
    return f'{{"channel": "WORLD", "message": "Fuel: {old} -> {new} ({new - old:+d})"}}'


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


def test_a_second_fuel_movement_excludes_the_window(tmp_path: Path) -> None:
    """A hit inside the window is excluded, not reported as bad physics.

    The measured cost is ``pre.fuel - post.fuel``, so it is the
    teleport's cost only while the teleport is the only thing moving
    fuel between the fixes. Every one of the 7 mismatches this validator
    used to report was a second movement folded into that difference --
    4 single hits at 45, one dual at 90, two pickups at +91 and +484 --
    and the teleport's own debit was exact in all seven.

    Here the 3-4-5 hop debits its exact 30 and then a 45 hit lands. The
    window must vanish from the evidence rather than count as a 75-cost
    teleport, because a claim that reports a contaminated measurement as
    a failure is worse than one that reports fewer samples.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            _fuel(500, 470),
            _fuel(470, 425),
            _fix(3, 4, 425),
        ],
    )

    evidence = validate_teleport_cost(tmp_path)

    assert (evidence["samples"], evidence["mismatches"]) == (0, 0)


def test_control_one_fuel_movement_still_pairs(tmp_path: Path) -> None:
    """Control: the same hop with only its own debit is measured.

    Distinguishes the exclusion above from the fuel lines simply
    breaking the pairing -- one movement is the normal shape of every
    clean window in the archive.

    The landing confirm rides along because the real batch carries one
    between the debit and the closing fix: a WORLD line with a message
    that is not a fuel movement must not be mistaken for one.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            _fuel(500, 470),
            '{"channel": "WORLD", "message": "TELEPORT_LANDED: server confirmed teleport"}',
            _fix(3, 4, 470),
        ],
    )

    evidence = validate_teleport_cost(tmp_path)

    assert (evidence["samples"], evidence["exact"]) == (1, 1)


def test_a_zero_delta_fuel_line_is_not_a_movement(tmp_path: Path) -> None:
    """``(+0)`` readings are re-reports, not a second movement.

    The wire re-states the same total routinely -- the landing batch
    carries one between the debit and the confirm -- so counting them
    would exclude every clean window in the archive.
    """
    _write_events(
        tmp_path,
        ELIGIBLE,
        [
            _fix(0, 0, 500),
            _teleport(3, 4),
            _fuel(500, 470),
            _fuel(470, 470),
            _fix(3, 4, 470),
        ],
    )

    evidence = validate_teleport_cost(tmp_path)

    assert (evidence["samples"], evidence["exact"]) == (1, 1)


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


def test_a_record_is_classified_once_by_its_diagnostic_kind() -> None:
    """A fix that also carries an action kind is a fix, and nothing else.

    ``_scan_record`` is a cascade: the two diagnostic kinds it knows are
    claimed first, and only an unclaimed record is read for an
    ``action_kind``. No archived record is both -- 38,945
    self_alignment_samples and 14,375 container_pickup_dispatched lines
    across the 427 runs, none carrying an action kind -- which is why
    removing either return currently changes no published number.

    The shape is constructed, and the reason to hold the line anyway is
    that it is one keyword away: an ``emit_diagnostic`` call that stamps
    action context onto an alignment sample would make every fix ALSO an
    action line, and action lines are what exclude a fix window from the
    pairing. The teleport-cost evidence would quietly lose samples with
    nothing reporting why.
    """
    from tankpit_bot.validate.events_validators import _EventScan, _scan_record

    scan = _EventScan(
        fixes=[],
        teleport_outcomes=[],
        action_lines=[],
        fuel_moves=[],
        skipped_lines=0,
    )

    _scan_record(
        scan,
        {
            "diagnostic_kind": "self_alignment_sample",
            "belief_x": 3,
            "belief_y": 4,
            "belief_fuel": 470,
            "action_kind": "walk",
        },
        7,
    )

    assert [(fix["x"], fix["y"], fix["fuel"]) for fix in scan["fixes"]] == [(3, 4, 470)]
    assert scan["action_lines"] == []


def test_a_pickup_dispatch_is_one_action_line_not_two() -> None:
    """The pickup arm claims the record; its action kind is not read again.

    Same cascade, second arm. Without its return the dispatch is
    appended once as ``pickup`` and then a second time under whatever
    ``action_kind`` it carries, so one dispatched pickup contaminates a
    fix window twice over and counts as two actions in the report.
    """
    from tankpit_bot.validate.events_validators import _EventScan, _scan_record

    scan = _EventScan(
        fixes=[],
        teleport_outcomes=[],
        action_lines=[],
        fuel_moves=[],
        skipped_lines=0,
    )

    _scan_record(
        scan,
        {"diagnostic_kind": "container_pickup_dispatched", "action_kind": "collect"},
        11,
    )

    assert [(line["line"], line["kind"]) for line in scan["action_lines"]] == [(11, "pickup")]
