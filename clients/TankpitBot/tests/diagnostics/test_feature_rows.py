"""Tests for the tick-level feature table.

The table is a DERIVATION over an events artifact, so every test here
drives the real builder over a real artifact written to disk -- no
stubbed rows. What is asserted is the shape of the join: which ticks
become rows, which events fold into which columns, and what an absent
field is allowed to look like.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.fs import WriteTextProtocol
from tankpit_bot.diagnostics.feature_rows import (
    NO_ACTION,
    build_feature_rows,
    decode_feature_row,
    encode_feature_row,
    main,
    render_feature_rows,
    write_feature_rows,
)


def _event(timestamp: str, **fields: str | int | bool) -> str:
    """Encode one DIAGNOSTIC event line.

    Args:
        timestamp: Event timestamp.
        **fields: Structured fields spread at the top level.

    Returns:
        One JSON line.
    """
    return dump_json_str(
        {
            "timestamp": timestamp,
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "DIAGNOSTIC",
            "message": f"diagnostic_kind={fields.get('diagnostic_kind', '')}",
            **fields,
        }
    )


def _write_session(path: Path, lines: list[str]) -> Path:
    """Write a synthetic events artifact.

    Args:
        path: Target file path.
        lines: JSONL lines.

    Returns:
        The written path.
    """
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_tick_becomes_one_row_carrying_its_action_and_its_counts(tmp_path: Path) -> None:
    """The join is per tick: one action, plus what else that tick emitted.

    Tick 5 dispatches a scan that completes while two hop lanes decline
    and a radar fires -- all four events land on one row rather than
    four, which is the whole point of the reshape.
    """
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="hop_declined", tick_n=5),
            _event("2026-09-01T00:00:01", diagnostic_kind="hop_declined", tick_n=5),
            _event("2026-09-01T00:00:01", diagnostic_kind="radar_dispatch", tick_n=5),
            _event(
                "2026-09-01T00:00:02",
                diagnostic_kind="action_outcome",
                tick_n=5,
                bot_state="COLLECT/SENSE",
                action_kind="scan",
                outcome="radar_complete",
                duration_ms=1484,
                attempt_id=1,
            ),
        ],
    )

    rows = build_feature_rows(source)

    assert len(rows) == 1
    assert rows[0] == {
        "tick_n": 5,
        "bot_state": "COLLECT/SENSE",
        "action_kind": "scan",
        "outcome": "radar_complete",
        "duration_ms": 1484,
        "attempt_id": 1,
        "hop_declined": 2,
        "radar_dispatch": 1,
        "container_pickup_dispatched": 0,
        "plan_released": 0,
        "command_error": 0,
        "fleet_knowledge_merged": 0,
    }


def test_every_counted_diagnostic_kind_folds_into_its_own_column(
    tmp_path: Path,
) -> None:
    """Each counted kind lands in its own column, none in another's.

    The fold spells the six kinds out as a branch rather than indexing
    the row by a variable, so nothing but a test per kind proves the
    branch goes where its name says.
    """
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="hop_declined", tick_n=3),
            _event("2026-09-01T00:00:01", diagnostic_kind="radar_dispatch", tick_n=3),
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="container_pickup_dispatched",
                tick_n=3,
            ),
            _event("2026-09-01T00:00:01", diagnostic_kind="plan_released", tick_n=3),
            _event("2026-09-01T00:00:01", diagnostic_kind="command_error", tick_n=3),
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="fleet_knowledge_merged",
                tick_n=3,
            ),
            # A kind the table does not count must leave every column
            # alone rather than falling into the last branch.
            _event("2026-09-01T00:00:01", diagnostic_kind="liveness_stall", tick_n=3),
        ],
    )

    rows = build_feature_rows(source)

    assert len(rows) == 1
    assert rows[0]["hop_declined"] == 1
    assert rows[0]["radar_dispatch"] == 1
    assert rows[0]["container_pickup_dispatched"] == 1
    assert rows[0]["plan_released"] == 1
    assert rows[0]["command_error"] == 1
    assert rows[0]["fleet_knowledge_merged"] == 1


def test_rows_are_tick_ordered_regardless_of_file_order(tmp_path: Path) -> None:
    """Ticks sort ascending even when the artifact interleaves them."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:03", diagnostic_kind="plan_released", tick_n=9),
            _event("2026-09-01T00:00:01", diagnostic_kind="plan_released", tick_n=2),
            _event("2026-09-01T00:00:02", diagnostic_kind="plan_released", tick_n=7),
        ],
    )

    assert [row["tick_n"] for row in build_feature_rows(source)] == [2, 7, 9]


def test_a_tick_that_emitted_nothing_is_absent_not_zero_filled(tmp_path: Path) -> None:
    """Only ticks the artifact recorded become rows.

    Zero-filling the gap would put rows in the table that no tick
    produced, which a model would read as real observations.
    """
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="plan_released", tick_n=1),
            _event("2026-09-01T00:00:09", diagnostic_kind="plan_released", tick_n=9),
        ],
    )

    assert [row["tick_n"] for row in build_feature_rows(source)] == [1, 9]


def test_session_level_records_without_a_tick_are_skipped(tmp_path: Path) -> None:
    """``session_room_joined`` belongs to no tick, so it makes no row."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="session_room_joined", room_id="6"),
            _event("2026-09-01T00:00:02", diagnostic_kind="plan_released", tick_n=4),
        ],
    )

    assert [row["tick_n"] for row in build_feature_rows(source)] == [4]


def test_non_diagnostic_lines_are_skipped(tmp_path: Path) -> None:
    """A record without a diagnostic_kind contributes no row."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            dump_json_str(
                {
                    "timestamp": "2026-09-01T00:00:01",
                    "level": "INFO",
                    "logger": "tankpit_bot.runtime.events",
                    "mode": "bot",
                    "channel": "WIRE",
                    "message": "radar",
                    "tick_n": 3,
                }
            ),
            _event("2026-09-01T00:00:02", diagnostic_kind="plan_released", tick_n=4),
        ],
    )

    assert [row["tick_n"] for row in build_feature_rows(source)] == [4]


def test_an_outcome_missing_its_numbers_keeps_the_absent_sentinel(tmp_path: Path) -> None:
    """A missing duration reads -1, never 0.

    Zero is a real duration; imputing it would invent a fact. The row
    still records the action and outcome that DID arrive.
    """
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="action_outcome",
                tick_n=2,
                bot_state="HUNT/ENGAGE",
                action_kind="shoot",
                outcome="hit",
            ),
        ],
    )

    row = build_feature_rows(source)[0]

    assert row["action_kind"] == "shoot"
    assert row["outcome"] == "hit"
    assert row["duration_ms"] == -1
    assert row["attempt_id"] == -1


def test_a_retried_tick_is_reported_by_how_it_ended(tmp_path: Path) -> None:
    """Two outcomes on one tick: the last wins, attempt_id carries depth."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="action_outcome",
                tick_n=3,
                action_kind="teleport",
                outcome="superseded",
                duration_ms=100,
                attempt_id=1,
            ),
            _event(
                "2026-09-01T00:00:02",
                diagnostic_kind="action_outcome",
                tick_n=3,
                action_kind="teleport",
                outcome="landed_exact",
                duration_ms=2141,
                attempt_id=2,
            ),
        ],
    )

    row = build_feature_rows(source)[0]

    assert row["outcome"] == "landed_exact"
    assert row["attempt_id"] == 2


def test_a_tick_with_no_action_records_the_absence_explicitly(tmp_path: Path) -> None:
    """A tick that only declined hops has no action, and says so."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [_event("2026-09-01T00:00:01", diagnostic_kind="hop_declined", tick_n=6)],
    )

    row = build_feature_rows(source)[0]

    assert row["action_kind"] == NO_ACTION
    assert row["outcome"] == NO_ACTION
    assert row["hop_declined"] == 1


def test_bot_state_is_taken_from_the_first_event_that_carries_one(tmp_path: Path) -> None:
    """State rides on many kinds; the tick's first sighting wins."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="hop_declined", tick_n=8),
            _event(
                "2026-09-01T00:00:02",
                diagnostic_kind="plan_released",
                tick_n=8,
                bot_state="COLLECT/PICKUP",
            ),
            _event(
                "2026-09-01T00:00:03",
                diagnostic_kind="command_error",
                tick_n=8,
                bot_state="HUNT/ENGAGE",
            ),
        ],
    )

    assert build_feature_rows(source)[0]["bot_state"] == "COLLECT/PICKUP"


def test_a_row_survives_an_encode_decode_round_trip() -> None:
    """Every field crosses the JSON boundary intact."""
    source_row: JSONObject = {
        "tick_n": 12,
        "bot_state": "COLLECT/PICKUP",
        "action_kind": "collect",
        "outcome": "pickup_empty",
        "duration_ms": 1918,
        "attempt_id": 1,
        "hop_declined": 0,
        "radar_dispatch": 1,
        "container_pickup_dispatched": 1,
        "plan_released": 1,
        "command_error": 1,
        "fleet_knowledge_merged": 3,
    }

    restored = decode_feature_row(encode_feature_row(decode_feature_row(source_row)))

    assert restored == source_row


def test_decoding_a_row_missing_a_field_raises() -> None:
    """A malformed row is surfaced, never defaulted into the table."""
    with pytest.raises(JSONTypeError):
        decode_feature_row({"tick_n": 1})


def test_the_table_is_written_as_one_json_row_per_line(tmp_path: Path) -> None:
    """JSONL so a corpus streams and many runs concatenate by cat."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event("2026-09-01T00:00:01", diagnostic_kind="plan_released", tick_n=1),
            _event("2026-09-01T00:00:02", diagnostic_kind="plan_released", tick_n=2),
        ],
    )
    written: list[tuple[Path, str]] = []

    def fake_write(path: Path, content: str) -> None:
        written.append((Path(path), content))

    original: WriteTextProtocol = _test_hooks.write_text
    _test_hooks.write_text = fake_write
    try:
        destination = write_feature_rows(source, build_feature_rows(source))
    finally:
        _test_hooks.write_text = original

    assert destination.name == "s.features.jsonl"
    path, content = written[0]
    assert path == destination
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert decode_feature_row(narrow_json_to_dict(load_json_str(lines[0])))["tick_n"] == 1


def test_the_report_names_every_tick_it_rendered(tmp_path: Path) -> None:
    """The human report carries the row count and one line per tick."""
    source = _write_session(
        tmp_path / "s.events.jsonl",
        [
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="action_outcome",
                tick_n=7,
                bot_state="COLLECT/APPROACH",
                action_kind="teleport",
                outcome="landed_exact",
                duration_ms=2141,
                attempt_id=1,
            ),
        ],
    )

    rendered = render_feature_rows(build_feature_rows(source))

    assert "FEATURE ROWS (1 ticks)" in rendered
    assert "teleport" in rendered
    assert "landed_exact" in rendered


def test_the_cli_writes_the_table_beside_its_source(tmp_path: Path) -> None:
    """End to end: the command persists real JSONL a consumer can read.

    Exercises the same entrypoint the ``tankpit-feature-rows`` script
    resolves to, over an artifact on disk -- no fake path, no stubbed
    writer, so the file this asserts on is the file a run produces.
    """
    source = _write_session(
        tmp_path / "run.events.jsonl",
        [
            _event(
                "2026-09-01T00:00:01",
                diagnostic_kind="action_outcome",
                tick_n=1,
                bot_state="COLLECT/SENSE",
                action_kind="scan",
                outcome="radar_complete",
                duration_ms=1484,
                attempt_id=1,
            ),
            _event("2026-09-01T00:00:02", diagnostic_kind="hop_declined", tick_n=2),
        ],
    )
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-feature-rows", str(source)]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv

    written = (tmp_path / "run.features.jsonl").read_text(encoding="utf-8")
    lines = written.strip().splitlines()
    rows = [decode_feature_row(narrow_json_to_dict(load_json_str(line))) for line in lines]
    assert [row["tick_n"] for row in rows] == [1, 2]
    assert rows[0]["outcome"] == "radar_complete"
    assert rows[1]["action_kind"] == NO_ACTION
