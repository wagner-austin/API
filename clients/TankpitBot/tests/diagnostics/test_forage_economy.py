"""Tests for :mod:`tankpit_bot.diagnostics.forage_economy`.

Fixtures are hand-written JSONL records (validated by the REAL
:func:`tankpit_bot.runtime_records.decode_runtime_event_record`
through the real :func:`load_event_records` path) because the
economy metrics attribute wall-clock time between records — the
live emit pipeline stamps real timestamps, which a test cannot
control. The bytes match what a production run writes; only the
timestamps are chosen.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.forage_economy import (
    build_forage_economy,
    main,
    render_forage_comparison,
    render_forage_economy,
)

_SOURCE = Path("runs/bot/test.events.jsonl")
_BASELINE = Path("runs/bot/baseline.events.jsonl")


def _record(
    timestamp: str,
    *,
    bot_state: str | None = None,
    **fields: str | int | float | bool,
) -> dict[str, str | int | float | bool]:
    """Build one JSONL record with controlled timestamp and fields."""
    row: dict[str, str | int | float | bool] = {
        "timestamp": timestamp,
        "level": "INFO",
        "logger": "tankpit_bot.runtime.events",
        "mode": "bot",
        "channel": "DIAGNOSTIC",
        "message": "diagnostic_kind=" + str(fields.get("diagnostic_kind", "test")),
    }
    if bot_state is not None:
        row["bot_state"] = bot_state
    row.update(fields)
    return row


def _write_jsonl(
    fs: FakeFileSystem,
    path: Path,
    rows: list[dict[str, str | int | float | bool]],
) -> None:
    """Write records as a JSONL artifact into the fake file system."""
    fs.write_text(path, "\n".join(dump_json_str(dict(row), compact=True) for row in rows) + "\n")


def _economy_rows() -> list[dict[str, str | int | float | bool]]:
    """A small run exercising every routed record kind."""
    return [
        _record(
            "2026-07-26T10:00:00",
            bot_state="HUNT/ENGAGE",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="hit",
        ),
        _record(
            "2026-07-26T10:00:10",
            bot_state="HUNT/ENGAGE",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="miss",
        ),
        _record(
            "2026-07-26T10:00:20",
            bot_state="COLLECT/SENSE",
            diagnostic_kind="action_outcome",
            action_kind="scan",
            outcome="radar_complete",
        ),
        _record(
            "2026-07-26T10:00:30",
            bot_state="HUNT/SCAN_ON_LANDING",
            diagnostic_kind="action_outcome",
            action_kind="scan",
            outcome="radar_complete",
        ),
        _record(
            "2026-07-26T10:00:40",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="container_consumed",
        ),
        _record(
            "2026-07-26T10:00:50",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="clamped_transfer",
        ),
        _record(
            "2026-07-26T10:01:00",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="equipment_gain",
            armor=0,
            dual=4,
            missile=0,
            homing=2,
            radar=3,
        ),
        _record(
            "2026-07-26T10:01:10",
            bot_state="COLLECT/SEARCH",
            diagnostic_kind="hop_selected",
            hop_kind="dot",
            target_x=10,
            target_y=20,
            score=0.5,
            cost=7,
        ),
        _record(
            "2026-07-26T10:01:20",
            bot_state="COLLECT/SEARCH",
            diagnostic_kind="hop_selected",
            hop_kind="equipment",
            target_x=30,
            target_y=40,
            landing_x=30,
            landing_y=41,
            cost=9,
        ),
        _record(
            "2026-07-26T10:01:30",
            bot_state="COLLECT/SEARCH",
            diagnostic_kind="hop_declined",
            hop_kind="equipment",
            external=8,
            no_landing=8,
            reserve_blocked=0,
            fuel=1100,
            landing_reserve=650,
        ),
        _record(
            "2026-07-26T10:01:40",
            bot_state="COLLECT/SEARCH",
            diagnostic_kind="hop_declined",
            hop_kind="equipment",
            no_candidates=1,
        ),
        _record(
            "2026-07-26T10:01:50",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="command_error",
            error_code=5,
            error_name="tank_full_clamp_receipt",
        ),
        _record(
            "2026-07-26T10:02:00",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="command_error",
            error_code=4,
            error_name="empty_container",
        ),
        _record(
            "2026-07-26T10:02:10",
            bot_state="UNSET",
            diagnostic_kind="noise",
        ),
        _record(
            "2026-07-26T10:02:20",
            bot_state="COLLECT/PICKUP",
            diagnostic_kind="equipment_gain",
            dual=1,
        ),
        _record(
            "2026-07-26T10:02:30",
            diagnostic_kind="noise",
        ),
        _record(
            "2026-07-26T10:02:40",
            diagnostic_kind="session_scorecard",
            ticks=60,
            kills=2,
            hits=1,
            misses=1,
        ),
    ]


def test_build_forage_economy_counts_every_metric(fake_fs: FakeFileSystem) -> None:
    """The builder routes every record kind into the right counter."""
    _write_jsonl(fake_fs, _SOURCE, _economy_rows())
    report = build_forage_economy(_SOURCE)
    assert report["source_path"] == str(_SOURCE)
    assert report["span_seconds"] == 160.0
    assert report["hunt_seconds"] == 30.0
    assert report["collect_seconds"] == 110.0
    assert report["other_seconds"] == 20.0
    assert report["kills"] == 2
    assert report["forage_scans"] == 1
    assert report["pickups_consumed"] == 1
    assert report["pickups_clamped"] == 1
    assert report["equipment_pickups"] == 2
    assert report["weapons_gained"] == 7
    assert report["radars_gained"] == 3
    assert report["hops_dot"] == 1
    assert report["hops_equipment"] == 1
    assert report["hops_declined"] == 2
    assert report["no_landing_rejections"] == 8
    assert report["shots_hit"] == 1
    assert report["shots_missed"] == 1
    assert report["clamp_receipts"] == 1
    assert report["other_command_errors"] == 1


def test_build_forage_economy_empty_artifact_is_all_zero(fake_fs: FakeFileSystem) -> None:
    """A single-record artifact yields zero spans and None kills."""
    _write_jsonl(fake_fs, _SOURCE, [_record("2026-07-26T10:00:00", diagnostic_kind="noise")])
    report = build_forage_economy(_SOURCE)
    assert report["span_seconds"] == 0.0
    assert report["kills"] is None
    assert report["forage_scans"] == 0


def test_render_forage_economy_shows_ratios(fake_fs: FakeFileSystem) -> None:
    """The renderer prints the deciding ratios."""
    _write_jsonl(fake_fs, _SOURCE, _economy_rows())
    text = render_forage_economy(build_forage_economy(_SOURCE))
    assert "=== FORAGE ECONOMY ===" in text
    assert "kills: 2" in text
    assert "forage viewports: 1 (0.50/kill)" in text
    assert "pickups: 2 (1 consumed + 1 clamped, 2.00/viewport)" in text
    assert "equipment pickups: 2 -> weapons 7 (3.50/pickup), radars 3" in text
    assert "hops: dot 1, equipment 1, declined 2 (no_landing candidate-evals 8)" in text
    assert "shots: 2 (1 hit, 1 missed)" in text
    assert "command errors: 1 clamp receipts, 1 other" in text


def test_render_forage_economy_handles_missing_scorecard(fake_fs: FakeFileSystem) -> None:
    """Without a scorecard, kills and per-kill ratios degrade loudly."""
    rows = [row for row in _economy_rows() if row.get("diagnostic_kind") != "session_scorecard"]
    _write_jsonl(fake_fs, _SOURCE, rows)
    text = render_forage_economy(build_forage_economy(_SOURCE))
    assert "kills: unknown (no scorecard)" in text
    assert "forage viewports: 1 (-/kill)" in text


def test_render_forage_economy_zero_denominators_render_dash(fake_fs: FakeFileSystem) -> None:
    """Zero forage scans and zero pickups render '-' ratios."""
    _write_jsonl(fake_fs, _SOURCE, [_record("2026-07-26T10:00:00", diagnostic_kind="noise")])
    text = render_forage_economy(build_forage_economy(_SOURCE))
    assert "pickups: 0 (0 consumed + 0 clamped, -/viewport)" in text
    assert "equipment pickups: 0 -> weapons 0 (-/pickup), radars 0" in text


def test_render_forage_comparison_shows_delta(fake_fs: FakeFileSystem) -> None:
    """The two-run renderer prints both reports and the delta section."""
    _write_jsonl(fake_fs, _SOURCE, _economy_rows())
    _write_jsonl(fake_fs, _BASELINE, _economy_rows())
    current = build_forage_economy(_SOURCE)
    baseline = build_forage_economy(_BASELINE)
    text = render_forage_comparison(current, baseline)
    assert text.count("=== FORAGE ECONOMY ===") == 2
    assert "=== DELTA (current vs baseline) ===" in text
    assert "span: 160 s vs 160 s" in text
    assert "weapons/pickup: 3.50 vs 3.50" in text


def test_main_single_path_renders_report(fake_fs: FakeFileSystem) -> None:
    """``main`` with one path renders the single-run report."""
    _write_jsonl(fake_fs, _SOURCE, _economy_rows())
    original = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-forage-economy", str(_SOURCE)]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original


def test_main_two_paths_renders_comparison(fake_fs: FakeFileSystem) -> None:
    """``main`` with two paths renders the comparison."""
    _write_jsonl(fake_fs, _SOURCE, _economy_rows())
    _write_jsonl(fake_fs, _BASELINE, _economy_rows())
    original = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: [
        "tankpit-forage-economy",
        str(_SOURCE),
        str(_BASELINE),
    ]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original


def test_main_defaults_to_latest_events(fake_fs: FakeFileSystem) -> None:
    """``main`` with no args reads runs/bot/latest.events.jsonl."""
    _write_jsonl(fake_fs, Path("runs/bot/latest.events.jsonl"), _economy_rows())
    original = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-forage-economy"]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original
