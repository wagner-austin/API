"""End-to-end tests for the in-game ``C`` statistics panel parsing.

The panel fixture mirrors the live probe capture from 20260610-2348
(``Play time: 41:25:23`` / ``Destroyed enemies: 65`` /
``Deactivated: 0`` / ``Promotion points: 121314``). Emission tests
drive the REAL pipeline through
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` and
the JSONL artifact via :class:`tests.conftest.FakeFileSystem`.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.diagnostics.account_stats import (
    AccountStatsDict,
    decode_account_stats,
    emit_account_stats_sample,
    encode_account_stats,
    parse_account_stats,
)
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.runtime_records import RuntimeEventRecordDict

_PANEL_TEXT = (
    "Name: Artax\n"
    " Troop: blue\n"
    " Rank: private (160)\n"
    " Awards: \n"
    " Inventory:\n"
    " 14 dual shots\n"
    " Statistics:\n"
    " Play time: 41:25:23\n"
    " Destroyed enemies: 65\n"
    " Deactivated: 0\n"
    " Promotion points: 121314\n"
    "LOCATION: 131,126\n"
)

_EXPECTED = AccountStatsDict(
    play_time_s=41 * 3600 + 25 * 60 + 23,
    destroyed_enemies=65,
    deactivated=0,
    promotion_points=121314,
    rank_name="private",
    leaderboard_position=160,
)


def _sample_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``session_account_stats`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "session_account_stats"
    ]


def test_parse_live_panel_text() -> None:
    """The live probe capture parses to the exact account stats."""
    assert parse_account_stats(_PANEL_TEXT) == _EXPECTED


def test_stat_lines_without_the_panel_marker_parse_to_nothing() -> None:
    """Stat-shaped text outside the panel is not account statistics.

    The marker is what says "this is the C panel". Without it the same
    five patterns would match anywhere they appear on the page -- a
    chat line quoting someone's play time is the obvious case -- and
    the session would record another player's numbers as its own
    account baseline.
    """
    text = (
        " Rank: private (139)\n"
        " Play time: 42:3:10\n"
        " Destroyed enemies: 89\n"
        " Deactivated: 0\n"
        " Promotion points: 157725\n"
    )

    assert parse_account_stats(text) is None


def test_panel_missing_the_play_time_line_parses_to_nothing() -> None:
    """A half-painted panel yields nothing rather than raising.

    Mid-render is a normal observable state: the marker paints before
    the stat lines, and a timed read can land between them. Reaching
    the integer conversion with an unmatched ``Play time`` calls
    ``.group()`` on ``None`` and raises ``AttributeError``, which
    crashed sessions 20260611-004251 / 004405 / 012807. Callers poll
    until non-``None``, so ``None`` is the correct answer.

    ``Rank`` is present on purpose: the rank check sits AFTER this one,
    so a page missing both would be caught by the later guard and the
    two could not be told apart.
    """
    text = (
        " Rank: private (139)\n"
        " Statistics:\n"
        " Destroyed enemies: 89\n"
        " Deactivated: 0\n"
        " Promotion points: 157725\n"
    )

    assert parse_account_stats(text) is None


def test_parse_unpadded_play_time_fields() -> None:
    """Single-digit minutes/seconds parse: the game does not zero-pad.

    Run 20260611-093904 rendered ``Play time: 42:3:10`` and all three
    capture attempts failed against a two-digit pattern, costing the
    session its account baseline.
    """
    text = (
        " Rank: private (139)\n"
        " Statistics:\n"
        " Play time: 42:3:10\n"
        " Destroyed enemies: 89\n"
        " Deactivated: 0\n"
        " Promotion points: 157725\n"
    )
    stats = parse_account_stats(text)
    if stats is None:
        raise AssertionError("expected unpadded play time to parse")
    assert stats["play_time_s"] == 42 * 3600 + 3 * 60 + 10
    assert stats["destroyed_enemies"] == 89
    assert stats["promotion_points"] == 157725
    assert stats["leaderboard_position"] == 139


def test_parse_returns_none_without_marker() -> None:
    """Page text without the Statistics marker is a defined absence."""
    assert parse_account_stats("Name: Artax\nLOCATION: 1,2\n") is None


def test_parse_returns_none_for_half_rendered_panel() -> None:
    """A marker without the stats lines is a not-yet-readable scrape.

    The panel paints incrementally; a read between the header and the
    stat lines is a normal transient, not corrupt input. Treating it
    as an error crashed sessions 20260611-004251/004405/012807.
    """
    assert parse_account_stats("Statistics:\n Play time: 41:25:23\n") is None


def test_parse_returns_none_for_missing_rank_line() -> None:
    """Stats lines without the rank line are still mid-render."""
    text = (
        "Statistics:\n"
        " Play time: 41:25:23\n"
        " Destroyed enemies: 65\n"
        " Deactivated: 0\n"
        " Promotion points: 121314\n"
    )
    assert parse_account_stats(text) is None


def test_encode_decode_round_trip() -> None:
    """Encode then decode preserves every field."""
    assert decode_account_stats(encode_account_stats(_EXPECTED)) == _EXPECTED


def test_emit_visible_sample(fake_fs: FakeFileSystem) -> None:
    """A parsed panel emits the full diagnostic through the artifact."""
    artifacts = configure_bot_runtime_logging("20260610-120000")

    emit_account_stats_sample(_EXPECTED, phase="startup")

    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "session_account_stats",
        "phase": "startup",
        "panel_visible": True,
        "play_time_s": 149123,
        "destroyed_enemies": 65,
        "deactivated": 0,
        "promotion_points": 121314,
        "rank_name": "private",
        "leaderboard_position": 160,
    }


def test_emit_absent_panel(fake_fs: FakeFileSystem) -> None:
    """A raced scrape emits a loud absence marker, not silence."""
    artifacts = configure_bot_runtime_logging("20260610-120000")

    emit_account_stats_sample(None, phase="startup", marker_present=True, scrape_chars=42)

    records = _sample_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "session_account_stats",
        "phase": "startup",
        "panel_visible": False,
        "marker_present": True,
        "scrape_chars": 42,
    }
