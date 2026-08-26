"""Tests for :mod:`tankpit_bot.diagnostics.engagement_ledger`.

Fixtures are hand-written JSONL records validated by the real decode
path (the forage-economy fixture discipline): the bytes match what a
production run writes; only timestamps and ids are chosen. The
flagship scenario is the 2026-08-26 return-fire/solvency collision —
an engagement with a break and a negative damage trade must flag.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.engagement_ledger import (
    build_engagement_ledger,
    main,
    render_engagement_ledger,
)

_SOURCE = Path("runs/bot/test.events.jsonl")


def _record(
    timestamp: str,
    *,
    channel: str = "DIAGNOSTIC",
    message: str = "",
    **fields: str | int | float | bool,
) -> dict[str, str | int | float | bool]:
    """Build one JSONL record with controlled timestamp and fields."""
    row: dict[str, str | int | float | bool] = {
        "timestamp": timestamp,
        "level": "INFO",
        "logger": "tankpit_bot.runtime.events",
        "mode": "bot",
        "channel": channel,
        "message": message or "diagnostic_kind=" + str(fields.get("diagnostic_kind", "test")),
    }
    row.update(fields)
    return row


def _shot(timestamp: str, target_id: int) -> dict[str, str | int | float | bool]:
    """One wire shot dispatch record."""
    return _record(timestamp, channel="WIRE", message=f"shoot(100,100,id={target_id})")


def _self_sample(timestamp: str, tank_id: int) -> dict[str, str | int | float | bool]:
    """One self-alignment sample naming our wire id."""
    return _record(
        timestamp,
        diagnostic_kind="self_alignment_sample",
        belief_tank_id=tank_id,
    )


def _write_jsonl(
    fs: FakeFileSystem,
    path: Path,
    rows: list[dict[str, str | int | float | bool]],
) -> None:
    """Write records as a JSONL artifact into the fake file system."""
    fs.write_text(path, "\n".join(dump_json_str(dict(row), compact=True) for row in rows) + "\n")


def _collision_rows() -> list[dict[str, str | int | float | bool]]:
    """The flagship run: a kill, the solvency-collision fight, a death.

    Enemy 50 is killed cleanly. Enemy 60 gets a break and loses the
    damage trade — the collision signature. Enemy 70 kills us. Enemy
    80 exists only in the damage ledger (we never fired). Ground fire
    (id 0) and an unrelated kill (90 kills 91) must not create rows.
    """
    return [
        _self_sample("2026-08-26T04:00:00", 601),
        _self_sample("2026-08-26T04:00:01", 601),
        _record(
            "2026-08-26T04:00:02",
            diagnostic_kind="tank_identity",
            tank_id=50,
            name="red-1",
        ),
        _record(
            "2026-08-26T04:00:03",
            channel="WIRE",
            message="teleport(10,10)",
        ),
        _shot("2026-08-26T04:00:04", 0),
        _shot("2026-08-26T04:00:05", 50),
        _shot("2026-08-26T04:00:07", 50),
        _record(
            "2026-08-26T04:00:09",
            diagnostic_kind="tank_deactivated",
            victim_id=50,
            killer_id=601,
        ),
        _shot("2026-08-26T04:00:20", 60),
        _record(
            "2026-08-26T04:00:22",
            diagnostic_kind="engagement_break",
            target_id=60,
            target_name="red-6",
        ),
        _shot("2026-08-26T04:00:24", 60),
        _record(
            "2026-08-26T04:00:30",
            diagnostic_kind="tank_deactivated",
            victim_id=90,
            killer_id=91,
        ),
        _shot("2026-08-26T04:00:40", 70),
        _record(
            "2026-08-26T04:00:42",
            diagnostic_kind="tank_deactivated",
            victim_id=601,
            killer_id=70,
        ),
        _record(
            "2026-08-26T04:00:50",
            diagnostic_kind="tank_identity",
            tank_id=70,
            name="purple-9",
        ),
        _record(
            "2026-08-26T04:00:55",
            diagnostic_kind="damage_ledger",
            dealt="red-1(50): dual=2 fuel=180; red-6(60): dual=1 fuel=90; junk-row",
            taken="red-6(60): dual=3 fuel=270; ghost(80): single=4 fuel=180; ",
        ),
    ]


def test_collision_run_builds_the_expected_ledger(fake_fs: FakeFileSystem) -> None:
    """Kills, deaths, name resolution, and the collision flag all land."""
    _write_jsonl(fake_fs, _SOURCE, _collision_rows())

    ledger = build_engagement_ledger(_SOURCE)

    assert ledger["self_id"] == 601
    assert ledger["deaths"] == 1
    assert ledger["kills"] == 1
    by_id = {row["target_id"]: row for row in ledger["engagements"]}
    assert sorted(by_id) == [50, 60, 70, 80]
    assert by_id[50]["target_name"] == "red-1"
    assert by_id[50]["outcome"] == "kill"
    assert by_id[50]["shots"] == 2
    assert by_id[50]["seconds_to_kill"] == 4.0
    assert by_id[60]["breaks"] == 1
    assert by_id[60]["target_name"] == "red-6"
    assert by_id[60]["dealt_fuel"] == 90
    assert by_id[60]["taken_fuel"] == 270
    assert by_id[70]["outcome"] == "killed_us"
    assert by_id[70]["target_name"] == "purple-9"
    assert by_id[80]["shots"] == 0
    assert by_id[80]["target_name"] == "ghost"
    assert by_id[80]["taken_fuel"] == 180
    # Ground fire (id 0) and the unrelated 90/91 kill created no rows.
    assert ledger["negative_trades"] == 1
    assert ledger["post_break_negative_trades"] == 1
    # Rows we shot at order by first shot; ledger-only rows sort last.
    assert [row["target_id"] for row in ledger["engagements"]] == [50, 60, 70, 80]


def test_collision_run_renders_the_flag(fake_fs: FakeFileSystem) -> None:
    """The report names the loser and prints the collision flag."""
    _write_jsonl(fake_fs, _SOURCE, _collision_rows())

    text = render_engagement_ledger(build_engagement_ledger(_SOURCE))

    assert "self id: 601 | engagements 4 | kills 1 | deaths 1" in text
    assert "red-6(60) taken 270 > dealt 90 AFTER A BREAK" in text
    assert "FLAG: 1 engagement(s) lost the damage trade after a break" in text
    # The ledger-only ghost row renders but never flags (shots=0).
    assert "ghost" in text
    assert "t2k" in text


def test_deactivations_before_the_self_id_are_unattributable(fake_fs: FakeFileSystem) -> None:
    """Without a self id no kill or death can be attributed."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _shot("2026-08-26T04:00:00", 50),
            _record(
                "2026-08-26T04:00:01",
                diagnostic_kind="tank_deactivated",
                victim_id=50,
                killer_id=601,
            ),
        ],
    )

    ledger = build_engagement_ledger(_SOURCE)

    assert ledger["self_id"] is None
    assert ledger["kills"] == 0
    assert ledger["deaths"] == 0
    assert ledger["engagements"][0]["outcome"] == "open"
    assert "self id: unknown" in render_engagement_ledger(ledger)


def test_death_by_a_tank_we_never_fought_books_no_row(fake_fs: FakeFileSystem) -> None:
    """Our death counts even when the killer has no engagement row."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _self_sample("2026-08-26T04:00:00", 601),
            _record(
                "2026-08-26T04:00:01",
                diagnostic_kind="tank_deactivated",
                victim_id=601,
                killer_id=999,
            ),
        ],
    )

    ledger = build_engagement_ledger(_SOURCE)

    assert ledger["deaths"] == 1
    assert ledger["engagements"] == []


def test_kill_of_a_ledger_only_row_has_no_time_to_kill(fake_fs: FakeFileSystem) -> None:
    """A kill with no recorded first shot leaves seconds_to_kill unset."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _self_sample("2026-08-26T04:00:00", 601),
            _record(
                "2026-08-26T04:00:01",
                diagnostic_kind="engagement_break",
                target_id=50,
            ),
            _record(
                "2026-08-26T04:00:02",
                diagnostic_kind="tank_deactivated",
                victim_id=50,
                killer_id=601,
            ),
        ],
    )

    ledger = build_engagement_ledger(_SOURCE)

    row = ledger["engagements"][0]
    assert row["outcome"] == "kill"
    assert row["seconds_to_kill"] is None
    assert row["target_name"] == "?"
    assert row["breaks"] == 1
    rendered = render_engagement_ledger(ledger)
    assert "kill" in rendered
    assert "FLAG" not in rendered


def test_break_without_a_target_id_books_nothing(fake_fs: FakeFileSystem) -> None:
    """A malformed break record creates no engagement row."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [_record("2026-08-26T04:00:00", diagnostic_kind="engagement_break")],
    )

    assert build_engagement_ledger(_SOURCE)["engagements"] == []


def test_identity_for_an_unengaged_tank_books_nothing(fake_fs: FakeFileSystem) -> None:
    """Identity records only name enemies we actually have rows for."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _record(
                "2026-08-26T04:00:00",
                diagnostic_kind="tank_identity",
                tank_id=50,
                name="red-1",
            )
        ],
    )

    assert build_engagement_ledger(_SOURCE)["engagements"] == []


def test_identity_before_the_first_shot_still_names_the_row(fake_fs: FakeFileSystem) -> None:
    """An identity that precedes the engagement names it on the late pass."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _record(
                "2026-08-26T04:00:00",
                diagnostic_kind="tank_identity",
                tank_id=50,
                name="red-1",
            ),
            _shot("2026-08-26T04:00:05", 50),
        ],
    )

    ledger = build_engagement_ledger(_SOURCE)

    assert ledger["engagements"][0]["target_name"] == "red-1"


def test_damage_ledger_with_only_a_dealt_side(fake_fs: FakeFileSystem) -> None:
    """A dealt-only ledger fills dealt fuel and leaves taken at zero."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _shot("2026-08-26T04:00:00", 50),
            _record(
                "2026-08-26T04:00:01",
                diagnostic_kind="damage_ledger",
                dealt="red-1(50): dual=2 fuel=180",
            ),
        ],
    )

    row = build_engagement_ledger(_SOURCE)["engagements"][0]
    assert row["dealt_fuel"] == 180
    assert row["taken_fuel"] == 0


def test_damage_ledger_with_only_a_taken_side(fake_fs: FakeFileSystem) -> None:
    """A taken-only ledger fills taken fuel and leaves dealt at zero."""
    _write_jsonl(
        fake_fs,
        _SOURCE,
        [
            _shot("2026-08-26T04:00:00", 50),
            _record(
                "2026-08-26T04:00:01",
                diagnostic_kind="damage_ledger",
                taken="red-1(50): dual=2 fuel=180",
            ),
        ],
    )

    row = build_engagement_ledger(_SOURCE)["engagements"][0]
    assert row["dealt_fuel"] == 0
    assert row["taken_fuel"] == 180


def test_main_renders_the_default_source(fake_fs: FakeFileSystem) -> None:
    """The CLI defaults to the latest events artifact."""
    _write_jsonl(fake_fs, Path("runs/bot/latest.events.jsonl"), _collision_rows())
    original = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-engagements"]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original


def test_main_renders_an_explicit_source(fake_fs: FakeFileSystem) -> None:
    """A path argument selects the artifact to analyze."""
    _write_jsonl(fake_fs, _SOURCE, _collision_rows())
    original = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-engagements", str(_SOURCE)]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original
