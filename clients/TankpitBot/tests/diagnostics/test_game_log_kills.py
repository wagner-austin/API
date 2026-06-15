"""End-to-end tests for game-log kill registration.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:func:`tankpit_bot.diagnostics.game_log_kills.register_kills_from_game_log`
-> real ``mark_tank_killed`` world-state mutation + real JSONL artifact
via :class:`tests.conftest.FakeFileSystem`. Nothing is mocked. The
banner shapes mirror the live game log captured in runs
20260610-005248 / 20260610-011x.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.browser.dom_scraper import GameLogEntry, parse_game_log
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.game_log_kills import register_kills_from_game_log
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_combat import drain_killed_tank_ids
from tankpit_bot.state import make_empty_world_state
from tankpit_bot.state.types import WorldStateDict, make_tank_state


def _world_with_tank(tank_id: int, name: str) -> WorldStateDict:
    """Return a world state tracking one named enemy tank."""
    world = make_empty_world_state()
    world["tanks"][str(tank_id)] = make_tank_state(
        tank_id=tank_id,
        x=100,
        y=100,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=1000,
    )
    return world


def _kill_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``tank_deactivated`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "tank_deactivated"
    ]


def test_two_line_banner_registers_kill(fake_fs: FakeFileSystem) -> None:
    """The live banner shape (name line, then suffix line) registers the kill.

    This is byte-for-byte the shape from live run 20260610-011x where
    the bot kept firing at purple-8 after the on-screen deactivation.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")
    state = parse_game_log(
        "You hit purple-8\n"
        "You fire\n"
        "********************************************\n"
        " purple-8\n"
        " has been deactivated by you\n"
        "********************************************\n"
        "You earned extra points\n"
    )

    kills = register_kills_from_game_log(state["entries"], _world_with_tank(516, "purple-8"))

    assert kills == 1
    assert drain_killed_tank_ids(get_world_service()) == {516}
    records = _kill_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "tank_deactivated",
        "origin": "game_log",
        "victim_name": "purple-8",
        "victim_id": 516,
        "killer_id": -1,
    }


def test_one_line_banner_registers_kill(fake_fs: FakeFileSystem) -> None:
    """A single-line ``X has been deactivated by you`` also registers."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    entries: list[GameLogEntry] = [
        GameLogEntry(text="red-8 has been deactivated by you", category="combat"),
    ]

    kills = register_kills_from_game_log(entries, _world_with_tank(512, "red-8"))

    assert kills == 1
    assert drain_killed_tank_ids(get_world_service()) == {512}
    assert len(_kill_records(artifacts["latest_events_path"])) == 1


def test_unresolved_victim_name_emits_but_does_not_mark(
    fake_fs: FakeFileSystem,
) -> None:
    """A kill banner for an untracked tank stays visible without a mark."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    entries: list[GameLogEntry] = [
        GameLogEntry(text="ghost-9 has been deactivated by you", category="combat"),
    ]

    kills = register_kills_from_game_log(entries, _world_with_tank(512, "red-8"))

    assert kills == 1
    assert drain_killed_tank_ids(get_world_service()) == set()
    records = _kill_records(artifacts["latest_events_path"])
    assert records[0]["fields"]["victim_id"] == -1
    assert records[0]["fields"]["victim_name"] == "ghost-9"


def test_non_kill_entries_register_nothing(fake_fs: FakeFileSystem) -> None:
    """Ordinary combat and action lines never register kills."""
    configure_bot_runtime_logging("20260610-120000")
    entries: list[GameLogEntry] = [
        GameLogEntry(text="You hit purple-8", category="combat"),
        GameLogEntry(text="purple-8 hit you", category="combat"),
        GameLogEntry(text="You fire", category="action"),
        GameLogEntry(text="You earned extra points", category="combat"),
    ]

    kills = register_kills_from_game_log(entries, _world_with_tank(516, "purple-8"))

    assert kills == 0
    assert drain_killed_tank_ids(get_world_service()) == set()


def test_suffix_as_first_entry_has_no_victim(fake_fs: FakeFileSystem) -> None:
    """The suffix line with no preceding name line cannot name a victim."""
    configure_bot_runtime_logging("20260610-120000")
    entries: list[GameLogEntry] = [
        GameLogEntry(text="has been deactivated by you", category="combat"),
    ]

    kills = register_kills_from_game_log(entries, _world_with_tank(516, "purple-8"))

    assert kills == 0
    assert drain_killed_tank_ids(get_world_service()) == set()
