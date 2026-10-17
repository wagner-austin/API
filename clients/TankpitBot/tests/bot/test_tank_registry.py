"""Tests for the per-colour tank recorder.

The registry is the only record of the three colours an account is NOT
playing, so a row that claims a reading the run never took is worse
than no row: every guard below refuses to write rather than guess.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import (
    PathExistsProtocol,
    ReadTextProtocol,
    ReplaceTextProtocol,
)
from tankpit_bot.bot.tank_registry import record_tank_sample
from tankpit_bot.runtime_artifacts import TANK_REGISTRY_PATH
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types.self_state import make_self_state


def _panelled_world(*, team: int, name: str = "Artax") -> WorldService:
    """Build a world service carrying a panel sample and a self tank.

    Args:
        team: The self tank's team id (its colour).
        name: The in-game tank name.

    Returns:
        The prepared world service.
    """
    ws = WorldService()
    ws.world_state["self_state"] = make_self_state(
        tank_id=1, x=10, y=10, team=team, rank=6, fuel=1234, leaderboard_position=1
    )
    ws.record_self_identity(name, 62913, "00", 1000)
    ws.record_account_stats(
        rank_name="major",
        leaderboard_position=18,
        promotion_points=674270,
        destroyed_enemies=1958,
        deactivated_total=5,
        play_time_s=365247,
        timestamp_ms=1000,
    )
    return ws


_ACCOUNTS_JSON = '[{"username": "Artax", "password": "x"}]'


class _Recorder:
    """Captures the registry write instead of touching the disk.

    Path-aware on purpose: the recorder resolves the ACCOUNT to key
    rows by, so a blanket read hook would starve accounts.json and the
    row would be skipped for the wrong reason.
    """

    def __init__(self, existing: str | OSError = "{}") -> None:
        self.existing = existing
        self.written: list[tuple[Path, str]] = []

    def read(self, path: Path) -> str:
        if Path(path).name == "accounts.json":
            return _ACCOUNTS_JSON
        if isinstance(self.existing, OSError):
            raise self.existing
        return self.existing

    def exists(self, path: Path) -> bool:
        return Path(path).name == "accounts.json"

    def replace(self, path: Path, content: str) -> None:
        self.written.append((Path(path), content))


def _with_recorder(
    recorder: _Recorder,
) -> tuple[ReadTextProtocol, ReplaceTextProtocol, PathExistsProtocol]:
    """Install the recorder over the filesystem hooks.

    Args:
        recorder: The double to install.

    Returns:
        The original ``(read_text, replace_text, path_exists)`` hooks.
    """
    originals = (top_hooks.read_text, top_hooks.replace_text, top_hooks.path_exists)
    top_hooks.read_text = recorder.read
    top_hooks.replace_text = recorder.replace
    top_hooks.path_exists = recorder.exists
    return originals


def _restore(
    originals: tuple[ReadTextProtocol, ReplaceTextProtocol, PathExistsProtocol],
) -> None:
    """Restore the filesystem hooks.

    Args:
        originals: The hooks returned by :func:`_with_recorder`.
    """
    top_hooks.read_text, top_hooks.replace_text, top_hooks.path_exists = originals


def test_records_the_played_colour_under_its_account_and_room() -> None:
    """A played tank lands under account -> room -> colour.

    Team 3 is orange ([[game-rules]]: the troop byte IS the team id),
    so this is the reading no lobby query could have produced — the
    lobby names only the last-played colour.
    """
    recorder = _Recorder()
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=3), "World")
    finally:
        _restore(originals)

    assert len(recorder.written) == 1
    path, content = recorder.written[0]
    assert path == TANK_REGISTRY_PATH
    cell = narrow_json_to_dict(load_json_str(content))
    accounts = narrow_json_to_dict(cell["accounts"])
    rooms = narrow_json_to_dict(accounts["Artax"])
    colours = narrow_json_to_dict(rooms["World"])
    orange = narrow_json_to_dict(colours["orange"])
    assert orange["rank"] == "major"
    assert orange["kills"] == 1958
    assert orange["deaths"] == 5
    assert orange["leaderboard"] == 18
    # Live state as the session left it: the cap for a major is 1600,
    # so a recorded 1600 would mean the cap leaked in place of fuel.
    assert orange["fuel"] == 1234
    assert orange["inventory"] == [0, 0, 0, 0, 0]


def test_merges_beside_an_existing_colour_without_erasing_it() -> None:
    """Recording one colour must not drop the other three.

    The registry accumulates across sessions — each colour is written
    only while it is being played, so a clobbering write would throw
    away readings that cost a live entry each.
    """
    recorder = _Recorder('{"accounts": {"Artax": {"World": {"red": {"rank": "lieutenant"}}}}}')
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=3), "World")
    finally:
        _restore(originals)

    colours = narrow_json_to_dict(
        narrow_json_to_dict(
            narrow_json_to_dict(
                narrow_json_to_dict(load_json_str(recorder.written[0][1]))["accounts"]
            )["Artax"]
        )["World"]
    )
    assert set(colours) == {"red", "orange"}


def test_writes_nothing_without_a_panel_sample() -> None:
    """No panel, no row: rank_name empty means the C press never landed."""
    ws = WorldService()
    ws.world_state["self_state"] = make_self_state(
        tank_id=1, x=10, y=10, team=3, rank=6, fuel=1600, leaderboard_position=1
    )
    ws.record_self_identity("Artax", 62913, "00", 1000)
    recorder = _Recorder()
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(ws, "World")
    finally:
        _restore(originals)

    assert recorder.written == []


def test_writes_nothing_without_a_self_tank() -> None:
    """No self state means no colour to file the reading under."""
    ws = WorldService()
    ws.record_self_identity("Artax", 62913, "00", 1000)
    recorder = _Recorder()
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(ws, "World")
    finally:
        _restore(originals)

    assert recorder.written == []


def test_writes_nothing_for_a_team_outside_the_colour_range() -> None:
    """An unsynced tank reads team -1; that is not a colour."""
    recorder = _Recorder()
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=9), "World")
    finally:
        _restore(originals)

    assert recorder.written == []


@pytest.mark.parametrize(
    "corrupt",
    [
        "[1, 2, 3]",
        # The shape two truncating writers left on sedona, 2026-10-17:
        # a whole document followed by the longer write's tail.
        '{"accounts": {}}\n}ld": {}\n}',
    ],
)
def test_a_corrupt_registry_is_kept_and_the_sample_skipped(corrupt: str) -> None:
    """Bookkeeping never takes down a live session, nor overwrites what it cannot read."""
    recorder = _Recorder(corrupt)
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=0), "Practice")
    finally:
        _restore(originals)

    assert recorder.written == []


@pytest.mark.parametrize(
    "window",
    [
        FileNotFoundError(errno.ENOENT, "rewritten"),
        PermissionError(errno.EACCES, "held open"),
        OSError(errno.ENODATA, "no data"),
    ],
)
def test_a_registry_hidden_by_a_rewrite_skips_the_sample(window: OSError) -> None:
    """A replace window is another bot's rewrite: merging into nothing would erase it."""
    recorder = _Recorder(window)
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=0), "Practice")
    finally:
        _restore(originals)

    assert recorder.written == []


def test_an_empty_registry_read_skips_the_sample() -> None:
    """A zero-length read is the 9p mount's other form of the same window."""
    recorder = _Recorder("")
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=0), "Practice")
    finally:
        _restore(originals)

    assert recorder.written == []


def test_a_read_failure_outside_a_window_propagates() -> None:
    """Out of memory is not a rewrite; the run reports it rather than hiding it."""
    recorder = _Recorder(OSError(errno.ENOMEM, "Cannot allocate memory"))
    originals = _with_recorder(recorder)
    try:
        with pytest.raises(OSError, match="Cannot allocate memory"):
            record_tank_sample(_panelled_world(team=0), "Practice")
    finally:
        _restore(originals)

    assert recorder.written == []
