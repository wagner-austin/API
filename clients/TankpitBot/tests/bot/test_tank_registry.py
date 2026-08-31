"""Tests for the per-colour tank recorder.

The registry is the only record of the three colours an account is NOT
playing, so a row that claims a reading the run never took is worse
than no row: every guard below refuses to write rather than guess.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import (
    PathExistsProtocol,
    ReadTextProtocol,
    WriteTextProtocol,
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
        tank_id=1, x=10, y=10, team=team, rank=6, fuel=1600, leaderboard_position=1
    )
    ws.record_self_identity(name, 62913, "00", 1000)
    ws.record_account_stats(
        rank_name="major",
        rank_number=18,
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

    def __init__(self, existing: str | None = None) -> None:
        self.existing = existing
        self.written: list[tuple[Path, str]] = []

    def read(self, path: Path) -> str:
        if Path(path).name == "accounts.json":
            return _ACCOUNTS_JSON
        if self.existing is None:
            raise OSError(f"no registry at {path}")
        return self.existing

    def exists(self, path: Path) -> bool:
        return Path(path).name == "accounts.json"

    def write(self, path: Path, content: str) -> None:
        self.written.append((Path(path), content))


def _with_recorder(
    recorder: _Recorder,
) -> tuple[ReadTextProtocol, WriteTextProtocol, PathExistsProtocol]:
    """Install the recorder over the filesystem hooks.

    Args:
        recorder: The double to install.

    Returns:
        The original ``(read_text, write_text, path_exists)`` hooks.
    """
    originals = (top_hooks.read_text, top_hooks.write_text, top_hooks.path_exists)
    top_hooks.read_text = recorder.read
    top_hooks.write_text = recorder.write
    top_hooks.path_exists = recorder.exists
    return originals


def _restore(
    originals: tuple[ReadTextProtocol, WriteTextProtocol, PathExistsProtocol],
) -> None:
    """Restore the filesystem hooks.

    Args:
        originals: The pair returned by :func:`_with_recorder`.
    """
    top_hooks.read_text, top_hooks.write_text, top_hooks.path_exists = originals


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


def test_a_corrupt_registry_starts_fresh_rather_than_killing_the_run() -> None:
    """Bookkeeping must never take down a live session mid-run."""
    recorder = _Recorder("[1, 2, 3]")
    originals = _with_recorder(recorder)
    try:
        record_tank_sample(_panelled_world(team=0), "Practice")
    finally:
        _restore(originals)

    accounts = narrow_json_to_dict(
        narrow_json_to_dict(load_json_str(recorder.written[0][1]))["accounts"]
    )
    assert "Artax" in accounts
