"""Tests for clearing a reused fleet slot before its next session starts.

The run directory is relative to the working directory, so the three
filesystem seams are faked: the stream files and the two sentinels a
predecessor can leave are what the fakes report, and every removal is
recorded. The stream-file half is also pinned through a real spawn in
``test_fleet.py``.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_slot import clear_reused_slot
from tests.service._fleet_fixtures import _FakeSpawner

_RUN = Path("runs/bot/alpha")


class _Slot:
    """A slot's leftovers, as the three seams report them.

    Attributes:
        present: Paths that exist.
        removed: Paths removed, in order.
    """

    def __init__(self, present: list[Path]) -> None:
        """Hold the leftovers."""
        self.present = list(present)
        self.removed: list[Path] = []

    def glob_paths(self, directory: Path, pattern: str) -> list[Path]:
        """List the present files directly under ``directory``.

        Args:
            directory: Directory to list.
            pattern: Must be ``*``.

        Returns:
            The present paths whose parent is ``directory``.
        """
        assert pattern == "*"
        return [path for path in self.present if path.parent == directory]

    def path_exists(self, path: Path) -> bool:
        """Report whether ``path`` is present.

        Args:
            path: Candidate path.

        Returns:
            Whether it is present.
        """
        return path in self.present

    def remove_file(self, path: Path) -> None:
        """Remove ``path``.

        Args:
            path: Path to remove.
        """
        self.present.remove(path)
        self.removed.append(path)


@pytest.fixture()
def install() -> Generator[list[_Slot], None, None]:
    """Hand the test a way to install a slot, and restore the seams after.

    Yields:
        A list the test appends its installed slot to.
    """
    originals = (top_hooks.glob_paths, top_hooks.path_exists, top_hooks.remove_file)
    installed: list[_Slot] = []
    yield installed
    top_hooks.glob_paths, top_hooks.path_exists, top_hooks.remove_file = originals


def _use(installed: list[_Slot], slot: _Slot) -> _Slot:
    """Point the seams at ``slot``.

    Args:
        installed: The fixture's list.
        slot: The slot to install.

    Returns:
        The slot.
    """
    top_hooks.glob_paths = slot.glob_paths
    top_hooks.path_exists = slot.path_exists
    top_hooks.remove_file = slot.remove_file
    installed.append(slot)
    return slot


def test_a_predecessors_stream_and_untaken_sentinels_are_all_removed(
    install: list[_Slot],
) -> None:
    """Stream files first, then an untaken STOP and CONTROL."""
    leftovers = [_RUN / "hls" / "index.m3u8", _RUN / "hls" / "seg00042.ts", _RUN / "STOP"]
    slot = _use(install, _Slot([*leftovers, _RUN / "CONTROL", _RUN / "latest.log"]))

    clear_reused_slot("alpha")

    assert slot.removed == [*leftovers, _RUN / "CONTROL"]
    assert slot.present == [_RUN / "latest.log"]


def test_a_clean_slot_removes_nothing(install: list[_Slot]) -> None:
    """No stream files and no sentinels: nothing to do."""
    slot = _use(install, _Slot([_RUN / "latest.log"]))

    clear_reused_slot("alpha")

    assert slot.removed == []


def test_a_spawn_clears_a_stop_the_predecessor_never_took(
    spawner: _FakeSpawner, install: list[_Slot]
) -> None:
    """The new session in a reused slot must not end at its first tick."""
    slot = _use(install, _Slot([_RUN / "STOP"]))

    FleetManager().spawn(instance="alpha", account="", kills=0, seconds=0)

    assert slot.removed == [_RUN / "STOP"]
    assert spawner.envs[0]["TANKPIT_BOT_INSTANCE"] == "alpha"
