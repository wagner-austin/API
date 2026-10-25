"""The fleet's scratch root and its reaper (MCPs board task fb505198).

The root is memory, ``/dev/shm``, which the default hook answers. The reaper
is run over real directories with a pinned clock and real modification
times.
"""

from __future__ import annotations

import os
import pathlib

from fleet.cli import rolled as rolled_cli
from fleet.cli import tick
from fleet.core import _test_hooks, scratch, scratch_root
from tests.conftest import DEMO_NOW, FakeClock, FakeTempRoot


def age(path: pathlib.Path, seconds: int) -> None:
    """Set a path's modification time to ``seconds`` before the pinned clock.

    Args:
        path: The file or directory.
        seconds: How old it is to read.
    """
    then = DEMO_NOW - seconds
    os.utime(path, (then, then))


class TestScratchRoot:
    def test_scratch_is_kept_in_memory(self) -> None:
        assert pathlib.Path("/dev/shm") == scratch_root.SCRATCH_ROOT

    def test_the_default_hook_answers_the_memory_root(self) -> None:
        assert _test_hooks._default_temp_root() == scratch_root.SCRATCH_ROOT

    def test_both_directories_hang_off_the_hooked_root(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.temp_root = FakeTempRoot(tmp_path)

        assert scratch.directory(scratch.EXTRACTIONS_DIR) == tmp_path / "fleet-rolled"
        assert scratch.directory(scratch.ARCHIVES_DIR) == tmp_path / "fleet-archives"


class TestReap:
    def test_a_directory_not_made_yet_reaps_nothing(self, tmp_path: pathlib.Path) -> None:
        assert scratch.reap(tmp_path / "absent", keep=frozenset()) == ()

    def test_only_entries_older_than_the_bound_and_not_kept_are_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.now = FakeClock(DEMO_NOW)
        (tmp_path / "old-file.tgz").write_bytes(b"x")
        age(tmp_path / "old-file.tgz", scratch.STALE_SECONDS + 1)
        (tmp_path / "old-dir" / "nested").mkdir(parents=True)
        (tmp_path / "old-dir" / "nested" / "bundle").write_bytes(b"x")
        age(tmp_path / "old-dir", scratch.STALE_SECONDS + 1)
        (tmp_path / "kept-dir").mkdir()
        (tmp_path / "kept-dir" / "inside").write_bytes(b"x")
        age(tmp_path / "kept-dir", scratch.STALE_SECONDS * 3)
        (tmp_path / "at-the-bound.tgz").write_bytes(b"x")
        age(tmp_path / "at-the-bound.tgz", scratch.STALE_SECONDS)

        removed = scratch.reap(tmp_path, keep=frozenset({"kept-dir"}))

        assert removed == ("old-dir", "old-file.tgz")
        assert sorted(path.name for path in tmp_path.iterdir()) == ["at-the-bound.tgz", "kept-dir"]
        assert (tmp_path / "kept-dir" / "inside").exists()

    def test_an_entry_gone_since_it_was_listed_is_not_stale(self, tmp_path: pathlib.Path) -> None:
        assert not scratch.stale(tmp_path / "unlinked-by-its-runner.tgz", now=DEMO_NOW)

    def test_the_bound_outlives_every_process_that_uses_an_entry(self) -> None:
        assert scratch.STALE_SECONDS > tick.TICK_WALL_SECONDS > rolled_cli.AGENT_WALL_SECONDS
