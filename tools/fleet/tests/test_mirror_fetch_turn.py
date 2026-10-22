"""A mirror's fetch lands in a ref, one export at a time (MCPs board task 86f70457).

On diphtheria on 2026-10-22 six node runners each received MCPs' whole
history, 133470 objects, for one commit, because the commit was fetched with
no destination ref and the mirror never held one. The first cases here run
real git against a real origin and a real bare mirror whose every fetch is
kept as a pack, so the objects a fetch received are counted rather than
inferred. The others hold the turn with a real lock file beside the mirror:
another export's turn is a lock this process did not create, and what that
export does is what the case's sleep does while this one waits.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, export, mirror_fetch
from tests._git_fixtures import git
from tests.conftest import FakeRun, failed, ok

SHA = "4e3c6bc1d9f0a7b2c3e4f5061728394a5b6c7d8e"


def _origin(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Build an origin of four files and one commit.

    Args:
        tmp_path: The case's directory.

    Returns:
        The origin and its commit.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--quiet")
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (origin / name).write_text(f"{name}\n", encoding="utf-8")
    git(origin, "add", "--all")
    git(origin, "commit", "--quiet", "-m", "first")
    return origin, git(origin, "rev-parse", "HEAD")


def _mirror(tmp_path: pathlib.Path) -> pathlib.Path:
    """Make a bare mirror that keeps every fetch as a pack.

    Args:
        tmp_path: The case's directory.

    Returns:
        The mirror.
    """
    mirror = export.mirror_path(tmp_path / "mirrors", "wagner-austin-origin")
    export.ensure_mirror(mirror)
    git(mirror, "config", "fetch.unpackLimit", "1")
    return mirror


def _in_pack(mirror: pathlib.Path) -> int:
    """How many objects the mirror's packs hold, counted once per pack.

    Args:
        mirror: The mirror.

    Returns:
        ``git count-objects -v``'s ``in-pack``.
    """
    counted = dict(line.split(": ", 1) for line in git(mirror, "count-objects", "-v").splitlines())
    return int(counted["in-pack"])


def _sleeping(steps: list[Callable[[], None]], slept: list[int]) -> _test_hooks.SleepProtocol:
    """A sleep that records each wait and runs the next step in it.

    Args:
        steps: Run in turn, one per sleep.
        slept: Every wait asked for, appended in order.

    Returns:
        The sleep.
    """

    def sleep(seconds: int) -> None:
        slept.append(seconds)
        steps.pop(0)()

    return sleep


class TestTheRef:
    def test_a_later_commit_receives_only_what_the_fetched_one_lacks(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The first fetch receives the commit, its tree and four blobs. The
        second, of a commit that changed one file, offers the first from the
        ref and receives three objects; fetched with no ref it received all
        nine reachable from it, the whole history again."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)

        mirror_fetch.fetch_commit(mirror, str(origin), first)

        assert git(mirror, "rev-parse", mirror_fetch.FETCHED_REF) == first
        assert _in_pack(mirror) == 6

        (origin / "a.txt").write_text("changed\n", encoding="utf-8")
        git(origin, "commit", "--quiet", "--all", "-m", "second")
        second = git(origin, "rev-parse", "HEAD")
        mirror_fetch.fetch_commit(mirror, str(origin), second)

        assert git(mirror, "rev-parse", mirror_fetch.FETCHED_REF) == second
        assert _in_pack(mirror) == 6 + 3
        assert not mirror_fetch.turn_lock(mirror).exists()

    def test_the_turn_is_held_beside_the_mirror(self, tmp_path: pathlib.Path) -> None:
        mirror = tmp_path / "mirrors" / "wagner-austin-MCPs.git"

        assert mirror_fetch.turn_lock(mirror) == (
            tmp_path / "mirrors" / "wagner-austin-MCPs.git.fetch.lock"
        )

    def test_the_turn_outlasts_a_probe_and_the_whole_fetch_budget(self) -> None:
        """A turn taken away while its fetch still runs would let a second
        fetch write the one ref beside it."""
        assert mirror_fetch.FETCH_TURN["stale_seconds"] > (
            mirror_fetch.INIT_TIMEOUT_SECONDS + mirror_fetch.FETCH_BUDGET_SECONDS
        )


class TestADeadFetchsPack:
    def test_is_removed_inside_the_turn_and_a_young_one_is_kept(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tmp_pack older than the turn's bound was left by a fetch that
        died; a young one may be a detached git gc's, written outside the
        turn, and is not this export's to remove."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        packs = mirror / "objects" / "pack"
        dead = packs / "tmp_pack_dead01"
        dead.write_bytes(b"PACK half a fetch")
        died = time.time() - mirror_fetch.FETCH_TURN["stale_seconds"] - 5
        os.utime(dead, (died, died))
        young = packs / "tmp_pack_live01"
        young.write_bytes(b"PACK a gc's")

        with caplog.at_level("INFO"):
            mirror_fetch.fetch_commit(mirror, str(origin), first)

        assert not dead.exists()
        assert young.exists()
        assert git(mirror, "rev-parse", mirror_fetch.FETCHED_REF) == first
        assert f"removed 1 dead fetch pack(s) from {mirror}: ['tmp_pack_dead01']" in [
            record.getMessage() for record in caplog.records
        ]

    def test_a_mirror_with_none_removes_nothing(self, tmp_path: pathlib.Path) -> None:
        mirror = _mirror(tmp_path)

        assert mirror_fetch.remove_dead_packs(mirror) == ()

    def test_a_pack_renamed_since_it_was_listed_has_no_age(self, tmp_path: pathlib.Path) -> None:
        """git renames a whole pack into place, so a listed tmp_pack can be
        gone by the time its age is read, and nothing is left to remove."""
        assert mirror_fetch._age(tmp_path / "tmp_pack_renamed") is None


class TestTheTurn:
    def test_a_second_export_of_one_commit_waits_then_finds_it_held(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The six fetches of 2026-10-22, made one: the other export holds the
        turn and fetches the commit while this one waits, and this one then
        probes, finds it and fetches nothing."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        lock = mirror_fetch.turn_lock(mirror)
        lock.write_text("the export of it pid 4242 turn fetching", encoding="utf-8")
        clock = [int(time.time())]
        slept: list[int] = []

        def the_other_export_fetches_it() -> None:
            git(mirror, "fetch", "--quiet", "--no-tags", str(origin), f"+{first}:refs/other")
            lock.unlink()
            clock[0] += mirror_fetch.FETCH_PAUSE_SECONDS

        _test_hooks.now = lambda: clock[0]
        _test_hooks.sleep = _sleeping([the_other_export_fetches_it], slept)
        with caplog.at_level("INFO"):
            mirror_fetch.fetch_commit(mirror, str(origin), first)

        assert slept == [mirror_fetch.FETCH_PAUSE_SECONDS]
        assert _in_pack(mirror) == 6
        assert not lock.exists()
        assert (
            f"the export of {first} took its mirror fetch turn after waiting 10 s for "
            "another export fetching into the same mirror"
        ) in [record.getMessage() for record in caplog.records]

    def test_a_turn_left_by_an_export_that_died_inside_it_is_taken_away(
        self, tmp_path: pathlib.Path
    ) -> None:
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        lock = mirror_fetch.turn_lock(mirror)
        lock.write_text("the export of it pid 4243 turn dead", encoding="utf-8")
        died = time.time() - mirror_fetch.FETCH_TURN["stale_seconds"] - 5
        os.utime(lock, (died, died))

        mirror_fetch.fetch_commit(mirror, str(origin), first)

        assert git(mirror, "rev-parse", mirror_fetch.FETCHED_REF) == first
        assert not lock.exists()

    def test_a_refused_fetch_gives_the_turn_back(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(
            [failed(128, "missing"), failed(128, f"fatal: not our ref {SHA}")]
        )

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, "https://github.com/wagner-austin/API.git", SHA)

        assert raised.value.code is FleetErrorCode.SHA_NOT_ON_REMOTE
        assert not mirror_fetch.turn_lock(tmp_path).exists()

    def test_a_companion_s_fetch_waits_for_the_turn_too(self, tmp_path: pathlib.Path) -> None:
        """The companion's one ref is written by every job of its project, so
        its fetch and its read of the ref share the mirror's turn."""
        lock = mirror_fetch.turn_lock(tmp_path)
        lock.write_text("the fetch of main pid 4244 turn fetching", encoding="utf-8")
        runner = FakeRun([ok(""), ok(f"{SHA}\n")])
        _test_hooks.run = runner
        clock = [int(time.time())]
        slept: list[int] = []

        def the_other_fetch_ends() -> None:
            assert runner.calls == []
            lock.unlink()
            clock[0] += mirror_fetch.FETCH_PAUSE_SECONDS

        _test_hooks.now = lambda: clock[0]
        _test_hooks.sleep = _sleeping([the_other_fetch_ends], slept)

        companion = "https://github.com/wagner-austin/MCPs.git"
        assert mirror_fetch.fetch_ref(tmp_path, companion, "main") == SHA
        assert slept == [mirror_fetch.FETCH_PAUSE_SECONDS]
        assert [call[3] for call in runner.calls] == ["fetch", "rev-parse"]
        assert not lock.exists()
