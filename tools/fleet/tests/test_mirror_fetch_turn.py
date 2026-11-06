"""A mirror's fetch lands in a ref, one export at a time (MCPs board tasks 86f70457 and 5280a25d).

On diphtheria on 2026-10-22 six node runners each received MCPs' whole
history, 133470 objects, for one commit, because the commit was fetched with
no destination ref and the mirror never held one. The first cases here run
real git against a real origin and a real bare mirror whose every fetch is
kept as a pack, so the objects a fetch received are counted rather than
inferred. The others hold the turn with a real ``flock`` on the lock file
beside the mirror (:class:`tests._held_turns.OtherTurn`): what that export
does is what its turn does before it lets go.
"""

from __future__ import annotations

import os
import pathlib
import re
import time

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, export, mirror_fetch
from tests._git_fixtures import git
from tests._held_turns import OtherTurn, is_held
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
    """Make a bare mirror as the hub makes one, with git's own settings.

    Args:
        tmp_path: The case's directory.

    Returns:
        The mirror.
    """
    mirror = export.mirror_path(tmp_path / "mirrors", "wagner-austin-origin")
    export.ensure_mirror(mirror)
    return mirror


def _counted(mirror: pathlib.Path) -> dict[str, int]:
    """The mirror's ``git count-objects -v``, each field a number.

    Args:
        mirror: The mirror.

    Returns:
        Loose objects as ``count``, packed ones as ``in-pack``, and so on.
    """
    lines = git(mirror, "count-objects", "-v").splitlines()
    return {name: int(value) for name, value in (line.split(": ", 1) for line in lines)}


def _in_pack(mirror: pathlib.Path) -> int:
    """How many objects the mirror's packs hold, counted once per pack.

    Args:
        mirror: The mirror.

    Returns:
        ``git count-objects -v``'s ``in-pack``.
    """
    return _counted(mirror)["in-pack"]


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
        assert not is_held(mirror_fetch.turn_lock(mirror))

    def test_a_small_fetch_lands_as_one_pack_and_no_loose_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Both fetches are far under git's ``fetch.unpackLimit`` of 100
        objects, which unpacked each object into a file of its own, a create
        per object on the hub's NFS; each is kept as the one pack the remote
        sent, which leaves no ``.keep`` file to bar git's repack."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        assert git(mirror, "config", "--default", "100", "--get", "fetch.unpackLimit") == "100"

        mirror_fetch.fetch_commit(mirror, str(origin), first)
        (origin / "b.txt").write_text("changed\n", encoding="utf-8")
        git(origin, "commit", "--quiet", "--all", "-m", "second")
        mirror_fetch.fetch_commit(mirror, str(origin), git(origin, "rev-parse", "HEAD"))

        counted = _counted(mirror)
        assert (counted["count"], counted["packs"], counted["in-pack"]) == (0, 2, 6 + 3)
        assert sorted((mirror / "objects" / "pack").glob("*.keep")) == []

    def test_the_turn_is_held_beside_the_mirror(self, tmp_path: pathlib.Path) -> None:
        mirror = tmp_path / "mirrors" / "wagner-austin-MCPs.git"

        assert mirror_fetch.turn_lock(mirror) == (
            tmp_path / "mirrors" / "wagner-austin-MCPs.git.fetch.turn"
        )


class TestADeadFetchsPack:
    def test_outlives_a_probe_and_the_whole_fetch_budget(self) -> None:
        """A tmp_pack removed while its fetch still runs would break that
        fetch, so none younger than a fetch can last is taken for dead."""
        assert mirror_fetch.DEAD_PACK_SECONDS > (
            mirror_fetch.INIT_TIMEOUT_SECONDS + mirror_fetch.FETCH_BUDGET_SECONDS
        )

    def test_is_removed_inside_the_turn_and_a_young_one_is_kept(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tmp_pack older than DEAD_PACK_SECONDS was left by a fetch that
        died; a young one may be a detached git gc's, written outside the
        turn, and is not this export's to remove."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        packs = mirror / "objects" / "pack"
        dead = packs / "tmp_pack_dead01"
        dead.write_bytes(b"PACK half a fetch")
        died = time.time() - mirror_fetch.DEAD_PACK_SECONDS - 5
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
        probes, finds it and fetches nothing. It is woken as the other lets
        go, not on a clock, which cost an export 21 s of lacrymal's push gate
        on 2026-11-04 (MCPs board task 1e0ee821)."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        other = OtherTurn(mirror_fetch.turn_lock(mirror))

        def the_other_export_fetches_it() -> None:
            git(mirror, "fetch", "--quiet", "--no-tags", str(origin), f"+{first}:refs/other")

        ending = other.end_after(0.1, the_other_export_fetches_it)

        with caplog.at_level("INFO"):
            mirror_fetch.fetch_commit(mirror, str(origin), first)
        ending.join()

        assert _in_pack(mirror) == 6
        said = [record.getMessage() for record in caplog.records]
        assert len(said) == 1
        assert re.fullmatch(
            rf"the export of {first} took its mirror fetch turn after waiting \d+ ms for "
            "another export fetching into the same mirror",
            said[0],
        ), said

    def test_a_commit_a_fetch_brought_is_exported_while_another_export_holds_the_turn(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Every job of a push gate exports one commit, and once the first has
        fetched it the rest wait for no turn: on lacrymal at 18:55:00Z on
        2026-11-05 one waited 6 s behind the others (MCPs board task
        5280a25d). The turn stays held throughout, so a wait would hang the
        case rather than pass it."""
        origin, first = _origin(tmp_path)
        mirror = _mirror(tmp_path)
        mirror_fetch.fetch_commit(mirror, str(origin), first)
        other = OtherTurn(mirror_fetch.turn_lock(mirror))

        mirror_fetch.fetch_commit(mirror, str(origin), first)

        held = is_held(mirror_fetch.turn_lock(mirror))
        other.end_after(0).join()
        assert (held, _in_pack(mirror)) == (True, 6)

    def test_a_refused_fetch_gives_the_turn_back(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(
            [failed(1, ""), failed(128, "missing"), failed(128, f"fatal: not our ref {SHA}")]
        )

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, "https://github.com/wagner-austin/API.git", SHA)

        assert raised.value.code is FleetErrorCode.SHA_NOT_ON_REMOTE
        assert not is_held(mirror_fetch.turn_lock(tmp_path))

    def test_a_companion_s_fetch_waits_for_the_turn_too(self, tmp_path: pathlib.Path) -> None:
        """The companion's one ref is written by every job of its project, so
        its fetch and its read of the ref share the mirror's turn: nothing is
        run while the other fetch holds it."""
        runner = FakeRun([ok(""), ok(f"{SHA}\n")])
        _test_hooks.run = runner
        other = OtherTurn(mirror_fetch.turn_lock(tmp_path))
        run_while_held: list[int] = []
        ending = other.end_after(0.1, lambda: run_while_held.append(len(runner.calls)))

        companion = "https://github.com/wagner-austin/MCPs.git"
        assert mirror_fetch.fetch_ref(tmp_path, companion, "main") == SHA
        ending.join()

        assert run_while_held == [0]
        assert [call[3] for call in runner.calls] == ["fetch", "rev-parse"]
        assert not is_held(mirror_fetch.turn_lock(tmp_path))
