"""Origin's rolled ref, fetched before every read (MCPs board task 8db0f536).

Two halves. The scripted half pins the two commands, their deadlines, the
stamp a fetch leaves, the window it answers for and the refusals. The real
half is the defect itself: a hub clone and a session-host clone of one bare
origin, the session host rolling as ``make fleet-roll`` does (``git
update-ref`` then a forced push), and the hub's own reads, which never saw
that ref before, naming the roll.
"""

from __future__ import annotations

import os
import pathlib

from fleet.core import _test_hooks, rolled, rolled_fetch
from tests._git_fixtures import committed_repository, git
from tests._rolled_fetch_script import FETCH_TIMEOUTS, fetch_calls, fetch_replies
from tests.conftest import DEMO_NOW, FakeClock, FakeRun, FakeTempRoot, failed, ok


def stamp_of(git_directory: pathlib.Path) -> pathlib.Path:
    """Where a fetch leaves its stamp.

    Args:
        git_directory: The repository's git directory.

    Returns:
        The stamp file.
    """
    return git_directory / "fleet-rolled-fetch"


def stamped(git_directory: pathlib.Path, text: str, *, seconds_ago: int) -> None:
    """Leave the stamp an earlier fetch left.

    Args:
        git_directory: The git directory, created.
        text: Empty for a fetch that moved the ref, else its refusal.
        seconds_ago: How long before the pinned clock it was left.
    """
    git_directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp_of(git_directory)
    stamp.write_text(text, encoding="utf-8")
    then = DEMO_NOW - seconds_ago
    os.utime(stamp, (then, then))


class TestScriptedFetch:
    def test_with_no_stamp_it_fetches_and_leaves_an_empty_one(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        runner = FakeRun(fetch_replies(git_directory))
        _test_hooks.run = runner

        assert rolled_fetch.fetch_rolled(tmp_path / "api") is None

        assert runner.calls == fetch_calls(tmp_path / "api")
        assert runner.calls[1] == (
            "git",
            "-C",
            str(tmp_path / "api"),
            "fetch",
            "--quiet",
            "origin",
            "+refs/fleet/rolled:refs/fleet/rolled",
        )
        assert runner.timeouts == [*FETCH_TIMEOUTS] == [30, 60]
        stamp = stamp_of(git_directory)
        assert stamp.read_text(encoding="utf-8") == ""
        assert int(stamp.stat().st_mtime) == DEMO_NOW
        # The turn's lock is given back.
        assert sorted(path.name for path in git_directory.iterdir()) == ["fleet-rolled-fetch"]

    def test_a_fresh_stamp_answers_without_asking_origin(self, tmp_path: pathlib.Path) -> None:
        """The reads of one boundary share one fetch."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        stamped(git_directory, "", seconds_ago=rolled_fetch.FRESH_SECONDS - 1)
        runner = FakeRun([ok(f"{git_directory}\n")])
        _test_hooks.run = runner

        assert rolled_fetch.fetch_rolled(tmp_path / "api") is None

        assert runner.calls == fetch_calls(tmp_path / "api")[:1]

    def test_a_fresh_refusal_is_answered_again_without_asking_origin(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        refusal = "FLEET_ROLL_FETCH_FAILED: git fetch ... exited 128: unreachable"
        stamped(git_directory, refusal, seconds_ago=0)
        runner = FakeRun([ok(f"{git_directory}\n")])
        _test_hooks.run = runner

        assert rolled_fetch.fetch_rolled(tmp_path / "api") == refusal

        assert len(runner.calls) == 1

    def test_a_stamp_as_old_as_the_window_is_fetched_again(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        stamped(git_directory, "an older refusal", seconds_ago=rolled_fetch.FRESH_SECONDS)
        runner = FakeRun(fetch_replies(git_directory))
        _test_hooks.run = runner

        assert rolled_fetch.fetch_rolled(tmp_path / "api") is None

        assert runner.calls == fetch_calls(tmp_path / "api")
        assert stamp_of(git_directory).read_text(encoding="utf-8") == ""
        assert int(stamp_of(git_directory).stat().st_mtime) == DEMO_NOW

    def test_a_failed_fetch_refuses_by_name_and_stamps_its_refusal(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        located, _ = fetch_replies(git_directory)
        _test_hooks.run = FakeRun([located, failed(128, "  fatal: unable to access origin  ")])
        api = tmp_path / "api"

        refusal = rolled_fetch.fetch_rolled(api)

        assert refusal == (
            f"FLEET_ROLL_FETCH_FAILED: git fetch origin +refs/fleet/rolled:refs/fleet/rolled "
            f"in {api} exited 128: fatal: unable to access origin"
        )
        assert stamp_of(git_directory).read_text(encoding="utf-8") == refusal

    def test_an_unreadable_git_directory_refuses_and_fetches_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([failed(128, "fatal: not a git repository")])
        _test_hooks.run = runner
        api = tmp_path / "api"

        assert rolled_fetch.fetch_rolled(api) == (
            f"FLEET_ROLL_GIT_DIR_UNREAD: git rev-parse --git-common-dir in {api} exited 128: "
            "fatal: not a git repository"
        )
        assert len(runner.calls) == 1


#: What each roll carries: the files an extraction requires.
ROLLED_FILES: dict[str, str] = {
    **dict.fromkeys(rolled.REQUIRED_FILES, ""),
    "tools/fleet/registry/settings.json": '{"roll": 1}\n',
}


def roll_from(session_host: pathlib.Path, settings: str) -> str:
    """Commit a change on the session host and roll it as ``make fleet-roll`` does.

    Args:
        session_host: The session host's clone.
        settings: The registry settings the roll carries.

    Returns:
        The rolled commit.
    """
    (session_host / "tools/fleet/registry/settings.json").write_text(settings, encoding="utf-8")
    git(session_host, "commit", "--quiet", "--all", "-m", "a roll")
    git(session_host, "update-ref", rolled.ROLLED_REF, "HEAD")
    git(session_host, "push", "--quiet", "origin", f"+{rolled.ROLLED_REF}:{rolled.ROLLED_REF}")
    return git(session_host, "rev-parse", "HEAD")


def unresolved_here(hub: pathlib.Path) -> bool:
    """Say whether the hub's own ref names no commit, before any read.

    Args:
        hub: The hub's clone.

    Returns:
        True when ``git rev-parse --verify`` of the ref fails there.
    """
    probe = _test_hooks.run(
        ("git", "-C", str(hub), "rev-parse", "--verify", "--quiet", rolled.ROLLED_REF),
        timeout_seconds=60,
    )
    return probe["returncode"] != 0


class TestARollMadeOffTheHub:
    def test_is_what_the_hubs_next_reads_run_and_a_later_one_moves_them(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The 2026-10-24 roll from ziggurat that the hub's runners never read."""
        clock = FakeClock(DEMO_NOW)
        _test_hooks.now = clock
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        origin = tmp_path / "origin.git"
        origin.mkdir()
        git(origin, "init", "--quiet", "--bare")
        session_host = tmp_path / "ziggurat"
        committed_repository(session_host, ROLLED_FILES, "refs/heads/main")
        git(session_host, "remote", "add", "origin", str(origin))
        git(session_host, "push", "--quiet", "origin", "refs/heads/main")
        hub = tmp_path / "diphtheria"
        git(tmp_path, "clone", "--quiet", "--branch", "main", str(origin), str(hub))
        records = hub / pathlib.PurePosixPath(rolled.RECORDS_DIR)

        first = roll_from(session_host, '{"roll": 2}\n')

        # The clone's refspec fetches branches only, as the hub's does: the
        # roll is on origin and not here, which is all the old reads saw.
        assert unresolved_here(hub)
        assert rolled.rolled_state(records) == first
        tree = rolled.extract_rolled_tree(hub)
        assert not isinstance(tree, str), tree
        assert tree["commit"] == first
        settings = pathlib.Path(tree["config"]) / "settings.json"
        assert settings.read_text(encoding="utf-8") == '{"roll": 2}\n'

        second = roll_from(session_host, '{"roll": 3}\n')
        # Inside the window the boundary's fetch still answers.
        assert rolled.rolled_state(records) == first
        clock.seconds += rolled_fetch.FRESH_SECONDS

        assert rolled.rolled_state(records) == second

    def test_a_ref_moved_here_and_never_pushed_is_put_back_to_origins(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Origin's ref is the roll; the push of ``make fleet-roll`` is what
        makes one."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        origin = tmp_path / "origin.git"
        origin.mkdir()
        git(origin, "init", "--quiet", "--bare")
        hub = tmp_path / "diphtheria"
        rolled_commit = committed_repository(hub, ROLLED_FILES, "refs/heads/main")
        git(hub, "remote", "add", "origin", str(origin))
        git(hub, "push", "--quiet", "origin", f"{rolled_commit}:{rolled.ROLLED_REF}")
        (hub / "tools/fleet/registry/settings.json").write_text('{"roll": 9}\n', encoding="utf-8")
        git(hub, "commit", "--quiet", "--all", "-m", "never pushed")
        git(hub, "update-ref", rolled.ROLLED_REF, "HEAD")

        assert rolled.rolled_state(hub) == rolled_commit
