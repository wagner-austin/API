"""The rolled ref's home, fetched before every read (MCPs board tasks 8db0f536 and 18ca6bab).

Two halves. The scripted half pins the two commands, their deadlines, the
stamp a fetch leaves, the window it answers for and the refusals. The real
half is the fleet as it is: the hub's own API checkout holding the ref, its
``fleet`` remote the checkout itself and its origin a repository nothing
serves, as github.com was on 2026-11-04; a session host rolling into that
checkout as ``make fleet-roll`` does; a duty host whose ``fleet`` remote is
the hub; and that duty host's reads refused by name once the hub cannot be
reached, never answered off the ref it already holds.
"""

from __future__ import annotations

import os
import pathlib

from fleet.core import _test_hooks, rolled, rolled_fetch
from tests._git_fixtures import committed_repository, git
from tests._held_turns import is_held
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
            "-c",
            "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15",
            "fetch",
            "--quiet",
            "fleet",
            "+refs/fleet/rolled:refs/fleet/rolled",
        )
        assert runner.timeouts == [*FETCH_TIMEOUTS] == [30, 60]
        stamp = stamp_of(git_directory)
        assert stamp.read_text(encoding="utf-8") == ""
        assert int(stamp.stat().st_mtime) == DEMO_NOW
        # The turn's lock file stays beside the stamp, given back.
        assert sorted(path.name for path in git_directory.iterdir()) == [
            "fleet-rolled-fetch",
            rolled_fetch.LOCK_NAME,
        ]
        assert not is_held(git_directory / rolled_fetch.LOCK_NAME)

    def test_a_fresh_stamp_answers_without_asking_the_home(self, tmp_path: pathlib.Path) -> None:
        """The reads of one boundary share one fetch."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        git_directory = tmp_path / "git"
        stamped(git_directory, "", seconds_ago=rolled_fetch.FRESH_SECONDS - 1)
        runner = FakeRun([ok(f"{git_directory}\n")])
        _test_hooks.run = runner

        assert rolled_fetch.fetch_rolled(tmp_path / "api") is None

        assert runner.calls == fetch_calls(tmp_path / "api")[:1]

    def test_a_fresh_refusal_is_answered_again_without_asking_the_home(
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
        _test_hooks.run = FakeRun([located, failed(128, "  fatal: unable to access the hub  ")])
        api = tmp_path / "api"

        refusal = rolled_fetch.fetch_rolled(api)

        assert refusal == (
            f"FLEET_ROLL_FETCH_FAILED: git fetch fleet +refs/fleet/rolled:refs/fleet/rolled "
            f"in {api} exited 128: fatal: unable to access the hub"
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


def roll_from(session_host: pathlib.Path, hub: pathlib.Path, settings: str) -> str:
    """Commit a change on the session host and roll it as ``make fleet-roll`` does.

    MCPs maketools ``fleet-roll-push`` pushes the commit into the hub's API
    checkout, the ref's home, before anything else; its copy to origin is
    left out, since nothing on the fleet reads it.

    Args:
        session_host: The session host's clone.
        hub: The hub's API checkout.
        settings: The registry settings the roll carries.

    Returns:
        The rolled commit.
    """
    (session_host / "tools/fleet/registry/settings.json").write_text(settings, encoding="utf-8")
    git(session_host, "commit", "--quiet", "--all", "-m", "a roll")
    git(session_host, "push", "--quiet", str(hub), f"+HEAD:{rolled.ROLLED_REF}")
    git(session_host, "update-ref", rolled.ROLLED_REF, "HEAD")
    return git(session_host, "rev-parse", "HEAD")


def hub_and_session_host(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The hub's API checkout and a session host's clone of it.

    The hub's ``fleet`` remote is the checkout itself, as MCPs release-sync
    points it on the machine holding control, and its origin is a
    repository that does not exist, as github.com was to the fleet's key.

    Args:
        tmp_path: Where both go.

    Returns:
        The hub's checkout and the session host's clone.
    """
    hub = tmp_path / "lacrymal"
    committed_repository(hub, ROLLED_FILES, "refs/heads/main")
    git(hub, "remote", "add", rolled_fetch.REMOTE, str(hub))
    git(hub, "remote", "add", "origin", str(tmp_path / "no-github.git"))
    session_host = tmp_path / "ziggurat"
    git(tmp_path, "clone", "--quiet", "--branch", "main", str(hub), str(session_host))
    return hub, session_host


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


class TestARollIntoTheHub:
    def test_is_what_the_hubs_next_reads_run_with_origin_unreachable(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A1: on 2026-11-04 GitHub refused the fleet's key and every read
        here refused with it; the ref now lives in the hub's own checkout."""
        clock = FakeClock(DEMO_NOW)
        _test_hooks.now = clock
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        hub, session_host = hub_and_session_host(tmp_path)
        records = hub / pathlib.PurePosixPath(rolled.RECORDS_DIR)
        assert unresolved_here(hub)

        first = roll_from(session_host, hub, '{"roll": 2}\n')

        assert rolled.rolled_state(records) == first
        tree = rolled.extract_rolled_tree(hub)
        assert not isinstance(tree, str), tree
        assert tree["commit"] == first
        settings = pathlib.Path(tree["config"]) / "settings.json"
        assert settings.read_text(encoding="utf-8") == '{"roll": 2}\n'

        second = roll_from(session_host, hub, '{"roll": 3}\n')
        # The roll lands in the very ref the hub reads, so no fetch window
        # stands between it and the hub's next read; the window only spaces
        # the reads of a duty host fetching from the hub.
        assert rolled.rolled_state(records) == second

    def test_reaches_a_duty_host_that_fetches_it_from_the_hub(self, tmp_path: pathlib.Path) -> None:
        """A2: diphtheria and emerald run rolled ticks too, and their
        ``fleet`` remote is the hub's checkout at its roster address."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        hub, session_host = hub_and_session_host(tmp_path)
        duty_host = tmp_path / "diphtheria"
        git(tmp_path, "clone", "--quiet", "--branch", "main", str(hub), str(duty_host))
        git(duty_host, "remote", "add", rolled_fetch.REMOTE, str(hub))

        commit = roll_from(session_host, hub, '{"roll": 4}\n')

        assert unresolved_here(duty_host)
        assert rolled.rolled_state(duty_host) == commit

    def test_a_duty_host_that_cannot_reach_the_hub_refuses_rather_than_read_its_own(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A3: the ref it already holds is the roll before, and is never read."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        duty_host = tmp_path / "diphtheria"
        committed_repository(duty_host, ROLLED_FILES, rolled.ROLLED_REF)
        gone = tmp_path / "lacrymal-unreachable"
        git(duty_host, "remote", "add", rolled_fetch.REMOTE, str(gone))

        refusal = rolled.rolled_state(duty_host)

        assert refusal.startswith(
            f"{rolled_fetch.FETCH_FAILED_CODE}: git fetch fleet "
            f"+refs/fleet/rolled:refs/fleet/rolled in {duty_host} exited 128: "
        )
        assert str(gone) in refusal

    def test_a_ref_moved_here_and_never_rolled_is_put_back_to_the_homes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The home's ref is the roll; the push of ``make fleet-roll`` is what
        makes one."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        home = tmp_path / "lacrymal.git"
        home.mkdir()
        git(home, "init", "--quiet", "--bare")
        duty_host = tmp_path / "diphtheria"
        rolled_commit = committed_repository(duty_host, ROLLED_FILES, "refs/heads/main")
        git(duty_host, "remote", "add", rolled_fetch.REMOTE, str(home))
        git(
            duty_host,
            "push",
            "--quiet",
            rolled_fetch.REMOTE,
            f"{rolled_commit}:{rolled.ROLLED_REF}",
        )
        (duty_host / "tools/fleet/registry/settings.json").write_text(
            '{"roll": 9}\n', encoding="utf-8"
        )
        git(duty_host, "commit", "--quiet", "--all", "-m", "never rolled")
        git(duty_host, "update-ref", rolled.ROLLED_REF, "HEAD")

        assert rolled.rolled_state(duty_host) == rolled_commit
