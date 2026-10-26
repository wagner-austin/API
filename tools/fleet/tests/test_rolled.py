"""The rolled fleet code a tick runs (board task 465689f5).

Two halves, as for the published session-audit. The scripted half pins the
three commands, their order and deadlines, the refusal each failure
produces, the tree a success describes, and the reuse and reaping one
extraction per commit brings (MCPs board task fb505198). The real half
builds a repository whose working tree differs from the rolled commit,
runs the real git and tar through the default command hook, and reads the
extraction back off the disk.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from fleet.core import _test_hooks, commit_tree, rolled, rolled_fetch, scratch
from tests._git_fixtures import committed_repository, git, rolled_at_origin
from tests._rolled_fetch_script import FETCH_TIMEOUTS, fetch_calls, fetch_replies
from tests.conftest import DEMO_NOW, FakeClock, FakeRun, FakeTempRoot, failed, ok

#: The commit the scripted ``git rev-parse`` answers with.
COMMIT = "4db6720b0e1f2a3b4c5d6e7f8091a2b3c4d5e6f7"

#: An earlier roll whose tree a test leaves behind.
EARLIER = "0123456789abcdef0123456789abcdef01234567"


def tree_of(scratch_root: pathlib.Path) -> pathlib.Path:
    """Where the one extraction of :data:`COMMIT` lives.

    Args:
        scratch_root: The pinned scratch root.

    Returns:
        The commit's directory under the extractions directory.
    """
    return scratch_root / "fleet-rolled" / COMMIT


def staging_of(scratch_root: pathlib.Path) -> pathlib.Path:
    """Where this process extracts :data:`COMMIT` before renaming it.

    Args:
        scratch_root: The pinned scratch root.

    Returns:
        This process's staging directory.
    """
    return scratch_root / "fleet-rolled" / f"{COMMIT}.pid-{os.getpid()}"


def scripted_calls(api: pathlib.Path, scratch_root: pathlib.Path) -> list[tuple[str, ...]]:
    """The five commands a fresh extraction runs, in order.

    Args:
        api: The API checkout.
        scratch_root: The pinned scratch root.

    Returns:
        The fetch's two (:mod:`tests._rolled_fetch_script`), then ``git
        rev-parse``, ``git archive`` and ``tar -x``, as argv tuples.
    """
    staging = staging_of(scratch_root)
    tarball = staging / "published.tar"
    return [
        *fetch_calls(api),
        ("git", "-C", str(api), "rev-parse", "--verify", "refs/fleet/rolled^{commit}"),
        (
            "git",
            "-C",
            str(api),
            "archive",
            "--format=tar",
            "-o",
            str(tarball),
            COMMIT,
            "--",
            "tools/fleet/src",
            "libs/platform_core/src",
            "libs/monorepo_guards/src",
            "tools/board-watch/src",
            "tools/fleet/registry",
        ),
        ("tar", "-x", "-f", str(tarball), "-C", str(staging)),
    ]


def refusal_of(outcome: rolled.RolledTree | str) -> str:
    """Narrow an extraction's outcome to the refusal a test expects.

    Args:
        outcome: What :func:`~fleet.core.rolled.extract_rolled_tree` returned.

    Returns:
        The refusal detail.

    Raises:
        AssertionError: When the extraction succeeded instead.
    """
    if not isinstance(outcome, str):
        raise AssertionError(f"expected a refusal, got the extraction {outcome!r}")
    return outcome


def tree_from(outcome: rolled.RolledTree | str) -> rolled.RolledTree:
    """Narrow an extraction's outcome to the tree a test expects.

    Args:
        outcome: What :func:`~fleet.core.rolled.extract_rolled_tree` returned.

    Returns:
        The tree.

    Raises:
        AssertionError: When the extraction was refused instead.
    """
    if isinstance(outcome, str):
        raise AssertionError(f"expected an extraction, got the refusal {outcome!r}")
    return outcome


def plant(directory: pathlib.Path, *, tarball: bool) -> None:
    """Leave the files a successful extraction leaves in a directory.

    Args:
        directory: The staging directory, or a commit's tree.
        tarball: Whether to leave the archive tar extracted from too, as a
            staging directory holds it until it is renamed.
    """
    for required in rolled.REQUIRED_FILES:
        path = directory / pathlib.PurePosixPath(required)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    if tarball:
        (directory / commit_tree.TARBALL_NAME).write_bytes(b"tar")


def age(path: pathlib.Path, seconds: int) -> None:
    """Set a path's modification time to ``seconds`` before the pinned clock.

    Args:
        path: The file or directory.
        seconds: How old it is to read.
    """
    then = DEMO_NOW - seconds
    os.utime(path, (then, then))


class TestScriptedExtraction:
    def test_three_steps_in_order_and_the_tree_they_describe(self, tmp_path: pathlib.Path) -> None:
        api = tmp_path / "api"
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.now = FakeClock(DEMO_NOW)
        plant(staging_of(scratch_root), tarball=True)
        runner = FakeRun([*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n"), ok(""), ok("")])
        _test_hooks.run = runner

        tree = rolled.extract_rolled_tree(api)

        destination = tree_of(scratch_root)
        assert tree == rolled.RolledTree(
            commit=COMMIT,
            directory=str(destination),
            python_path=os.pathsep.join(
                (
                    str(destination / "tools" / "fleet" / "src"),
                    str(destination / "libs" / "platform_core" / "src"),
                    str(destination / "libs" / "monorepo_guards" / "src"),
                    str(destination / "tools" / "board-watch" / "src"),
                )
            ),
            config=str(destination / "tools" / "fleet" / "registry"),
            reaped=(),
        )
        assert runner.calls == scripted_calls(api, scratch_root)
        assert runner.timeouts == [*FETCH_TIMEOUTS, *[commit_tree.TREE_STEP_TIMEOUT_SECONDS] * 3]
        assert runner.timeouts == [30, 60, 120, 120, 120]
        assert runner.set_env == [(), (), (), (), ()]
        # Renamed into place whole, without the tarball, and touched now;
        # the turn's lock is given back.
        assert not staging_of(scratch_root).exists()
        assert not (destination / commit_tree.TARBALL_NAME).exists()
        assert int(destination.stat().st_mtime) == DEMO_NOW
        assert sorted(path.name for path in (scratch_root / "fleet-rolled").iterdir()) == [COMMIT]

    def test_a_later_launch_reuses_the_tree_and_runs_only_the_fetch_and_the_ref_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The SSD wear this removes: every tick extracted its own copy."""
        api = tmp_path / "api"
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.now = FakeClock(DEMO_NOW)
        plant(tree_of(scratch_root), tarball=False)
        age(tree_of(scratch_root), 600)
        runner = FakeRun([*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n")])
        _test_hooks.run = runner

        tree = tree_from(rolled.extract_rolled_tree(api))

        assert tree["directory"] == str(tree_of(scratch_root))
        assert runner.calls == scripted_calls(api, scratch_root)[:3]
        assert int(tree_of(scratch_root).stat().st_mtime) == DEMO_NOW

    def test_a_tree_missing_a_file_is_removed_and_refused(self, tmp_path: pathlib.Path) -> None:
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        plant(tree_of(scratch_root), tarball=False)
        damaged = tree_of(scratch_root) / pathlib.PurePosixPath(rolled.REQUIRED_FILES[1])
        damaged.unlink()
        runner = FakeRun([*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n")])
        _test_hooks.run = runner

        refusal = rolled.extract_rolled_tree(tmp_path / "api")

        assert refusal == (
            f"FLEET_ROLL_INCOMPLETE: the extraction of {COMMIT} at {tree_of(scratch_root)} "
            "lacks tools/fleet/src/fleet/cli/node_agent.py; nothing ran"
        )
        assert len(runner.calls) == 3
        assert not tree_of(scratch_root).exists()

    @pytest.mark.parametrize(
        "reply",
        [failed(128, "fatal: Needed a single revision"), ok("refs/fleet/rolled\n")],
    )
    def test_no_roll_refuses_and_nothing_else_runs(
        self, tmp_path: pathlib.Path, reply: _test_hooks.CommandResult
    ) -> None:
        api = tmp_path / "api"
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        runner = FakeRun([*fetch_replies(tmp_path / "git"), reply])
        _test_hooks.run = runner

        refusal = refusal_of(rolled.extract_rolled_tree(api))

        assert refusal.startswith(
            f"FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in {api} did not resolve to a "
            f"commit (exit {reply['returncode']}"
        )
        assert refusal.endswith("; nothing ran")
        assert len(runner.calls) == 3

    def test_the_state_a_serve_compares_refuses_when_the_fetched_ref_names_no_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        records = tmp_path / "api" / "tools" / "fleet"
        runner = FakeRun(
            [*fetch_replies(tmp_path / "git"), failed(128, "fatal: Needed a single revision")]
        )
        _test_hooks.run = runner

        state = rolled.rolled_state(records)

        assert state.startswith(
            f"FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in {records} did not resolve to a "
            "commit (exit 128"
        )
        assert runner.calls[:2] == fetch_calls(records)
        assert len(runner.calls) == 3

    def test_a_failed_fetch_refuses_before_the_ref_is_read(self, tmp_path: pathlib.Path) -> None:
        """A roll origin holds is not read off a ref the fetch could not move
        (MCPs board task 8db0f536)."""
        api = tmp_path / "api"
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        located, _ = fetch_replies(tmp_path / "git")
        runner = FakeRun([located, failed(128, "fatal: unable to access origin")])
        _test_hooks.run = runner

        refusal = refusal_of(rolled.extract_rolled_tree(api))

        assert refusal == (
            f"{rolled_fetch.FETCH_FAILED_CODE}: git fetch origin "
            f"+refs/fleet/rolled:refs/fleet/rolled in {api} exited 128: "
            "fatal: unable to access origin"
        )
        assert runner.calls == fetch_calls(api)
        assert not (scratch_root / "fleet-rolled").exists()

    @pytest.mark.parametrize(
        ("failing_step", "code", "step_words"),
        [
            (1, "FLEET_ROLL_ARCHIVE_FAILED", f"git archive {COMMIT}"),
            (2, "FLEET_ROLL_EXTRACT_FAILED", "tar -x "),
        ],
    )
    def test_a_failed_step_refuses_by_its_own_code_and_stops_there(
        self, tmp_path: pathlib.Path, failing_step: int, code: str, step_words: str
    ) -> None:
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        replies = [ok(f"{COMMIT}\n"), ok("")][:failing_step]
        runner = FakeRun(
            [*fetch_replies(tmp_path / "git"), *replies, failed(2, "  the step's own complaint  ")]
        )
        _test_hooks.run = runner

        refusal = refusal_of(rolled.extract_rolled_tree(tmp_path / "api"))

        assert refusal.startswith(f"{code}: ")
        assert step_words in refusal
        assert refusal.endswith(" exited 2: the step's own complaint")
        assert len(runner.calls) == failing_step + 3
        # MCPs board task 94ac1c4f: a refused launch leaves no directory.
        assert not staging_of(scratch_root).exists()
        assert not tree_of(scratch_root).exists()

    def test_an_extraction_missing_an_entry_point_refuses_naming_each_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.run = FakeRun(
            [*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n"), ok(""), ok("")]
        )

        refusal = rolled.extract_rolled_tree(tmp_path / "api")

        assert refusal == (
            f"FLEET_ROLL_INCOMPLETE: the extraction of {COMMIT} at {staging_of(scratch_root)} "
            "lacks tools/fleet/src/fleet/cli/agent.py, tools/fleet/src/fleet/cli/node_agent.py, "
            "libs/platform_core/src/platform_core/__init__.py, "
            "libs/monorepo_guards/src/monorepo_guards/__init__.py, "
            "tools/board-watch/src/board_watch/__init__.py, "
            "tools/fleet/registry/settings.json; nothing ran"
        )
        assert not staging_of(scratch_root).exists()
        assert not tree_of(scratch_root).exists()


class TestReaping:
    def test_the_turn_removes_what_is_stale_and_keeps_the_rolled_tree(
        self, tmp_path: pathlib.Path
    ) -> None:
        """873 dead launches' copies and 3.6 GB of archives, measured on
        diphtheria on 2026-10-24, are what this removes."""
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.now = FakeClock(DEMO_NOW)
        extractions = scratch_root / "fleet-rolled"
        archives = scratch_root / "fleet-archives"
        plant(tree_of(scratch_root), tarball=False)
        age(tree_of(scratch_root), scratch.STALE_SECONDS * 2)
        plant(extractions / EARLIER, tarball=False)
        age(extractions / EARLIER, scratch.STALE_SECONDS + 1)
        dead_staging = extractions / f"{EARLIER}.pid-7"
        plant(dead_staging, tarball=True)
        age(dead_staging, scratch.STALE_SECONDS + 1)
        # Exactly the bound old: an agent of that roll may still run from it.
        plant(extractions / ("f" * 40), tarball=False)
        age(extractions / ("f" * 40), scratch.STALE_SECONDS)
        archives.mkdir()
        (archives / "dead-run.tgz").write_bytes(b"left by a failed stage")
        age(archives / "dead-run.tgz", scratch.STALE_SECONDS + 1)
        (archives / "live-run.tgz").write_bytes(b"a stage under way")
        _test_hooks.run = FakeRun([*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n")])

        tree = tree_from(rolled.extract_rolled_tree(tmp_path / "api"))

        assert tree["reaped"] == (
            f"fleet-rolled/{EARLIER}",
            f"fleet-rolled/{EARLIER}.pid-7",
            "fleet-archives/dead-run.tgz",
        )
        assert sorted(path.name for path in extractions.iterdir()) == [COMMIT, "f" * 40]
        assert sorted(path.name for path in archives.iterdir()) == ["live-run.tgz"]

    def test_a_refused_launch_still_reaps(self, tmp_path: pathlib.Path) -> None:
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)
        _test_hooks.now = FakeClock(DEMO_NOW)
        archives = scratch_root / "fleet-archives"
        archives.mkdir(parents=True)
        (archives / "dead-run.tgz").write_bytes(b"left by a failed stage")
        age(archives / "dead-run.tgz", scratch.STALE_SECONDS + 1)
        _test_hooks.run = FakeRun(
            [*fetch_replies(tmp_path / "git"), ok(f"{COMMIT}\n"), failed(128, "bad object")]
        )

        refusal_of(rolled.extract_rolled_tree(tmp_path / "api"))

        assert list(archives.iterdir()) == []


#: What the rolled commit carries, and one file the archive must leave out.
ROLLED_FILES: dict[str, str] = {
    "tools/fleet/src/fleet/cli/agent.py": "ROLLED = True\n",
    "tools/fleet/src/fleet/cli/node_agent.py": "ROLLED = True\n",
    "tools/fleet/pyproject.toml": "[tool.poetry]\n",
    "libs/platform_core/src/platform_core/__init__.py": "",
    "libs/monorepo_guards/src/monorepo_guards/__init__.py": "",
    "tools/board-watch/src/board_watch/__init__.py": "",
    "tools/fleet/registry/settings.json": '{"rolled": true}\n',
}


class TestRealExtraction:
    def test_the_rolled_commit_is_extracted_once_and_the_working_tree_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        api = tmp_path / "api"
        commit = rolled_at_origin(api, ROLLED_FILES)
        agent_source = api / "tools/fleet/src/fleet/cli/agent.py"
        agent_source.write_text("ROLLED = False\n", encoding="utf-8")
        (api / "tools/fleet/registry/settings.json").write_text(
            '{"rolled": false}\n', encoding="utf-8"
        )
        git(api, "commit", "--quiet", "--all", "-m", "committed after the roll")
        scratch_root = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch_root)

        tree = tree_from(rolled.extract_rolled_tree(api))

        assert tree["commit"] == commit
        destination = scratch_root / "fleet-rolled" / commit
        assert tree["directory"] == str(destination)
        agent = destination / "tools/fleet/src/fleet/cli/agent.py"
        assert agent.read_text(encoding="utf-8") == "ROLLED = True\n"
        settings = pathlib.Path(tree["config"]) / "settings.json"
        assert settings.read_text(encoding="utf-8") == '{"rolled": true}\n'
        assert not (destination / "tools/fleet/pyproject.toml").exists()
        assert not (destination / commit_tree.TARBALL_NAME).exists()

        # The next launch of the same roll reads the same tree: what an
        # agent wrote into it, its bytecode, is still there.
        (destination / "written-by-the-agent").write_text("", encoding="utf-8")

        again = tree_from(rolled.extract_rolled_tree(api))

        assert again["directory"] == str(destination)
        assert (destination / "written-by-the-agent").exists()

    def test_the_state_a_serve_compares_is_read_from_its_records_directory_for_real(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A serving agent knows only its records directory, tools/fleet in
        the checkout, and reads the ref from there; a roll moves what it
        reads (MCPs board task 8993c306) once origin holds it and the last
        fetch is no longer fresh (MCPs board task 8db0f536)."""
        clock = FakeClock(DEMO_NOW)
        _test_hooks.now = clock
        api = tmp_path / "api"
        first = rolled_at_origin(api, ROLLED_FILES)
        records = api / pathlib.PurePosixPath(rolled.RECORDS_DIR)

        assert rolled.rolled_state(records) == first

        (api / "tools/fleet/registry/settings.json").write_text(
            '{"rolled": "again"}\n', encoding="utf-8"
        )
        git(api, "commit", "--quiet", "--all", "-m", "the next roll")
        second = git(api, "rev-parse", "HEAD")
        git(api, "push", "--quiet", "--force", "origin", f"{second}:{rolled.ROLLED_REF}")
        clock.seconds += rolled_fetch.FRESH_SECONDS

        assert rolled.rolled_state(records) == second
        assert second != first

    def test_a_repository_never_rolled_answers_with_the_refusal(
        self, tmp_path: pathlib.Path
    ) -> None:
        api = tmp_path / "api"
        never_rolled(api)

        assert rolled.rolled_state(api).startswith(
            f"{rolled_fetch.FETCH_FAILED_CODE}: git fetch origin "
            f"+refs/fleet/rolled:refs/fleet/rolled in {api} exited 128: "
            "fatal: couldn't find remote ref refs/fleet/rolled"
        )

    def test_a_repository_never_rolled_refuses_for_real(self, tmp_path: pathlib.Path) -> None:
        api = tmp_path / "api"
        never_rolled(api)
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")

        refusal = refusal_of(rolled.extract_rolled_tree(api))

        assert refusal.startswith(
            f"{rolled_fetch.FETCH_FAILED_CODE}: git fetch origin "
            f"+refs/fleet/rolled:refs/fleet/rolled in {api} exited 128"
        )
        assert not (tmp_path / "scratch" / "fleet-rolled").exists()


def never_rolled(api: pathlib.Path) -> None:
    """Build a repository whose origin holds a branch and no roll.

    Args:
        api: Where to build it; its origin is ``<api>-origin.git`` beside it.
    """
    origin = api.parent / f"{api.name}-origin.git"
    origin.mkdir()
    git(origin, "init", "--quiet", "--bare")
    committed_repository(api, ROLLED_FILES, "refs/heads/elsewhere")
    git(api, "remote", "add", "origin", str(origin))
    git(api, "push", "--quiet", "origin", "refs/heads/elsewhere")
