"""The rolled fleet code a tick runs (board task 465689f5).

Two halves, as for the published session-audit. The scripted half pins the
three commands, their order and deadlines, the refusal each failure
produces and the tree a success describes. The real half builds a
repository whose working tree differs from the rolled commit, runs the real
git and tar through the default command hook, and reads the extraction
back off the disk.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from fleet.core import _test_hooks, commit_tree, rolled
from tests._git_fixtures import committed_repository, git
from tests.conftest import FakeRun, FakeTempRoot, failed, ok

#: The commit the scripted ``git rev-parse`` answers with.
COMMIT = "4db6720b0e1f2a3b4c5d6e7f8091a2b3c4d5e6f7"


def destination_of(scratch: pathlib.Path) -> pathlib.Path:
    """Where the extraction of :data:`COMMIT` lands.

    Args:
        scratch: The pinned scratch root.

    Returns:
        This process's own directory under the commit's.
    """
    return scratch / "fleet-rolled" / COMMIT / f"pid-{os.getpid()}"


def scripted_calls(api: pathlib.Path, scratch: pathlib.Path) -> list[tuple[str, ...]]:
    """The three commands a successful extraction runs, in order.

    Args:
        api: The API checkout.
        scratch: The pinned scratch root.

    Returns:
        ``git rev-parse``, ``git archive`` and ``tar -x``, as argv tuples.
    """
    destination = destination_of(scratch)
    tarball = destination / "published.tar"
    return [
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
            "tools/fleet/fleet.json",
        ),
        ("tar", "-x", "-f", str(tarball), "-C", str(destination)),
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


def plant(scratch: pathlib.Path) -> None:
    """Leave the files a successful extraction of :data:`COMMIT` leaves.

    Args:
        scratch: The pinned scratch root.
    """
    for required in rolled.REQUIRED_FILES:
        path = destination_of(scratch) / pathlib.PurePosixPath(required)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


class TestScriptedExtraction:
    def test_three_steps_in_order_and_the_tree_they_describe(self, tmp_path: pathlib.Path) -> None:
        api = tmp_path / "api"
        scratch = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch)
        plant(scratch)
        runner = FakeRun([ok(f"{COMMIT}\n"), ok(""), ok("")])
        _test_hooks.run = runner

        tree = rolled.extract_rolled_tree(api)

        destination = destination_of(scratch)
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
            config=str(destination / "tools" / "fleet" / "fleet.json"),
        )
        assert runner.calls == scripted_calls(api, scratch)
        assert runner.timeouts == [commit_tree.TREE_STEP_TIMEOUT_SECONDS] * 3 == [120] * 3
        assert runner.set_env == [(), (), ()]

    @pytest.mark.parametrize(
        "reply",
        [failed(128, "fatal: Needed a single revision"), ok("refs/fleet/rolled\n")],
    )
    def test_no_roll_refuses_and_nothing_else_runs(
        self, tmp_path: pathlib.Path, reply: _test_hooks.CommandResult
    ) -> None:
        api = tmp_path / "api"
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        runner = FakeRun([reply])
        _test_hooks.run = runner

        refusal = refusal_of(rolled.extract_rolled_tree(api))

        assert refusal.startswith(
            f"FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in {api} did not resolve to a "
            f"commit (exit {reply['returncode']}"
        )
        assert refusal.endswith("; nothing ran")
        assert len(runner.calls) == 1

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
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        replies = [ok(f"{COMMIT}\n"), ok("")][:failing_step]
        runner = FakeRun([*replies, failed(2, "  the step's own complaint  ")])
        _test_hooks.run = runner

        refusal = refusal_of(rolled.extract_rolled_tree(tmp_path / "api"))

        assert refusal.startswith(f"{code}: ")
        assert step_words in refusal
        assert refusal.endswith(" exited 2: the step's own complaint")
        assert len(runner.calls) == failing_step + 1

    def test_an_extraction_missing_an_entry_point_refuses_naming_each_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        scratch = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch)
        _test_hooks.run = FakeRun([ok(f"{COMMIT}\n"), ok(""), ok("")])

        refusal = rolled.extract_rolled_tree(tmp_path / "api")

        assert refusal == (
            f"FLEET_ROLL_INCOMPLETE: the extraction of {COMMIT} at {destination_of(scratch)} "
            "lacks tools/fleet/src/fleet/cli/agent.py, tools/fleet/src/fleet/cli/node_agent.py, "
            "libs/platform_core/src/platform_core/__init__.py, "
            "libs/monorepo_guards/src/monorepo_guards/__init__.py, "
            "tools/board-watch/src/board_watch/__init__.py, tools/fleet/fleet.json; nothing ran"
        )


#: What the rolled commit carries, and one file the archive must leave out.
ROLLED_FILES: dict[str, str] = {
    "tools/fleet/src/fleet/cli/agent.py": "ROLLED = True\n",
    "tools/fleet/src/fleet/cli/node_agent.py": "ROLLED = True\n",
    "tools/fleet/pyproject.toml": "[tool.poetry]\n",
    "libs/platform_core/src/platform_core/__init__.py": "",
    "libs/monorepo_guards/src/monorepo_guards/__init__.py": "",
    "tools/board-watch/src/board_watch/__init__.py": "",
    "tools/fleet/fleet.json": '{"rolled": true}\n',
}


class TestRealExtraction:
    def test_the_rolled_commit_is_extracted_and_the_working_tree_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        api = tmp_path / "api"
        commit = committed_repository(api, ROLLED_FILES, rolled.ROLLED_REF)
        agent_source = api / "tools/fleet/src/fleet/cli/agent.py"
        agent_source.write_text("ROLLED = False\n", encoding="utf-8")
        (api / "tools/fleet/fleet.json").write_text('{"rolled": false}\n', encoding="utf-8")
        git(api, "commit", "--quiet", "--all", "-m", "committed after the roll")
        scratch = tmp_path / "scratch"
        _test_hooks.temp_root = FakeTempRoot(scratch)

        tree = rolled.extract_rolled_tree(api)

        if isinstance(tree, str):
            raise AssertionError(f"expected an extraction, got the refusal {tree!r}")
        assert tree["commit"] == commit
        destination = scratch / "fleet-rolled" / commit / f"pid-{os.getpid()}"
        assert tree["directory"] == str(destination)
        agent = destination / "tools/fleet/src/fleet/cli/agent.py"
        assert agent.read_text(encoding="utf-8") == "ROLLED = True\n"
        assert pathlib.Path(tree["config"]).read_text(encoding="utf-8") == '{"rolled": true}\n'
        assert not (destination / "tools/fleet/pyproject.toml").exists()

    def test_a_repository_never_rolled_refuses_for_real(self, tmp_path: pathlib.Path) -> None:
        api = tmp_path / "api"
        committed_repository(api, ROLLED_FILES, "refs/heads/elsewhere")
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")

        refusal = refusal_of(rolled.extract_rolled_tree(api))

        assert refusal.startswith(
            f"FLEET_ROLL_REF_UNRESOLVED: refs/fleet/rolled in {api} did not resolve to a "
            "commit (exit 128"
        )
        assert not (tmp_path / "scratch" / "fleet-rolled").exists()
