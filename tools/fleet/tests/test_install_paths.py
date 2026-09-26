"""An install step naming a path the exported commit lacks is refused by
name, before any lease (MCPs board task 8454b6a9).

Measured 2026-09-26: fleet checks of MCPs/packages/db at ce86f4d9 and
MCPs/pcsession-mcp at 69bc7948, both 2026-09-21 commits, closed exit=127
on diphtheria, because today's fleet.json runs ``bash
scripts/testdb-setup.sh`` and those commits predate the script. The probe
is asked of the hub's mirror with real git here, against a real commit, so
what ``git cat-file -e <sha>:<path>`` answers is measured rather than
assumed; the tick's case drives the node agent end to end with the git and
ssh calls faked, as its own suite does.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.core import _test_hooks, export
from tests._node_agent_fixtures import PROBED, _credentials_in_env, node_argv, sourced_document
from tests._queue_fakes import DEFAULT_SHA, FakeQueue, queue_job
from tests.conftest import FakeRun, failed, ok

__all__ = ["_credentials_in_env"]

STEP = ("bash", "scripts/testdb-setup.sh", "--container", "corvis-fleet-testdb")


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run real git in ``repo`` and return its stdout.

    Args:
        repo: The repository.
        *args: git's arguments.

    Returns:
        Standard output, stripped.
    """
    done = subprocess.run(
        ("git", "-C", str(repo), *args), check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


@pytest.fixture(name="commit")
def _commit(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A real repository with one commit carrying ``scripts/present.sh``.

    Args:
        tmp_path: pytest's directory.

    Returns:
        The repository and its commit's sha.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "present.sh").write_text("echo ok\n", encoding="utf-8")
    _git(repo, "add", "scripts/present.sh")
    _git(repo, "commit", "--quiet", "-m", "one")
    return repo, _git(repo, "rev-parse", "HEAD")


class TestInstallPaths:
    def test_a_slashed_non_flag_token_in_the_path_alphabet_is_a_path(self) -> None:
        assert export.install_paths(STEP) == ("scripts/testdb-setup.sh",)

    def test_words_flags_and_flag_values_are_not(self) -> None:
        assert export.install_paths(("npm", "ci")) == ()
        assert export.install_paths(("npm", "--workspace=packages/db", "-w/x")) == ()
        assert export.install_paths(("npx", "a/b", "c/d.sh")) == ("a/b", "c/d.sh")


class TestRequireInstallPaths:
    def test_a_commit_carrying_every_named_path_passes(
        self, commit: tuple[pathlib.Path, str]
    ) -> None:
        repo, sha = commit
        export.require_install_paths(repo, sha, (("npm", "ci"), ("bash", "scripts/present.sh")))

    def test_a_commit_lacking_a_named_path_is_refused_naming_the_step_and_the_path(
        self, commit: tuple[pathlib.Path, str]
    ) -> None:
        repo, sha = commit
        with pytest.raises(AppError) as raised:
            export.require_install_paths(repo, sha, (("bash", "scripts/present.sh"), STEP))
        assert raised.value.code is FleetErrorCode.INSTALL_PATH_NOT_IN_COMMIT
        assert raised.value.message == (
            f"the install step 'bash scripts/testdb-setup.sh --container corvis-fleet-testdb' "
            f"names scripts/testdb-setup.sh, which commit {sha} does not contain: the step was "
            "declared in fleet.json after this commit, so today's registry cannot build it. "
            "Check a commit that carries scripts/testdb-setup.sh, or run this one from a "
            "working tree with fleet-run"
        )

    def test_steps_naming_no_path_ask_git_nothing(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner

        export.require_install_paths(tmp_path, DEFAULT_SHA, (("npm", "ci"), ("npm", "rebuild")))

        assert runner.calls == []


class TestTheTickRefusesBeforeAnyLease:
    def test_a_missing_install_path_closes_the_job_refused_with_nothing_leased(
        self, config_path: pathlib.Path
    ) -> None:
        config_path.write_text(dump_json_str(sourced_document((STEP,))), encoding="utf-8")
        runner = FakeRun(
            [
                *PROBED,
                ok(""),  # git init --bare
                ok(""),  # cat-file -e <sha>^{commit}: held, no fetch
                failed(128, "fatal: path 'scripts/testdb-setup.sh' does not exist"),
            ]
        )
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(config_path)) == 0

        closed = endpoint.arguments[2]
        assert closed["action"] == "close"
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        assert narrow_json_to_str(closed["detail"]).startswith(
            "INSTALL_PATH_NOT_IN_COMMIT: the install step 'bash scripts/testdb-setup.sh "
            "--container corvis-fleet-testdb' names scripts/testdb-setup.sh, which commit "
            f"{DEFAULT_SHA} does not contain"
        )
        assert runner.calls[-1][-3:] == ("cat-file", "-e", f"{DEFAULT_SHA}:scripts/testdb-setup.sh")
        assert not (config_path.parent / "runs" / "leases.json").exists()
