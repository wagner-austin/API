"""``.githooks/published-maketools.sh`` runs MCPs' maketools as origin/main carries it.

MCPs board task 691b0067. THE SCRIPT IS EXECUTED, NOT READ: what can be
wrong in it is the binding, which MCPs it finds, which commit of it runs,
whether the command's exit status reaches the hook that called it. So each
case copies the shipped file byte for byte into a temporary layout, a
repository beside a real MCPs git repository whose origin/main carries a
maketools launcher that echoes its arguments and exits with the last one,
and runs it with sh. The launcher is the fixture's own; the real
commit-tasks is tested in MCPs against the board's answers.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from maketools import _test_hooks
from maketools.cli import repository_root

SCRIPT: Final[Path] = repository_root() / ".githooks" / "published-maketools.sh"

#: The launcher origin/main carries: it echoes and exits with its last argument.
PUBLISHED: Final[str] = (
    'import sys\nprint("published", *sys.argv[1:])\nsys.exit(int(sys.argv[-1]))\n'
)

#: An uncommitted edit to that launcher, which must never be what runs.
EDITED: Final[str] = 'import sys\nprint("edited", *sys.argv[1:])\nsys.exit(0)\n'

#: Wall for each child, every one of which exits at once.
PROBE_WALL_SECONDS: Final[int] = 120


def _git(root: Path, *args: str) -> None:
    """Run git in ``root`` with a fixed identity, requiring success.

    Args:
        root: The repository.
        args: git's arguments.
    """
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=PROBE_WALL_SECONDS,
    )


def _layout(parent: Path, *, published: bool) -> Path:
    """A repository holding the shipped script, beside an MCPs repository.

    Args:
        parent: The directory both are made in.
        published: Whether MCPs has an origin/main carrying the launcher.

    Returns:
        The repository's root.
    """
    repo = parent / "repo"
    (repo / ".githooks").mkdir(parents=True)
    shutil.copyfile(SCRIPT, repo / ".githooks" / "published-maketools.sh")
    launcher = parent / "MCPs" / "packages" / "maketools" / "scripts" / "run.py"
    launcher.parent.mkdir(parents=True)
    _git(parent / "MCPs", "init", "--quiet")
    launcher.write_text(PUBLISHED, encoding="utf-8")
    _git(parent / "MCPs", "add", ".")
    _git(parent / "MCPs", "commit", "--quiet", "--message", "the published launcher")
    if published:
        _git(parent / "MCPs", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _run(repo: Path, git_env: Mapping[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copied script with sh.

    Args:
        repo: The repository holding it.
        git_env: Variables git would export to a hook, laid over this environment.
        args: The maketools command and its arguments.

    Returns:
        The finished child.
    """
    return subprocess.run(
        ["sh", ".githooks/published-maketools.sh", *args],
        cwd=repo,
        env={**_test_hooks.environ(), **git_env},
        capture_output=True,
        text=True,
        timeout=PROBE_WALL_SECONDS,
    )


def test_it_runs_origin_mains_command_and_exits_with_its_status(tmp_path: Path) -> None:
    finished = _run(_layout(tmp_path, published=True), {}, "commit-tasks", "../MCPs", "3")
    assert finished.stdout.strip() == "published commit-tasks ../MCPs 3"
    assert finished.returncode == 3


def test_it_never_runs_an_uncommitted_edit_in_the_mcps_checkout(tmp_path: Path) -> None:
    repo = _layout(tmp_path, published=True)
    launcher = tmp_path / "MCPs" / "packages" / "maketools" / "scripts" / "run.py"
    launcher.write_text(EDITED, encoding="utf-8")
    finished = _run(repo, {}, "commit-message-tasks", "4")
    assert finished.stdout.strip() == "published commit-message-tasks 4"
    assert finished.returncode == 4


def test_it_reads_mcps_inside_a_hook_where_git_exported_this_repositorys_git_dir(
    tmp_path: Path,
) -> None:
    """A hook runs with GIT_DIR naming the repository being committed.

    GIT_DIR outranks ``git -C``, so a script naming MCPs with -C read this
    repository instead and refused every commit, which is how the first real
    commit through the hook failed while every case above passed.
    """
    repo = _layout(tmp_path, published=True)
    _git(repo, "init", "--quiet")
    finished = _run(repo, {"GIT_DIR": str(repo / ".git")}, "commit-message-tasks", "5")
    assert finished.stdout.strip() == "published commit-message-tasks 5"
    assert finished.returncode == 5


def test_it_refuses_naming_what_did_not_run_when_mcps_has_no_origin_main(
    tmp_path: Path,
) -> None:
    repo = _layout(tmp_path, published=False)
    finished = _run(repo, {}, "commit-tasks", "0")
    # The script names the directory as sh's pwd spells it, so that is what
    # the expected line is built from.
    shell_repo = subprocess.run(
        ["sh", "-c", "pwd"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=PROBE_WALL_SECONDS,
    ).stdout.strip()
    assert finished.stdout == ""
    assert finished.stderr.splitlines()[-1] == (
        f"published-maketools: {shell_repo}/../MCPs has no origin/main carrying "
        "packages/maketools, so commit-tasks did not run and what called it is refused"
    )
    assert finished.returncode == 1
