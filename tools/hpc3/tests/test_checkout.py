"""Tests for the one ledger every worktree of a repository shares.

Every case runs REAL git against repositories built in the test: a main
checkout, a linked worktree of it, a submodule, a worktree of a bare
repository, and a directory in no repository at all. The layout being read
is git's, so a case that wrote the pointer files by hand would only prove the
reader agrees with the test's idea of git.

The incident is board task 747c391e: on 2026-10-05 a campaign resumed in
``API-research-w2`` read its own empty ledger while ``API-research-w3`` held
the rows for twelve live members, and submitted them again.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.cli._config import load_workspace, load_workspace_connection
from hpc3.core.checkout import shared_directory
from tests.conftest import workspace_document, write_file, write_workspace

_WALL_SECONDS = 60
"""Bound on each git call; every one of them answers in milliseconds."""

_RUNS = pathlib.Path("tools", "hpc3", "runs")


def _git(cwd: pathlib.Path, *arguments: str) -> None:
    """Run git in a directory.

    Args:
        cwd: Where to run it.
        *arguments: Git's arguments.
    """
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "protocol.file.allow=always",
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
        timeout=_WALL_SECONDS,
    )


def _repository(path: pathlib.Path) -> pathlib.Path:
    """Make a main checkout with one commit carrying the runs directory.

    Args:
        path: An absent directory to create.

    Returns:
        The checkout's root, resolved.
    """
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    write_file(path / _RUNS / "hpc3.json", b"{}\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "runs")
    return path.resolve()


def _worktree(main: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    """Add a linked worktree of a main checkout.

    Args:
        main: The main checkout.
        path: An absent directory for the worktree.

    Returns:
        The worktree's root, resolved.
    """
    _git(main, "worktree", "add", "-q", "--detach", str(path))
    return path.resolve()


class TestSharedDirectory:
    def test_a_worktree_maps_to_the_main_checkout(self, tmp_path: pathlib.Path) -> None:
        """The whole fix: w2 and w3 both name the main checkout's ledger."""
        main = _repository(tmp_path / "API")
        w2 = _worktree(main, tmp_path / "API-research-w2")
        w3 = _worktree(main, tmp_path / "API-research-w3")
        assert shared_directory(w2 / _RUNS) == main / _RUNS
        assert shared_directory(w3 / _RUNS) == main / _RUNS

    def test_the_worktree_root_itself_maps_to_the_main_root(self, tmp_path: pathlib.Path) -> None:
        main = _repository(tmp_path / "API")
        tree = _worktree(main, tmp_path / "API-tree")
        assert shared_directory(tree) == main

    def test_the_main_checkout_is_its_own_answer(self, tmp_path: pathlib.Path) -> None:
        main = _repository(tmp_path / "API")
        _worktree(main, tmp_path / "API-tree")
        assert shared_directory(main / _RUNS) == main / _RUNS

    def test_a_directory_in_no_repository_is_its_own_answer(self, tmp_path: pathlib.Path) -> None:
        """Nothing else can be sharing a ledger with it."""
        loose = tmp_path / "loose" / "runs"
        loose.mkdir(parents=True)
        assert shared_directory(loose) == loose

    def test_a_submodule_checkout_is_its_own_answer(self, tmp_path: pathlib.Path) -> None:
        """Its .git is a FILE, like a worktree's, but its git directory names
        no common directory: a submodule shares nothing with its parent."""
        child = _repository(tmp_path / "child")
        main = _repository(tmp_path / "API")
        _git(main, "submodule", "add", "-q", str(child), "vendor/child")
        inside = main / "vendor" / "child" / _RUNS
        assert (main / "vendor" / "child" / ".git").is_file()
        assert shared_directory(inside) == inside

    def test_a_worktree_of_a_bare_repository_is_refused(self, tmp_path: pathlib.Path) -> None:
        """There is no main working tree, so no one ledger to share."""
        seed = _repository(tmp_path / "seed")
        _git(tmp_path, "clone", "-q", "--bare", str(seed), "bare.git")
        bare = (tmp_path / "bare.git").resolve()
        tree = tmp_path / "tree"
        _git(bare, "worktree", "add", "-q", "--detach", str(tree))
        with pytest.raises(AppError) as excinfo:
            shared_directory(tree.resolve() / _RUNS)
        assert excinfo.value.code is Hpc3ErrorCode.LEDGER_CHECKOUT_UNREADABLE
        assert str(bare) in excinfo.value.message

    def test_a_dot_git_file_that_is_no_pointer_is_refused(self, tmp_path: pathlib.Path) -> None:
        write_file(tmp_path / "odd" / ".git", b"not a pointer\n")
        (tmp_path / "odd" / "runs").mkdir()
        with pytest.raises(AppError) as excinfo:
            shared_directory(tmp_path / "odd" / "runs")
        assert excinfo.value.code is Hpc3ErrorCode.LEDGER_CHECKOUT_UNREADABLE
        assert "'not a pointer'" in excinfo.value.message


class TestTheCommandsShareTheLedger:
    """What the commands read, through the one loader every entry point uses."""

    def test_a_worktree_workspace_names_the_main_ledger(self, tmp_path: pathlib.Path) -> None:
        main = _repository(tmp_path / "API")
        tree = _worktree(main, tmp_path / "API-tree")
        config = write_workspace(tree / _RUNS / "hpc3.json", workspace_document())
        workspace = load_workspace({"--config": config})
        assert pathlib.Path(workspace["ledger"]) == main / _RUNS / "ledger.jsonl"

    def test_a_worktree_workspace_keeps_its_own_repo(self, tmp_path: pathlib.Path) -> None:
        """The code a run names is the worktree's; only the record is shared."""
        main = _repository(tmp_path / "API")
        tree = _worktree(main, tmp_path / "API-tree")
        config = write_workspace(tree / _RUNS / "hpc3.json", workspace_document())
        workspace = load_workspace({"--config": config})
        assert pathlib.Path(workspace["projects"]["abl"]["repo"]) == tree / _RUNS / "repos/abl"

    def test_the_connection_names_the_same_ledger(self, tmp_path: pathlib.Path) -> None:
        main = _repository(tmp_path / "API")
        tree = _worktree(main, tmp_path / "API-tree")
        config = write_workspace(tree / _RUNS / "hpc3.json", workspace_document())
        connection = load_workspace_connection({"--config": config})
        assert pathlib.Path(connection["ledger"]) == main / _RUNS / "ledger.jsonl"
