"""Real git for the suites that extract a commit's trees.

Shared by ``test_rolled`` and ``test_cli_rolled``, which each build a
small repository, point a ref at a commit and dirty the working tree, so
the extraction is only right if it reads the commit. The command runs
through the default command hook, which the conftest reset restores
before every test.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

from fleet.core import _test_hooks


def git(repo: pathlib.Path, *arguments: str) -> str:
    """Run one real git command in the repository and return its stdout.

    Args:
        repo: The repository.
        arguments: The git arguments.

    Returns:
        Its standard output, stripped.
    """
    result = _test_hooks.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=fleet-test",
            "-c",
            "user.email=fleet-test@example.invalid",
            "-c",
            "core.autocrlf=false",
            *arguments,
        ),
        timeout_seconds=60,
    )
    assert result["returncode"] == 0, result["stderr"]
    return result["stdout"].strip()


def committed_repository(repo: pathlib.Path, files: Mapping[str, str], ref: str) -> str:
    """Commit ``files`` in a new repository and point ``ref`` at the commit.

    Args:
        repo: Where to build the repository.
        files: Repository-relative paths and their contents.
        ref: The full ref name to set.

    Returns:
        The commit the ref names.
    """
    repo.mkdir()
    git(repo, "init", "--quiet")
    for relative, body in files.items():
        path = repo / pathlib.PurePosixPath(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "committed")
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", ref, commit)
    return commit


__all__ = ["committed_repository", "git"]
