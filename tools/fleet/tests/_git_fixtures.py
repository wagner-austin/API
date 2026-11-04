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

from fleet.core import _test_hooks, rolled_fetch


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


def rolled_at_home(repo: pathlib.Path, files: Mapping[str, str]) -> str:
    """Commit ``files`` in a new repository and roll the commit at its home only.

    The roll a session host makes (MCPs board tasks 8db0f536 and 18ca6bab):
    the commit is pushed as ``refs/fleet/rolled`` to the rolled ref's home, a
    repository beside this one that its ``fleet`` remote names, as a duty
    host's names the hub's checkout, and the repository's own
    ``refs/fleet/rolled`` is never set, so only a fetch of the home's ref can
    bring the roll here.

    Args:
        repo: Where to build the repository; its home is
            ``<repo>-home.git`` beside it.
        files: Repository-relative paths and their contents.

    Returns:
        The commit the home's ``refs/fleet/rolled`` names.
    """
    home = repo.parent / f"{repo.name}-home.git"
    home.mkdir()
    git(home, "init", "--quiet", "--bare")
    commit = committed_repository(repo, files, "refs/heads/rolled-source")
    git(repo, "remote", "add", rolled_fetch.REMOTE, str(home))
    git(repo, "push", "--quiet", rolled_fetch.REMOTE, f"{commit}:refs/fleet/rolled")
    return commit


__all__ = ["committed_repository", "git", "rolled_at_home"]
