"""Real repositories and registry reads for the main-history cases (MCPs board tasks 8b3dcb31 and ecd088e0).

``test_main_history.py`` reads and publishes main with them and
``test_main_ancestry.py`` proves main's descent with them; each builds its
repositories with real git, so a case asserts the history git itself lists.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject

from fleet.core import declared, export
from tests._git_fixtures import git

#: The instant every publication here is read at.
READ_AT = "2026-11-13T22:00:00+00:00"

#: This runner's identity arguments.
IDENTITY: JSONObject = {
    "agent": "fleet-hub-lacrymal",
    "sessionId": "s",
    "cwd": "/home/corvis/fleet",
}

#: Any instant: the clock is pinned so a lost fetch's pauses pass at once.
PINNED = 1_794_000_000


def commit(repo: pathlib.Path, name: str, body: str, message: str) -> str:
    """Write one file, commit it and answer the commit.

    Args:
        repo: The repository.
        name: The file.
        body: Its contents.
        message: The commit's message.

    Returns:
        The commit's id.
    """
    (repo / name).write_text(body, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def merged_repository(repo: pathlib.Path) -> dict[str, str]:
    """A repository whose main merged a side branch: first, second, then the merge.

    Args:
        repo: Where to build it.

    Returns:
        Each commit by name: ``first``, ``side``, ``second`` and ``merge``.
    """
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet", "--initial-branch=main")
    first = commit(repo, "a.txt", "1", "first")
    git(repo, "checkout", "--quiet", "-b", "side")
    side = commit(repo, "b.txt", "side", "side")
    git(repo, "checkout", "--quiet", "main")
    second = commit(repo, "a.txt", "2", "second")
    git(repo, "merge", "--quiet", "--no-ff", "-m", "merge side", "side")
    merge = git(repo, "rev-parse", "HEAD")
    return {"first": first, "side": side, "second": second, "merge": merge}


def rerouted_repository(repo: pathlib.Path) -> dict[str, str]:
    """A main fast-forwarded to a branch that merged it, as osrs-lab's e6b6a0a8 was.

    main is first, second, third; a branch off first made b1, b2 and b3,
    merged main as its second parent, and main fast-forwarded to the merge,
    so main's first-parent chain now runs through the branch.

    Args:
        repo: Where to build it.

    Returns:
        Each commit by name: ``first``, ``b1``, ``b2``, ``b3``, ``second``,
        ``third`` and ``merge``.
    """
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet", "--initial-branch=main")
    first = commit(repo, "a.txt", "1", "first")
    git(repo, "checkout", "--quiet", "-b", "branch")
    built = {"first": first}
    for index in (1, 2, 3):
        built[f"b{index}"] = commit(repo, "b.txt", str(index), f"b{index}")
    git(repo, "checkout", "--quiet", "main")
    built["second"] = commit(repo, "a.txt", "2", "second")
    built["third"] = commit(repo, "a.txt", "3", "third")
    git(repo, "checkout", "--quiet", "branch")
    git(repo, "merge", "--quiet", "--no-ff", "-m", "merge main", "main")
    built["merge"] = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--quiet", "main")
    git(repo, "merge", "--quiet", "--ff-only", "branch")
    return built


def mirror_of(mirrors_root: pathlib.Path, remote: pathlib.Path) -> pathlib.Path:
    """The hub's mirror a read of a local remote fetches into.

    Args:
        mirrors_root: The mirrors directory.
        remote: The remote.

    Returns:
        The mirror's path.
    """
    return export.mirror_path(mirrors_root, export.remote_mirror_key(str(remote)))


def source(remote: str, repository: str) -> declared.DeclaredProject:
    """A declared project sourced from a remote.

    Args:
        remote: The remote.
        repository: The repository its remote names.

    Returns:
        The project.
    """
    return declared.DeclaredProject(
        project=f"{repository}/pkg",
        required_tags=(),
        dispatch_nodes=(),
        source=declared.DeclaredSource(repository=repository, directory="pkg", remote=remote),
    )


def registry(*projects: declared.DeclaredProject) -> declared.RegistryRead:
    """A read of the rolled registry declaring these projects.

    Args:
        projects: The projects.

    Returns:
        The read, at :data:`READ_AT`.
    """
    return declared.RegistryRead(
        commit="c" * 40,
        projects=projects,
        runners=("fleet-node-sedona",),
        machines=(declared.DeclaredMachine(name="sedona", client=False, enabled=True),),
        checked_at=READ_AT,
    )


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every line logged.

    Args:
        caplog: The capture.

    Returns:
        Each record's message.
    """
    return [record.getMessage() for record in caplog.records]
