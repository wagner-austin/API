"""Each source repository's default-branch history, published to the queue.

WHY (MCPs board tasks 8b3dcb31 and 1a316615). The queue never knew which
commits are on a repository's main or in what order, and task_update's cancel
gate (MCPs board task 8654d277) must tell a check at a main commit newer than
a failure's from one at a worktree commit main never held: its audit found a
passing check of a commit that never reached main ending the refusal while
main still failed. The gate's caller has no git; the hub has a bare mirror of
every remote a job exports from (:mod:`fleet.core.export`) and the
credentials for each. So each hub tick reads, for every repository the
registry's projects are sourced from, the remote's default branch and its
first-parent history, and hands the queue ``dispatch_main_publish``. The
repositories are the ones the same tick's read of the rolled registry
sources its projects from (:mod:`fleet.core.declared`), so a tick that could
not read the registry reads no main either.

THE DEFAULT BRANCH, NOT A NAMED ONE. The remote's ``HEAD`` names it, ``main``
for most repositories and ``master`` for some; the registry declares a
``base_branch`` only for a project whose check asks what its commit changes.

A COMMIT'S DEPTH is ``git rev-list --first-parent --count`` of it: the head's
is counted once, and the queue gives each older commit one less than the
commit after it, so only the newest :data:`MAX_COMMITS` are sent each tick
and two main commits are ordered by depth with no git.

A REPOSITORY THAT CANNOT BE READ IS LOGGED AND LEFT OUT: a refused credential
or a lost fetch for one remote says nothing about another's, and the queue
keeps what it held for a repository a publication leaves out.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    require_int,
    require_str,
)
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.core import _test_hooks, export, mirror_fetch
from fleet.core.declared import DeclaredProject

#: The queue's tool a publication calls.
PUBLISH_TOOL: Final = "dispatch_main_publish"

#: The ref a remote's default branch is fetched by.
DEFAULT_BRANCH_REF: Final = "HEAD"

#: The most commits one repository's read sends: corvis-db's
#: ``MAX_MAIN_HISTORY_COMMITS``, which refuses more. MCPs, the busiest
#: repository, landed about 190 first-parent commits a day in the week to
#: 2026-11-13, and a tick runs every three minutes.
MAX_COMMITS: Final = 1000

#: Detail prefix for a repository whose projects name two remotes.
REMOTES_DISAGREE_CODE: Final = "FLEET_MAIN_REMOTES_DISAGREE"

#: Detail prefix for a history git could not list.
HISTORY_UNREADABLE_CODE: Final = "FLEET_MAIN_HISTORY_UNREADABLE"


class MainHistoryRead(TypedDict):
    """One repository's default branch as the hub read it.

    Attributes:
        repository: The repository, as the registry's source declarations name it.
        head_depth: The head's first-parent depth.
        commits: The head and its first-parent ancestors, newest first.
    """

    repository: str
    head_depth: int
    commits: tuple[str, ...]


class MainHistoryHead(TypedDict):
    """What the queue answered for one repository.

    Attributes:
        repository: The repository.
        head: Its head as read.
        depth: The head's depth.
        added: How many of the read's commits the queue did not hold before.
    """

    repository: str
    head: str
    depth: int
    added: int


def source_remotes(projects: Sequence[DeclaredProject]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Each source repository and the remote its projects declare.

    Args:
        projects: The projects one read of the rolled registry found
            (:func:`fleet.core.declared.read_declared`).

    Returns:
        The remote of every repository its projects agree on, keyed by
        repository; and a :data:`REMOTES_DISAGREE_CODE` line for each
        repository whose projects name more than one remote, which is left
        out, sorted.
    """
    named: dict[str, set[str]] = {}
    for project in projects:
        source = project["source"]
        if source is not None:
            named.setdefault(source["repository"], set()).add(source["remote"])
    agreed = {
        repository: next(iter(remotes))
        for repository, remotes in named.items()
        if len(remotes) == 1
    }
    disagreeing = tuple(
        f"{REMOTES_DISAGREE_CODE}: {repository}'s projects name {', '.join(sorted(remotes))}; "
        "its main was not read"
        for repository, remotes in sorted(named.items())
        if len(remotes) > 1
    )
    return agreed, disagreeing


def _listed(mirror: pathlib.Path, argv: Sequence[str]) -> str:
    """Run one ``git rev-list`` in the mirror and answer its output.

    Args:
        mirror: The mirror.
        argv: The ``rev-list`` arguments.

    Returns:
        Its standard output, stripped.

    Raises:
        AppError: ``EXPORT_FAILED``, :data:`HISTORY_UNREADABLE_CODE` and git's
            words, when it exits non-zero.
    """
    listed = _test_hooks.run(
        ("git", "-C", str(mirror), "rev-list", *argv),
        timeout_seconds=mirror_fetch.INIT_TIMEOUT_SECONDS,
    )
    if listed["returncode"] != 0:
        raise mirror_fetch.git_failure(
            FleetErrorCode.EXPORT_FAILED,
            f"{HISTORY_UNREADABLE_CODE}: git rev-list {' '.join(argv)} in {mirror}",
            listed["stderr"],
        )
    return listed["stdout"].strip()


def read_main(mirrors_root: pathlib.Path, repository: str, remote: str) -> MainHistoryRead:
    """Fetch a remote's default branch into its mirror and read its history.

    Args:
        mirrors_root: The hub's mirrors directory.
        repository: The repository.
        remote: Its declared remote.

    Returns:
        Its head's depth and its newest :data:`MAX_COMMITS` first-parent
        commits, newest first.

    Raises:
        AppError: As :func:`fleet.core.export.ensure_mirror` and
            :func:`fleet.core.mirror_fetch.fetch_ref` say, or as
            :func:`_listed` does.
    """
    mirror = export.mirror_path(mirrors_root, export.remote_mirror_key(remote))
    export.ensure_mirror(mirror)
    head = mirror_fetch.fetch_ref(
        mirror, remote, DEFAULT_BRANCH_REF, declared_as="the repository's default branch"
    )
    depth = int(_listed(mirror, ("--first-parent", "--count", head)))
    newest = _listed(mirror, ("--first-parent", f"--max-count={MAX_COMMITS}", head))
    commits = tuple(newest.splitlines())
    return MainHistoryRead(repository=repository, head_depth=depth, commits=commits)


def read_mains(
    mirrors_root: pathlib.Path, remotes: Mapping[str, str]
) -> tuple[tuple[MainHistoryRead, ...], tuple[str, ...]]:
    """Read every repository's default branch, leaving out each that fails.

    Args:
        mirrors_root: The hub's mirrors directory.
        remotes: Each repository's remote.

    Returns:
        Each read, in repository order; and, for each repository that could
        not be read, a line naming it and the refusal, in repository order.
    """
    reads: list[MainHistoryRead] = []
    unread: list[str] = []
    for repository, remote in sorted(remotes.items()):
        try:
            reads.append(read_main(mirrors_root, repository, remote))
        except AppError as refused:
            unread.append(f"{repository}: {refused.code}: {refused.message}")
    return tuple(reads), tuple(unread)


def publish_mains(
    credentials: McpCredentials,
    reads: Sequence[MainHistoryRead],
    *,
    read_at: str,
    identity: JSONObject,
) -> tuple[MainHistoryHead, ...]:
    """Hand the queue each repository's history as read.

    Args:
        credentials: The queue's endpoint and headers.
        reads: Each repository's read.
        read_at: An ISO-8601 instant no later than the reads.
        identity: From :func:`fleet.core.queue.identity_arguments`.

    Returns:
        Each repository's head as the queue answered it.

    Raises:
        AppError: Any transport or contract failure from the call, the
            queue's refusal of a stale or rewritten read among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer is not an object carrying a ``heads``
            list of objects with a text ``repository`` and ``head`` and a
            whole ``depth`` and ``added``.
    """
    arguments: JSONObject = {
        "repositories": [
            {
                "repository": read["repository"],
                "headDepth": read["head_depth"],
                "commits": list(read["commits"]),
            }
            for read in reads
        ],
        "readAt": read_at,
        **identity,
    }
    body = load_json_str(call_mcp_tool(_test_hooks.http_post, credentials, PUBLISH_TOOL, arguments))
    if not isinstance(body, dict):
        raise JSONTypeError(f"{PUBLISH_TOOL} must answer an object, got {type(body).__name__}")
    heads = body.get("heads")
    if not isinstance(heads, list):
        raise JSONTypeError(f"{PUBLISH_TOOL} must answer a heads list")
    answered: list[MainHistoryHead] = []
    for head in heads:
        if not isinstance(head, dict):
            raise JSONTypeError(f"{PUBLISH_TOOL} heads must be objects, got {type(head).__name__}")
        answered.append(
            MainHistoryHead(
                repository=require_str(head, "repository"),
                head=require_str(head, "head"),
                depth=require_int(head, "depth"),
                added=require_int(head, "added"),
            )
        )
    return tuple(answered)


__all__ = [
    "DEFAULT_BRANCH_REF",
    "HISTORY_UNREADABLE_CODE",
    "MAX_COMMITS",
    "PUBLISH_TOOL",
    "REMOTES_DISAGREE_CODE",
    "MainHistoryHead",
    "MainHistoryRead",
    "publish_mains",
    "read_main",
    "read_mains",
    "source_remotes",
]
