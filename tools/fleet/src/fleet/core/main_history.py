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

A COMMIT'S GENERATION (MCPs board task ecd088e0, mig 990) is 1 for a root
commit, else 1 plus the greatest of its parents', computed in one pass over
``git rev-list --topo-order --reverse --parents`` of the head, which lists
every parent before its children. It is the commit's own and rises from a
commit to each of its descendants, so the queue orders two main commits by
it however main's first-parent chain runs. The first-parent depth sent until
then did not: osrs-lab's merge e6b6a0a8 took its side branch as first parent
and reached main as a fast forward, the held head bf635924 left the
first-parent chain at depth 760, the branch's commits took depths 684..1147,
and from 2026-11-18T00:06Z the queue refused every tick's publication as main
rewritten. Only the newest :data:`MAX_COMMITS` first-parent commits are sent.

MAIN REWRITTEN IS DECIDED BY GIT. The queue answers each repository's held
head (``dispatch_main_heads``); the mirror says whether main descends from it
(``git merge-base --is-ancestor``), and the read sends the held head it
proved as ``since``, or None. The queue writes a repository only when
``since`` is the head it still holds, and answers every other repository it
refuses by code beside the heads it wrote, so one refused repository leaves
the rest current and the tick's later passes run.

A REPOSITORY THAT CANNOT BE READ IS LOGGED AND LEFT OUT: a refused credential
or a lost fetch for one remote says nothing about another's, and the queue
keeps what it held for a repository a publication leaves out.

A HOST THAT CANNOT BE RESOLVED IS ASKED ONCE A TICK (MCPs board task
84ee4f44). A fetch git refused because this host could not resolve the
remote's host name does say something about every other remote on that host.
From 21:54 to 22:08Z on 2026-11-14 lacrymal resolved no ``github.com``, and
its tick sent each of 15 repositories' fetches through all of
:data:`fleet.core.mirror_fetch.FETCH_PAUSES_SECONDS`, 75 attempts over 14
minutes, before the same tick's observe pass ran; ziggurat's native evidence
read at 21:51:21Z for 600 s was next renewed 1095.6 s later, and
``execution/tests/served/observe-pass-gaps-live.test.ts`` failed
``OBSERVE_GAP_EXPIRED``. So once one repository's fetch is refused naming an
unresolved host, each later repository whose remote is on that host is left
out unfetched, as :data:`HOST_UNRESOLVED_CODE`, naming the repository whose
fetch found it; the next tick asks again.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Final, TypedDict
from urllib.parse import urlsplit

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_int,
    require_str,
)
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.core import _test_hooks, export, mirror_fetch
from fleet.core.declared import DeclaredProject

#: The queue's tool a publication calls.
PUBLISH_TOOL: Final = "dispatch_main_publish"

#: The queue's tool that answers each repository's held head.
HEADS_TOOL: Final = "dispatch_main_heads"

#: The most commits one repository's read sends: corvis-db's
#: ``MAX_MAIN_HISTORY_COMMITS``, which refuses more. MCPs, the busiest
#: repository, landed about 190 first-parent commits a day in the week to
#: 2026-11-13, and a tick runs every three minutes.
MAX_COMMITS: Final = 1000

#: Detail prefix for a repository whose projects name two remotes.
REMOTES_DISAGREE_CODE: Final = "FLEET_MAIN_REMOTES_DISAGREE"

#: Detail prefix for a history git could not list.
HISTORY_UNREADABLE_CODE: Final = "FLEET_MAIN_HISTORY_UNREADABLE"

#: Detail prefix for a held head whose ancestry git could not answer.
ANCESTRY_UNREADABLE_CODE: Final = "FLEET_MAIN_ANCESTRY_UNREADABLE"

#: Detail prefix for a repository left unfetched because an earlier fetch of
#: the same tick could not resolve its remote's host.
HOST_UNRESOLVED_CODE: Final = "FLEET_MAIN_HOST_UNRESOLVED"


class MainCommit(TypedDict):
    """One main commit and its rank.

    Attributes:
        sha: The commit.
        generation: 1 for a root commit, else 1 plus the greatest of its parents'.
    """

    sha: str
    generation: int


class MainRead(TypedDict):
    """One repository's default branch as the hub fetched and listed it.

    Attributes:
        repository: The repository, as the registry's source declarations name it.
        mirror: The hub's mirror it was read in, which the proof of descent asks.
        head: The fetched head.
        commits: The head and its first-parent ancestors, newest first.
    """

    repository: str
    mirror: pathlib.Path
    head: str
    commits: tuple[MainCommit, ...]


class MainHistoryRead(TypedDict):
    """One repository's read as the queue is handed it.

    Attributes:
        repository: The repository.
        commits: The head and its first-parent ancestors, newest first.
        since: The held head main was proved to descend from, or None when
            the queue holds none or main does not descend from it.
    """

    repository: str
    commits: tuple[MainCommit, ...]
    since: str | None


class HeldHead(TypedDict):
    """A repository's head as the queue holds it.

    Attributes:
        repository: The repository.
        head: The held commit.
        generation: Its generation.
        read_at: The instant the publication that wrote it was read at.
    """

    repository: str
    head: str
    generation: int
    read_at: str


class MainHistoryHead(TypedDict):
    """What the queue answered for one repository it wrote.

    Attributes:
        repository: The repository.
        head: Its head as read.
        generation: The head's generation.
        added: How many of the read's commits the queue did not hold before.
    """

    repository: str
    head: str
    generation: int
    added: int


class MainHistoryRefused(TypedDict):
    """One repository the queue wrote nothing for.

    Attributes:
        repository: The repository.
        code: Why, as the queue names it.
        message: The refusal, opening with its code.
    """

    repository: str
    code: str
    message: str


class MainHistoryPublication(TypedDict):
    """What the queue answered for one publication.

    Attributes:
        heads: Each repository written, in repository order.
        refused: Each repository refused, in the publication's order.
    """

    heads: tuple[MainHistoryHead, ...]
    refused: tuple[MainHistoryRefused, ...]


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
    agreed: dict[str, str] = {}
    disagreeing: list[str] = []
    for repository, remotes in sorted(named.items()):
        match sorted(remotes):
            case [remote]:
                agreed[repository] = remote
            case several:
                disagreeing.append(
                    f"{REMOTES_DISAGREE_CODE}: {repository}'s projects name "
                    f"{', '.join(several)}; its main was not read"
                )
    return agreed, tuple(disagreeing)


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


def generations(mirror: pathlib.Path, head: str) -> dict[str, int]:
    """Every commit the head contains, with its generation.

    Args:
        mirror: The mirror.
        head: The head.

    Returns:
        Each commit's generation: 1 for a root commit, else 1 plus the
        greatest of its parents'.

    Raises:
        AppError: As :func:`_listed` says.
    """
    ranked: dict[str, int] = {}
    for line in _listed(mirror, ("--topo-order", "--reverse", "--parents", head)).splitlines():
        commit, *parents = line.split()
        ranked[commit] = 1 + max((ranked[parent] for parent in parents), default=0)
    return ranked


def read_main(mirrors_root: pathlib.Path, repository: str, remote: str) -> MainRead:
    """Fetch a remote's default branch into its mirror and read its history.

    Args:
        mirrors_root: The hub's mirrors directory.
        repository: The repository.
        remote: Its declared remote.

    Returns:
        Its head and its newest :data:`MAX_COMMITS` first-parent commits,
        newest first, each with its generation.

    Raises:
        AppError: As :func:`fleet.core.export.prepare_default_branch` says,
            or as :func:`_listed` does.
    """
    fetched = export.prepare_default_branch(mirrors_root, remote=remote)
    mirror, head = fetched["mirror"], fetched["head"]
    ranked = generations(mirror, head)
    newest = _listed(mirror, ("--first-parent", f"--max-count={MAX_COMMITS}", head))
    return MainRead(
        repository=repository,
        mirror=mirror,
        head=head,
        commits=tuple(
            MainCommit(sha=commit, generation=ranked[commit]) for commit in newest.splitlines()
        ),
    )


def read_mains(
    mirrors_root: pathlib.Path, remotes: Mapping[str, str]
) -> tuple[tuple[MainRead, ...], tuple[str, ...]]:
    """Read every repository's default branch, leaving out each that fails.

    A repository whose remote is on a host an earlier fetch of this call
    could not resolve is left out without a fetch.

    Args:
        mirrors_root: The hub's mirrors directory.
        remotes: Each repository's remote.

    Returns:
        Each read, in repository order; and, for each repository that could
        not be read, a line naming it and the refusal, or the repository
        whose fetch found its host unresolved, in repository order.
    """
    reads: list[MainRead] = []
    unread: list[str] = []
    unresolved: dict[str, str] = {}
    for repository, remote in sorted(remotes.items()):
        host = urlsplit(remote).hostname
        found_by = None if host is None else unresolved.get(host)
        if found_by is not None:
            unread.append(
                f"{repository}: {HOST_UNRESOLVED_CODE}: not fetched from {remote}: "
                f"this tick's fetch for {found_by} could not resolve {host}"
            )
            continue
        try:
            reads.append(read_main(mirrors_root, repository, remote))
        except AppError as refused:
            unread.append(f"{repository}: {refused.code}: {refused.message}")
            named = mirror_fetch.unresolved_host(refused.message)
            if named is not None:
                unresolved[named] = repository
    return tuple(reads), tuple(unread)


def _git_answers(mirror: pathlib.Path, argv: Sequence[str]) -> bool:
    """Ask the mirror a yes-or-no question git answers by exit status.

    Args:
        mirror: The mirror.
        argv: The git arguments, for a command that exits 0 for yes and 1 for no.

    Returns:
        Whether it exited 0.

    Raises:
        AppError: ``EXPORT_FAILED``, :data:`ANCESTRY_UNREADABLE_CODE` and
            git's words, for any other exit.
    """
    answered = _test_hooks.run(
        ("git", "-C", str(mirror), *argv), timeout_seconds=mirror_fetch.INIT_TIMEOUT_SECONDS
    )
    match answered["returncode"]:
        case 0:
            return True
        case 1:
            return False
        case _:
            raise mirror_fetch.git_failure(
                FleetErrorCode.EXPORT_FAILED,
                f"{ANCESTRY_UNREADABLE_CODE}: git {' '.join(argv)} in {mirror}",
                answered["stderr"],
            )


def descends(mirror: pathlib.Path, held: str, head: str) -> bool:
    """Whether a head contains a held commit.

    Args:
        mirror: The mirror the head was read in.
        held: The commit the queue holds as the repository's head.
        head: The head as read.

    Returns:
        Whether ``held`` is the head or one of its ancestors. A commit the
        mirror does not hold is no ancestor of a head it does.

    Raises:
        AppError: As :func:`_git_answers` says.
    """
    return _git_answers(mirror, ("cat-file", "-e", held)) and _git_answers(
        mirror, ("merge-base", "--is-ancestor", held, head)
    )


def held_heads(credentials: McpCredentials) -> dict[str, HeldHead]:
    """Each repository's head as the queue holds it.

    Args:
        credentials: The queue's endpoint and headers.

    Returns:
        Each held head, keyed by repository.

    Raises:
        AppError: Any transport or contract failure from the call.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer is not an object carrying a ``heads``
            list of objects with a text ``repository``, ``head`` and
            ``readAt`` and a whole ``generation``.
    """
    body = load_json_str(call_mcp_tool(_test_hooks.http_post, credentials, HEADS_TOOL, {}))
    held: dict[str, HeldHead] = {}
    for head in _objects(body, HEADS_TOOL, "heads"):
        repository = require_str(head, "repository")
        held[repository] = HeldHead(
            repository=repository,
            head=require_str(head, "head"),
            generation=require_int(head, "generation"),
            read_at=require_str(head, "readAt"),
        )
    return held


def prove_reads(
    reads: Sequence[MainRead], held: Mapping[str, HeldHead]
) -> tuple[tuple[MainHistoryRead, ...], tuple[str, ...]]:
    """Each read with the held head it was proved to descend from.

    Args:
        reads: Each repository's read.
        held: Each repository's held head, from :func:`held_heads`.

    Returns:
        Each read the mirror could answer for, with ``since`` the held head
        when main descends from it and None otherwise, in the reads' order;
        and, for each it could not, a line naming it and the refusal.
    """
    proved: list[MainHistoryRead] = []
    unproved: list[str] = []
    for read in reads:
        held_head = held.get(read["repository"])
        try:
            since = (
                held_head["head"]
                if held_head is not None
                and descends(read["mirror"], held_head["head"], read["head"])
                else None
            )
        except AppError as refused:
            unproved.append(f"{read['repository']}: {refused.code}: {refused.message}")
            continue
        proved.append(
            MainHistoryRead(repository=read["repository"], commits=read["commits"], since=since)
        )
    return tuple(proved), tuple(unproved)


def _objects(body: JSONValue, tool: str, field: str) -> list[JSONObject]:
    """The list of objects an answer carries under one field.

    Args:
        body: The decoded answer.
        tool: The tool that answered, for the message.
        field: The field.

    Returns:
        Each object, in order.

    Raises:
        JSONTypeError: If the answer is not an object, the field not a list,
            or an entry not an object.
    """
    if not isinstance(body, dict):
        raise JSONTypeError(f"{tool} must answer an object, got {type(body).__name__}")
    entries = body.get(field)
    if not isinstance(entries, list):
        raise JSONTypeError(f"{tool} must answer a {field} list")
    objects: list[JSONObject] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise JSONTypeError(f"{tool} {field} must be objects, got {type(entry).__name__}")
        objects.append(entry)
    return objects


def publish_mains(
    credentials: McpCredentials,
    reads: Sequence[MainHistoryRead],
    *,
    read_at: str,
    identity: JSONObject,
) -> MainHistoryPublication:
    """Hand the queue each repository's history as read.

    Args:
        credentials: The queue's endpoint and headers.
        reads: Each repository's read, with its since.
        read_at: An ISO-8601 instant no later than the reads.
        identity: From :func:`fleet.core.queue.identity_arguments`.

    Returns:
        Each repository the queue wrote, with its head, and each it refused,
        with its code and message.

    Raises:
        AppError: Any transport or contract failure from the call, the
            queue's refusal of a malformed publication among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer is not an object carrying a ``heads``
            list of objects with a text ``repository`` and ``head`` and a
            whole ``generation`` and ``added``, and a ``refused`` list of
            objects with a text ``repository``, ``code`` and ``message``.
    """
    arguments: JSONObject = {
        "repositories": [
            {
                "repository": read["repository"],
                "commits": [
                    {"sha": commit["sha"], "generation": commit["generation"]}
                    for commit in read["commits"]
                ],
                "since": read["since"],
            }
            for read in reads
        ],
        "readAt": read_at,
        **identity,
    }
    body = load_json_str(call_mcp_tool(_test_hooks.http_post, credentials, PUBLISH_TOOL, arguments))
    heads = tuple(
        MainHistoryHead(
            repository=require_str(head, "repository"),
            head=require_str(head, "head"),
            generation=require_int(head, "generation"),
            added=require_int(head, "added"),
        )
        for head in _objects(body, PUBLISH_TOOL, "heads")
    )
    refused = tuple(
        MainHistoryRefused(
            repository=require_str(each, "repository"),
            code=require_str(each, "code"),
            message=require_str(each, "message"),
        )
        for each in _objects(body, PUBLISH_TOOL, "refused")
    )
    return MainHistoryPublication(heads=heads, refused=refused)


__all__ = [
    "ANCESTRY_UNREADABLE_CODE",
    "HEADS_TOOL",
    "HISTORY_UNREADABLE_CODE",
    "MAX_COMMITS",
    "PUBLISH_TOOL",
    "REMOTES_DISAGREE_CODE",
    "HeldHead",
    "MainCommit",
    "MainHistoryHead",
    "MainHistoryPublication",
    "MainHistoryRead",
    "MainHistoryRefused",
    "MainRead",
    "descends",
    "generations",
    "held_heads",
    "prove_reads",
    "publish_mains",
    "read_main",
    "read_mains",
    "source_remotes",
]
