"""Every project the rolled fleet registry declares, published to the queue.

WHY (MCPs board tasks 1915b042 and 0be88a36). A node runner claims only the
projects it fits this tick, and those come from the registry, so a queued job
naming a key the registry does not declare was claimed by nobody and closed
by nobody. Measured 2026-10-14: a check submitted as ``MCPs/wiki-check``,
whose key is ``MCPs/packages/wiki-check``, sat queued about 80 minutes until
it was cancelled by hand. And every submitter read the registry from an API
checkout beside it to learn a project's ``required_tags``, so a submitter
with no API checkout, CI's gateway among them, could not submit at all. So
the hub tick publishes what it read, every project with its tags
(``dispatch_registry_publish``): the queue keeps it, and in the same
transaction refuses each queued make-target job queued before the read whose
key is not among them.

READ AT THE REF, NOT FROM THE EXTRACTION. A serve runs the registry it was
extracted with, and a roll made between that extraction and the serve's
first read of the ref keeps it on the older code until its time is up
(:func:`fleet.cli.node_serve.handover_policy`). A publication of that
registry would put back what the roll had just changed. So the projects are
read from the commit :data:`fleet.core.rolled.ROLLED_REF` names at the read
itself, and the instant sent with them is taken before the ref is resolved:
a job queued before it named a key the rolled registry did not declare at a
moment after the job existed.

ONE GIT PROCESS FOR EVERY FILE. The registry declared 144 projects on
2026-10-15, and the hub tick runs every three minutes; one ``git cat-file
--batch`` answers every file at the commit, where a ``git show`` per file
would spawn 144 processes a tick.

AND EVERY NODE RUNNER THE ROSTER DECLARES (MCPs board task 1e9ab738). The
same read takes the machine files and names each runner the hub runs:
``fleet-node-<machine>`` for a machine whose dispatch is enabled and not
held, the rule MCPs session-audit's ``fleet_lanes`` lays the hub's lanes by.
The queue deletes the tick of every runner the publication does not name
(MCPs mig 824): until then fleet-node-lavender-wsl stayed on
``fleet_status`` a day after lavender's reimage took its machine.
"""

from __future__ import annotations

import datetime
import pathlib
from collections.abc import Sequence
from typing import Final, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    require_int,
    require_str_list,
)
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.machine import Machine, decode_machine
from fleet.contracts.project import decode_project_config
from fleet.contracts.tags import NodeTag, encode_tags
from fleet.core import _test_hooks, commit_tree, names, rolled
from fleet.core.workspace_files import (
    JSON_SUFFIX,
    MACHINES_DIRECTORY,
    PROJECTS_DIRECTORY,
    project_key,
)

#: The registry's per-project files, as paths in the API repository.
PROJECTS_PATH: Final = f"{rolled.REGISTRY_DIRECTORY}/{PROJECTS_DIRECTORY}"

#: The roster's per-machine files, as paths in the API repository.
MACHINES_PATH: Final = f"{rolled.REGISTRY_DIRECTORY}/{MACHINES_DIRECTORY}"

#: The queue's tool a publication calls.
PUBLISH_TOOL: Final = "dispatch_registry_publish"

#: Detail prefix when the registry's files could not be listed.
LIST_FAILED_CODE: Final = "FLEET_REGISTRY_LIST_FAILED"

#: Detail prefix when the listing named no project.
EMPTY_CODE: Final = "FLEET_REGISTRY_EMPTY"

#: Detail prefix when the roster declared no node runner.
NO_RUNNER_CODE: Final = "FLEET_ROSTER_NO_RUNNER"

#: Detail prefix when the files could not be read at the commit.
READ_FAILED_CODE: Final = "FLEET_REGISTRY_READ_FAILED"

#: Detail prefix when ``git cat-file --batch`` answered something not a blob.
BATCH_UNREADABLE_CODE: Final = "FLEET_REGISTRY_BATCH_UNREADABLE"


class DeclaredProject(TypedDict):
    """One project as the rolled registry declares it.

    Attributes:
        project: Its key, the file's path under ``projects`` without the suffix.
        required_tags: The tags its file names, decoded by the project contract.
    """

    project: str
    required_tags: tuple[NodeTag, ...]


class RegistryRead(TypedDict):
    """The registry as one read of the rolled ref found it.

    Attributes:
        commit: The full commit id :data:`fleet.core.rolled.ROLLED_REF` named.
        projects: Every project that commit declares, sorted by key.
        runners: Every node runner its roster declares, in machine order
            (:func:`declared_runners`).
        checked_at: An ISO-8601 instant in UTC taken before the ref was
            resolved, floored to the second.
    """

    commit: str
    projects: tuple[DeclaredProject, ...]
    runners: tuple[str, ...]
    checked_at: str


class RegistryPublication(TypedDict):
    """What the queue did with one publication.

    Attributes:
        published: How many projects it now holds, all from the publication.
        removed: How many the previous publication held and this one dropped.
        refused: How many queued jobs it refused as undeclared.
        retired: The runners whose ticks it deleted, as not declared.
    """

    published: int
    removed: int
    refused: int
    retired: tuple[str, ...]


def declared_files(listing: str) -> tuple[tuple[str, str], ...]:
    """Each project key and its file, from a ``git ls-tree`` listing.

    Args:
        listing: One repository path per line, as ``git ls-tree -r
            --name-only`` prints them.

    Returns:
        ``(key, path)`` for each per-project file, sorted by key, the key read
        with :func:`fleet.core.workspace_files.project_key` exactly as the
        workspace loader keys them; a path without the suffix is no project,
        and one outside :data:`PROJECTS_PATH` is a machine's.
    """
    projects = pathlib.Path(PROJECTS_PATH)
    return tuple(
        sorted(
            (project_key(projects, pathlib.Path(line)), line)
            for line in listing.splitlines()
            if line.startswith(f"{PROJECTS_PATH}/") and line.endswith(JSON_SUFFIX)
        )
    )


def machine_files(listing: str) -> tuple[tuple[str, str], ...]:
    """Each machine's name and its file, from the same ``git ls-tree`` listing.

    Args:
        listing: One repository path per line.

    Returns:
        ``(name, path)`` for each file directly under :data:`MACHINES_PATH`,
        sorted by name, the name its file's stem as the workspace loader
        reads it.
    """
    return tuple(
        sorted(
            (line.removeprefix(f"{MACHINES_PATH}/").removesuffix(JSON_SUFFIX), line)
            for line in listing.splitlines()
            if line.startswith(f"{MACHINES_PATH}/") and line.endswith(JSON_SUFFIX)
        )
    )


def declared_runners(machines: Sequence[Machine]) -> tuple[str, ...]:
    """Every node runner a roster declares, in the machines' order.

    A machine runs its node runner when its dispatch is enabled and not held
    (the node's ``enabled``): the rule MCPs session-audit's ``fleet_lanes``
    lays the hub's lanes by, so a label here is a runner the hub ticks.

    Args:
        machines: The roster's machines.

    Returns:
        Each runner's board label (:func:`fleet.core.names.node_runner_label`).
    """
    return tuple(
        names.node_runner_label(machine["name"])
        for machine in machines
        if (node := machine["dispatch"]) is not None and node["enabled"]
    )


def split_batch(output: bytes, paths: Sequence[str]) -> tuple[str, ...] | str:
    """The blobs ``git cat-file --batch`` answered, one per requested path.

    Args:
        output: The command's standard output as bytes. Each object is a
            ``<oid> blob <size>`` line, ``size`` bytes of content and a
            newline, in the order the paths were asked for.
        paths: The paths asked for, for the refusal's message.

    Returns:
        Each blob's text in request order, or a ``CODE: message`` refusal when
        an answer is not a blob header (a path missing at the commit answers
        ``<name> missing``) or the output ends early.
    """
    blobs: list[str] = []
    offset = 0
    for path in paths:
        end = output.find(b"\n", offset)
        header = output[offset:end].decode("utf-8") if end >= 0 else ""
        parts = header.split(" ")
        if len(parts) != 3 or parts[1] != "blob" or not parts[2].isdigit():
            return (
                f"{BATCH_UNREADABLE_CODE}: git cat-file --batch answered {header!r} for "
                f"{path}, not a blob header"
            )
        start = end + 1
        size = int(parts[2])
        blobs.append(output[start : start + size].decode("utf-8"))
        offset = start + size + 1
    return tuple(blobs)


def read_declared(checkout: pathlib.Path) -> RegistryRead | str:
    """Read every project and node runner the rolled registry declares, now.

    Args:
        checkout: A directory inside the API checkout whose object store
            holds the ref: the agent's records directory.

    Returns:
        The read, or a ``CODE: message`` refusal when the ref names no
        commit, the listing or the read failed, or it declared no project
        or no node runner; nothing is sent to the queue then.

    Raises:
        InvalidJsonError: When a project or machine file at the commit is
            not JSON.
        JSONTypeError: When one does not decode as a project or a machine:
            the rolled registry is then broken, which every runner's
            extraction meets too, so it is raised rather than published
            around.
    """
    checked_at = datetime.datetime.fromtimestamp(_test_hooks.now(), tz=datetime.UTC).isoformat()
    resolved = commit_tree.resolve_commit(checkout, rolled.ROLLED_REF, rolled.REF_UNRESOLVED_CODE)
    if isinstance(resolved, str):
        return resolved
    commit = resolved["commit"]
    listed = _test_hooks.run(
        (
            "git",
            "-C",
            str(checkout),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            f"{PROJECTS_PATH}/",
            f"{MACHINES_PATH}/",
        ),
        timeout_seconds=commit_tree.TREE_STEP_TIMEOUT_SECONDS,
    )
    if listed["returncode"] != 0:
        return commit_tree.step_refusal(LIST_FAILED_CODE, f"git ls-tree {commit}", listed)
    files = declared_files(listed["stdout"])
    if not files:
        return (
            f"{EMPTY_CODE}: {PROJECTS_PATH} at {commit} declares no project; nothing was published"
        )
    machine_paths = machine_files(listed["stdout"])
    paths = [path for _, path in files] + [path for _, path in machine_paths]
    batch = _test_hooks.run(
        ("git", "-C", str(checkout), "cat-file", "--batch"),
        timeout_seconds=commit_tree.TREE_STEP_TIMEOUT_SECONDS,
        stdin_bytes="".join(f"{commit}:{path}\n" for path in paths).encode("utf-8"),
    )
    if batch["returncode"] != 0:
        return commit_tree.step_refusal(READ_FAILED_CODE, f"git cat-file --batch {commit}", batch)
    # The seam hands stdout back decoded; registry files are UTF-8 JSON, so
    # encoding it again restores the bytes the blob sizes count.
    blobs = split_batch(batch["stdout"].encode("utf-8"), paths)
    if isinstance(blobs, str):
        return blobs
    projects = tuple(
        DeclaredProject(
            project=key,
            required_tags=decode_project_config(load_json_str(blob))["required_tags"],
        )
        for (key, _), blob in zip(files, blobs[: len(files)], strict=True)
    )
    runners = declared_runners(
        [
            decode_machine(name, load_json_str(blob))
            for (name, _), blob in zip(machine_paths, blobs[len(files) :], strict=True)
        ]
    )
    if not runners:
        return (
            f"{NO_RUNNER_CODE}: {MACHINES_PATH} at {commit} declares no node runner; "
            "nothing was published, since the queue would retire every runner"
        )
    return RegistryRead(commit=commit, projects=projects, runners=runners, checked_at=checked_at)


def publish_registry(
    credentials: McpCredentials, read: RegistryRead, *, identity: JSONObject
) -> RegistryPublication:
    """Hand the queue a registry read, so it keeps it and refuses undeclared work.

    Args:
        credentials: The queue's endpoint and headers.
        read: What :func:`read_declared` found.
        identity: From :func:`fleet.core.queue.identity_arguments`; the
            publication and its refusals are this runner's.

    Returns:
        The queue's counts.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            the queue's refusal of a stale or malformed read among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer is not an object carrying a whole
            ``published``, ``removed`` and ``refused`` and a ``retired`` list
            of labels.
    """
    arguments: JSONObject = {
        "projects": [
            {"project": project["project"], "requiredTags": encode_tags(project["required_tags"])}
            for project in read["projects"]
        ],
        "runners": list(read["runners"]),
        "commit": read["commit"],
        "checkedAt": read["checked_at"],
        **identity,
    }
    body = load_json_str(call_mcp_tool(_test_hooks.http_post, credentials, PUBLISH_TOOL, arguments))
    if not isinstance(body, dict):
        raise JSONTypeError(f"{PUBLISH_TOOL} must answer an object, got {type(body).__name__}")
    return RegistryPublication(
        published=require_int(body, "published"),
        removed=require_int(body, "removed"),
        refused=require_int(body, "refused"),
        retired=tuple(require_str_list(body, "retired")),
    )


__all__ = [
    "BATCH_UNREADABLE_CODE",
    "EMPTY_CODE",
    "LIST_FAILED_CODE",
    "MACHINES_PATH",
    "NO_RUNNER_CODE",
    "PROJECTS_PATH",
    "PUBLISH_TOOL",
    "READ_FAILED_CODE",
    "DeclaredProject",
    "RegistryPublication",
    "RegistryRead",
    "declared_files",
    "declared_runners",
    "machine_files",
    "publish_registry",
    "read_declared",
    "split_batch",
]
