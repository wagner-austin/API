"""Every project the rolled fleet registry declares, as the queue's refusal reads it.

WHY (MCPs board task 1915b042). A node runner claims only the projects it
fits this tick, and those come from the registry, so a queued job naming a
key the registry does not declare was claimed by nobody and closed by
nobody. Measured 2026-10-14: a check submitted as ``MCPs/wiki-check``, whose
key is ``MCPs/packages/wiki-check``, sat queued about 80 minutes until it was
cancelled by hand. MCPs fleet-mcp holds no copy of the registry, so a runner
hands the queue what it read (``dispatch_refuse_undeclared``), and the queue
refuses each queued make-target job queued before the read whose key is
not among them.

READ AT THE REF, NOT FROM THE EXTRACTION. A serve runs the registry it was
extracted with, and a roll made between that extraction and the serve's
first read of the ref keeps it on the older code until its time is up
(:func:`fleet.cli.node_serve.handover_policy`). A refusal from that
registry would refuse a project the roll had just declared. So the keys are
listed from the commit :data:`fleet.core.rolled.ROLLED_REF` names at the read
itself, and the instant sent with them is taken before the ref is resolved:
a job queued before it named a key the rolled registry did not declare at a
moment after the job existed.
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Final, TypedDict

from platform_core.json_utils import JSONObject, JSONTypeError, load_json_str, require_int
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.core import _test_hooks, commit_tree, rolled
from fleet.core.workspace_files import PROJECT_SUFFIX, PROJECTS_DIRECTORY, project_key

#: The registry's per-project files, as paths in the API repository.
PROJECTS_PATH: Final = f"{rolled.REGISTRY_DIRECTORY}/{PROJECTS_DIRECTORY}"

#: Detail prefix when the registry's keys could not be listed.
LIST_FAILED_CODE: Final = "FLEET_REGISTRY_LIST_FAILED"

#: Detail prefix when the listing named no project.
EMPTY_CODE: Final = "FLEET_REGISTRY_EMPTY"


class RegistryRead(TypedDict):
    """The registry's keys as one read of the rolled ref found them.

    Attributes:
        commit: The full commit id :data:`fleet.core.rolled.ROLLED_REF` named.
        declared: Every project key that commit declares, sorted.
        checked_at: An ISO-8601 instant in UTC taken before the ref was
            resolved, floored to the second.
    """

    commit: str
    declared: tuple[str, ...]
    checked_at: str


def declared_keys(listing: str) -> tuple[str, ...]:
    """The project keys a ``git ls-tree`` listing of the projects directory names.

    Args:
        listing: One repository path per line, as ``git ls-tree -r
            --name-only`` prints them.

    Returns:
        The key of each per-project file, sorted, read with
        :func:`fleet.core.workspace_files.project_key` exactly as the
        workspace loader keys them; a path without the suffix is no project.
    """
    projects = pathlib.Path(PROJECTS_PATH)
    return tuple(
        sorted(
            project_key(projects, pathlib.Path(line))
            for line in listing.splitlines()
            if line.endswith(PROJECT_SUFFIX)
        )
    )


def read_declared(checkout: pathlib.Path) -> RegistryRead | str:
    """List the project keys the rolled registry declares, now.

    Args:
        checkout: A directory inside the API checkout whose object store
            holds the ref: the agent's records directory.

    Returns:
        The read, or a ``CODE: message`` refusal when the ref names no
        commit, the listing failed, or it declared no project; nothing is
        sent to the queue then.
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
        ),
        timeout_seconds=commit_tree.TREE_STEP_TIMEOUT_SECONDS,
    )
    if listed["returncode"] != 0:
        return commit_tree.step_refusal(LIST_FAILED_CODE, f"git ls-tree {commit}", listed)
    declared = declared_keys(listed["stdout"])
    if not declared:
        return f"{EMPTY_CODE}: {PROJECTS_PATH} at {commit} declares no project; nothing was refused"
    return RegistryRead(commit=commit, declared=declared, checked_at=checked_at)


def refuse_undeclared(
    credentials: McpCredentials, read: RegistryRead, *, identity: JSONObject
) -> int:
    """Hand the queue a registry read, so it refuses the jobs no runner will take.

    Args:
        credentials: The queue's endpoint and headers.
        read: What :func:`read_declared` found.
        identity: From :func:`fleet.core.queue.identity_arguments`; the
            refusals are this runner's.

    Returns:
        How many queued jobs the queue refused; 0 on most ticks.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            a refusal of the read's shape among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer is not an object carrying a whole
            ``refused`` count.
    """
    arguments: JSONObject = {
        "declared": list(read["declared"]),
        "commit": read["commit"],
        "checkedAt": read["checked_at"],
        **identity,
    }
    body = load_json_str(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_refuse_undeclared", arguments)
    )
    if not isinstance(body, dict):
        raise JSONTypeError(
            f"dispatch_refuse_undeclared must answer an object, got {type(body).__name__}"
        )
    return require_int(body, "refused")


__all__ = [
    "EMPTY_CODE",
    "LIST_FAILED_CODE",
    "PROJECTS_PATH",
    "RegistryRead",
    "declared_keys",
    "read_declared",
    "refuse_undeclared",
]
