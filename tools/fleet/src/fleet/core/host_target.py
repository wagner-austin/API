"""Running a project's host target on its node, from the hub (MCPs board task 57791f39).

A project's host target (:mod:`fleet.contracts.host_target`) is one make
target run on one node's own account, outside any check: osrs-lab's
manual-desktop declares ``update`` on loki, which builds the recorder and
the login plugin and installs them into every OSRS desktop home there. The
hub runs it, because the hub holds the remote's credential and its mirror
and a node holds neither (:mod:`fleet.core.export`).

ONE RUN IS A LEASED RUN LIKE A CHECK'S, WITHOUT A LEDGER ROW. Its lease
(:func:`fleet.core.run_lease.take`, held for :func:`fleet.core.run_lease.host_target_window`)
holds the project on the node and records its retire owed, so no collect
pass retires its tree while make runs and a run whose caller died is
retired by a later pass. Its commit is staged through the checks' own
commit staging (:func:`fleet.core.commit_stage.stage_commit`), and the
retire that ends it keeps make's transcript at
:func:`fleet.core.names.retained_log_path`, as a check's is kept.

THE NODE'S SCRIPT EXITS 0 AND SAYS make's STATUS LAST. A script's output
comes back only when it exits 0 (:func:`fleet.core.remote.run_script`), so
make's status is a result the script prints, never its own exit: make runs
under ``timeout`` with its output in the run's transcript, the script prints
the transcript, then :data:`STATUS_WORD` and the status. ``timeout`` stops
make at the deadline with TERM, then KILL :data:`KILL_AFTER_SECONDS` later,
and its status, one of :data:`DEADLINE_STATUSES`, is the run's.
"""

from __future__ import annotations

import re
import shlex
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.host_target import HostTarget
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import ProjectConfig
from fleet.contracts.workspace import FleetWorkspace, require_node
from fleet.core import dialect, names, remote

#: The word the node's script prints before make's status, on its last line.
STATUS_WORD: Final = "host-target-status"

#: The seconds between ``timeout``'s TERM at the deadline and its KILL.
KILL_AFTER_SECONDS: Final = 30

#: ``timeout``'s statuses for a target it stopped: 124 when TERM ended it,
#: 137 (128 plus KILL's 9) when it outlived TERM by :data:`KILL_AFTER_SECONDS`.
DEADLINE_STATUSES: Final = frozenset({124, 137})

#: The stem of the script that runs the target, inside the run's enclosure,
#: which the retire removes with it.
RUN_STEM: Final = "host-target"

_STATUS_LINE: Final = re.compile(rf"{STATUS_WORD} ([0-9]+)")


class HostTargetRun(TypedDict):
    """What one run of a host target came to on its node.

    Attributes:
        lines: make's output, standard error merged, one line per entry.
        status: make's exit status, or ``timeout``'s when the deadline
            stopped it (:data:`DEADLINE_STATUSES`).
    """

    lines: tuple[str, ...]
    status: int


def require_host_target(project: str, plan: ProjectConfig) -> HostTarget:
    """The project's host target, or the refusal for a project declaring none.

    Args:
        project: The project key.
        plan: Its declaration.

    Returns:
        The host target.

    Raises:
        AppError: ``HOST_TARGET_UNDECLARED``, naming the registry entry to
            give one.
    """
    host_target = plan["host_target"]
    if host_target is None:
        raise AppError(
            FleetErrorCode.HOST_TARGET_UNDECLARED,
            f"project {project!r} declares no host_target in registry/projects/{project}.json, "
            "so no node, workers or deadline are written for one; declare its target, node, "
            "workers and deadline_seconds there",
        )
    return host_target


def require_host_node(workspace: FleetWorkspace, host_target: HostTarget) -> NodeConfig:
    """The node a host target runs on, refusing one the registry holds disabled.

    Args:
        workspace: The decoded workspace.
        host_target: The declared host target.

    Returns:
        The node's declaration.

    Raises:
        AppError: ``WORKSPACE_NODE_UNKNOWN`` from
            :func:`fleet.contracts.workspace.require_node` for a node the
            registry does not hold, and ``NODE_DISABLED`` for one it holds
            disabled, which is asked nothing.
    """
    name = host_target["node"]
    node = require_node(workspace, name)
    if not node["enabled"]:
        raise AppError(
            FleetErrorCode.NODE_DISABLED,
            f"{name} is declared disabled in this workspace, so its host target "
            f"{host_target['target']!r} was not run and nothing was asked of it. Set enabled in "
            f"registry/machines/{name}.json once it is expected to answer.",
        )
    return node


def run_script(*, recipe: str, log: str, host_target: HostTarget) -> str:
    """The node's script: make the target under its deadline, print its transcript and status.

    Args:
        recipe: The project's directory in the staged tree
            (:func:`fleet.core.names.recipe_directory`).
        log: The run's transcript (:func:`fleet.core.names.log_path`), which
            the retire keeps.
        host_target: The declared host target.

    Returns:
        The script's text. Only the ``cd`` and the prints can fail it; make's
        own status is printed last after :data:`STATUS_WORD`.
    """
    quoted_log = shlex.quote(log)
    make = " ".join(
        (
            f"{names.WORKERS_VARIABLE}={host_target['workers']}",
            "timeout",
            f"--kill-after={KILL_AFTER_SECONDS}",
            str(host_target["deadline_seconds"]),
            "make",
            host_target["target"],
        )
    )
    lines = (
        f"cd {shlex.quote(recipe)}",
        "status=0",
        f"{make} > {quoted_log} 2>&1 || status=$?",
        f"cat {quoted_log}",
        f"printf '%s %s\\n' {STATUS_WORD} \"$status\"",
    )
    return dialect.PROLOGUE + "".join(f"{line}\n" for line in lines)


def read_run(host: str, output: str) -> HostTargetRun:
    """Read make's output and status out of the node's answer.

    Args:
        host: SSH destination, for the refusal.
        output: The script's standard output.

    Returns:
        make's lines and its status.

    Raises:
        AppError: ``DISPATCH_FAILED`` when the last line is not
            :data:`STATUS_WORD` and a status, naming the line it ended on.
    """
    lines = output.splitlines()
    last = lines[-1] if lines else ""
    said = _STATUS_LINE.fullmatch(last)
    if said is None:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{host} answered the host target without its status: the last line was {last!r}, "
            f"not '{STATUS_WORD} <status>'",
        )
    return HostTargetRun(lines=tuple(lines[:-1]), status=int(said.group(1)))


def run_on_node(
    node: NodeConfig, *, run_id: str, target: str, path: str, host_target: HostTarget
) -> HostTargetRun:
    """Run the host target in its staged tree on the node.

    Args:
        node: The node's declaration.
        run_id: The run, whose enclosure holds the script.
        target: The staged tree (:func:`fleet.core.names.dispatch_directory`).
        path: The project's directory in its repository.
        host_target: The declared host target.

    Returns:
        make's lines and status.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, waited for until the deadline, ``timeout``'s KILL and
            the transport's own bound have all passed; ``DISPATCH_FAILED``
            from :func:`read_run`.
    """
    host = node["host"]
    output = remote.run_script(
        host,
        dialect.script_path(names.run_enclosure(node["stage_root"], run_id), RUN_STEM),
        run_script(
            recipe=names.recipe_directory(target, path),
            log=names.log_path(target),
            host_target=host_target,
        ),
        timeout_seconds=host_target["deadline_seconds"]
        + KILL_AFTER_SECONDS
        + remote.SSH_TIMEOUT_SECONDS,
    )
    return read_run(host, output)


def describe(run: HostTargetRun, host_target: HostTarget) -> str:
    """Say how a run ended, for the feed and the summary line.

    Args:
        run: The run.
        host_target: Its declared host target.

    Returns:
        make's status, or that the deadline stopped it.
    """
    status = run["status"]
    if status in DEADLINE_STATUSES:
        return (
            f"stopped at its deadline of {host_target['deadline_seconds']} s "
            f"(timeout exited {status})"
        )
    return f"make {host_target['target']} exited {status}"


__all__ = [
    "DEADLINE_STATUSES",
    "KILL_AFTER_SECONDS",
    "RUN_STEM",
    "STATUS_WORD",
    "HostTargetRun",
    "describe",
    "read_run",
    "require_host_node",
    "require_host_target",
    "run_on_node",
    "run_script",
]
