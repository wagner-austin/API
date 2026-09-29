"""A node's ordinary runner steps aside while its elevated runner has work.

MCPs board task a98d7083. A node that declares ``elevated`` has two runners,
the ordinary one and the elevated one (:mod:`fleet.cli.node_agent`), and they
share the node's capacity: serendipity's limit is one fleet run for both.
The ordinary runner collects its finished build and claims the next job in
the same tick, so while its lane has a backlog the slot passes from one
ordinary job straight to the next, and the elevated runner's ticks only ever
read the node as full. Measured 2026-09-29: MCPs/execution-elevated job
58feb9d1 sat queued from 10:04Z to past 11:55Z while serendipity's ordinary
runner claimed five jobs back to back.

So the ordinary runner of such a node asks the queue first whether a job
requiring the ``elevated`` tag is waiting for this node, or for any node,
and if one is, it claims nothing this tick and says why. The slot then goes
to the elevated runner on its next tick. An elevated job is rare, so the
ordinary lane gives up a few minutes at a time, never its backlog.

WHICH JOBS COUNT. The queue lists by project, not by tag, so the question is
asked once per project the registry declares with the ``elevated`` tag, the
only projects whose jobs can require it (a job's tags must equal its
project's, :func:`fleet.cli.node_agent.tags_refusal`). A job pinned to
another node does not count: that node's elevated runner takes it.
"""

from __future__ import annotations

from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.node import NodeConfig
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import FleetWorkspace
from fleet.core import queue

_log = get_logger(__name__)


def elevated_projects(workspace: FleetWorkspace) -> tuple[str, ...]:
    """The registry's projects whose jobs require the ``elevated`` tag.

    Args:
        workspace: The decoded workspace.

    Returns:
        Their keys, sorted, so the queue is asked in a fixed order.
    """
    return tuple(
        sorted(
            name
            for name, project in workspace["projects"].items()
            if NodeTag.ELEVATED in project["required_tags"]
        )
    )


def waiting_elevated_job(
    credentials: McpCredentials, workspace: FleetWorkspace, *, alias: str
) -> DispatchJob | None:
    """The first queued elevated job this node's elevated runner could take.

    Args:
        credentials: The queue's endpoint and headers.
        workspace: The decoded workspace.
        alias: This node's workspace name.

    Returns:
        A queued job of an elevated project that names this node or no node,
        or None when there is none.

    Raises:
        AppError: Any transport or contract failure from the queue call.
    """
    for project in elevated_projects(workspace):
        for job in queue.queued_for(credentials, project=project):
            if job["requested_node"] in (None, alias):
                return job
    return None


def yields_to_elevated(
    credentials: McpCredentials,
    workspace: FleetWorkspace,
    *,
    alias: str,
    node: NodeConfig,
    elevated: bool,
) -> bool:
    """Whether this runner should claim nothing so the elevated runner can.

    Args:
        credentials: The queue's endpoint and headers.
        workspace: The decoded workspace.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner.

    Returns:
        True only for the ORDINARY runner of a node that declares
        ``elevated``, and only while an elevated job waits for this node;
        the waiting job is logged. False for the elevated runner itself, and
        for every node without one, which asks the queue nothing.

    Raises:
        AppError: Any transport or contract failure from the queue call.
    """
    if elevated or not node["elevated"]:
        return False
    waiting = waiting_elevated_job(credentials, workspace, alias=alias)
    if waiting is None:
        return False
    _log.info(
        "%s claims nothing so its elevated runner takes the next slot: %s",
        alias,
        encode_job_line(waiting),
    )
    return True


__all__ = ["elevated_projects", "waiting_elevated_job", "yields_to_elevated"]
