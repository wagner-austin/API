"""A node's ordinary runner steps aside while its elevated runner has work.

MCPs board task a98d7083. A node that declares ``elevated`` has two runners,
the ordinary one and the elevated one (:mod:`fleet.cli.node_agent`), and they
share the node's live capacity. The ordinary runner collects its finished
build and claims again in the same tick, so while its lane has a backlog the
room a run frees passes from one ordinary job straight to the next, and the
elevated runner's ticks only ever read the node as full. Measured 2026-09-29:
MCPs/execution-elevated job 58feb9d1 sat queued from 10:04Z to past 11:55Z
while serendipity's ordinary runner claimed five jobs back to back.

So the ordinary runner of such a node asks the queue first whether a job
requiring the ``elevated`` tag is waiting for this node, or for any node,
and if one is, it claims nothing this tick and says why. The slot then goes
to the elevated runner on its next tick. An elevated job is rare, so the
ordinary lane gives up a few minutes at a time, never its backlog.

WHICH JOBS COUNT. The queue lists by project, not by tag, so the question is
asked once per project the registry declares with the ``elevated`` tag, the
only projects whose jobs need it (what a job needs is its project's
declaration, :func:`fleet.cli.node_agent.tags_refusal`). A job pinned to
another node does not count: that node's elevated runner takes it.

NOR DOES A JOB THE ELEVATED RUNNER COULD NOT TAKE NOW (MCPs board task
939ec5c7). A project a lease on this node already holds is left out of the
elevated runner's fits (:func:`fleet.core.run_lease.held_on_node`), so a
job of it waits whoever yields. On 2026-10-03 serendipity's ordinary runner
yielded at 09:03Z and 09:06Z to a queued MCPs/scripts/ps-harness job while
its elevated runner was running another ps-harness job and matched nothing,
so neither lane claimed with sixteen jobs queued. The ordinary runner asks
the same function the claim gate asks, and yields only to a project it
does not hold.

A JOB SUBMITTED WITHOUT THE ``elevated`` TAG COUNTS TOO (MCPs board task
939ec5c7). MCPs/scripts/ps-harness job 7c16305c was queued on 2026-10-03
17:16Z requiring only ``windows``, and the queue's exclusive rule hands the
elevated lane only jobs that carry the tag, so for a while no runner took
it: serendipity's ordinary runner yielded to it on every tick while
libs/covenant_ml, which only serendipity can run, waited behind it, and
board task eaf80425 then stopped counting such a job. The elevated runner
now claims it as well (:func:`fleet.cli.node_claim.claim_untagged`), so it
is a job that runner takes, and the ordinary runner yields to it like any
other.
"""

from __future__ import annotations

import pathlib

from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.node import NodeConfig
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import FleetWorkspace
from fleet.core import queue, run_lease

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
    credentials: McpCredentials,
    workspace: FleetWorkspace,
    *,
    alias: str,
    leases: pathlib.Path,
) -> DispatchJob | None:
    """The first queued elevated job this node's elevated runner could take.

    Args:
        credentials: The queue's endpoint and headers.
        workspace: The decoded workspace.
        alias: This node's workspace name.
        leases: The workspace's lease file, for the elevated projects a run
            on this node holds now; the queue is not asked about those.

    Returns:
        A queued job of an elevated project that no lease on this node holds
        and that names this node or no node, whatever tags it was submitted
        with, or None when there is none.

    Raises:
        AppError: Any transport or contract failure from the queue call.
    """
    candidates = elevated_projects(workspace)
    held = run_lease.held_on_node(
        leases,
        node=alias,
        names=candidates,
        projects=workspace["projects"],
        node_local=workspace["node_local_resources"],
    )
    if held:
        _log.info(
            "%s does not yield for what a lease on it holds now, which its elevated runner "
            "could not take either: %s",
            alias,
            ", ".join(held),
        )
    for project in (name for name in candidates if name not in held):
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
    leases: pathlib.Path,
) -> bool:
    """Whether this runner should claim nothing so the elevated runner can.

    Args:
        credentials: The queue's endpoint and headers.
        workspace: The decoded workspace.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner.
        leases: The workspace's lease file (:func:`waiting_elevated_job`).

    Returns:
        True only for the ORDINARY runner of a node that declares
        ``elevated``, and only while an elevated job waits for this node
        whose project no lease on it holds; the waiting job is logged. False
        for the elevated runner itself, and for every node without one,
        which asks the queue nothing.

    Raises:
        AppError: Any transport or contract failure from the queue call.
    """
    if elevated or not node["elevated"]:
        return False
    waiting = waiting_elevated_job(credentials, workspace, alias=alias, leases=leases)
    if waiting is None:
        return False
    _log.info(
        "%s claims nothing so its elevated runner takes the next slot: %s",
        alias,
        encode_job_line(waiting),
    )
    return True


__all__ = ["elevated_projects", "waiting_elevated_job", "yields_to_elevated"]
