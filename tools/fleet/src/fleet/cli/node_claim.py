"""Asking the queue for a ready node's job, and recording what the tick decided.

Lifted out of :mod:`fleet.cli.node_agent`, which runs the tick and stages
what it claims, when A4 of MCPs board task 939ec5c7 made every tick record
itself (:mod:`fleet.core.tick_report`): the claim is where the tick's last
verdict is decided, so the claim and its record live together and the
staging module stays one concern.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli.node_collect import CLAIM_LEASE_SECONDS
from fleet.cli.node_ready import Ready
from fleet.contracts.dispatch import ClosingStatus, DispatchJob, DispatchLane, encode_job_line
from fleet.contracts.node import NodeConfig
from fleet.contracts.runner_tick import RunnerTick
from fleet.contracts.tags import NodeTag
from fleet.core import elevated_yield, queue, tick_report

_log = get_logger(__name__)


def refuse(
    credentials: McpCredentials, job: DispatchJob, identity: JSONObject, *, detail: str
) -> None:
    """Close a claimed job as refused, with its named reason, and log it.

    Moved here from :mod:`fleet.cli.node_agent` when its tick's watch
    (:mod:`fleet.cli.node_watch`) took that module to the file ceiling: a
    refusal is how a claim ends when it cannot be launched.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed job.
        identity: This runner's identity arguments.
        detail: The ``CODE: message`` refusal for the queue.

    Raises:
        AppError: Only from the queue call itself.
    """
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=ClosingStatus.REFUSED,
        exit_code=None,
        detail=detail,
        identity=identity,
    )
    _log.info("refused %s: %s", job["job_id"], detail)


def ask_queue(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    tick: RunnerTick,
    ready: Ready,
    *,
    alias: str,
    node: NodeConfig,
    elevated: bool,
) -> DispatchJob | None:
    """Claim one job for a ready node, and record the tick with what came of it.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        tick: The tick the gate composed for this ready node.
        ready: Its state, claim tags and fitting projects.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner.

    Returns:
        The claimed job, or None when the ordinary runner yields to a waiting
        elevated job or nothing in the lane matched; the elevated runner
        whose lane matched nothing also asks :func:`claim_untagged`. The tick is recorded
        either way, before anything is staged, so ``fleet_status`` reads the
        claim while its build is still being prepared.

    Raises:
        AppError: From the queue calls.
    """
    if elevated_yield.yields_to_elevated(
        credentials,
        loaded.workspace,
        alias=alias,
        node=node,
        elevated=elevated,
        leases=loaded.leases,
    ):
        verdict = "yields to its elevated runner; claiming nothing"
        tick_report.record_tick(
            credentials, decided(tick, claiming=False, verdict=verdict), identity=identity
        )
        return None
    job = queue.claim_next(
        credentials,
        lane=DispatchLane.NODE,
        tags=tuple(sorted(ready["tags"])),
        node=alias,
        projects=ready["fits"],
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    if job is None and elevated:
        job = claim_untagged(loaded, credentials, identity, ready, alias=alias)
    verdict = (
        f"asked for {len(ready['fits'])} fitting project(s); nothing in the node lane matched"
        if job is None
        else f"claimed {encode_job_line(job)}{untagged_note(job, elevated=elevated)}"
    )
    _log.info("%s %s", alias, verdict)
    tick_report.record_tick(
        credentials, decided(tick, claiming=True, verdict=verdict), identity=identity
    )
    return job


def claim_untagged(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    ready: Ready,
    *,
    alias: str,
) -> DispatchJob | None:
    """Claim, for the elevated runner, a job only it can run that lacks the ``elevated`` tag.

    The queue's exclusive rule (MCPs ``claimNextDispatchJob``) hands a
    runner that carries ``elevated`` only jobs that require it, and every
    ordinary runner's fits leave out a project that declares it
    (:func:`fleet.core.capacity.assess`), so a job of such a project
    submitted with fewer tags reached no runner on any node. MCPs board
    task 939ec5c7: MCPs/scripts/ps-harness job 7c16305c, submitted with
    ``[windows]``, waited from 2026-10-03 17:16Z while serendipity had room
    in both lanes. So the elevated runner asks the lane once more as the
    ordinary runner would, its tags less ``elevated``, for only the
    projects that declare it among those it fits: the jobs it alone can run,
    whatever tags they were submitted with. The project's declaration is
    what the build needs (:func:`fleet.cli.node_agent.tags_refusal`).

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        ready: The elevated runner's state, claim tags and fitting projects.
        alias: This node's workspace name.

    Returns:
        The claimed job, or None when it fits no elevated project or no
        such job waits; the queue is not asked when it fits none.

    Raises:
        AppError: From the queue call.
    """
    declaring = elevated_yield.elevated_projects(loaded.workspace)
    own = tuple(name for name in ready["fits"] if name in declaring)
    if not own:
        return None
    return queue.claim_next(
        credentials,
        lane=DispatchLane.NODE,
        tags=tuple(sorted(ready["tags"] - {NodeTag.ELEVATED})),
        node=alias,
        projects=own,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )


def untagged_note(job: DispatchJob, *, elevated: bool) -> str:
    """What the elevated runner's verdict adds for a job submitted without the tag.

    Args:
        job: The claimed job.
        elevated: Whether this is the node's elevated runner.

    Returns:
        A clause naming the missing tag, so the tick log says why a job
        the lane's own rule would never hand this runner was taken; empty
        for every other claim.
    """
    if not elevated or NodeTag.ELEVATED in job["required_tags"]:
        return ""
    return ", submitted without the elevated tag its project declares"


def decided(tick: RunnerTick, *, claiming: bool, verdict: str) -> RunnerTick:
    """The gate's tick with the claim's outcome in place of its verdict.

    Args:
        tick: The tick the gate composed.
        claiming: Whether the runner asked the queue.
        verdict: What came of it.

    Returns:
        The tick to record.
    """
    return RunnerTick(
        node=tick["node"],
        elevated=tick["elevated"],
        tags=tick["tags"],
        fits=tick["fits"],
        load=tick["load"],
        claiming=claiming,
        verdict=verdict,
    )


__all__ = ["ask_queue", "claim_untagged", "decided", "refuse", "untagged_note"]
