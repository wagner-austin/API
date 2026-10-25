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

from fleet.cli.node_collect import CLAIM_LEASE_SECONDS
from fleet.cli.node_ready import Ready
from fleet.contracts.dispatch import DispatchJob, DispatchLane, encode_job_line
from fleet.contracts.runner_tick import RunnerTick
from fleet.core import queue, tick_report

_log = get_logger(__name__)


def ask_queue(
    credentials: McpCredentials,
    identity: JSONObject,
    tick: RunnerTick,
    ready: Ready,
    *,
    alias: str,
) -> DispatchJob | None:
    """Claim one job for a ready node, and record the tick with what came of it.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        tick: The tick the gate composed for this ready node.
        ready: Its state, claim tags and fitting projects.
        alias: This node's workspace name.

    Returns:
        The claimed job, or None when nothing in the lane matched. The tick
        is recorded either way, before anything is staged, so
        ``fleet_status`` reads the claim while its build is still being
        prepared.

    Raises:
        AppError: From the queue call.
    """
    job = queue.claim_next(
        credentials,
        lane=DispatchLane.NODE,
        tags=tuple(sorted(ready["tags"])),
        node=alias,
        projects=ready["fits"],
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    verdict = (
        f"asked for {len(ready['fits'])} fitting project(s); nothing in the node lane matched"
        if job is None
        else f"claimed {encode_job_line(job)}"
    )
    _log.info("%s %s", alias, verdict)
    tick_report.record_tick(
        credentials, decided(tick, claiming=True, verdict=verdict), identity=identity
    )
    return job


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
        tags=tick["tags"],
        fits=tick["fits"],
        load=tick["load"],
        claiming=claiming,
        verdict=verdict,
    )


__all__ = ["ask_queue", "decided"]
