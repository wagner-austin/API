"""Reconcile research in the shared dispatch queue using existing resource leases.

One bounded controller operation per scheduled tick. Native scheduler waits
retain the queue receipt but consume no local fleet resource lease. Four live
requests may wait concurrently; least-recently-renewed work is polled first.
"""

from __future__ import annotations

import pathlib
import shlex

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, require_str
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import (
    ClosingStatus,
    DispatchCommand,
    DispatchJob,
    DispatchLane,
    DispatchStatus,
)
from fleet.contracts.lease import Lease
from fleet.contracts.research_worker import (
    ResearchState,
    ResearchWorker,
    decode_observation,
    encode_observation,
)
from fleet.core import _test_hooks, leases, queue, queue_listing, remote

MAX_ACTIVE = 4
QUEUE_LEASE_SECONDS = 7200
CONTROL_LEASE_SECONDS = 1260
CONTROL_TIMEOUT_SECONDS = 1050


def _script(worker: ResearchWorker, job: DispatchJob) -> str:
    """Render a fixed native CLI command from a validated queue reference.

    Args:
        worker: Operator-provisioned placement.
        job: Queue-owned immutable identity.

    Returns:
        Script sent and executed by path through the existing fleet transport.

    Raises:
        ValueError: RESEARCH_QUEUE_ROUTING for mismatched or incomplete routing.
    """
    request = job["research_request"]
    if (
        job["command"] is not DispatchCommand.RESEARCH
        or request is None
        or job["requested_node"] != worker["node"]
        or job["task_id"] is None
    ):
        raise ValueError("RESEARCH_QUEUE_ROUTING: command, reference, node and task are required")
    if job["project"] != "research/" + request["backend"].value:
        raise ValueError("RESEARCH_QUEUE_ROUTING: project differs from its native backend")
    argv = (
        "timeout",
        "--signal=TERM",
        "--kill-after=10",
        "1020",
        worker["python"],
        "-m",
        "fleet.cli.research_native",
        "--root",
        worker["root"],
        "--backend",
        request["backend"].value,
        "--digest",
        request["spec_sha256"],
        "--attempt",
        request["attempt"],
        "--tenant",
        worker["tenant"],
        "--task",
        job["task_id"],
        "--env-file",
        worker["env_file"],
    )
    return "#!/bin/sh\nset -eu\numask 077\nexec " + shlex.join(argv) + "\n"


def reconcile(
    credentials: McpCredentials,
    identity: JSONObject,
    worker: ResearchWorker,
    lease_path: pathlib.Path,
    job: DispatchJob,
) -> DispatchJob:
    """Keep native receipts across failed transport and close only verified outcomes.

    Args:
        credentials: Authenticated tenant queue.
        identity: Stable fleet runner identity.
        worker: Authorized private controller placement.
        lease_path: The existing fleet's shared resource lease file.
        job: Claimed or running research request.

    Returns:
        The queue's updated row.

    Raises:
        ValueError: Invalid native evidence or routing.
        AppError: Queue transport or resource admission failure.
    """
    body = _script(worker, job)
    run_id = job["run_id"] if job["run_id"] else job["job_id"]
    now = _test_hooks.now()
    lease = Lease(
        node=worker["node"],
        project="research/controller",
        run_id=run_id,
        agent=require_str(identity, "agent"),
        session_id=require_str(identity, "sessionId"),
        acquired_unix=now,
        expires_unix=now + CONTROL_LEASE_SECONDS,
        resources=("research-controller@" + worker["node"],),
    )
    leases.acquire(lease_path, lease, now_unix=now)
    try:
        if job["status"] is DispatchStatus.CLAIMED:
            job = queue.report_start(
                credentials,
                job_id=job["job_id"],
                node=worker["node"],
                run_id=run_id,
                lease_seconds=QUEUE_LEASE_SECONDS,
                identity=identity,
            )
        outcome = remote.attempt_script(
            worker["host"], worker["script"], body, timeout_seconds=CONTROL_TIMEOUT_SECONDS
        )
        failure = outcome["failure"]
        if failure is not None:
            _test_hooks.append_text(
                lease_path.parent / "research-errors.jsonl",
                dump_json_str(
                    {
                        "job": job["job_id"],
                        "at": _test_hooks.now(),
                        "message": failure["message"],
                    }
                ),
            )
            return queue.report_progress(
                credentials,
                job_id=job["job_id"],
                identity=identity,
                lease_seconds=QUEUE_LEASE_SECONDS,
                note=(
                    f"{failure['code'].value}: controller tick failed; "
                    "native receipt retained for reconciliation"
                ),
            )
        observation = decode_observation(load_json_str(outcome["output"]))
        detail = dump_json_str(encode_observation(observation))
        if observation["state"] is ResearchState.RUNNING:
            return queue.report_progress(
                credentials,
                job_id=job["job_id"],
                note=detail,
                lease_seconds=QUEUE_LEASE_SECONDS,
                identity=identity,
            )
        passed = observation["state"] is ResearchState.COMPLETE
        return queue.report_close(
            credentials,
            job_id=job["job_id"],
            identity=identity,
            status=ClosingStatus.PASSED if passed else ClosingStatus.FAILED,
            exit_code=0 if passed else 1,
            detail="Native controller outcome (adapter exit status): " + detail,
        )
    finally:
        leases.release_if_held(lease_path, run_id=run_id, now_unix=_test_hooks.now())


def tick(
    credentials: McpCredentials,
    identity: JSONObject,
    worker: ResearchWorker,
    lease_path: pathlib.Path,
) -> DispatchJob | None:
    """Admit bounded work and poll the oldest renewed receipt, without another queue.

    Args:
        credentials: Shared queue credentials.
        identity: Stable runner identity.
        worker: Private placement for this tenant.
        lease_path: Existing shared fleet leases.

    Returns:
        Updated request, or None when idle or the controller resource is occupied.

    Raises:
        ValueError: RESEARCH_QUEUE_TENANT if authenticated and configured tenants differ.
        AppError: Existing queue or resource error, with native identity preserved.
    """
    if credentials["tenant_id"] != worker["tenant"]:
        raise ValueError("RESEARCH_QUEUE_TENANT: private controller belongs to another tenant")
    now = _test_hooks.now()
    if (
        leases.find_holder(
            lease_path, node=worker["node"], project="research/controller", now_unix=now
        )
        is not None
    ):
        return None
    held = queue_listing.held_by(credentials, agent=require_str(identity, "agent"))
    if len(held) < MAX_ACTIVE:
        claimed = queue.claim_next(
            credentials,
            lane=DispatchLane.RESEARCH,
            tags=(),
            node=worker["node"],
            projects=("research/hpc3", "research/wrds"),
            lease_seconds=QUEUE_LEASE_SECONDS,
            identity=identity,
        )
        if claimed is not None:
            retained = tuple(job for job in held if job["job_id"] != claimed["job_id"])
            held = (*retained, claimed)
    if not held:
        return None
    selected = min(held, key=lambda job: job["lease_expires_unix"] or 0)
    return reconcile(credentials, identity, worker, lease_path, selected)


__all__ = [
    "CONTROL_LEASE_SECONDS",
    "CONTROL_TIMEOUT_SECONDS",
    "MAX_ACTIVE",
    "QUEUE_LEASE_SECONDS",
    "reconcile",
    "tick",
]
