"""The jobs ``fleet-agent`` runs where it claims them, one function per kind.

Split out of :mod:`fleet.cli.agent` when the netboot lane (MCPs board task
83102d9d) gave it a third kind of job and took it past the file-size
ceiling: the agent parses its command line and claims, and each function
here runs one claimed job to its close.

EVERY KIND HAS THE SAME SHAPE. Each is local to the machine that claimed
it and closes in the tick that claimed it, so nothing crosses ticks:

* a REBUILD (``build-bases``, the control duty's) runs a bake in the MCPs
  checkout (:mod:`fleet.core.rebuild`);
* a SESSION VERB (restart, revive, the two kills, the compact and the exit,
  the sessions duty's) runs the active sealed session-verbs release's
  session-audit (:mod:`fleet.core.restart`);
* a NETBOOT ARM (the netboot duty's) runs the installed sealed netboot
  release's ``fleet-netboot arm`` (:mod:`fleet.core.netboot_arm`).

Each refuses before it starts what it cannot run, by a named code, with
nothing run; then reports the job started on this machine, runs it, and
closes it with what it said. A refusal is reported to the queue and does
not propagate: that is transport, not recovery, for the reason the agent's
module docstring gives.
"""

from __future__ import annotations

import pathlib
import sys

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import job_close
from fleet.contracts.dispatch import ClosingStatus, DispatchJob
from fleet.core import netboot_arm, queue, rebuild, restart, session_release

_log = get_logger(__name__)

#: How long a claim survives without a report.
#:
#: An hour: a hub verb closes in the tick that claimed it, so the lease only
#: has to outlive the longest of them (an arm's first ISO fetch, bounded at
#: 1800 s, or a bake, bounded at 1800 s by its own deadline) and the tick's
#: own ExecutionTimeLimit ends anything past that.
CLAIM_LEASE_SECONDS = 3600


def _start(
    credentials: McpCredentials, job: DispatchJob, identity: JSONObject, *, node: str, run_id: str
) -> None:
    """Report a claimed job started on this machine.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed job.
        identity: This runner's identity arguments.
        node: This machine's alias.
        run_id: What the job's run is called.
    """
    queue.report_start(
        credentials,
        job_id=job["job_id"],
        node=node,
        run_id=run_id,
        lease_seconds=CLAIM_LEASE_SECONDS,
        identity=identity,
    )
    _log.info("started %s on %s as %s", job["job_id"], node, run_id)


def rebuild_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    mcps_root: pathlib.Path,
    node: str,
) -> DispatchJob:
    """Execute a claimed ``build-bases`` job on the hub, start to close.

    Synchronous, unlike a suite dispatch: the bake is a local make, minutes
    long, and closing it in the same tick leaves no cross-tick state to
    collect (module docstring of :mod:`fleet.core.rebuild`). A runner that
    dies mid-bake leaves the job ``running`` until its lease lapses, after
    which a later tick reclaims it and re-runs the idempotent make.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed ``build-bases`` job.
        identity: This runner's identity arguments.
        mcps_root: The MCPs checkout the bake runs in.
        node: This machine's alias, which the job is reported started on:
            the job names none, since holding the control duty is the
            placement (MCPs board task c39ee056).

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves.
    """
    refusal = rebuild.refusal_for(mcps_root, job["submitted_by"])
    if refusal is not None:
        return job_close.refuse(credentials, job, identity, detail=refusal)
    _start(credentials, job, identity, node=node, run_id=f"bases-{job['job_id']}")
    result = rebuild.run_build_bases(mcps_root, submitted_by=job["submitted_by"])
    return job_close.close(
        credentials,
        job,
        identity,
        status=ClosingStatus.PASSED if result["returncode"] == 0 else ClosingStatus.FAILED,
        exit_code=result["returncode"],
        detail=rebuild.describe_result(result),
    )


def session_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    node: str,
) -> DispatchJob:
    """Execute a claimed session job (restart, revive, kill, compact or exit)
    on this machine, start to close.

    The same shape as :func:`rebuild_job` and for the same reasons: the
    work is local, well under a minute, and closing it in one tick leaves
    nothing to collect. The keystroke sequence and every rail around it
    belong to ``session_audit`` (``rollover``, ``revive``, ``kill`` and
    ``compact``); this runner composes one invocation per verb through
    :func:`fleet.core.restart.session_invocation` and reports what it said
    (module docstring of :mod:`fleet.core.restart`).

    Named for the job kind rather than one of its five verbs. It was
    ``restart_job``, which named one verb and claimed a state change this
    runner does not make: the change, and the check of each step before
    the next (the old process gone before a resume is typed, a new record
    on the installed build before a restart is reported), are the sealed
    ``session_audit`` release's, whose exit status this closes the job
    with (``state-change-verified``, board task cc7222ca).

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed session job.
        identity: This runner's identity arguments.
        node: This machine's alias, which the job is reported started on.

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves.
    """
    refusal = restart.refusal_for(job["session_target"])
    if refusal is not None:
        return job_close.refuse(credentials, job, identity, detail=refusal)
    # Narrowed by the refusal above: a None target was refused there.
    target = job["session_target"] if job["session_target"] is not None else ""
    # The verb runs the active sealed session-verbs release, verified for
    # this job, and nothing else (MCPs board task c7c2527d); no release, or
    # one that fails its seal, refuses the job with nothing run. Its store
    # is this machine's platform's (board task 315111e4).
    release = session_release.active_session_release(sys.platform)
    if isinstance(release, str):
        return job_close.refuse(credentials, job, identity, detail=release)
    invocation = restart.session_invocation(
        release, job["command"], target, job["submitted_by"], node
    )
    # A revive types the submitter's label into its brief and a kill passes
    # it as an argument (MCPs migs 525 and 526), so the label is judged
    # before it can become an argv element, the same way the target is.
    if invocation["types_requester"]:
        requester_refusal = restart.requester_refusal(job["submitted_by"])
        if requester_refusal is not None:
            return job_close.refuse(credentials, job, identity, detail=requester_refusal)
    _start(credentials, job, identity, node=node, run_id=f"{invocation['verb']}-{job['job_id']}")
    result = restart.run_session_job(invocation)
    return job_close.close(
        credentials,
        job,
        identity,
        status=ClosingStatus.PASSED if result["returncode"] == 0 else ClosingStatus.FAILED,
        exit_code=result["returncode"],
        detail=restart.describe_result(result, invocation),
    )


def netboot_job(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    place: netboot_arm.NetbootPlace,
    node: str,
) -> DispatchJob:
    """Execute a claimed ``netboot-arm`` job on this machine, start to close.

    The job names the machine to arm as its requested node (MCPs mig 807),
    which is why the lane claims with no node filter: the runner is placed
    by holding the netboot duty, and the requested node is the target. The
    installed netboot release's seal is checked and the key read before
    the job is reported started, so a release that fails its seal or an
    env file with no key refuses the job with nothing minted
    (:mod:`fleet.core.netboot_arm` carries the rest).

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed ``netboot-arm`` job.
        identity: This runner's identity arguments.
        place: Where arms run from and write to.
        node: This machine's alias, which the job is reported started on.

    Returns:
        The job, whatever became of it.

    Raises:
        AppError: Only from the queue calls themselves.
    """
    machine = job["requested_node"]
    if machine is None:
        return job_close.refuse(
            credentials,
            job,
            identity,
            detail=(
                f"{netboot_arm.NO_MACHINE_CODE}: the job names no machine to arm; "
                "dispatch_netboot_arm queues one naming a roster machine; nothing ran"
            ),
        )
    ready = netboot_arm.ready(place)
    if isinstance(ready, str):
        return job_close.refuse(credentials, job, identity, detail=ready)
    _start(credentials, job, identity, node=node, run_id=f"netboot-arm-{job['job_id']}")
    result = netboot_arm.run_cli(
        netboot_arm.arm_argv(place, ready["release"], machine),
        ready["key"],
        timeout_seconds=netboot_arm.ARM_TIMEOUT_SECONDS,
    )
    outcome = netboot_arm.arm_outcome(result, ready["release"], machine)
    return job_close.close(
        credentials,
        job,
        identity,
        status=outcome["status"],
        exit_code=outcome["exit_code"],
        detail=outcome["detail"],
    )


__all__ = [
    "CLAIM_LEASE_SECONDS",
    "netboot_job",
    "rebuild_job",
    "session_job",
]
