"""Give back each held job whose run is gone, before the claim (MCPs board task 05dd6588).

WHY. A fleet-agent tick runs the job it claims to its close, and the MCPs
hub runner binds every tick's unit to its own, so a restart of the runner
ends the run, by design (:mod:`fleet.core.tick_run`). Nothing gave the job
back. On diphtheria on 2026-10-21 the runner's ship restarted it at
01:09:58Z, 22 s into job e2f2a964's netboot arm: no arm file was written,
the row read running under a lease to 02:09:36Z, nothing reached the task's
thread, and every later tick of fleet-netboot-diphtheria logged an empty
lane. The hub runner's journal recorded 11 such ships between 2026-10-19
19:13 and 2026-10-20 18:10 local, about one every two hours, each ending
whatever every lane had in flight.

WHAT. The runner that knows its run is gone is the claimant: each tick
first lists the jobs the queue holds under its own label and asks this
machine's user manager about the unit each one's run id names. A unit that
reads stopped with no control group is a run that ended without closing its
job, so the job is reported lost, which requeues it the first time and fails
it the second (:mod:`fleet.core.queue_lost`); the claim that follows in the
same tick takes it back. Nothing else is reported: a run still live is
another tick of this label working, and a job whose run id names no unit
(claimed and never started, or started before runs were named by unit) or a
unit on another machine cannot be proven gone from here, so each is left to
its lease, which the claim's short pre-start lease keeps to minutes for the
first (:data:`fleet.cli.hub_jobs.PRESTART_LEASE_SECONDS`).
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.core import queue_listing, queue_lost, tick_run

_log = get_logger(__name__)


def lost_detail(job: DispatchJob, own: tick_run.TickRun) -> str | None:
    """Say how a held job's run is known to be gone, if it is.

    Args:
        job: A job the queue holds under this runner's label.
        own: This tick's run, which names the machine it can ask about.

    Returns:
        The line naming the stopped run and its readings, or None when the
        run is live or cannot be proven gone from here.

    Raises:
        AppError: When this machine's manager cannot be asked about the unit.
    """
    held = tick_run.decode_run_id(job["run_id"])
    if held is None:
        _log.info(
            "job %s is %s under this label with run id %r, which names no unit, so it is "
            "left to its lease",
            job["job_id"],
            job["status"].value,
            job["run_id"],
        )
        return None
    if held["machine"] != own["machine"]:
        _log.info(
            "job %s runs as %s, on %s rather than this machine, %s, so it is left to its lease",
            job["job_id"],
            held["unit"],
            held["machine"],
            own["machine"],
        )
        return None
    state = tick_run.read_run_state(held["unit"])
    if state["live"]:
        _log.info(
            "job %s runs as %s, which reads %s: live", job["job_id"], held["unit"], state["shown"]
        )
        return None
    return (
        f"run {job['run_id']} is gone: systemctl --user show read {state['shown']}, so the "
        f"make {job['command'].value} it ran ended without closing the job"
    )


def orphan_pass(
    credentials: McpCredentials, identity: JSONObject, *, agent: str, own: tick_run.TickRun
) -> tuple[DispatchJob, ...]:
    """Report lost every job held under this label whose run is gone.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        agent: This runner's label.
        own: This tick's run.

    Returns:
        Each job reported lost, as the queue left it: ``queued``, or
        ``failed`` when it had been requeued before.

    Raises:
        AppError: From the queue calls, or when this machine's manager
            cannot be asked about a unit.
    """
    reported: list[DispatchJob] = []
    for job in queue_listing.held_by(credentials, agent=agent):
        detail = lost_detail(job, own)
        if detail is None:
            continue
        answered = queue_lost.report_lost(
            credentials, job_id=job["job_id"], detail=detail, identity=identity
        )
        _log.info("reported lost: %s; the queue left it %s", detail, encode_job_line(answered))
        reported.append(answered)
    return tuple(reported)


__all__ = ["lost_detail", "orphan_pass"]
