"""How a runner closes a job it claimed and ran where it claimed it.

ONE CLOSE PER OUTCOME (MCPs board task af8b5662). The node lanes and the
hub jobs each carried their own refusal, the same queue report and the same
log line in :mod:`fleet.cli.node_claim` and :mod:`fleet.cli.hub_jobs`, and
the hub jobs repeated their finished close once per kind. Every close is
now made here: :func:`refuse` for a job that ran nothing, :func:`close` for
one that ran and ended with a status of its own. That holds for a node
lane's collect too, which refuses a claim an earlier tick launched nothing
for (:func:`fleet.cli.node_collect.reconcile_claim`) and closes an ended
run with its verdict (:func:`fleet.cli.node_settle.settle`); check review
678e4470 found both still building the queue's close by hand, and
``tests/test_one_job_close.py`` now fails on a ``report_close`` call in any
module but this one.

A RUN ENDED FOR MEMORY IS GIVEN BACK, NOT CLOSED (MCPs board task d111690a).
A check yields memory to the production stack on its host (it runs at
``OOMScoreAdjust=1000``, :mod:`fleet.core.linux_unit_end`), so its being
killed says the host was short, not that its suite failed. :func:`give_back`
reports the run lost, and the queue's own bound decides what follows
(MCPs ``reportDispatchRunLost``): the first loss requeues the job, and a job
requeued before is closed ``failed``, so a job whose own working set is more
than any node can give is retried once and never loops.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import ClosingStatus, DispatchJob, encode_job_line
from fleet.core import queue, queue_lost

_log = get_logger(__name__)


def refuse(
    credentials: McpCredentials, job: DispatchJob, identity: JSONObject, *, detail: str
) -> DispatchJob:
    """Close a claimed job as refused, with its named reason, and log it.

    Args:
        credentials: The queue's endpoint and headers.
        job: The claimed job, which ran nothing.
        identity: This runner's identity arguments.
        detail: The ``CODE: message`` refusal for the queue.

    Returns:
        The job, which a hub job hands back whatever became of it.

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
    return job


def close(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    status: ClosingStatus,
    exit_code: int | None,
    detail: str,
) -> DispatchJob:
    """Close a job this runner started, with how it ended, and log it.

    Args:
        credentials: The queue's endpoint and headers.
        job: The job, reported started on this machine.
        identity: This runner's identity arguments.
        status: How it ended; the queue pairs ``passed`` with exit 0,
            ``failed`` with a non-zero exit and ``refused`` with none.
        exit_code: Its exit status, or None for ``refused``.
        detail: The line the submitter reads.

    Returns:
        The job, which a hub job hands back whatever became of it.

    Raises:
        AppError: Only from the queue call itself.
    """
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=status,
        exit_code=exit_code,
        detail=detail,
        identity=identity,
    )
    _log.info("%s: %s", encode_job_line(job), detail)
    return job


def give_back(
    credentials: McpCredentials, job: DispatchJob, identity: JSONObject, *, detail: str
) -> DispatchJob:
    """Give the queue back a job whose run the kernel or systemd-oomd ended for memory, and log it.

    Args:
        credentials: The queue's endpoint and headers.
        job: The job, reported started on this machine.
        identity: This runner's identity arguments.
        detail: The verdict line, naming the run and its ending.

    Returns:
        The job as the queue answered: ``queued`` the first time, ``failed``
        when it was requeued before.

    Raises:
        AppError: Only from the queue call itself.
    """
    answered = queue_lost.report_lost(
        credentials, job_id=job["job_id"], detail=detail, identity=identity
    )
    _log.info(
        "%s: ended for memory, the queue answered %s: %s",
        encode_job_line(job),
        answered["status"],
        detail,
    )
    return answered


__all__ = ["close", "give_back", "refuse"]
