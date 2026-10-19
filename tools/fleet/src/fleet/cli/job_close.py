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
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.dispatch import ClosingStatus, DispatchJob, encode_job_line
from fleet.core import queue

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


__all__ = ["close", "refuse"]
