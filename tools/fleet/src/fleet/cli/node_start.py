"""The last step of a node runner's claim: report the launched run started.

Split out of :mod:`fleet.cli.node_agent`, which was at the file ceiling, for
MCPs board task 88b8fe61. A claim tick launches the run first and reports its
start second, because the run id the report carries is the one the launch
wrote. A job can be cancelled in between, and that is a legal state of the
queue rather than a fault: measured 2026-09-29, lavender-wsl claimed
MCPs/packages/db b4c6c447 at 18:57:14Z and launched it, its submitter
cancelled it at 18:57:25Z, and the start report was answered
``DISPATCH_BAD_TRANSITION: a job that is 'cancelled' cannot become
'running'``. The error left the tick with exit 1 and the suite running on the
node until the next tick's collect stopped it three minutes later.

So a refused start report is followed by one read of the job. A job the queue
now lists as cancelled is settled here, by the same stop the next tick would
have used (:func:`fleet.cli.node_collect.stop_cancelled_run`); a job in any
other status re-raises the refusal untouched, and a queue that cannot be
reached for the read fails with its own error, leaving the claimed job for
the next tick to adopt (:mod:`fleet.cli.node_collect`, board task 5a4f9b3e).
"""

from __future__ import annotations

from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli.node_collect import CLAIM_LEASE_SECONDS, stop_cancelled_run
from fleet.contracts.dispatch import DispatchJob, DispatchStatus
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import NodeConfig
from fleet.core import queue

_log = get_logger(__name__)


def report_started(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    identity: JSONObject,
    *,
    job: DispatchJob,
    row: LedgerEntry,
    alias: str,
    node: NodeConfig,
    agent: str,
) -> None:
    """Report the launched run started, or stop it when its job was cancelled.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        job: The claimed job.
        row: The run's live ledger row, as the launch wrote it.
        alias: This node's workspace name.
        node: Its declaration.
        agent: This runner's label.

    Raises:
        AppError: The start report's own refusal when the job is not
            cancelled, the read's failure when the queue cannot answer it, or
            a node failure from the stop.
    """
    try:
        queue.report_start(
            credentials,
            job_id=job["job_id"],
            node=alias,
            run_id=row["run_id"],
            lease_seconds=CLAIM_LEASE_SECONDS,
            identity=identity,
        )
    except AppError:
        current = queue.get_job(credentials, job_id=job["job_id"])
        if current["status"] is not DispatchStatus.CANCELLED:
            raise
        _log.info("%s was cancelled while %s launched", job["job_id"], row["run_id"])
        stop_cancelled_run(loaded, node=node, row=row, job=current, agent=agent)
        return
    _log.info("started %s on %s as %s at %s", job["job_id"], alias, row["run_id"], job["sha"])


__all__ = ["report_started"]
