"""Telling the dispatch queue that the run of a job this runner holds is gone.

Beside :mod:`fleet.core.queue` rather than in it, because that module stands
at the file-size ceiling; the call is one MCP call and one decode like each
of its own. The queue decides what becomes of the job (MCPs
``reportDispatchRunLost``, board task 05dd6588): the first loss puts it back
in the queue with a ``requeued`` trail row carrying ``detail``, and a loss of
a job requeued before closes it ``failed`` with no exit code, its outcome
posted to its task's thread.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.dispatch import DispatchJob, decode_reported
from fleet.core import _test_hooks


def report_lost(
    credentials: McpCredentials, *, job_id: str, detail: str, identity: JSONObject
) -> DispatchJob:
    """Report that the run of a held job is gone.

    Args:
        credentials: Endpoint and headers.
        job_id: The held job.
        detail: The run that is gone and how it was found gone, in one line.
        identity: From :func:`fleet.core.queue.identity_arguments`.

    Returns:
        The job: ``queued`` the first time, ``failed`` the second.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            including the queue refusing a runner that does not hold the job.
    """
    arguments: JSONObject = {"action": "lost", "jobId": job_id, "detail": detail, **identity}
    return decode_reported(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_report", arguments)
    )


__all__ = ["report_lost"]
