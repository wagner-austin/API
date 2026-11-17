"""Telling the dispatch queue a catalog model is served, and that it stopped.

Beside :mod:`fleet.core.queue` rather than in it, as
:mod:`fleet.core.queue_lost` is, since that module stands near the
file-size ceiling; each call is one MCP call and one decode. Both are
``dispatch_report`` actions the queue keeps for model loads (MCPs board
tasks 62b1bb0a and b9235198): ``loaded`` passes a load with the url its
model answers on, which ``close`` refuses to do, and ``ended`` records on a
passed load that its server stopped with no unload job, idle or exited, so
the next load of the model queues a new job rather than answering with it.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.dispatch import DispatchJob, decode_reported
from fleet.core import _test_hooks
from fleet.core.queue import report_target


def report_loaded(
    credentials: McpCredentials,
    *,
    job: DispatchJob,
    endpoint: str,
    detail: str,
    identity: JSONObject,
) -> DispatchJob:
    """Pass a running model load with the url its model answers on.

    Args:
        credentials: Endpoint and headers.
        job: The running load, as the queue last answered it.
        endpoint: ``http://<address>:<port>``, the queue's endpoint grammar.
        detail: What was served and how it was known to answer, in one line.
        identity: From :func:`fleet.core.queue.identity_arguments`.

    Returns:
        The passed job.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            including the queue refusing an endpoint outside its grammar.
    """
    arguments: JSONObject = {
        "action": "loaded",
        **report_target(job),
        "endpoint": endpoint,
        "detail": detail,
        **identity,
    }
    return decode_reported(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_report", arguments)
    )


def report_ended(
    credentials: McpCredentials, *, job_id: str, detail: str, identity: JSONObject
) -> DispatchJob:
    """Record that a passed model load's server stopped with no unload job.

    The one report naming no claim generation (MCPs board task 8f130298): a
    passed load is never claimed again, so its generation cannot move under
    it, and the node's record of a served model names only its job.

    Args:
        credentials: Endpoint and headers.
        job_id: The passed load.
        detail: Why it stopped, in one line.
        identity: From :func:`fleet.core.queue.identity_arguments`.

    Returns:
        The load, still ``passed``.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            including the queue refusing a load that ended already.
    """
    arguments: JSONObject = {"action": "ended", "jobId": job_id, "detail": detail, **identity}
    return decode_reported(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_report", arguments)
    )


__all__ = ["report_ended", "report_loaded"]
