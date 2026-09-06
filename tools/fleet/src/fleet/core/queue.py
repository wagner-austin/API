"""Talking to the corvis dispatch queue: claim, report, and find own work.

Every function here is one MCP call and one decode. The transport is
:func:`platform_core.mcp_client.call_mcp_tool`, shared with
``tools/board-watch`` -- the JSON-RPC envelope, the Server-Sent-Events framing
and the two required headers are properties of MCP-over-HTTP and were lifted
out of that package rather than copied into this one.

THE CREDENTIALS ARE READ FROM THE ENVIRONMENT AND NOT DISCOVERED. An earlier
prototype elsewhere in this monorepo shelled out to ``docker inspect`` and
``psql`` to find them, which made every poll depend on the container runtime
being present and on the caller having permission to inspect containers.
Requiring them in the environment moves that work to the operator's shell
once, where it is visible.

NOTHING HERE LOOPS. One call, one answer. The interval belongs to whatever
schedules :mod:`fleet.cli.agent`, where it is visible at the call site --
the same decision ``board-watch`` made, and for the same reason: a loop inside
a library is an interval nobody can see or change without editing code.
"""

from __future__ import annotations

from typing import Final

from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.dispatch import (
    ClosingStatus,
    DispatchJob,
    decode_claim,
    decode_listing,
    decode_reported,
)
from fleet.core import _test_hooks

#: Environment variable holding fleet-mcp's ``x-api-key`` value.
API_KEY_VARIABLE: Final = "FLEET_MCP_API_KEY"

#: Environment variable holding the tenant whose queue is being served.
TENANT_ID_VARIABLE: Final = "CORVIS_TENANT_ID"

#: Environment variable overriding the endpoint, for a non-default deployment.
URL_VARIABLE: Final = "FLEET_DISPATCH_URL"

#: Where fleet-mcp is published on the host by default.
DEFAULT_URL: Final = "http://127.0.0.1:8035/mcp"


def load_credentials() -> McpCredentials:
    """Read the endpoint and both secrets from the environment.

    Returns:
        The credentials.

    Raises:
        AppError: ``QUEUE_CREDENTIALS_MISSING``, naming which variable is
            unset and where its value comes from. One code rather than two
            because both are fixed in the same place -- the shell that
            schedules the agent -- unlike ``board-watch``'s pair, whose two
            values live in a container's environment and a database row.
    """
    api_key = _test_hooks.env(API_KEY_VARIABLE)
    if api_key is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_CREDENTIALS_MISSING,
            message=(
                f"{API_KEY_VARIABLE} is unset; it is fleet-mcp's own "
                "MCP_INTERNAL_KEY, read from the mcp-fleet container's "
                "environment, and must be exported before the agent runs"
            ),
        )
    tenant_id = _test_hooks.env(TENANT_ID_VARIABLE)
    if tenant_id is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_CREDENTIALS_MISSING,
            message=(
                f"{TENANT_ID_VARIABLE} is unset; it is the tenants row whose "
                "queue this runner serves, and the queue has no default tenant"
            ),
        )
    url = _test_hooks.env(URL_VARIABLE)
    return McpCredentials(
        url=DEFAULT_URL if url is None else url,
        api_key=api_key,
        tenant_id=tenant_id,
    )


def identity_arguments(agent: str, session_id: str, cwd: str) -> JSONObject:
    """Build the three identity fields every mutating dispatch tool requires.

    Args:
        agent: This runner's label.
        session_id: Its stable UUID.
        cwd: Its working directory.

    Returns:
        The arguments fragment.
    """
    return {"agent": agent, "sessionId": session_id, "cwd": cwd}


def claim_next(
    credentials: McpCredentials,
    *,
    node: str | None,
    lease_seconds: int,
    identity: JSONObject,
) -> DispatchJob | None:
    """Take the next claimable job, if there is one.

    Args:
        credentials: Endpoint and headers.
        node: Restrict to jobs for this node plus jobs that named none, or
            None to take anything.
        lease_seconds: How long the claim survives without a report. Always
            set, never omitted: an unbounded claim means a runner that dies
            holds the job forever, and this runner is scheduled rather than
            supervised.
        identity: From :func:`identity_arguments`.

    Returns:
        The claimed job, or None when the queue is empty.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {"leaseSeconds": lease_seconds, **identity}
    if node is not None:
        arguments["node"] = node
    return decode_claim(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_claim", arguments)
    )


def report_start(
    credentials: McpCredentials,
    *,
    job_id: str,
    node: str,
    run_id: str,
    lease_seconds: int,
    identity: JSONObject,
) -> DispatchJob:
    """Record that the job is running, on which node and under which run id.

    Args:
        credentials: Endpoint and headers.
        job_id: The claimed job.
        node: The node this run committed to.
        run_id: The fleet ledger's run id, so the queue row and this
            machine's own records name the same run.
        lease_seconds: Renewed lease.
        identity: From :func:`identity_arguments`.

    Returns:
        The updated job.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "action": "start",
        "jobId": job_id,
        "node": node,
        "runId": run_id,
        "leaseSeconds": lease_seconds,
        **identity,
    }
    return decode_reported(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_report", arguments)
    )


def report_close(
    credentials: McpCredentials,
    *,
    job_id: str,
    status: ClosingStatus,
    exit_code: int | None,
    detail: str,
    identity: JSONObject,
) -> DispatchJob:
    """Report a job's result.

    ``exit_code`` must agree with ``status`` and the queue refuses the pair if
    it does not -- ``passed`` needs 0, ``failed`` needs non-zero, ``refused``
    needs none. That is not a rule this function softens: a runner that got it
    wrong should hear about it here rather than store a wrong answer.

    Args:
        credentials: Endpoint and headers.
        job_id: The claimed job.
        status: ``passed``, ``failed`` or ``refused``.
        exit_code: The command's exit code, or None when it never ran.
        detail: One line the submitter will read.
        identity: From :func:`identity_arguments`.

    Returns:
        The closed job.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            including the queue refusing an inconsistent status/exit pair.
    """
    arguments: JSONObject = {
        "action": "close",
        "jobId": job_id,
        "status": status,
        "detail": detail,
        **identity,
    }
    if exit_code is not None:
        arguments["exitCode"] = exit_code
    return decode_reported(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_report", arguments)
    )


def held_by(credentials: McpCredentials, *, agent: str) -> tuple[DispatchJob, ...]:
    """List the live jobs this runner is holding.

    Asked of the QUEUE rather than remembered locally. A runner that crashed
    between launching a suite and writing a note to itself would otherwise
    leave a job running on a node with nothing left that knows to collect it.

    Args:
        credentials: Endpoint and headers.
        agent: This runner's label.

    Returns:
        Its live jobs, newest first.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {"claimedBy": agent, "status": "live"}
    return decode_listing(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_list", arguments)
    )


__all__ = [
    "API_KEY_VARIABLE",
    "DEFAULT_URL",
    "TENANT_ID_VARIABLE",
    "URL_VARIABLE",
    "claim_next",
    "held_by",
    "identity_arguments",
    "load_credentials",
    "report_close",
    "report_start",
]
