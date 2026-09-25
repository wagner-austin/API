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
from platform_core.json_utils import JSONObject, JSONValue
from platform_core.mcp_client import McpCredentials, call_mcp_tool
from platform_core.stack_endpoints import FLEET_SERVICE, STACK_ENDPOINTS_PATH, declared_url

from fleet.contracts.dispatch import (
    ClosingStatus,
    DispatchJob,
    DispatchLane,
    ListingPage,
    decode_claim,
    decode_listing,
    decode_listing_page,
    decode_reported,
)
from fleet.contracts.tags import NodeTag
from fleet.core import _test_hooks

#: Environment variable holding fleet-mcp's ``x-api-key`` value.
API_KEY_VARIABLE: Final = "FLEET_MCP_API_KEY"

#: Environment variable holding the tenant whose queue is being served.
TENANT_ID_VARIABLE: Final = "CORVIS_TENANT_ID"

#: Environment variable overriding the endpoint, for a non-default deployment.
URL_VARIABLE: Final = "FLEET_DISPATCH_URL"

#: The page size a paged listing asks for: ``dispatch_list``'s own maximum,
#: so a reader walking every page makes as few calls as the tool allows.
LISTING_PAGE_LIMIT: Final = 100


def load_credentials() -> McpCredentials:
    """Read both secrets from the environment and the endpoint from the
    stack's declaration, unless the environment overrides it.

    The default endpoint is where the MCPs repository declares fleet-mcp
    (:mod:`platform_core.stack_endpoints`), read after both secrets so a
    shell missing one hears about that first. It was a literal
    ``http://127.0.0.1:8035/mcp`` until 2026-09-25, when fleet-mcp moved to
    diphtheria and the hub's landing for that port was retired (MCPs board
    tasks 91ca67f4 and 60df277e): every fleet-agent tick failed from that
    minute, the same break :mod:`board_watch.config` took for the taskboard
    the day before.

    Returns:
        The credentials.

    Raises:
        AppError: ``QUEUE_CREDENTIALS_MISSING``, naming which variable is
            unset and where its value comes from. One code rather than two
            because both are fixed in the same place -- the shell that
            schedules the agent -- unlike ``board-watch``'s pair, whose two
            values live in a container's environment and a database row.
            ``STACK_ENDPOINT_UNDECLARED`` when no override is set and the
            declaration names no fleet-mcp url.
        OSError: When no override is set and the declaration cannot be read.
    """
    api_key = _test_hooks.env(API_KEY_VARIABLE)
    if api_key is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_CREDENTIALS_MISSING,
            message=(
                f"{API_KEY_VARIABLE} is unset; it is fleet-mcp's own "
                "MCP_INTERNAL_KEY, exported by the hpc-wake runs/env.ps1 the "
                "tick scripts dot-source, and must be set before the agent runs"
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
    if url is None:
        url = declared_url(_test_hooks.read_text(STACK_ENDPOINTS_PATH), FLEET_SERVICE)
    return McpCredentials(url=url, api_key=api_key, tenant_id=tenant_id)


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
    lane: DispatchLane,
    tags: tuple[NodeTag, ...],
    node: str | None,
    lease_seconds: int,
    identity: JSONObject,
) -> DispatchJob | None:
    """Take the next claimable job in this runner's lane, if there is one.

    Args:
        credentials: Endpoint and headers.
        lane: Which jobs this runner drains (MCPs mig 532, board task
            fd5cabfa A5): ``hub`` for the rebuild and the session verbs the
            hub runs itself, ``node`` for the make targets a fleet node runs
            on a checked-out commit. Two lanes so a revive never queues
            behind a check.
        tags: The capabilities the claiming node carries; the queue hands
            back only a job whose required tags are all among them.
        node: Restrict to jobs for this node plus jobs that named none, or
            None to take anything.
        lease_seconds: How long the claim survives without a report. Always
            set, never omitted: an unbounded claim means a runner that dies
            holds the job forever, and this runner is scheduled rather than
            supervised.
        identity: From :func:`identity_arguments`.

    Returns:
        The claimed job, or None when nothing in the lane matches.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "lane": lane,
        "tags": list(tags),
        "leaseSeconds": lease_seconds,
        **identity,
    }
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


def report_progress(
    credentials: McpCredentials,
    *,
    job_id: str,
    note: str,
    lease_seconds: int,
    identity: JSONObject,
) -> DispatchJob:
    """Renew a running job's lease with a progress note.

    The node lane's collect pass calls this on every job it holds that is
    still running (MCPs board task fd5cabfa): a suite that outlives the
    claim's lease would otherwise be handed to a second node while the first
    was still writing, and the lease is renewed by the runner that can see
    the run rather than sized in advance for the slowest node.

    Args:
        credentials: Endpoint and headers.
        job_id: The running job.
        note: One line of progress for the trail.
        lease_seconds: Renewed lease.
        identity: From :func:`identity_arguments`.

    Returns:
        The updated job.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "action": "progress",
        "jobId": job_id,
        "note": note,
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


def cancelled_page(credentials: McpCredentials, *, agent: str, offset: int) -> ListingPage:
    """List one page of the jobs cancelled while this runner held them.

    A cancel keeps the row's ``claimed_by`` and ``run_id`` (MCPs
    ``cancelDispatchJob`` sets only the status, detail, lease and close
    time), so the runner that launched a build can still find it here after
    :func:`held_by` has stopped returning it, and that is the only route a
    cancel has to the node: the queue has no route to the tailnet.

    Args:
        credentials: Endpoint and headers.
        agent: This runner's label.
        offset: Where the page begins, ``0`` for the newest.

    Returns:
        The page, and where the next one begins.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "claimedBy": agent,
        "status": "cancelled",
        "offset": offset,
        "limit": LISTING_PAGE_LIMIT,
    }
    return decode_listing_page(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_list", arguments)
    )


def observe_sessions(
    credentials: McpCredentials,
    *,
    machine: str,
    observations: list[JSONValue],
    identity: JSONObject,
) -> str:
    """Record one node's session records in the board's session ledger.

    The one call in this module that goes to the TASKBOARD rather than the
    dispatch queue: ``task_session_observe`` lives beside the board's other
    ``task_*`` tools, so ``credentials`` here are the board's (see
    :func:`board_watch.config.load_credentials`), not the queue's.

    Args:
        credentials: The board's endpoint and headers.
        machine: The node's machine id in the harness's ``pidDomain``
            spelling, ``win32:serendipity``.
        observations: Every record read from the node, already in the tool's
            wire shape.
        identity: From :func:`identity_arguments`.

    Returns:
        The tool's rendered line, which names what the pass wrote.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {"machine": machine, "observations": observations, **identity}
    return call_mcp_tool(_test_hooks.http_post, credentials, "task_session_observe", arguments)


#: The room a verdict lands in when the job names no task: the fleet's own,
#: where the runners live, addressed to the submitting label so it reaches
#: that session's feed as a mention.
VERDICT_ROOM: Final = "fleet"

#: What a runner tells the ledger it is, on its registering check-in.
RUNNER_HARNESS: Final = "fleet-agent"


def post_verdict(
    credentials: McpCredentials,
    *,
    task_id: str | None,
    submitted_by: str,
    line: str,
    identity: JSONObject,
) -> str:
    """Post a check's verdict where the submitter will read it.

    The second call here that goes to the TASKBOARD (MCPs board task
    fd5cabfa, A3): ``task_post`` on the job's task thread, or, when the job
    names no task, a board-level note in :data:`VERDICT_ROOM` that opens
    with the submitter's label so it lands in their feed.

    Args:
        credentials: The board's endpoint and headers.
        task_id: The job's task, or None.
        submitted_by: The label that enqueued the job.
        line: The verdict line (:func:`fleet.core.verdict.render_verdict`).
        identity: From :func:`identity_arguments`, the runner's own.

    Returns:
        The tool's rendered answer.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            including the board refusing a runner it has never ledgered.
    """
    arguments: JSONObject = {"kind": "note", **identity}
    if task_id is None:
        arguments["room"] = VERDICT_ROOM
        arguments["body"] = f"@{submitted_by} {line}"
    else:
        arguments["taskId"] = task_id
        arguments["body"] = line
    return call_mcp_tool(_test_hooks.http_post, credentials, "task_post", arguments)


def announce(
    credentials: McpCredentials,
    *,
    machine: str,
    body: str,
    identity: JSONObject,
) -> str:
    """Register this runner on the board's session ledger with a check-in.

    A runner posts verdicts under its own label, and the board binds work
    only to a session it has ledgered (MCPs mig 530). The hub's sessions
    are registered by the session-ledger hook; a runner has no hook, so it
    registers itself once, at registration time, with the harness it is.

    Args:
        credentials: The board's endpoint and headers.
        machine: ``<platform>:<hostname>`` in the harness's spelling.
        body: What the check-in says.
        identity: From :func:`identity_arguments`, the runner's own.

    Returns:
        The tool's rendered answer.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "kind": "checkin",
        "harness": RUNNER_HARNESS,
        "machine": machine,
        "room": VERDICT_ROOM,
        "body": body,
        **identity,
    }
    return call_mcp_tool(_test_hooks.http_post, credentials, "task_post", arguments)


__all__ = [
    "API_KEY_VARIABLE",
    "LISTING_PAGE_LIMIT",
    "RUNNER_HARNESS",
    "TENANT_ID_VARIABLE",
    "URL_VARIABLE",
    "VERDICT_ROOM",
    "announce",
    "cancelled_page",
    "claim_next",
    "held_by",
    "identity_arguments",
    "load_credentials",
    "observe_sessions",
    "post_verdict",
    "report_close",
    "report_progress",
    "report_start",
]
