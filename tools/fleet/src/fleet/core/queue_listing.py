"""Reading the dispatch queue's listings: own work, waiting work, and pages.

Beside :mod:`fleet.core.queue` rather than in it, because that module stood
at the file-size ceiling; each function is one ``dispatch_list`` call and one
decode like each of its own.

EVERY LISTING ASKS FOR THE WHOLE ROW BY NAME (``full``, MCPs board task
beae2ca4). :func:`fleet.contracts.dispatch.decode_listing` reads every field
of a job, and fleet-mcp's ``dispatch_list`` is to answer the compact row of at
most 600 characters by default (MCPs board task a8d3a736), which a runner
could not decode. Naming ``full`` here keeps what the runner reads unchanged
when that default flips.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.dispatch import (
    DispatchJob,
    ListingPage,
    decode_listing,
    decode_listing_page,
)
from fleet.core import _test_hooks

#: The page size a paged listing asks for: ``dispatch_list``'s own maximum,
#: so a reader walking every page makes as few calls as the tool allows.
LISTING_PAGE_LIMIT: Final = 100


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
    arguments: JSONObject = {"claimedBy": agent, "status": "live", "full": True}
    return decode_listing(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_list", arguments)
    )


def queued_for(credentials: McpCredentials, *, project: str | None) -> tuple[DispatchJob, ...]:
    """List the first page of the jobs still waiting for a runner.

    One page is enough for both questions this answers: whether ANY job of a
    project waits, since the first page holds up to
    :data:`LISTING_PAGE_LIMIT` jobs and a project with more queued than that
    has one on it; and, for a serving node runner, whether a job has arrived
    since it last looked (:mod:`fleet.cli.node_serve`), since the page is
    newest first and an arrival is always on it.

    Args:
        credentials: Endpoint and headers.
        project: The project's key in the registry, or None for every
            project's.

    Returns:
        The queued jobs, newest first.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {"status": "queued", "limit": LISTING_PAGE_LIMIT, "full": True}
    if project is not None:
        arguments["project"] = project
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
        "full": True,
    }
    return decode_listing_page(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_list", arguments)
    )


def submitted_page(
    credentials: McpCredentials, *, project: str, submitted_by: str, offset: int
) -> ListingPage:
    """List one page of one submitter's jobs for one project, any status.

    The route from a ledger row back to the job that launched it once the
    job has left this runner (board task fd402617): the row names its
    project and submitter, and a reclaim overwrites every field of the job
    that named this runner.

    Args:
        credentials: Endpoint and headers.
        project: The project's key in the registry.
        submitted_by: The submitting session's label.
        offset: Where the page begins, ``0`` for the newest.

    Returns:
        The page, and where the next one begins.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    arguments: JSONObject = {
        "project": project,
        "submittedBy": submitted_by,
        "offset": offset,
        "limit": LISTING_PAGE_LIMIT,
        "full": True,
    }
    return decode_listing_page(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_list", arguments)
    )


__all__ = [
    "LISTING_PAGE_LIMIT",
    "cancelled_page",
    "held_by",
    "queued_for",
    "submitted_page",
]
