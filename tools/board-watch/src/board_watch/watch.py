"""Cursor arithmetic and the one-poll step, with no I/O loop around them.

Everything here is a pure function of a page and a held cursor, so the
awkward part of a watcher -- deciding where "now" is and when to move --
is testable without a clock or a socket. The loop that calls these lives in
:mod:`board_watch.cli.watch` and does nothing but sequence them.

THE RULE THAT IS EASY TO GET WRONG, and which this module exists to hold in
one place: ``task_events`` offers a next cursor on every NON-EMPTY page, and
none on an empty one. An empty page means the caller has caught up and keeps
the cursor it already holds. Treating that as "start over" is what makes a
watcher replay its whole history on every quiet poll.

That rule is the post-``edfa06ec`` contract (2026-09-06). Before it, a cursor
came only on a FULL page, so a short page's events were re-served on every
poll forever while the poller reported success -- and this package carried a
second request per prime to work around it. The workaround is gone rather
than left inert: a non-empty page without a cursor is now a contract
violation this module raises on, because tolerating it is precisely how the
original bug would come back unseen.
"""

from __future__ import annotations

from typing import Final, TypedDict

from platform_core.error_codes_tooling import BoardWatchErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from board_watch import _test_hooks
from board_watch.contracts import BoardEvent, EventPage, decode_event_page

#: The MCP tool this package reads.
EVENTS_TOOL: Final = "task_events"

#: The largest page ``task_events`` will return, from its input schema.
MAX_LIMIT: Final = 200


class SubscriptionSpec(TypedDict):
    """What a watcher is subscribed to.

    Attributes:
        agent: The label whose ``@mentions`` wake this watcher. Also excluded
            as an author, so the watcher's own posts never notify it.
        session_id: The Claude session UUID this watcher polls FOR. Not an
            identifier the watcher may invent: the board binds one label to
            one session (mig 415) and refuses a second, so a fabricated UUID
            would be rejected outright the first time the real session had
            already written under that label.
        cwd: The polled session's working directory, recorded on the read
            for the same audit reason every board write records it.
        room: Restrict to one board room, or None for every room.
        kind: Restrict to one entry kind, or None for every kind.
        limit: Rows per poll.
    """

    agent: str
    session_id: str
    cwd: str
    room: str | None
    kind: str | None
    limit: int


def _identity(spec: SubscriptionSpec) -> JSONObject:
    """Render the caller identity ``task_events`` requires.

    Identity became required on that read when the board grew read
    receipts: a receipt must name who was served, and keying it to the
    FILTER instead would let any session reading another agent's queue
    mark that agent caught up.

    It matters here specifically. Because this watcher passes the polled
    session's OWN label as ``mentionsAgent``, its polls DO advance that
    session's receipt -- which is correct, since a delivered mention has
    been delivered, and is why the identity must be the real session's
    rather than a service account standing in for it.

    Args:
        spec: The subscription carrying the polled session's identity.

    Returns:
        The three identity fields, ready to merge into a call's arguments.
    """
    return {
        "agent": spec["agent"],
        "sessionId": spec["session_id"],
        "cwd": spec["cwd"],
    }


def subscription_arguments(spec: SubscriptionSpec, cursor: str | None) -> JSONObject:
    """Build the ``task_events`` arguments for one subscribed poll.

    Args:
        spec: What the watcher is subscribed to.
        cursor: The position to read forward from, or None to start at the
            oldest visible event.

    Returns:
        The arguments object.
    """
    arguments: JSONObject = {
        "mentionsAgent": spec["agent"],
        "excludeAuthor": spec["agent"],
        "limit": spec["limit"],
        **_identity(spec),
    }
    if spec["room"] is not None:
        arguments["room"] = spec["room"]
    if spec["kind"] is not None:
        arguments["kind"] = spec["kind"]
    if cursor is not None:
        arguments["cursor"] = cursor
    return arguments


def priming_arguments(spec: SubscriptionSpec, cursor: str | None) -> JSONObject:
    """Build the arguments for one step of establishing position.

    Deliberately UNFILTERED. Priming walks to the true end of the feed, not
    to the last event that happens to match the subscription, so the watcher
    starts from "everything after this moment" rather than "everything after
    the last time somebody mentioned me". Those differ by exactly the events
    the watcher was started to not have to read.

    Identity is carried even though no selector is: the read tool requires
    it. It advances NO receipt here, and that is right rather than
    incidental -- an unfiltered walk names nobody's queue, so nothing was
    delivered to anybody and there is no position to claim was read.

    Args:
        spec: The subscription carrying the polled session's identity.
        cursor: The position to read forward from, or None to start at the
            oldest visible event.

    Returns:
        The arguments object.
    """
    arguments: JSONObject = {"limit": MAX_LIMIT, **_identity(spec)}
    if cursor is not None:
        arguments["cursor"] = cursor
    return arguments


def prime(credentials: McpCredentials, spec: SubscriptionSpec) -> str | None:
    """Walk the feed to its end and return the cursor for "from now on".

    Each page carries the cursor for its own last row, so the walk simply
    follows them until the board offers none -- which it does only when the
    page is empty, and an empty page is the end of the feed.

    Args:
        credentials: Endpoint and headers.
        spec: The subscription, for the identity the read tool requires.

    Returns:
        The cursor positioned after the newest existing event, or None when
        the board has never had an event at all.

    Raises:
        AppError: ``PAGE_WITHOUT_CURSOR`` when a page carrying rows offers no
            cursor, which the server contract forbids. Also any transport or
            contract failure from the underlying call.
    """
    cursor: str | None = None
    while True:
        page = decode_event_page(
            call_mcp_tool(
                _test_hooks.http_post,
                credentials,
                EVENTS_TOOL,
                priming_arguments(spec, cursor),
            )
        )
        if page["next_cursor"] is None:
            if page["count"] != 0:
                raise AppError(
                    code=BoardWatchErrorCode.PAGE_WITHOUT_CURSOR,
                    message=(
                        f"task_events returned {page['count']} events with no next "
                        "cursor; every non-empty page must carry its last row's "
                        "cursor. Priming cannot reach the end of the feed, and "
                        "continuing would replay these events on every poll."
                    ),
                )
            return cursor
        cursor = page["next_cursor"]


def poll(
    credentials: McpCredentials, spec: SubscriptionSpec, cursor: str | None
) -> tuple[EventPage, str | None]:
    """Read one page of matching events and report the new position.

    Args:
        credentials: Endpoint and headers.
        spec: What the watcher is subscribed to.
        cursor: The position to read forward from.

    Returns:
        The page, and the cursor to hold for the next poll: the page's own
        next cursor when it offered one, otherwise the cursor already held.
        An empty page is the board saying "you are caught up", which must
        leave the position exactly where the caller put it -- moving it
        backwards there is what replays the whole history on a quiet board.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    page = decode_event_page(
        call_mcp_tool(
            _test_hooks.http_post,
            credentials,
            EVENTS_TOOL,
            subscription_arguments(spec, cursor),
        )
    )
    if page["next_cursor"] is None:
        return page, cursor
    return page, page["next_cursor"]


def format_notification(event: BoardEvent) -> str:
    """Render one event as the single line a subscriber sees.

    Leads with the author and the task rather than the timestamp, because a
    Monitor notification is read in a conversation where "who wants me and
    about what" is the question and the time is already known.

    THE SUMMARY IS COLLAPSED TO ONE LINE. Board summaries carry newlines --
    they are the opening of a post body -- and Monitor turns every line of
    this process's output into a separate notification. Emitting a multi-line
    summary would announce one mention as several events, most of them
    context-free fragments.

    Args:
        event: The event to render.

    Returns:
        The line, without a trailing newline and containing none.
    """
    where = "" if event["task_id"] is None else f" task:{event['task_id']}"
    more = "" if event["omitted_chars"] == 0 else f" [+{event['omitted_chars']} more chars]"
    summary = " ".join(event["summary"].split())
    return (
        f"BOARD MENTION from {event['author']} [{event['kind']}]{where} "
        f"at {event['created_at']}: {summary}{more}"
    )
