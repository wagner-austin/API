"""Cursor arithmetic and the one-poll step, with no I/O loop around them.

Everything here is a pure function of a page and a held cursor, so the
awkward part of a watcher -- deciding where "now" is and when to move --
is testable without a clock or a socket. The loop that calls these lives in
:mod:`board_watch.cli.watch` and does nothing but sequence them.

THE RULE THAT IS EASY TO GET WRONG, and which this module exists to hold in
one place: ``task_events`` offers a next cursor ONLY on a full page. A short
page means the caller has caught up, carries no cursor, and the caller keeps
the one it already holds. Treating a missing cursor as "start over" is what
makes a watcher replay its whole history on every quiet poll.
"""

from __future__ import annotations

from typing import Final, TypedDict

from platform_core.json_utils import JSONObject

from board_watch.client import call_tool
from board_watch.config import BoardCredentials
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
        room: Restrict to one board room, or None for every room.
        kind: Restrict to one entry kind, or None for every kind.
        limit: Rows per poll.
    """

    agent: str
    room: str | None
    kind: str | None
    limit: int


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
    }
    if spec["room"] is not None:
        arguments["room"] = spec["room"]
    if spec["kind"] is not None:
        arguments["kind"] = spec["kind"]
    if cursor is not None:
        arguments["cursor"] = cursor
    return arguments


def priming_arguments(cursor: str | None, limit: int = MAX_LIMIT) -> JSONObject:
    """Build the arguments for one step of establishing position.

    Deliberately UNFILTERED. Priming walks to the true end of the feed, not
    to the last event that happens to match the subscription, so the watcher
    starts from "everything after this moment" rather than "everything after
    the last time somebody mentioned me". Those differ by exactly the events
    the watcher was started to not have to read.

    Args:
        cursor: The position to read forward from, or None to start at the
            oldest visible event.
        limit: Rows to request. Defaults to the largest page; :func:`prime`
            passes an exact row count on its final request, to turn a short
            page into a full one so the board mints a cursor for its last row.

    Returns:
        The arguments object.
    """
    arguments: JSONObject = {"limit": limit}
    if cursor is not None:
        arguments["cursor"] = cursor
    return arguments


def advance(held: str | None, page: EventPage) -> str | None:
    """Decide which cursor to hold after reading a page.

    Args:
        held: The cursor used to fetch this page.
        page: What came back.

    Returns:
        The page's next cursor when it offered one, otherwise the cursor
        already held. A short page is the board saying "you are caught up",
        which leaves the position exactly where the caller put it.
    """
    if page["next_cursor"] is None:
        return held
    return page["next_cursor"]


def prime(credentials: BoardCredentials) -> str | None:
    """Walk the feed to its end and return the cursor for "from now on".

    THE LAST PARTIAL PAGE NEEDS A SECOND REQUEST, and getting that wrong is
    not visible from inside the walk. A cursor is offered only on a FULL
    page, so walking with the maximum limit stops at the last full-page
    BOUNDARY and never learns a position inside the partial page after it.
    Measured against the live board on 2026-09-05: priming landed at
    00:29:05 while the feed already held events at 00:42 and 02:17, so the
    very first poll announced two mentions that predated arming -- the exact
    backlog priming exists to skip.

    The fix costs one request. A short page of ``k`` rows re-requested with
    ``limit=k`` is by definition a full page, so the board mints a cursor for
    its last row. Events arriving between the two calls are simply carried
    into the next poll, which is correct: they are genuinely new.

    Args:
        credentials: Endpoint and headers.

    Returns:
        The cursor positioned after the newest existing event, or None when
        the board has never had an event at all.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    cursor: str | None = None
    while True:
        page = decode_event_page(call_tool(credentials, EVENTS_TOOL, priming_arguments(cursor)))
        if page["next_cursor"] is not None:
            cursor = page["next_cursor"]
            continue
        if page["count"] == 0:
            return cursor
        exact = decode_event_page(
            call_tool(credentials, EVENTS_TOOL, priming_arguments(cursor, limit=page["count"]))
        )
        return advance(cursor, exact)


def poll(
    credentials: BoardCredentials, spec: SubscriptionSpec, cursor: str | None
) -> tuple[EventPage, str | None]:
    """Read one page of matching events and report the new position.

    Args:
        credentials: Endpoint and headers.
        spec: What the watcher is subscribed to.
        cursor: The position to read forward from.

    Returns:
        The page and the cursor to hold for the next poll.

    Raises:
        AppError: Any transport or contract failure from the underlying call.
    """
    page = decode_event_page(
        call_tool(credentials, EVENTS_TOOL, subscription_arguments(spec, cursor))
    )
    return page, advance(cursor, page)


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
