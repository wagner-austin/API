"""The wire and domain types this package reads, and their decoders.

Every type here is a :class:`~typing.TypedDict` with a ``decode_*`` that
validates rather than assumes. The board answers in RENDERED PROSE built for
a model to read, not in JSON keyed by field, so "validate" means checking a
grammar rather than checking a schema -- and a grammar has no library to do
it for you.

The grammar is not invented here. It is the observable output of three
functions in ``taskboard-mcp``, read from their source on 2026-09-05:

* ``encodeAgentTaskEventLine``  (``taskboard-mcp/src/encode.ts``)
* ``taskReference``             (same file)
* ``encodeCursorPaginationFooter`` (``mcp-shared/src/pagination.ts``)

:data:`EVENT_LINE_PATTERN` and :data:`FOOTER_PATTERN` transcribe those, and
``tests/test_contracts.py`` pins each against a literal captured from the
live board so a server-side change fails here rather than in the field.
"""

from __future__ import annotations

import re
from typing import Final, TypedDict

from platform_core.error_codes import BoardWatchErrorCode
from platform_core.errors import AppError

#: One event row, as ``encodeAgentTaskEventLine`` renders it.
#:
#: ``{ISO8601} [{kind}] {author}{ task:{id}}?{ mentions:@a,@b}?: {summary}``
#: with an optional ``` [+{n} more chars]``` suffix when the body was longer
#: than the summary.
#:
#: The author group is non-greedy and the optional groups are anchored to
#: their literal prefixes, because an author label may itself contain no
#: spaces but a summary certainly may contain ``task:`` and ``mentions:``.
#: A SUMMARY MAY CONTAIN NEWLINES, so a row is not a line. ``re.DOTALL`` lets
#: the summary group span them, and :data:`ROW_ANCHOR_PATTERN` is what
#: separates one row from the next. Found against the live board on
#: 2026-09-05: every captured fixture happened to have a single-line summary,
#: so a decoder that split on ``\n`` passed 69 tests and failed on the first
#: real page, at ``'Done this session:'``.
EVENT_LINE_PATTERN: Final = re.compile(
    r"^(?P<created_at>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z) "
    r"\[(?P<kind>[a-z_]+)\] "
    r"(?P<author>\S+?)"
    r"(?: task:(?P<task_id>[0-9a-fA-F-]{36}))?"
    r"(?: mentions:(?P<mentions>@\S*))?"
    r": (?P<summary>.*?)"
    r"(?: \[\+(?P<more_chars>\d+) more chars\])?$",
    re.DOTALL,
)

#: What begins a row: the timestamp and kind that open every rendered event.
#:
#: This, not the newline, is the record separator. It is anchored to the start
#: of a physical line so a timestamp quoted inside somebody's post body cannot
#: be mistaken for the start of a new event.
ROW_ANCHOR_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \[[a-z_]+\] ")

#: The pagination footer, as ``encodeCursorPaginationFooter`` renders it.
#:
#: ``[showing {n} events]`` when the caller is caught up, or
#: ``[showing {n} events; next cursor: {token}]`` when a full page implies
#: more may follow.
#:
#: THE SPELLING IS THE WHOLE POINT. A watcher that looked for ``nextCursor``
#: -- the name of the field in the tool's own documentation -- matched
#: nothing, advanced no cursor, and replayed the same events indefinitely
#: while reporting success. The rendered form is ``next cursor:``.
FOOTER_PATTERN: Final = re.compile(
    r"^\[showing (?P<count>\d+) events(?:; next cursor: (?P<next_cursor>\S+))?\]$"
)

#: What ``handleTaskEvents`` returns in place of rows when there are none.
EMPTY_FEED_SENTINEL: Final = "no board events after this cursor — you are caught up"


class BoardEvent(TypedDict):
    """One row of the board's change feed.

    Attributes:
        created_at: ISO-8601 UTC timestamp with millisecond precision,
            exactly as rendered.
        kind: The entry kind: ``checkin``, ``note``, ``status_change`` or
            ``handoff``.
        author: The agent label that wrote the entry.
        task_id: The task whose thread the entry belongs to, or None for a
            board-level post.
        mentions: Every agent label named with an ``@`` prefix, without the
            prefix, in render order.
        summary: The bounded summary the board renders. NOT the full body.
        omitted_chars: How many characters of the body the summary leaves
            out. Zero when the summary is the whole body.
    """

    created_at: str
    kind: str
    author: str
    task_id: str | None
    mentions: tuple[str, ...]
    summary: str
    omitted_chars: int


class EventPage(TypedDict):
    """One response from ``task_events``.

    Attributes:
        events: The rows, oldest first.
        next_cursor: The token to pass on the next call, or None when the
            caller has caught up. A short page carries no cursor and the
            caller keeps the one it already holds -- see
            :func:`board_watch.watch.advance`.
        count: The row count the footer reports. Held separately from
            ``len(events)`` so :func:`decode_event_page` can assert the two
            agree rather than trusting either.
    """

    events: tuple[BoardEvent, ...]
    next_cursor: str | None
    count: int


def decode_event_line(line: str) -> BoardEvent:
    """Decode one rendered event row.

    Args:
        line: A single line of the tool's response, without its newline.

    Returns:
        The decoded row.

    Raises:
        AppError: ``EVENT_LINE_MALFORMED`` when the line does not match the
            grammar ``encodeAgentTaskEventLine`` produces. The message
            carries the offending line so the reader can see which element
            moved.
    """
    match = EVENT_LINE_PATTERN.match(line)
    if match is None:
        raise AppError(
            code=BoardWatchErrorCode.EVENT_LINE_MALFORMED,
            message=(
                f"board event line does not match the taskboard-mcp render contract: {line!r}"
            ),
        )
    # Every group is annotated at its assignment. ``re.Match.group`` is typed
    # loosely enough that its result is Any, and a decoder whose fields are
    # Any would defeat the point of decoding at all.
    raw_mentions: str | None = match.group("mentions")
    mentions: tuple[str, ...] = (
        ()
        if raw_mentions is None
        else tuple(label[1:] for label in raw_mentions.split(",") if label.startswith("@"))
    )
    raw_more: str | None = match.group("more_chars")
    created_at: str = match.group("created_at")
    kind: str = match.group("kind")
    author: str = match.group("author")
    task_id: str | None = match.group("task_id")
    summary: str = match.group("summary")
    return BoardEvent(
        created_at=created_at,
        kind=kind,
        author=author,
        task_id=task_id,
        mentions=mentions,
        summary=summary,
        omitted_chars=0 if raw_more is None else int(raw_more),
    )


def decode_footer(line: str) -> tuple[int, str | None]:
    """Decode the pagination footer.

    Args:
        line: The footer line, without its newline.

    Returns:
        The row count it reports and the next cursor, or None for a short
        page where the caller is caught up.

    Raises:
        AppError: ``FOOTER_MALFORMED`` when the line is not a footer.
    """
    match = FOOTER_PATTERN.match(line)
    if match is None:
        raise AppError(
            code=BoardWatchErrorCode.FOOTER_MALFORMED,
            message=(
                f"board pagination footer does not match the mcp-shared render contract: {line!r}"
            ),
        )
    count: str = match.group("count")
    next_cursor: str | None = match.group("next_cursor")
    return int(count), next_cursor


def split_rows(body: list[str]) -> tuple[str, ...]:
    """Group the body's physical lines into logical event rows.

    A row begins at :data:`ROW_ANCHOR_PATTERN` and continues until the next
    one, because a summary may carry newlines. Splitting on ``\\n`` instead
    treats a post's second paragraph as a malformed event, which is what the
    first live run did.

    Args:
        body: Every line above the footer, in order, blank lines included.

    Returns:
        One string per row, each with its continuation lines rejoined.

    Raises:
        AppError: ``EVENT_LINE_MALFORMED`` when the body does not begin with
            a row anchor, which means the response is not a list of events at
            all and no row boundary can be found.
    """
    if len(body) > 0 and ROW_ANCHOR_PATTERN.match(body[0]) is None:
        raise AppError(
            code=BoardWatchErrorCode.EVENT_LINE_MALFORMED,
            message=(f"board response body does not begin with an event row: {body[0]!r}"),
        )
    rows: list[list[str]] = []
    for line in body:
        if ROW_ANCHOR_PATTERN.match(line) is not None:
            rows.append([line])
        else:
            rows[-1].append(line)
    return tuple("\n".join(row) for row in rows)


def decode_event_page(text: str) -> EventPage:
    """Decode a whole ``task_events`` response body.

    The response is the rows joined by newlines, a blank line, then the
    footer. An empty feed replaces the rows with
    :data:`EMPTY_FEED_SENTINEL` and still carries a footer.

    Args:
        text: The tool's rendered text content.

    Returns:
        The decoded page.

    Raises:
        AppError: ``FOOTER_MISSING`` when the body carries no footer line, or
            ``EVENT_LINE_MALFORMED`` / ``FOOTER_MALFORMED`` from the
            element decoders.
    """
    lines = text.rstrip().splitlines()
    if len(lines) == 0:
        raise AppError(
            code=BoardWatchErrorCode.FOOTER_MISSING,
            message="board response carried no lines at all, so no footer to read",
        )
    count, next_cursor = decode_footer(lines[-1])
    # Everything above the footer, minus the blank separator the renderer puts
    # between the rows and it. Blank lines INSIDE a row are kept, because a
    # post body may contain them and dropping them would corrupt the summary.
    body = lines[:-1]
    while len(body) > 0 and body[-1].strip() == "":
        body.pop()
    events = (
        ()
        if body == [EMPTY_FEED_SENTINEL]
        else tuple(decode_event_line(row) for row in split_rows(body))
    )
    if len(events) != count:
        raise AppError(
            code=BoardWatchErrorCode.FOOTER_MALFORMED,
            message=(
                f"board footer reports {count} events but {len(events)} rows "
                "were rendered; the two are produced by different functions "
                "and have diverged"
            ),
        )
    return EventPage(events=events, next_cursor=next_cursor, count=count)


def encode_event_line(event: BoardEvent) -> str:
    """Render one event back to the board's own line format.

    The inverse of :func:`decode_event_line`. It exists so a round-trip test
    can assert the decoder loses nothing, which is the only check that
    catches a field being parsed and then silently dropped.

    Args:
        event: The row to render.

    Returns:
        The line, without a trailing newline.
    """
    task_ref = "" if event["task_id"] is None else f" task:{event['task_id']}"
    mentions = (
        ""
        if len(event["mentions"]) == 0
        else " mentions:" + ",".join(f"@{label}" for label in event["mentions"])
    )
    truncation = "" if event["omitted_chars"] == 0 else f" [+{event['omitted_chars']} more chars]"
    return (
        f"{event['created_at']} [{event['kind']}] {event['author']}"
        f"{task_ref}{mentions}: {event['summary']}{truncation}"
    )
