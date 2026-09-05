"""Cursor arithmetic, priming, and one poll.

The assertions that matter here are about NOT moving: a short page must leave
the position where it was. Getting that backwards is what makes a watcher
replay its history on every quiet poll while reporting success.
"""

from __future__ import annotations

from board_watch import _test_hooks
from board_watch.contracts import decode_event_line, decode_event_page
from board_watch.watch import (
    MAX_LIMIT,
    SubscriptionSpec,
    advance,
    format_notification,
    poll,
    prime,
    priming_arguments,
    subscription_arguments,
)
from tests.conftest import (
    LIVE_CHECKIN_LINE,
    LIVE_MENTION_LINE,
    LIVE_MULTILINE_ROW,
    LIVE_TASK_LINE,
    TEST_CREDENTIALS,
    FakeHttpPost,
    ok,
    page_text,
    sent_arguments,
    tool_text,
)

SPEC = SubscriptionSpec(agent="opus-nclex-licensure-0904", room=None, kind=None, limit=50)


def test_a_short_page_leaves_the_cursor_where_it_was() -> None:
    """The board offers no cursor when the caller has caught up.

    Treating that as "start over" is the replay bug; treating it as "stay"
    is the contract.
    """
    page = decode_event_page(page_text([LIVE_CHECKIN_LINE], None))
    assert advance("held", page) == "held"


def test_a_full_page_moves_the_cursor_forward() -> None:
    """A full page is the only case that implies more may exist."""
    page = decode_event_page(page_text([LIVE_CHECKIN_LINE], "onward"))
    assert advance("held", page) == "onward"


def test_a_short_page_from_no_cursor_stays_at_no_cursor() -> None:
    """Priming an empty board leaves the watcher with nothing to hold."""
    page = decode_event_page(page_text([], None))
    assert advance(None, page) is None


def test_subscription_arguments_filter_to_the_agent_both_ways() -> None:
    """Mentions select what wakes you; excludeAuthor stops self-waking."""
    arguments = subscription_arguments(SPEC, None)
    assert arguments["mentionsAgent"] == SPEC["agent"]
    assert arguments["excludeAuthor"] == SPEC["agent"]
    assert arguments["limit"] == 50
    assert "cursor" not in arguments
    assert "room" not in arguments
    assert "kind" not in arguments


def test_subscription_arguments_carry_every_optional_filter() -> None:
    """Each optional flag has to reach the board to have any effect."""
    spec = SubscriptionSpec(agent="a-b-c", room="main", kind="status_change", limit=7)
    arguments = subscription_arguments(spec, "here")
    assert arguments["room"] == "main"
    assert arguments["kind"] == "status_change"
    assert arguments["cursor"] == "here"


def test_priming_is_unfiltered_and_uses_the_largest_page() -> None:
    """Priming walks to the end of the FEED, not to the last mention.

    A filtered prime would leave the watcher positioned after the last time
    somebody happened to mention it, so every mention between then and now
    would arrive as new -- which is the backlog a subscription exists to
    avoid announcing.
    """
    arguments = priming_arguments(None)
    assert arguments == {"limit": MAX_LIMIT}
    assert "mentionsAgent" not in arguments
    assert priming_arguments("mid")["cursor"] == "mid"


def test_prime_re_requests_the_last_partial_page_to_reach_its_end() -> None:
    """The bug the live board found: a short page offers no cursor.

    Walking with the maximum limit stops at the last full-page BOUNDARY, so
    the events in the partial page after it would all arrive as new on the
    first poll -- measured as two real mentions announced after arming. The
    fix re-requests that page with ``limit`` equal to its own row count,
    which makes it a full page and mints a cursor for its last row.
    """
    poster = FakeHttpPost(
        [
            ok(tool_text(page_text([LIVE_CHECKIN_LINE], "boundary"))),
            ok(tool_text(page_text([LIVE_TASK_LINE, LIVE_MENTION_LINE], None))),
            ok(tool_text(page_text([LIVE_TASK_LINE, LIVE_MENTION_LINE], "true-end"))),
        ]
    )
    _test_hooks.http_post = poster
    assert prime(TEST_CREDENTIALS) == "true-end"
    assert sent_arguments(poster.bodies[0]) == {"limit": MAX_LIMIT}
    assert sent_arguments(poster.bodies[1]) == {"limit": MAX_LIMIT, "cursor": "boundary"}
    # The exact-count re-request is what turns the short page into a full one.
    assert sent_arguments(poster.bodies[2]) == {"limit": 2, "cursor": "boundary"}


def test_prime_on_an_empty_board_holds_no_cursor() -> None:
    """A board with no events has no position to hold, and that is not an error.

    No re-request is made either: a page of zero rows cannot be turned into
    a full page, so asking again would be a call that could not answer.
    """
    poster = FakeHttpPost([ok(tool_text(page_text([], None)))])
    _test_hooks.http_post = poster
    assert prime(TEST_CREDENTIALS) is None
    assert len(poster.bodies) == 1


def test_prime_keeps_its_cursor_if_the_re_request_comes_back_short() -> None:
    """Events can vanish between the two calls, and that is not a restart."""
    _test_hooks.http_post = FakeHttpPost(
        [
            ok(tool_text(page_text([LIVE_CHECKIN_LINE], "boundary"))),
            ok(tool_text(page_text([LIVE_TASK_LINE], None))),
            ok(tool_text(page_text([], None))),
        ]
    )
    assert prime(TEST_CREDENTIALS) == "boundary"


def test_poll_returns_the_page_and_the_new_position() -> None:
    """One poll reads matching rows and reports where to resume."""
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([LIVE_MENTION_LINE], "next")))])
    page, moved = poll(TEST_CREDENTIALS, SPEC, "start")
    assert moved == "next"
    assert len(page["events"]) == 1
    assert page["events"][0]["author"] == "opus-lavender-gpu-0824"


def test_poll_on_a_quiet_board_holds_its_position() -> None:
    """The quiet case is the common one and must not move the cursor."""
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    page, moved = poll(TEST_CREDENTIALS, SPEC, "start")
    assert page["events"] == ()
    assert moved == "start"


def test_a_notification_leads_with_who_wants_you_and_why() -> None:
    """The line is read in a conversation where the time is already known."""
    line = format_notification(decode_event_line(LIVE_TASK_LINE))
    assert line.startswith("BOARD MENTION from fable-brain-audit-0903 [status_change]")
    assert "task:8793517e-5c6b-4edd-a127-0234b40404d4" in line
    assert "claimed -> done" in line


def test_a_notification_reports_a_truncated_body() -> None:
    """A summary is bounded, and a reader has to know the body was longer."""
    line = format_notification(decode_event_line(LIVE_MENTION_LINE))
    assert "[+3149 more chars]" in line


def test_a_notification_collapses_a_multi_line_summary() -> None:
    """Monitor makes one notification per LINE of output.

    A multi-line summary emitted raw would announce a single mention as
    several events, most of them context-free fragments.
    """
    line = format_notification(decode_event_line(LIVE_MULTILINE_ROW))
    assert "\n" not in line
    assert "Done this session:" in line


def test_a_notification_omits_the_task_when_there_is_none() -> None:
    """A board-level post has no thread and must not render an empty one."""
    line = format_notification(decode_event_line(LIVE_CHECKIN_LINE))
    assert "task:" not in line


__all__ = [
    "test_a_full_page_moves_the_cursor_forward",
    "test_a_notification_collapses_a_multi_line_summary",
    "test_a_notification_leads_with_who_wants_you_and_why",
    "test_a_notification_omits_the_task_when_there_is_none",
    "test_a_notification_reports_a_truncated_body",
    "test_a_short_page_from_no_cursor_stays_at_no_cursor",
    "test_a_short_page_leaves_the_cursor_where_it_was",
    "test_poll_on_a_quiet_board_holds_its_position",
    "test_poll_returns_the_page_and_the_new_position",
    "test_prime_keeps_its_cursor_if_the_re_request_comes_back_short",
    "test_prime_on_an_empty_board_holds_no_cursor",
    "test_prime_re_requests_the_last_partial_page_to_reach_its_end",
    "test_priming_is_unfiltered_and_uses_the_largest_page",
    "test_subscription_arguments_carry_every_optional_filter",
    "test_subscription_arguments_filter_to_the_agent_both_ways",
]
