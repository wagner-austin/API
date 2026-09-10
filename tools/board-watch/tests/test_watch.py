"""Cursor arithmetic, priming, and one poll.

The assertions that matter here are about NOT moving: an empty page must
leave the position where it was. Getting that backwards is what makes a
watcher replay its history on every quiet poll while reporting success.

The cursor cases are driven through :func:`poll` and :func:`prime` rather
than through a pure helper, because the decision is only ever made on a real
response and a helper tested in isolation can agree with a contract the
server no longer has -- which is what happened here before ``edfa06ec``.
"""

from __future__ import annotations

import pytest
from platform_core.error_codes_tooling import BoardWatchErrorCode
from platform_core.errors import AppError

from board_watch import _test_hooks
from board_watch.contracts import decode_event_line
from board_watch.watch import (
    MAX_LIMIT,
    SubscriptionSpec,
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

SPEC = SubscriptionSpec(
    agent="opus-nclex-licensure-0904",
    session_id="55555555-5555-4555-8555-555555555555",
    cwd="C:/Users/Test/PROJECTS/MCPs",
    room=None,
    kind=None,
    limit=50,
)


def test_an_empty_page_from_no_cursor_stays_at_no_cursor() -> None:
    """A watcher that has never held a cursor must not invent one.

    Distinct from the quiet-board case below: there the held position is a
    real token, here it is None, and the two are different states of the
    ``str | None`` this package threads through every call.
    """
    _test_hooks.http_post = FakeHttpPost([ok(tool_text(page_text([], None)))])
    _, moved = poll(TEST_CREDENTIALS, SPEC, None)
    assert moved is None


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
    spec = SubscriptionSpec(
        agent="a-b-c",
        session_id="66666666-6666-4666-8666-666666666666",
        cwd="/tmp",
        room="main",
        kind="status_change",
        limit=7,
    )
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
    arguments = priming_arguments(SPEC, None)
    # Identity rides along because the read tool requires it, but NO
    # selector does -- so this walk names nobody's queue and advances no
    # receipt. Asserting the selectors are absent is the whole point:
    # a primed watcher must not have marked its session caught up on
    # mentions it never showed anyone.
    assert arguments["limit"] == MAX_LIMIT
    assert arguments["agent"] == SPEC["agent"]
    assert arguments["sessionId"] == SPEC["session_id"]
    assert arguments["cwd"] == SPEC["cwd"]
    assert "mentionsAgent" not in arguments
    assert "relevantToAgent" not in arguments
    assert "cursor" not in arguments
    assert priming_arguments(SPEC, "mid")["cursor"] == "mid"


def test_subscription_arguments_carry_the_polled_sessions_identity() -> None:
    """The receipt names who was served, so the poll must say who that is.

    A watcher polling with its OWN invented identity would either be
    refused -- one label is bound to one session -- or, worse, advance a
    receipt under a session that does not exist.
    """
    arguments = subscription_arguments(SPEC, None)
    assert arguments["agent"] == SPEC["agent"]
    assert arguments["sessionId"] == SPEC["session_id"]
    assert arguments["cwd"] == SPEC["cwd"]


def test_prime_follows_each_pages_cursor_until_the_board_offers_none() -> None:
    """The walk ends at the empty page, and every step carries the cursor forward.

    Asserting the ARGUMENTS, not just the result: a walk that reached the
    right answer while re-sending the same cursor would pass a return-value
    check and loop forever against a live board.
    """
    poster = FakeHttpPost(
        [
            ok(tool_text(page_text([LIVE_CHECKIN_LINE], "first"))),
            ok(tool_text(page_text([LIVE_TASK_LINE, LIVE_MENTION_LINE], "true-end"))),
            ok(tool_text(page_text([], None))),
        ]
    )
    _test_hooks.http_post = poster
    assert prime(TEST_CREDENTIALS, SPEC) == "true-end"
    # Identity is on every page of the walk because the read tool
    # requires it; the CURSOR is what this test is about, so each page is
    # checked for the position it followed rather than for whole-object
    # equality that would break again on the next required field.
    identity = {
        "agent": SPEC["agent"],
        "sessionId": SPEC["session_id"],
        "cwd": SPEC["cwd"],
    }
    assert sent_arguments(poster.bodies[0]) == {"limit": MAX_LIMIT, **identity}
    assert sent_arguments(poster.bodies[1]) == {
        "limit": MAX_LIMIT,
        "cursor": "first",
        **identity,
    }
    assert sent_arguments(poster.bodies[2]) == {
        "limit": MAX_LIMIT,
        "cursor": "true-end",
        **identity,
    }


def test_prime_on_an_empty_board_holds_no_cursor() -> None:
    """A board with no events has no position to hold, and that is not an error."""
    poster = FakeHttpPost([ok(tool_text(page_text([], None)))])
    _test_hooks.http_post = poster
    assert prime(TEST_CREDENTIALS, SPEC) is None
    assert len(poster.bodies) == 1


def test_prime_refuses_a_page_that_carries_rows_but_no_cursor() -> None:
    """The server contract forbids it, and tolerating it reinstates the old bug.

    Before ``edfa06ec`` this was the NORMAL response for a short page, and
    this package worked around it. Now it can only mean the server has
    regressed -- and returning the held cursor here would leave the watcher
    permanently short of the feed's end, re-announcing those rows on every
    poll while reporting success. That is the exact failure the package
    exists to make impossible, so it raises instead.
    """
    _test_hooks.http_post = FakeHttpPost(
        [ok(tool_text(page_text([LIVE_CHECKIN_LINE, LIVE_TASK_LINE], None)))]
    )
    with pytest.raises(AppError) as caught:
        prime(TEST_CREDENTIALS, SPEC)
    assert caught.value.code is BoardWatchErrorCode.PAGE_WITHOUT_CURSOR
    assert "2 events with no next cursor" in caught.value.message


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
    "test_a_notification_collapses_a_multi_line_summary",
    "test_a_notification_leads_with_who_wants_you_and_why",
    "test_a_notification_omits_the_task_when_there_is_none",
    "test_a_notification_reports_a_truncated_body",
    "test_an_empty_page_from_no_cursor_stays_at_no_cursor",
    "test_poll_on_a_quiet_board_holds_its_position",
    "test_poll_returns_the_page_and_the_new_position",
    "test_prime_follows_each_pages_cursor_until_the_board_offers_none",
    "test_prime_on_an_empty_board_holds_no_cursor",
    "test_prime_refuses_a_page_that_carries_rows_but_no_cursor",
    "test_priming_is_unfiltered_and_uses_the_largest_page",
    "test_subscription_arguments_carry_every_optional_filter",
    "test_subscription_arguments_filter_to_the_agent_both_ways",
]
