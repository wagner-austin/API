"""The rendered-text grammar, pinned against bytes the live board produced.

Every assertion here is about a contract owned by another repository. If
``taskboard-mcp`` changes how it renders a row or a footer, this file is
where that shows up -- which is the point, because the alternative is a
watcher that parses nothing and reports quiet.
"""

from __future__ import annotations

import pytest
from platform_core.error_codes import BoardWatchErrorCode
from platform_core.errors import AppError

from board_watch.contracts import (
    EMPTY_FEED_SENTINEL,
    decode_event_line,
    decode_event_page,
    decode_footer,
    encode_event_line,
)
from tests.conftest import (
    LIVE_CHECKIN_LINE,
    LIVE_MENTION_LINE,
    LIVE_MULTILINE_ROW,
    LIVE_TASK_LINE,
    page_text,
)


def test_decodes_a_real_row_with_mentions_and_truncation() -> None:
    """The captured line exercises every optional element at once."""
    event = decode_event_line(LIVE_MENTION_LINE)
    assert event["created_at"] == "2026-09-04T22:57:47.747Z"
    assert event["kind"] == "note"
    assert event["author"] == "opus-lavender-gpu-0824"
    assert event["task_id"] is None
    assert event["mentions"] == (
        "all-sessions",
        "opus-nclex-licensure-0904",
        "opus-artifact-sweep-0902",
    )
    assert event["summary"].startswith("@opus-nclex-licensure-0904 URGENT")
    assert event["omitted_chars"] == 3149


def test_decodes_a_row_with_no_mentions_and_no_truncation() -> None:
    """The optional groups are genuinely optional, not defaulted."""
    event = decode_event_line(LIVE_CHECKIN_LINE)
    assert event["kind"] == "checkin"
    assert event["mentions"] == ()
    assert event["task_id"] is None
    assert event["omitted_chars"] == 0
    assert event["summary"] == "LANDING fleet NOW"


def test_decodes_a_task_scoped_row() -> None:
    """A thread post carries the task id between the author and the mentions."""
    event = decode_event_line(LIVE_TASK_LINE)
    assert event["task_id"] == "8793517e-5c6b-4edd-a127-0234b40404d4"
    assert event["author"] == "fable-brain-audit-0903"
    assert event["mentions"] == ("opus-nclex-licensure-0904",)
    assert event["summary"] == "claimed -> done"


@pytest.mark.parametrize(
    "line",
    [LIVE_MENTION_LINE, LIVE_CHECKIN_LINE, LIVE_TASK_LINE],
)
def test_round_trips_every_captured_row(line: str) -> None:
    """Re-rendering a decoded row reproduces it exactly.

    This is the assertion that catches a field being parsed and then
    silently dropped, which no field-by-field check can: a decoder that
    ignored ``task_id`` would satisfy every other test in this file.
    """
    assert encode_event_line(decode_event_line(line)) == line


def test_decodes_a_row_whose_summary_spans_several_lines() -> None:
    """A row is delimited by the timestamp anchor, not by a newline.

    This is the case that broke the first live run: every other fixture here
    is single-line, so a decoder splitting on ``\\n`` passed the whole suite
    and then rejected a real page at its second physical line.
    """
    event = decode_event_line(LIVE_MULTILINE_ROW)
    assert event["author"] == "opus-portaclaude-0815"
    assert event["mentions"] == ("100",)
    assert "\n" in event["summary"]
    assert event["summary"].endswith("Done this session:")


def test_a_multi_line_row_round_trips() -> None:
    """Newlines inside a summary survive decoding, so nothing is lost."""
    assert encode_event_line(decode_event_line(LIVE_MULTILINE_ROW)) == LIVE_MULTILINE_ROW


def test_a_page_separates_rows_on_the_anchor_not_the_newline() -> None:
    """Two rows, one of them multi-line, must decode as exactly two events."""
    page = decode_event_page(page_text([LIVE_MULTILINE_ROW, LIVE_CHECKIN_LINE], None))
    assert page["count"] == 2
    assert page["events"][0]["author"] == "opus-portaclaude-0815"
    assert page["events"][1]["author"] == "opus-fleet-mcp-0904"


def test_a_body_that_does_not_begin_with_a_row_is_rejected() -> None:
    """Without a first anchor there is no boundary to split on at all."""
    with pytest.raises(AppError) as raised:
        decode_event_page("orphaned continuation text\n\n[showing 1 events]")
    assert raised.value.code is BoardWatchErrorCode.EVENT_LINE_MALFORMED


def test_a_line_that_is_not_a_row_is_rejected_by_code() -> None:
    """The code names the element, so a reader knows which half moved."""
    with pytest.raises(AppError) as raised:
        decode_event_line("not a board event at all")
    assert raised.value.code is BoardWatchErrorCode.EVENT_LINE_MALFORMED
    assert "not a board event at all" in raised.value.message


def test_footer_without_a_cursor_reports_none() -> None:
    """A short page is the board saying the caller has caught up."""
    assert decode_footer("[showing 3 events]") == (3, None)


def test_footer_with_a_cursor_reports_it() -> None:
    """The rendered spelling is ``next cursor:``, not ``nextCursor``.

    A watcher that looked for the documented FIELD name rather than the
    rendered form matched nothing and replayed its history forever. That is
    the defect this assertion exists to prevent recurring.
    """
    assert decode_footer("[showing 200 events; next cursor: abc123==]") == (200, "abc123==")


def test_a_line_that_is_not_a_footer_is_rejected_by_code() -> None:
    """Including the near-miss spelling, which is the one that actually happened."""
    with pytest.raises(AppError) as raised:
        decode_footer("[showing 5 events; nextCursor: abc]")
    assert raised.value.code is BoardWatchErrorCode.FOOTER_MALFORMED


def test_decodes_a_whole_page() -> None:
    """Rows plus a blank line plus a footer is the whole response shape."""
    page = decode_event_page(page_text([LIVE_CHECKIN_LINE, LIVE_TASK_LINE], "cursor-1"))
    assert page["count"] == 2
    assert page["next_cursor"] == "cursor-1"
    assert page["events"][0]["author"] == "opus-fleet-mcp-0904"
    assert page["events"][1]["author"] == "fable-brain-audit-0903"


def test_decodes_an_empty_page() -> None:
    """The sentinel replaces the rows and is not parsed as one."""
    page = decode_event_page(page_text([], None))
    assert page["events"] == ()
    assert page["count"] == 0
    assert page["next_cursor"] is None


def test_the_empty_sentinel_is_the_string_the_board_sends() -> None:
    """Pinned separately because it carries a non-ASCII dash.

    An em dash typed by hand instead of copied would make the sentinel never
    match, and an unmatched sentinel is parsed as an event row -- so a quiet
    board would raise a malformed-line error on every poll.
    """
    assert EMPTY_FEED_SENTINEL == "no board events after this cursor — you are caught up"


def test_a_body_with_no_footer_is_rejected_by_code() -> None:
    """An empty body has no footer to read, and that is its own failure."""
    with pytest.raises(AppError) as raised:
        decode_event_page("   \n\n  ")
    assert raised.value.code is BoardWatchErrorCode.FOOTER_MISSING


def test_a_footer_disagreeing_with_the_rows_is_rejected() -> None:
    """The count and the rows come from different functions on the server.

    Trusting either one alone would hide a divergence between them, and the
    count is what a caller would naturally believe.
    """
    with pytest.raises(AppError) as raised:
        decode_event_page(f"{LIVE_CHECKIN_LINE}\n\n[showing 7 events]")
    assert raised.value.code is BoardWatchErrorCode.FOOTER_MALFORMED
    assert "reports 7 events but 1 rows" in raised.value.message


__all__ = [
    "test_a_body_that_does_not_begin_with_a_row_is_rejected",
    "test_a_body_with_no_footer_is_rejected_by_code",
    "test_a_footer_disagreeing_with_the_rows_is_rejected",
    "test_a_line_that_is_not_a_footer_is_rejected_by_code",
    "test_a_line_that_is_not_a_row_is_rejected_by_code",
    "test_a_multi_line_row_round_trips",
    "test_a_page_separates_rows_on_the_anchor_not_the_newline",
    "test_decodes_a_real_row_with_mentions_and_truncation",
    "test_decodes_a_row_whose_summary_spans_several_lines",
    "test_decodes_a_row_with_no_mentions_and_no_truncation",
    "test_decodes_a_task_scoped_row",
    "test_decodes_a_whole_page",
    "test_decodes_an_empty_page",
    "test_footer_with_a_cursor_reports_it",
    "test_footer_without_a_cursor_reports_none",
    "test_round_trips_every_captured_row",
    "test_the_empty_sentinel_is_the_string_the_board_sends",
]
