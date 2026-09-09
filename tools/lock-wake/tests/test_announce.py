"""The digest: one post per cycle, boundaries only, counts folded in."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from lock_wake.announce import Announcement, announcement, parse_journal_ts
from lock_wake.journal import LockEvent, decode_lock_event


def _event(
    *,
    ts: str,
    kind: str,
    holder_pid: int = 2688,
    label: str = "up-transcriber",
    op: str = "service-up",
    detail: str = "",
    agent: str = "opus-mosh-reboot-0909",
) -> LockEvent:
    """One event built through the production decoder, never by hand."""
    return decode_lock_event(
        {
            "ts": ts,
            "kind": kind,
            "pid": holder_pid,
            "label": label,
            "op": op,
            "only": "all",
            "detail": detail,
            "agent": agent,
        },
        1,
    )


def _required(events: Sequence[LockEvent]) -> Announcement:
    """The announcement a test expects to exist, or a named failure.

    Args:
        events: The slice under test.

    Returns:
        The post.

    Raises:
        AssertionError: When the slice produced no post -- the progress-only
            outcome, which the tests that expect it assert directly.
    """
    post = announcement(tuple(events))
    if post is None:
        raise AssertionError("expected an announcement, got the progress-only outcome")
    return post


class TestParseJournalTs:
    def test_reads_the_wrappers_seven_digit_form(self) -> None:
        parsed = parse_journal_ts("2026-09-09T19:28:00.0608673Z")
        assert (parsed.hour, parsed.minute, parsed.second, parsed.microsecond) == (
            19,
            28,
            0,
            60867,
        )

    def test_reads_a_fractionless_form(self) -> None:
        assert parse_journal_ts("2026-09-09T19:28:00Z").second == 0

    def test_refuses_a_timestamp_without_z(self) -> None:
        with pytest.raises(ValueError, match="does not end in Z"):
            parse_journal_ts("2026-09-09T19:28:00.0608673")


class TestAnnouncement:
    def test_a_completed_hold_reports_its_duration_steps_and_agent(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="acquired"),
            _event(ts="2026-09-09T19:28:00.1000000Z", kind="step", detail="compose up"),
            _event(ts="2026-09-09T19:30:45.0000000Z", kind="released"),
        )

        post = _required(events)
        assert post["holds"] == 1
        assert post["agents"] == ("opus-mosh-reboot-0909",)
        body = post["body"]
        assert "FLEET-LOCK: 1 hold(s) transitioned" in body
        assert "up-transcriber (service-up, pid 2688):" in body
        assert "acquired 19:28:00Z" in body
        assert "RELEASED after 165s" in body
        assert "+1 step(s) this window" in body
        assert "@opus-mosh-reboot-0909 your fleet-lock operation transitioned" in body

    def test_a_failure_carries_its_detail(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="acquired"),
            _event(
                ts="2026-09-09T19:29:00.0000000Z",
                kind="failed",
                detail="operation did not reach its last phase",
            ),
        )

        post = _required(events)
        assert "FAILED after 60s" in post["body"]
        assert "(operation did not reach its last phase)" in post["body"]

    def test_an_ending_without_its_acquire_reports_the_time_instead(self) -> None:
        # The acquire landed in an earlier slice; the duration is not
        # computable from this window and is not invented.
        events = (_event(ts="2026-09-09T19:30:45.0000000Z", kind="released"),)

        post = _required(events)
        assert "RELEASED 19:30:45Z" in post["body"]
        assert "after" not in post["body"]

    def test_progress_only_slices_are_not_posts(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="requested"),
            _event(ts="2026-09-09T19:28:01.0000000Z", kind="waiting", detail="pid=1 ..."),
            _event(ts="2026-09-09T19:28:02.0000000Z", kind="step", detail="build"),
        )
        assert announcement(events) is None

    def test_two_holds_share_one_post_and_deduplicated_mentions(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="acquired", holder_pid=1),
            _event(ts="2026-09-09T19:28:05.0000000Z", kind="released", holder_pid=1),
            _event(
                ts="2026-09-09T19:29:00.0000000Z",
                kind="acquired",
                holder_pid=2,
                label="up-wiki-check-mcp",
            ),
            _event(
                ts="2026-09-09T19:29:30.0000000Z",
                kind="released",
                holder_pid=2,
                label="up-wiki-check-mcp",
            ),
        )

        post = _required(events)
        assert post["holds"] == 2
        assert post["body"].count("FLEET-LOCK") == 1
        assert post["agents"] == ("opus-mosh-reboot-0909",)
        # Each hold line attributes its own agent; the MENTION is the
        # trailer, once, deduplicated.
        assert post["body"].splitlines()[-1] == (
            "@opus-mosh-reboot-0909 your fleet-lock operation transitioned"
        )

    def test_a_waiting_hold_beside_a_boundary_hold_is_counted_not_lined(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="acquired", holder_pid=1),
            _event(ts="2026-09-09T19:28:05.0000000Z", kind="released", holder_pid=1),
            _event(
                ts="2026-09-09T19:28:01.0000000Z",
                kind="waiting",
                holder_pid=9,
                label="deploy-ts",
                agent="",
            ),
        )

        post = _required(events)
        assert post["holds"] == 1
        assert "deploy-ts" not in post["body"]

    def test_unlabelled_holds_make_an_unaddressed_post(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="acquired", agent=""),
            _event(ts="2026-09-09T19:28:05.0000000Z", kind="released", agent=""),
        )

        post = _required(events)
        assert post["agents"] == ()
        assert "@" not in post["body"]

    def test_waits_are_folded_as_counts(self) -> None:
        events = (
            _event(ts="2026-09-09T19:28:00.0000000Z", kind="requested"),
            _event(ts="2026-09-09T19:28:01.0000000Z", kind="waiting"),
            _event(ts="2026-09-09T19:28:31.0000000Z", kind="waiting"),
            _event(ts="2026-09-09T19:29:00.0000000Z", kind="acquired"),
        )

        post = _required(events)
        assert "+2 wait(s)" in post["body"]
