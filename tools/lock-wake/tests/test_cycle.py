"""The cycle: post before position, nothing swallowed, quiet paths honest."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import require_str
from platform_core.mcp_testing import (
    FakeHttpPost,
    announcing_poster,
    notes_sent,
    sent_arguments,
)

from lock_wake import _test_hooks
from lock_wake.cycle import run_cycle
from lock_wake.identity import BRIDGE_AGENT, HARNESS, IDENTITY, PURPOSE
from lock_wake.position import position_path, read_offset, write_offset
from tests.conftest import CONFIGURED_ENV, TASK_ID, journal_line, pin_env, stage_journal

COMPLETED_HOLD = (
    journal_line(ts="2026-09-09T19:28:00.0000000Z", kind="acquired")
    + journal_line(ts="2026-09-09T19:28:01.0000000Z", kind="step", detail="compose up")
    + journal_line(ts="2026-09-09T19:30:45.0000000Z", kind="released")
)


class TestRunCycle:
    def test_posts_the_digest_then_advances_the_offset(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, COMPLETED_HOLD.encode("utf-8"))
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        # The registering checkin is bodies[0] since 2026-09-21; the
        # digest is the post behind it. Its own shape is pinned by
        # TestLedgerRegistration below.
        arguments = notes_sent(poster)[0]
        assert arguments["taskId"] == TASK_ID
        assert arguments["agent"] == BRIDGE_AGENT
        body = require_str(arguments, "body")
        assert "FLEET-LOCK: 1 hold(s) transitioned" in body
        assert "RELEASED after 165s" in body
        assert read_offset(position_path(journal)) == len(COMPLETED_HOLD.encode("utf-8"))
        assert emitted == [
            "posted 1 hold(s) and 0 check run(s) from 3 line(s): tagged @opus-mosh-reboot-0909"
        ]

    def test_a_refused_post_leaves_the_offset_unmoved(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """At-least-once: the crash between post and mark repeats, never
        loses."""
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, COMPLETED_HOLD.encode("utf-8"))
        _test_hooks.http_post = FakeHttpPost(
            [{"status": 500, "content_type": "text/plain", "body": "board down"}]
        )

        with pytest.raises(AppError):
            run_cycle(journal)

        assert read_offset(position_path(journal)) == 0
        assert emitted == []

    def test_the_second_cycle_repeats_what_the_first_failed_to_mark(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, COMPLETED_HOLD.encode("utf-8"))
        _test_hooks.http_post = FakeHttpPost(
            [{"status": 500, "content_type": "text/plain", "body": "board down"}]
        )
        with pytest.raises(AppError):
            run_cycle(journal)

        retry_poster = announcing_poster()
        _test_hooks.http_post = retry_poster
        run_cycle(journal)

        assert len(notes_sent(retry_poster)) == 1
        assert read_offset(position_path(journal)) == len(COMPLETED_HOLD.encode("utf-8"))

    def test_a_quiet_journal_posts_nothing_and_says_so(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, COMPLETED_HOLD.encode("utf-8"))
        marks = position_path(journal)
        write_offset(marks, len(COMPLETED_HOLD.encode("utf-8")))
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        run_cycle(journal)

        assert poster.bodies == []
        assert emitted == [f"journal quiet; offset {len(COMPLETED_HOLD.encode('utf-8'))}"]

    def test_progress_only_lines_advance_without_a_post(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """The noise budget: a long build's step lines are not posts."""
        pin_env(CONFIGURED_ENV)
        content = (
            journal_line(ts="2026-09-09T19:28:00.0000000Z", kind="requested")
            + journal_line(ts="2026-09-09T19:28:01.0000000Z", kind="step", detail="build")
        ).encode("utf-8")
        journal = stage_journal(tmp_path, content)
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        run_cycle(journal)

        assert poster.bodies == []
        assert read_offset(position_path(journal)) == len(content)
        assert emitted == [f"2 progress line(s), no boundary; offset {len(content)}"]

    def test_an_unlabelled_hold_posts_unaddressed_and_says_so(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = (
            journal_line(ts="2026-09-09T19:28:00.0000000Z", kind="acquired", agent=None)
            + journal_line(ts="2026-09-09T19:28:05.0000000Z", kind="released", agent=None)
        ).encode("utf-8")
        journal = stage_journal(tmp_path, content)
        _test_hooks.http_post = announcing_poster()

        run_cycle(journal)

        assert emitted == ["posted 1 hold(s) and 0 check run(s) from 2 line(s): unaddressed"]

    def test_a_finished_check_run_posts_unaddressed(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = journal_line(
            ts="2026-09-24T05:00:00.000000Z",
            kind="checked",
            holder_pid=7,
            label="packages/claude-hooks",
            op="check-lock",
            only="",
            detail="PASSED exit 0 in 18s at 42988420",
            agent="opus-coordination-w2-0924",
        ).encode("utf-8")
        journal = stage_journal(tmp_path, content)
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        body = require_str(notes_sent(poster)[0], "body")
        assert "CHECKS: 1 make test run(s) finished" in body
        assert "@" not in body
        assert read_offset(position_path(journal)) == len(content)
        assert emitted == ["posted 0 hold(s) and 1 check run(s) from 1 line(s): unaddressed"]

    def test_missing_credentials_refuse_before_any_read(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env({})
        with pytest.raises(AppError):
            run_cycle(tmp_path / "never-read.jsonl")
        assert emitted == []


class TestLedgerRegistration:
    """MCPs mig 514 refuses a write from a session no ledger surface knows,
    and a service session is exactly one. All three wake bridges died on
    TASK_SESSION_UNLEDGERED from 2026-09-16 to 2026-09-21, every tick, with
    the traceback going only to runs/cycle.log."""

    def test_the_digest_is_preceded_by_a_registering_checkin(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, COMPLETED_HOLD.encode("utf-8"))
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        checkin = sent_arguments(poster.bodies[0])
        assert checkin["kind"] == "checkin"
        assert checkin["harness"] == HARNESS
        assert checkin["agent"] == BRIDGE_AGENT
        assert checkin["sessionId"] == IDENTITY["session_id"]
        # Board-level: a checkin carrying a taskId registers nothing while
        # looking like it had.
        assert "taskId" not in checkin
        assert PURPOSE in require_str(checkin, "body")

    def test_a_quiet_journal_registers_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """A checkin per three-minute tick would be 480 board posts a day
        from this bridge alone, so the registration rides the digest. The
        poster is scripted with no replies, which raises on the first
        call."""
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, b"")
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        run_cycle(journal)

        assert poster.bodies == []
