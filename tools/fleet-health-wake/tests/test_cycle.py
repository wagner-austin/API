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

from fleet_health_wake import _test_hooks
from fleet_health_wake.cycle import run_cycle
from fleet_health_wake.identity import BRIDGE_AGENT, HARNESS, IDENTITY, PURPOSE
from tests.conftest import (
    CONFIGURED_ENV,
    REFUSED_BODY,
    REFUSED_LINE,
    TASK_ID,
    TRANSITIONS_BODY,
    TRANSITIONS_LINE,
    offset_of,
    pin_env,
    set_offset,
    stage_journal,
)


class TestRunCycle:
    def test_posts_every_unread_line_as_one_note_then_advances_the_offset(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = (TRANSITIONS_LINE + REFUSED_LINE).encode("utf-8")
        journal = stage_journal(tmp_path, content)
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        [note] = notes_sent(poster)
        assert note["taskId"] == TASK_ID
        assert note["agent"] == BRIDGE_AGENT
        assert note["kind"] == "note"
        assert require_str(note, "body") == TRANSITIONS_BODY + "\n\n" + REFUSED_BODY
        assert offset_of(journal) == len(content)
        assert emitted == [f"posted 2 health line(s) (transitions, refused); offset {len(content)}"]

    def test_registers_on_the_session_ledger_before_posting(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, TRANSITIONS_LINE.encode("utf-8"))
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        checkin = sent_arguments(poster.bodies[0])
        assert checkin["kind"] == "checkin"
        assert checkin["sessionId"] == IDENTITY["session_id"]
        assert checkin["harness"] == HARNESS
        assert PURPOSE in require_str(checkin, "body")
        assert len(emitted) == 1

    def test_posts_only_what_is_past_the_recorded_offset(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        first = TRANSITIONS_LINE.encode("utf-8")
        journal = stage_journal(tmp_path, first + REFUSED_LINE.encode("utf-8"))
        set_offset(journal, len(first))
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(journal)

        assert [require_str(n, "body") for n in notes_sent(poster)] == [REFUSED_BODY]
        assert emitted[0].startswith("posted 1 health line(s) (refused)")

    def test_a_refused_post_leaves_the_offset_unmoved_and_the_retry_posts_it(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """At-least-once: the failure between post and mark repeats, never loses."""
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, TRANSITIONS_LINE.encode("utf-8"))
        _test_hooks.http_post = FakeHttpPost(
            [{"status": 500, "content_type": "text/plain", "body": "board down"}]
        )
        with pytest.raises(AppError):
            run_cycle(journal)
        assert offset_of(journal) == 0
        assert emitted == []

        retry = announcing_poster()
        _test_hooks.http_post = retry
        run_cycle(journal)
        assert [require_str(n, "body") for n in notes_sent(retry)] == [TRANSITIONS_BODY]
        assert offset_of(journal) == len(TRANSITIONS_LINE.encode("utf-8"))

    def test_a_quiet_journal_posts_nothing_and_says_so(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        content = TRANSITIONS_LINE.encode("utf-8")
        journal = stage_journal(tmp_path, content)
        set_offset(journal, len(content))
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        run_cycle(journal)

        assert poster.bodies == []
        assert emitted == [f"health journal quiet; offset {len(content)}"]

    def test_a_torn_tail_is_left_for_the_next_cycle(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        journal = stage_journal(tmp_path, REFUSED_LINE.encode("utf-8")[:40])
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(journal)

        assert offset_of(journal) == 0
        assert emitted == ["health journal quiet; offset 0"]

    def test_an_unset_task_id_refuses_before_reading_anything(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        env = dict(CONFIGURED_ENV)
        del env["FLEET_HEALTH_WAKE_TASK_ID"]
        pin_env(env)
        with pytest.raises(AppError, match="FLEET_HEALTH_WAKE_TASK_ID"):
            run_cycle(stage_journal(tmp_path, TRANSITIONS_LINE.encode("utf-8")))
        assert emitted == []
