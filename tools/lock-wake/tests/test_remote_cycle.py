"""diphtheria's journal, driven through the CLI in the pump row's own form.

MCPs board task 03590bf9: every deploy takes its fleet lock on diphtheria,
and until this row nothing read that journal. These tests run
:func:`lock_wake.cli.wake.main` with the flags the hub's pump passes
(``tools/hpc-wake/scripts/run_cycle.py``, the ``lock-wake-diphtheria``
row), against a journal staged in diphtheria's byte form and served by
:class:`tests.conftest.FakeSsh`, and assert the post the standing task gets.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.errors import AppError
from platform_core.journal_cursor import (
    file_is_present,
    read_file_bytes,
    read_offset,
    write_file_text,
    write_offset,
)
from platform_core.json_utils import require_str
from platform_core.mcp_testing import FakeHttpPost, announcing_poster, notes_sent

from lock_wake import _test_hooks
from lock_wake.cli import wake
from lock_wake.identity import BRIDGE_AGENT
from lock_wake.remote import parse_remote_journal, remote_cursor_path
from tests.conftest import CONFIGURED_ENV, TASK_ID, FakeSsh, install_ssh, journal_line, pin_env

#: The pump row's journal flag value, verbatim.
REMOTE_JOURNAL: Final = "diphtheria:/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl"

_HOLDER: Final = "opus-coordination-w79-1004"


def _deploy_line(ts: str, kind: str, detail: str = "") -> str:
    """One line of a deploy's hold, as diphtheria's lock writes it.

    Args:
        ts: The timestamp, six fractional digits as diphtheria writes them.
        kind: The transition.
        detail: Kind-specific text.

    Returns:
        The line.
    """
    return journal_line(
        ts=ts,
        kind=kind,
        holder_pid=1544021,
        label="deploy",
        op="deploy",
        detail=detail,
        agent=_HOLDER,
    )


#: The 02:59:15Z release the task measured, with its acquisition and a step.
RELEASED_DEPLOY: Final = (
    _deploy_line("2026-10-05T02:41:07.120533Z", "acquired")
    + _deploy_line("2026-10-05T02:41:09.000210Z", "step", "build bases")
    + _deploy_line("2026-10-05T02:59:15.430001Z", "released")
).encode("utf-8")


def _stage(tmp_path: pathlib.Path, content: bytes) -> tuple[FakeSsh, pathlib.Path]:
    """Stage diphtheria's journal, bind the ssh serving it, name the cursor dir.

    Args:
        tmp_path: The test's temporary directory.
        content: The remote journal's bytes.

    Returns:
        The fake ssh and the local cursor directory.
    """
    local = tmp_path / "remote" / "fleet-events.jsonl"
    local.parent.mkdir()
    local.write_bytes(content)
    fake = FakeSsh("diphtheria", {"/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl": local}, None)
    install_ssh(fake)
    cursor_dir = tmp_path / "MCPs"
    cursor_dir.mkdir()
    return fake, cursor_dir


def _argv(cursor_dir: pathlib.Path) -> list[str]:
    """The pump row's arguments after ``lock-wake``, cursor dir aside.

    Args:
        cursor_dir: The local cursor directory.

    Returns:
        The argv.
    """
    return ["--remote-journal", REMOTE_JOURNAL, "--cursor-dir", str(cursor_dir)]


def _offset(cursor_dir: pathlib.Path) -> int:
    """This bridge's recorded position in diphtheria's journal.

    Args:
        cursor_dir: The local cursor directory.

    Returns:
        The offset, 0 when never written.
    """
    marks = remote_cursor_path(cursor_dir, parse_remote_journal(REMOTE_JOURNAL))
    return read_offset(file_is_present, read_file_bytes, marks)


class TestDiphtheriaRow:
    def test_a_deploys_release_is_posted_naming_its_holder(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _, cursor_dir = _stage(tmp_path, RELEASED_DEPLOY)
        poster = announcing_poster()
        _test_hooks.http_post = poster

        assert wake.main(_argv(cursor_dir)) == 0

        (note,) = notes_sent(poster)
        assert note["taskId"] == TASK_ID
        assert note["agent"] == BRIDGE_AGENT
        assert require_str(note, "body").splitlines() == [
            "FLEET-LOCK on diphtheria: 1 hold(s) transitioned",
            "deploy (deploy, pid 1544021): acquired 02:41:07Z RELEASED after 1088s "
            f"+1 step(s) this window by @{_HOLDER}",
            f"@{_HOLDER} your fleet-lock operation on diphtheria transitioned",
        ]
        assert _offset(cursor_dir) == len(RELEASED_DEPLOY)
        assert emitted == [
            f"diphtheria: posted 1 hold(s) and 0 check run(s) from 3 line(s): tagged @{_HOLDER}"
        ]

    def test_a_failed_deploy_is_posted_from_the_cursor_on(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        failed = _deploy_line("2026-10-05T03:10:00.000001Z", "failed", "build exited 1")
        content = RELEASED_DEPLOY + failed.encode("utf-8")
        _, cursor_dir = _stage(tmp_path, content)
        marks = remote_cursor_path(cursor_dir, parse_remote_journal(REMOTE_JOURNAL))
        write_offset(write_file_text, marks, len(RELEASED_DEPLOY))
        pin_env(CONFIGURED_ENV)
        poster = announcing_poster()
        _test_hooks.http_post = poster

        wake.main(_argv(cursor_dir))

        body = require_str(notes_sent(poster)[0], "body")
        assert "FAILED 03:10:00Z (build exited 1)" in body
        assert f"by @{_HOLDER}" in body
        assert _offset(cursor_dir) == len(content)

    def test_a_quiet_journal_posts_nothing_and_says_so(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        _, cursor_dir = _stage(tmp_path, RELEASED_DEPLOY)
        marks = remote_cursor_path(cursor_dir, parse_remote_journal(REMOTE_JOURNAL))
        write_offset(write_file_text, marks, len(RELEASED_DEPLOY))
        pin_env(CONFIGURED_ENV)
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        wake.main(_argv(cursor_dir))

        assert poster.bodies == []
        assert emitted == [f"diphtheria journals quiet; offsets {len(RELEASED_DEPLOY)}"]

    def test_a_refused_post_leaves_the_cursor_unmoved(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _, cursor_dir = _stage(tmp_path, RELEASED_DEPLOY)
        _test_hooks.http_post = FakeHttpPost(
            [{"status": 500, "content_type": "text/plain", "body": "board down"}]
        )

        with pytest.raises(AppError):
            wake.main(_argv(cursor_dir))

        assert _offset(cursor_dir) == 0
        assert emitted == []

    def test_missing_credentials_refuse_before_any_ssh(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env({})
        fake, cursor_dir = _stage(tmp_path, RELEASED_DEPLOY)

        with pytest.raises(AppError):
            wake.main(_argv(cursor_dir))

        assert fake.calls == []
        assert emitted == []
