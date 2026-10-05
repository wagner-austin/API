"""The remote read: one ssh, the unread window only, and every refusal named.

The journal served is a real file in diphtheria's byte form (six fractional
digits, a deploy's label and op), answered by :class:`tests.conftest.FakeSsh`
the way ``stat -c %s`` and ``tail -c +N`` answer on the host.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.error_codes_tooling import LockWakeErrorCode
from platform_core.errors import AppError

from lock_wake.remote import (
    SSH_TIMEOUT_SECONDS,
    RemoteJournal,
    parse_remote_journal,
    read_remote_slice,
    remote_cursor_path,
    window_command,
)
from tests.conftest import FakeSsh, SshReply, install_ssh, journal_line

#: Where the journal lives on diphtheria, as the pump's row names it.
REMOTE_PATH: Final = "/home/corvis/PROJECTS/MCPs/.fleet-events.jsonl"

DIPHTHERIA: Final[RemoteJournal] = {"host": "diphtheria", "path": REMOTE_PATH}

#: A deploy's step and release, in the form diphtheria's lock writes them.
DEPLOY_STEP: Final = journal_line(
    ts="2026-10-05T05:01:31.638973Z",
    kind="step",
    holder_pid=1637266,
    label="deploy",
    op="deploy",
    detail="build [only=all]",
    agent="opus-coordination-w14-1005",
)
DEPLOY_RELEASED: Final = journal_line(
    ts="2026-10-05T05:20:02.004117Z",
    kind="released",
    holder_pid=1637266,
    label="deploy",
    op="deploy",
    agent="opus-coordination-w14-1005",
)


def _serve(tmp_path: pathlib.Path, content: bytes) -> FakeSsh:
    """Stage diphtheria's journal and bind an ssh that serves it.

    Args:
        tmp_path: The test's temporary directory.
        content: The remote journal's bytes.

    Returns:
        The bound fake.
    """
    local = tmp_path / "diphtheria.fleet-events.jsonl"
    local.write_bytes(content)
    fake = FakeSsh("diphtheria", {REMOTE_PATH: local}, None)
    install_ssh(fake)
    return fake


class TestParseRemoteJournal:
    def test_reads_host_and_absolute_path(self) -> None:
        assert parse_remote_journal(f"diphtheria:{REMOTE_PATH}") == DIPHTHERIA

    @pytest.mark.parametrize(
        "value",
        [
            REMOTE_PATH,
            "diphtheria:relative/.fleet-events.jsonl",
            "diphtheria:/home/corvis/my journal.jsonl",
            "diphtheria:/home/corvis/$(reboot)",
            "-oProxyCommand=x:/home/corvis/j.jsonl",
            ":/home/corvis/j.jsonl",
        ],
    )
    def test_refuses_anything_the_remote_shell_would_need_quoted(self, value: str) -> None:
        with pytest.raises(ValueError, match="is not <host>:<absolute path>"):
            parse_remote_journal(value)


class TestRemoteCursorPath:
    def test_lives_in_the_named_directory_per_host(self, tmp_path: pathlib.Path) -> None:
        assert remote_cursor_path(tmp_path, DIPHTHERIA) == (
            tmp_path / ".fleet-events.jsonl.lock-wake-diphtheria-offset.json"
        )


class TestWindowCommand:
    def test_prints_the_size_then_tails_from_the_next_byte(self) -> None:
        assert window_command(REMOTE_PATH, 41) == (
            f"stat -c %s -- {REMOTE_PATH} && tail -c +42 -- {REMOTE_PATH}"
        )


class TestReadRemoteSlice:
    def test_decodes_every_complete_line_past_the_offset(self, tmp_path: pathlib.Path) -> None:
        step = DEPLOY_STEP.encode("utf-8")
        released = DEPLOY_RELEASED.encode("utf-8")
        fake = _serve(tmp_path, step + released)

        result = read_remote_slice(DIPHTHERIA, len(step))

        assert [event["kind"] for event in result["events"]] == ["released"]
        assert result["events"][0]["agent"] == "opus-coordination-w14-1005"
        assert result["next_offset"] == len(step) + len(released)
        ((argv, timeout),) = fake.calls
        assert argv[-1] == window_command(REMOTE_PATH, len(step))
        assert timeout == SSH_TIMEOUT_SECONDS

    def test_leaves_a_torn_tail_for_the_next_tick(self, tmp_path: pathlib.Path) -> None:
        step = DEPLOY_STEP.encode("utf-8")
        _serve(tmp_path, step + b'{"ts":"2026-10-05T05:2')

        result = read_remote_slice(DIPHTHERIA, 0)

        assert len(result["events"]) == 1
        assert result["next_offset"] == len(step)

    def test_an_offset_at_the_end_reads_nothing(self, tmp_path: pathlib.Path) -> None:
        step = DEPLOY_STEP.encode("utf-8")
        _serve(tmp_path, step)

        assert read_remote_slice(DIPHTHERIA, len(step)) == {
            "events": (),
            "next_offset": len(step),
        }

    def test_a_position_past_the_end_refuses(self, tmp_path: pathlib.Path) -> None:
        _serve(tmp_path, DEPLOY_STEP.encode("utf-8"))

        with pytest.raises(ValueError, match=r"diphtheria:/home/.*truncated or replaced"):
            read_remote_slice(DIPHTHERIA, 10_000)

    def test_an_ssh_failure_names_the_host_and_carries_its_stderr(self) -> None:
        install_ssh(
            FakeSsh(
                "diphtheria",
                {},
                SshReply(255, b"", b"ssh: connect to host 100.119.95.50 port 22: timed out\n"),
            )
        )

        with pytest.raises(AppError) as caught:
            read_remote_slice(DIPHTHERIA, 0)

        assert caught.value.code is LockWakeErrorCode.REMOTE_JOURNAL_UNREADABLE
        assert str(caught.value) == (
            f"ssh diphtheria exited 255 reading {REMOTE_PATH}: "
            f"ssh: connect to host 100.119.95.50 port 22: timed out"
        )

    @pytest.mark.parametrize("stdout", [b"", b"596416", b"Welcome to Debian\n596416\n"])
    def test_output_without_the_size_line_is_malformed(self, stdout: bytes) -> None:
        install_ssh(FakeSsh("diphtheria", {}, SshReply(0, stdout, b"")))

        with pytest.raises(AppError) as caught:
            read_remote_slice(DIPHTHERIA, 0)

        assert caught.value.code is LockWakeErrorCode.REMOTE_JOURNAL_MALFORMED
