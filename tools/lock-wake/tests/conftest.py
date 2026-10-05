"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a
mock, matching the three sibling bridges' conventions. The seams rebound
are this package's own (``lock_wake._test_hooks``) and the one sanctioned
environment reader (``platform_core.config.config_test_hooks.get_env``,
which also feeds ``board_watch.config.load_credentials``, so pinning it
configures the whole credential chain from one place).

THE JOURNAL AND THE POSITION ARE REAL FILES UNDER ``tmp_path``. The
journal fixture writes lines byte-for-byte in the lock wrapper's own form
(compact JSON, ``\\n``-terminated, seven-digit fractional timestamps), so
these tests read rows shaped the way production shapes them rather than
rows a fixture invented -- including the torn tail and the pre-``agent``
history rows, which are the cases that decide whether the cursor is
correct.

The MCP poster fake is :class:`platform_core.mcp_testing.FakeHttpPost`,
shared with every sibling's suite rather than copied a fifth time.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Sequence
from typing import Final

import pytest
from platform_core.config import config_test_hooks
from platform_core.journal_cursor import (
    cursor_path,
    file_is_present,
    read_file_bytes,
    read_offset,
    write_file_text,
    write_offset,
)
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from lock_wake import _test_hooks
from lock_wake.identity import CURSOR_READER, TASK_ID_VARIABLE
from lock_wake.remote import SSH_OPTIONS

#: The standing task id every configured test posts into.
TASK_ID: Final = "0b892f1e-0000-4000-8000-00000000c0de"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    # The override, so no test reads the MCPs checkout's endpoint declaration
    # (board-watch's own suite tests that default).
    "BOARD_WATCH_URL": DECLARED_TASKBOARD_URL,
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}


def journal_line(
    *,
    ts: str = "2026-09-09T19:28:00.0608673Z",
    kind: str = "acquired",
    holder_pid: int = 2688,
    label: str = "up-transcriber",
    op: str = "service-up",
    only: str = "all",
    detail: str = "",
    agent: str | None = "opus-mosh-reboot-0909",
) -> str:
    """One journal line in the lock wrapper's own byte form.

    Args:
        ts: The transition's timestamp, wrapper-form (seven fractional
            digits, trailing ``Z``).
        kind: The transition kind.
        holder_pid: The lock-taking process id (the journal's ``pid``).
        label: The make target's label.
        op: The wrapper operation.
        only: The language scope.
        detail: Kind-specific text.
        agent: The session label, or None for a pre-2026-09-09 history row
            that lacks the key entirely.

    Returns:
        The line, ``\\n``-terminated.
    """
    agent_part = "" if agent is None else f',"agent":"{agent}"'
    return (
        f'{{"ts":"{ts}","kind":"{kind}","pid":{holder_pid},"label":"{label}",'
        f'"op":"{op}","only":"{only}","detail":"{detail}"{agent_part}}}\n'
    )


def stage_journal(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write a journal file with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The journal's full contents, torn tails included.

    Returns:
        The journal's path.
    """
    journal = tmp_path / ".fleet-events.jsonl"
    journal.write_bytes(content)
    return journal


def offset_of(journal: pathlib.Path) -> int:
    """This bridge's recorded position in a journal, as the cycle reads it.

    Args:
        journal: The journal's path.

    Returns:
        The offset in its cursor file, 0 when the file was never written.
    """
    return read_offset(file_is_present, read_file_bytes, cursor_path(journal, CURSOR_READER))


def set_offset(journal: pathlib.Path, offset: int) -> None:
    """Record this bridge's position in a journal, as the cycle writes it.

    Args:
        journal: The journal's path.
        offset: The offset to record.
    """
    write_offset(write_file_text, cursor_path(journal, CURSOR_READER), offset)


def check_journal_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Where the check journal lives for a test: beside the fleet journal.

    Absent until :func:`stage_check_journal` writes it, which is the state
    of a clone where no locked ``make test`` has finished yet.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The check journal's path.
    """
    return tmp_path / ".check-events.jsonl"


def stage_check_journal(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write the check journal with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The journal's full contents.

    Returns:
        The check journal's path.
    """
    journal = check_journal_path(tmp_path)
    journal.write_bytes(content)
    return journal


class SshReply:
    """A finished ssh, as the production seam's Protocol reads one.

    Attributes:
        returncode: The exit status.
        stdout: Standard output.
        stderr: Standard error.
    """

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        """Record the outcome.

        Args:
            returncode: The exit status.
            stdout: Standard output.
            stderr: Standard error.
        """
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSsh:
    """An ssh to one host, answering the window command from a REAL file.

    The journal it serves is a file under ``tmp_path`` written in
    diphtheria's own byte form, and the answer is computed from that file
    exactly as ``stat -c %s`` and ``tail -c +N`` compute it on the host: the
    size in bytes on one line, then every byte from offset ``N - 1``. A reply
    passed in instead is returned as given, for ssh's own failures.

    Attributes:
        calls: Every argv received, in order, with its timeout.
    """

    def __init__(
        self, host: str, journals: dict[str, pathlib.Path], reply: SshReply | None
    ) -> None:
        """Serve journals on one host.

        Args:
            host: The only host this ssh reaches.
            journals: Remote path to the local file holding its bytes.
            reply: A fixed answer to every call, or None to compute it.
        """
        self._host = host
        self._journals = journals
        self._reply = reply
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def __call__(self, args: Sequence[str], timeout_seconds: int) -> SshReply:
        """Answer one ssh.

        Args:
            args: The argv, ``ssh`` first.
            timeout_seconds: The deadline the caller set.

        Returns:
            The reply.

        Raises:
            AssertionError: For an argv that is not ``ssh <options> <host>
                <window command>`` against a served journal.
        """
        self.calls.append((tuple(args), timeout_seconds))
        if self._reply is not None:
            return self._reply
        *head, host, command = args
        assert tuple(head) == ("ssh", *SSH_OPTIONS)
        assert host == self._host
        tokens = command.split(" ")
        assert len(tokens) == 11
        assert tokens[:4] == ["stat", "-c", "%s", "--"]
        assert tokens[5:8] == ["&&", "tail", "-c"]
        assert tokens[9] == "--"
        assert tokens[10] == tokens[4]
        assert tokens[8].startswith("+")
        data = self._journals[tokens[4]].read_bytes()
        start = int(tokens[8][1:]) - 1
        return SshReply(returncode=0, stdout=f"{len(data)}\n".encode() + data[start:], stderr=b"")


def install_ssh(fake: FakeSsh) -> None:
    """Bind the ssh seam to a fake.

    Args:
        fake: The fake to answer every ssh.
    """
    _test_hooks.run_ssh = fake


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Rebind every touched seam to production before and after each test."""
    _test_hooks.reset_hooks()
    original_env = config_test_hooks.get_env
    yield
    _test_hooks.reset_hooks()
    config_test_hooks.get_env = original_env


def pin_env(values: dict[str, str]) -> None:
    """Answer environment reads from a dictionary and nothing else.

    Args:
        values: The variables that are set. Every other variable reads as
            unset, so a test's environment is this call, not the
            developer's shell.
    """

    def _env(name: str) -> str | None:
        return values.get(name)

    config_test_hooks.get_env = _env


def _make_emitted() -> Generator[list[str], None, None]:
    """Capture report lines instead of writing them to stdout.

    Yields:
        The list the ``emit`` hook appends to, in emission order.
    """
    lines: list[str] = []

    def _emit(line: str) -> None:
        lines.append(line)

    _test_hooks.emit = _emit
    yield lines
    _test_hooks.reset_hooks()


emitted = pytest.fixture(_make_emitted)
