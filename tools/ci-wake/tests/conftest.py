"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a mock,
matching ``hpc-wake``'s, ``fleet-wake``'s and ``board-watch``'s conventions.
The seams rebound are this package's own (``ci_wake._test_hooks``) and the
one sanctioned environment reader
(``platform_core.config.config_test_hooks.get_env``, which also feeds
``board_watch.config.load_credentials``, so pinning it configures the whole
credential chain from one place).

THE TWO RECORDS ARE REAL FILES UNDER ``tmp_path``, WRITTEN BY THIS PACKAGE'S
OWN WRITERS. ``append_attempt`` and ``append_announced`` do the I/O exactly as
the hook and the cycle do, so these tests read rows produced the way
production produces them rather than rows a fixture invented. That is what
makes the append-only/collapse distinction testable at all: a fixture that
wrote one row per sha could never catch a reader that skipped
``latest_attempts``.

:class:`FakeGh` ANSWERS BY ARGUMENT VECTOR, NOT BY CALL ORDER. The cycle
makes one runs query per outstanding push and then one jobs query per run of
each announceable push, and the interleaving of those is the thing under
test in ``test_cycle``. A fake answering positionally would pass whatever
order the cycle happened to use, including a wrong one.

The MCP poster fake is :class:`platform_core.mcp_testing.FakeHttpPost`,
shared with ``platform_core``'s, ``hpc-wake``'s and ``fleet-wake``'s suites
rather than copied a fourth time.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Final

import pytest
from platform_core.config import config_test_hooks

from ci_wake import _test_hooks
from ci_wake.identity import TASK_ID_VARIABLE

#: The clock every test runs against, so timestamps are assertable.
FROZEN_NOW: Final = 1788700000

#: The standing task id every configured test posts into.
TASK_ID: Final = "9406cfd9-b208-42d3-a14c-669c10590f48"

#: The repository every test enrols against.
REPO: Final = "wagner-austin/MCPs"

#: A full-length sha, so the enrolment validator sees a real one.
SHA: Final = "6a0e66af21d79b5b969f26e798f47798de4059ee"

#: A second full-length sha, for the multi-push cases.
OTHER_SHA: Final = "167cbb63f49dfb5ed7a9cac6c5386c8b785dbb85"

#: The board label the addressed tests push under.
AGENT: Final = "opus-ci-wake-0909"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}


class FakeCompleted:
    """A finished process, satisfying :class:`ci_wake._test_hooks.CompletedProto`.

    Attributes:
        stdout: What the command wrote to standard output.
        stderr: What it wrote to standard error.
        returncode: Its exit status.
    """

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        """Build a finished process.

        Args:
            stdout: Standard output.
            stderr: Standard error.
            returncode: Exit status.
        """
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode

    @property
    def stdout(self) -> str:
        """Captured standard output."""
        return self._stdout

    @property
    def stderr(self) -> str:
        """Captured standard error."""
        return self._stderr

    @property
    def returncode(self) -> int:
        """The process's exit status."""
        return self._returncode


class FakeGh:
    """A process runner answering from a table keyed by argument vector.

    Satisfies :class:`ci_wake._test_hooks.RunProcessProtocol` exactly -- same
    parameter names, same keyword-only split, same return type -- so a test
    exercises the real ``gh`` boundary against it rather than against a
    signature that merely resembles the real one.

    An UNSCRIPTED command raises rather than answering emptily: a cycle that
    asked something the test did not declare has changed behaviour the test
    did not mean to assert on, and answering it would hide exactly that.

    Attributes:
        calls: Every argument vector it was handed, in order.
    """

    calls: list[tuple[str, ...]]

    def __init__(self, replies: dict[tuple[str, ...], FakeCompleted]) -> None:
        """Build a runner that answers these commands.

        Args:
            replies: Argument vector to the outcome it produces.
        """
        self._replies = dict(replies)
        self.calls = []

    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> FakeCompleted:
        """Answer one scripted command.

        Args:
            args: The argument vector.
            capture_output: Ignored; asserted by the caller's tests.
            text: Ignored; asserted by the caller's tests.
            timeout: Ignored; asserted by the caller's tests.

        Returns:
            The scripted outcome.

        Raises:
            AssertionError: If the command was not scripted.
        """
        vector = tuple(args)
        self.calls.append(vector)
        reply = self._replies.get(vector)
        if reply is None:
            raise AssertionError(f"unscripted gh call: {vector!r}")
        return reply


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


def _make_frozen_clock() -> Generator[int, None, None]:
    """Pin the bridge clock so timestamps and ages are assertable.

    Yields:
        The timestamp every row will record.
    """

    def _now() -> int:
        return FROZEN_NOW

    _test_hooks.now = _now
    yield FROZEN_NOW
    _test_hooks.reset_hooks()


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
frozen_clock = pytest.fixture(_make_frozen_clock)
