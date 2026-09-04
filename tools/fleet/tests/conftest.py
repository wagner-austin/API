"""Shared fakes and the hook reset that keeps tests independent.

:class:`FakeRun` and :class:`FakeClock` are FAKES, not mocks. Each implements
the same Protocol the production implementation does and records what it was
asked to do, so an assertion is about the commands this package builds rather
than about a patching library's call-recording API. Nothing here patches
anything: the hooks in :mod:`fleet.core._test_hooks` are module-level names,
and a test rebinds them.

HOOKS ARE RESET BEFORE AND AFTER EVERY TEST. A rebinding that leaked would
produce a test that fails only when it runs after a specific other one, and
``-n auto`` reorders freely -- so the symptom would be an intermittent failure
whose cause is invisible in the failing test.

THE CLOCK IS A FAKE BECAUSE THE PACKAGE IS ABOUT EXPIRY. Every question about
whether a resource is free is a question about the time, and a test that could
not move the clock could only assert that an unexpired lease is unexpired --
the case that never breaks.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence

import pytest

from fleet.core import _test_hooks


class FakeRun:
    """A command runner that answers from a script and records the calls.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`.

    Attributes:
        calls: Every argv it was given, in order.
        stdin: Every stdin payload it was given, in order, with None for the
            calls that had none. A separate list rather than a field on the
            call, so a test asserting on argv does not have to mention bytes.
    """

    calls: list[tuple[str, ...]]
    stdin: list[bytes | None]
    _replies: list[_test_hooks.CommandResult]

    def __init__(self, replies: Sequence[_test_hooks.CommandResult]) -> None:
        """Build a runner that will answer with these results in order.

        Args:
            replies: One result per expected call. Running out is an error
                rather than a default: a test that made more calls than it
                declared has changed behaviour it did not mean to assert on.
        """
        self.calls = []
        self.stdin = []
        self._replies = list(replies)

    def __call__(
        self, argv: Sequence[str], *, stdin_bytes: bytes | None = None
    ) -> _test_hooks.CommandResult:
        """Record a call and answer with the next scripted result.

        Args:
            argv: The command.
            stdin_bytes: Its standard input, or None.

        Returns:
            The next scripted result.

        Raises:
            AssertionError: If more calls are made than results were given.
        """
        self.calls.append(tuple(argv))
        self.stdin.append(stdin_bytes)
        assert self._replies, f"unscripted call: {list(argv)}"
        return self._replies.pop(0)


class FakeClock:
    """A clock a test moves by hand.

    Satisfies :class:`~fleet.core._test_hooks.NowProtocol`.

    Attributes:
        seconds: The current time, whole seconds since the epoch. Assign to
            it to move time.
    """

    seconds: int

    def __init__(self, seconds: int) -> None:
        """Start the clock.

        Args:
            seconds: Initial time, whole seconds since the epoch.
        """
        self.seconds = seconds

    def __call__(self) -> int:
        """Read the current time.

        Returns:
            Whatever ``seconds`` currently holds.
        """
        return self.seconds


def ok(stdout: str) -> _test_hooks.CommandResult:
    """Build a successful command result.

    Args:
        stdout: What the command printed.

    Returns:
        The result, exit status zero and empty stderr.
    """
    return _test_hooks.CommandResult(returncode=0, stdout=stdout, stderr="")


def failed(returncode: int, stderr: str) -> _test_hooks.CommandResult:
    """Build a failing command result.

    Args:
        returncode: The exit status.
        stderr: What the command wrote to standard error.

    Returns:
        The result, with empty stdout.
    """
    return _test_hooks.CommandResult(returncode=returncode, stdout="", stderr=stderr)


@pytest.fixture(name="reset_hooks", autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Put every hook back to its real implementation around each test."""
    _restore()
    yield None
    _restore()


def _restore() -> None:
    """Rebind every hook to the implementation it starts life with."""
    _test_hooks.run = _test_hooks._default_run
    _test_hooks.now = _test_hooks._default_now
    _test_hooks.read_text = _test_hooks._default_read_text
    _test_hooks.file_exists = _test_hooks._default_file_exists
    _test_hooks.append_text = _test_hooks._default_append_text
    _test_hooks.write_text = _test_hooks._default_write_text
