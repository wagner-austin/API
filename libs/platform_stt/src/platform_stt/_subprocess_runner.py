"""Running a child process for this package, and handing it its input safely.

Split out of ``_test_hooks`` when that module crossed the 600-line ceiling
(board task 0d891468). This is one cohesive concern: the subprocess contract
this package injects, and the production implementation behind it. Everything
else ``_test_hooks`` holds is a different seam.

THE PAYLOAD IS A FILE, NOT A PIPE, AND THAT IS THE WHOLE REASON THIS MODULE
READS THE WAY IT DOES. ``Popen.communicate(input=...)`` hands a payload over
by WRITING IT FROM THE CALLING THREAD, before the ``timeout`` governs
anything, so a call passing both a deadline and a payload reads as bounded
and is not. See :class:`_PayloadOnAFile` for the measurement.
"""

from __future__ import annotations

import subprocess
import tempfile
from types import TracebackType
from typing import IO, Protocol


class SubprocessRunResult(Protocol):
    """Protocol for subprocess.run result."""

    returncode: int
    stdout: bytes | str | None
    stderr: bytes | str | None


class SubprocessRunProtocol(Protocol):
    """Protocol for subprocess.run function."""

    def __call__(
        self,
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        """Run subprocess with given arguments."""
        ...


class _SubprocessRunResultImpl:
    """Concrete implementation of SubprocessRunResult from subprocess.run output."""

    __slots__ = ("returncode", "stderr", "stdout")

    def __init__(
        self,
        returncode: int,
        stdout: bytes | str | None,
        stderr: bytes | str | None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _PayloadOnAFile:
    """Give a child its standard input as a file, never as a pipe this fills.

    THE PIPE IS THE DEFECT AND THE FILE IS THE REMEDY.
    ``communicate(input=...)`` hands the payload over by WRITING IT FROM THE
    CALLING THREAD, and that write happens before the ``timeout`` governs
    anything, so a call passing both a deadline and a payload reads as
    bounded and is not. Measured on Windows CPython 2026-09-24 against a
    child that never drains its pipe: 60 MB with ``timeout=5`` was still
    blocked at 100 seconds, and 8 MB returned after 60.2 seconds, ended by
    the CHILD exiting rather than by the clock (board tasks 1e57ebe5 and
    0d891468). A file the child reads itself costs this thread nothing, so
    the deadline governs the whole call.

    SIZE IS NOT THE CRITERION. Whether the write blocks depends on whether
    the child drains its pipe, not on how many bytes there are, so a small
    payload that happens to fit a buffer is not bounded, it is lucky, and it
    stops being lucky when the child changes.

    A CLASS RATHER THAN A ``@contextmanager`` GENERATOR because this monorepo
    forbids importing ``Iterator`` from either ``typing`` or
    ``collections.abc`` (the ``import-collections-iterator`` guard), and a
    generator context manager cannot be annotated without it.

    Attributes:
        _payload: What the child should read, or None for nothing.
        _handle: The temporary file, once opened, so ``__exit__`` closes
            exactly what ``__enter__`` opened.
    """

    __slots__ = ("_handle", "_payload")

    def __init__(self, payload: bytes | str | None) -> None:
        """Hold the payload until the child is started.

        Args:
            payload: Bytes or text for standard input, or None.
        """
        self._payload = payload
        self._handle: IO[bytes] | None = None

    def __enter__(self) -> int | IO[bytes]:
        """Prepare the child's standard input.

        Returns:
            :data:`subprocess.DEVNULL` for no payload, or a temporary file
            already positioned at the first byte. A child with no payload
            gets a CLOSED stdin rather than this process's own, because an
            inherited one is a handle a child can wait on forever.
        """
        if self._payload is None:
            return subprocess.DEVNULL
        encoded = self._payload.encode("utf-8") if isinstance(self._payload, str) else self._payload
        handle = tempfile.TemporaryFile()
        handle.write(encoded)
        handle.seek(0)
        self._handle = handle
        return handle

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the temporary file, if one was opened.

        Returns None, never True, so nothing raised inside the block is
        swallowed here.

        Args:
            exc_type: The exception class, if the block raised.
            exc: The exception, if the block raised.
            traceback: Its traceback, if the block raised.
        """
        if self._handle is not None:
            self._handle.close()


def _run_subprocess_bytes(
    args: list[str],
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_data: bytes | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> _SubprocessRunResultImpl:
    """Run subprocess and return bytes output."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None

    with _PayloadOnAFile(input_data) as stdin_source:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            args,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
            stdin=stdin_source,
            cwd=cwd,
            env=env,
        )
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
    returncode: int = proc.returncode

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, args, stdout_bytes, stderr_bytes)

    return _SubprocessRunResultImpl(returncode, stdout_bytes, stderr_bytes)


def _run_subprocess_text(
    args: list[str],
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_data: str | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> _SubprocessRunResultImpl:
    """Run subprocess and return text output."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None

    # The payload goes on a FILE even here, where ``text=True``: that flag
    # governs how this process DECODES the child's output, and the child
    # reads bytes either way, so the text is encoded once and handed over as
    # a file for the reason :class:`_PayloadOnAFile` states.
    with _PayloadOnAFile(input_data) as stdin_source:
        proc: subprocess.Popen[str] = subprocess.Popen(
            args,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
            stdin=stdin_source,
            text=True,
            cwd=cwd,
            env=env,
        )
        stdout_str, stderr_str = proc.communicate(timeout=timeout)
    returncode: int = proc.returncode

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, args, stdout_str, stderr_str)

    return _SubprocessRunResultImpl(returncode, stdout_str, stderr_str)


def _default_subprocess_run(
    args: list[str],
    *,
    capture_output: bool = False,
    check: bool = False,
    timeout: float | None = None,
    text: bool = False,
    input: bytes | str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessRunResult:
    """Production implementation - uses typed Popen to avoid Any types."""
    if text:
        input_str: str | None = input if isinstance(input, str) else None
        return _run_subprocess_text(args, capture_output, check, timeout, input_str, cwd, env)
    input_bytes: bytes | None = None
    if isinstance(input, str):
        input_bytes = input.encode()
    elif isinstance(input, bytes):
        input_bytes = input
    return _run_subprocess_bytes(args, capture_output, check, timeout, input_bytes, cwd, env)


__all__ = [
    "SubprocessRunProtocol",
    "SubprocessRunResult",
]
