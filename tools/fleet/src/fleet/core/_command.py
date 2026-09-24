"""Running one local command to completion, and what that produced.

Split out of :mod:`fleet.core._test_hooks` on 2026-09-24, which had
reached its line ceiling: the seam there binds a dozen impure acts, and
running a child process is the only one of them with a deadline, a
payload and two streams to reason about. The seam still re-exports
:class:`CommandResult` and :data:`TIMED_OUT_RETURNCODE`, so every module
that imports them from ``_test_hooks`` is unaffected.

NOTHING HERE CATCHES, WITH ONE NAMED EXCEPTION. ``subprocess.run`` is
called with ``check=False`` and the return code is inspected explicitly,
so a remote failure becomes a typed ``AppError`` at the call site that
knows what the command was for, rather than a ``CalledProcessError``
caught and re-raised somewhere that does not. The exception is the
deadline: ``subprocess`` reports an expired ``timeout`` only by raising
``TimeoutExpired``, so :func:`_awaited` converts that one report into the
result's own ``timed_out`` field at the boundary and returns it. Nothing
is retried, softened or defaulted; the caller reads ``timed_out`` exactly
as it reads ``returncode``.

A DEADLINE IS ONLY REAL IF THE CALL CANNOT BLOCK OUTSIDE IT, and until
2026-09-24 this one could (board task 1e57ebe5).
``subprocess.run(input=...)`` writes the payload from the calling thread
BEFORE its ``timeout`` governs anything, so a call passing both a
deadline and a bulk payload reads as bounded and is not. Measured: 60 MB
to a child that never drains its pipe, with ``timeout=5``, was still
blocked at 100 seconds; at 8 MB the call returned after 60.2 s, ended by
the CHILD exiting rather than by the clock. It cost a fleet staging send
40 minutes against a 120-second bound, reporting nothing while its
dispatch row stayed claimed, four days after every remote command in this
package was given a deadline for exactly that reason.

So a payload is handed to the child as a FILE it reads itself and this
process writes nothing to a pipe it cannot walk away from. :func:`_awaited`
here takes that file already positioned and never writes to the child at
all; preparing it is the seam's job, in
:func:`fleet.core._test_hooks._default_run`, which is where the
environment is assembled too. A future caller adding a stdin path should
keep that shape: passing ``input=`` restores the defect silently, and a
suite that only drives draining children stays green through it, which is
how it survived here.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import IO

from typing_extensions import TypedDict

#: The ``returncode`` a result carries when the command was ended for
#: outliving its deadline. Negative, so no caller reading ``returncode != 0``
#: as failure can mistake it for success, and distinct from every status a
#: process can exit with on its own; the fact itself is ``timed_out``.
TIMED_OUT_RETURNCODE = -1


class CommandResult(TypedDict):
    """What running a command produced.

    Attributes:
        returncode: Process exit status, or :const:`TIMED_OUT_RETURNCODE`
            when the command was ended at its deadline.
        stdout: Standard output, decoded as UTF-8.
        stderr: Standard error, decoded as UTF-8. Carried because ssh puts
            the reason for a refusal here, and a failure that discards it
            sends the reader to the node to rediscover what happened. For a
            command ended at its deadline it ends with ``timed out after
            <n> s``, so a caller that only prints stderr still says so.
        timed_out: True when the command was ended for outliving its
            ``timeout_seconds``; the streams then hold what it had produced.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _decode_captured(captured: bytes | None) -> str:
    """Decode one captured stream.

    Args:
        captured: The bytes, or None when the process was ended before the
            stream was collected, which ``TimeoutExpired`` reports that way.

    Returns:
        The text, decoded as UTF-8 with undecodable bytes replaced -- a
        mangled character in a diagnostic is better than losing the
        diagnostic -- and empty for None.
    """
    return "" if captured is None else captured.decode("utf-8", errors="replace")


def _awaited(
    argv: Sequence[str],
    *,
    stdin_source: int | IO[bytes],
    environment: dict[str, str],
    timeout_seconds: int,
) -> CommandResult:
    """Run one command against a prepared standard input and collect it.

    Args:
        argv: Executable and arguments.
        stdin_source: What the child reads as standard input:
            :data:`subprocess.DEVNULL` for a closed one, or a file already
            positioned at the byte the child should read first.
        environment: The child's complete environment.
        timeout_seconds: The deadline; the child is ended when it passes.

    Returns:
        The command's exit status and captured streams, decoded through
        :func:`_decode_captured`; or, when the deadline passed first,
        :const:`TIMED_OUT_RETURNCODE`, whatever the streams held, stderr
        ending ``timed out after <n> s``, and ``timed_out`` set.
    """
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            stdin=stdin_source,
            capture_output=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as expired:
        partial = _decode_captured(expired.stderr).rstrip()
        note = f"timed out after {timeout_seconds} s"
        return CommandResult(
            returncode=TIMED_OUT_RETURNCODE,
            stdout=_decode_captured(expired.stdout),
            stderr=note if not partial else f"{partial}\n{note}",
            timed_out=True,
        )
    return CommandResult(
        returncode=completed.returncode,
        stdout=_decode_captured(completed.stdout),
        stderr=_decode_captured(completed.stderr),
        timed_out=False,
    )


__all__ = ["TIMED_OUT_RETURNCODE", "CommandResult"]
