"""Running a script on a node, and the one rule about how.

THE RULE, AND IT WAS LEARNED THE EXPENSIVE WAY ON 2026-09-04. A remote command
is never built by interpolating text into a command line. Probing three nodes
by that route failed twice: quotes were stripped passing through the local
shell, ssh, and ``cmd`` into ``powershell`` (the fleet was Windows then), and
the second attempt arrived as
``@(python,poetry,git,...)`` -- unquoted bare words, a parser error on the far
side. The form that works is the one :mod:`hpc3.core.preflight` already uses
for its batch scripts: RENDER THE SCRIPT, SEND IT, RUN IT BY PATH.

So :func:`run_script` takes a script body, writes it to the node, and executes
it by path. The bytes that run are the bytes that were sent, and no quoting
rule of any intermediate shell can change them.

NOTHING HERE CATCHES. Commands run with ``check=False`` and their status is
inspected, so a remote failure becomes an :class:`~platform_core.errors.AppError`
naming the node and carrying its own stderr.

EVERY OPERATION EXISTS TWICE: an ``attempt_*`` that returns the failure as a
VALUE, and the raising boundary built on top of it. Not two implementations --
the raising one is three lines over the other -- and not a "best-effort"
variant either. It exists because a node being off is a different question in
two different callers, and only one of them wants an exception.

  fleet-run --node lavender   asked for a specific machine. If it is down that
                              is an ERROR: silently running somewhere else
                              would answer a question nobody asked.
  fleet-run (auto-select)     asked for ANY machine with room. A node being
                              off is a REFUSAL to fold in beside "full" and
                              "no disk", not a reason to abandon the fleet.

Measured 2026-09-05, and it is why this split exists: the first real
auto-select dispatch was refused with ``NODE_UNREACHABLE: ssh to loki failed``
while lavender had already answered and had room. loki was powered off for a
trip. One laptop being asleep disabled the whole fleet, because the probe loop
raised instead of collecting.

HOW A FILE IS WRITTEN AND HOW A SCRIPT IS RUN are :mod:`fleet.core.dialect`'s
``sh`` acts, the same on every node since the fleet's last Windows machine
left it (board task e705f08c).
"""

from __future__ import annotations

import math
import pathlib
from typing import TypedDict

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, dialect
from fleet.core.ssh_options import SSH_OPTIONS
from fleet.core.stderr_tail import stderr_tail


class RemoteFailure(TypedDict):
    """Why a remote operation did not produce output.

    Attributes:
        code: The typed code the raising boundary will carry.
            ``NODE_UNREACHABLE`` when ssh itself could not reach the node,
            ``DISPATCH_FAILED`` when it reached it and the command exited
            non-zero. Two faults with two different fixes -- one is the
            tailnet, the other is the work.
        message: The full explanation, naming the node and carrying its own
            stderr.
    """

    code: FleetErrorCode
    message: str


class RemoteOutcome(TypedDict):
    """What a remote operation produced, or why it produced nothing.

    Attributes:
        output: The command's standard output. Empty when ``failure`` is set.
        failure: ``None`` on success; otherwise the reason.
    """

    output: str
    failure: RemoteFailure | None


def _failure_for(
    host: str, context: str, result: _test_hooks.CommandResult
) -> RemoteFailure | None:
    """Classify one command result, or report success.

    The single place a remote exit status becomes a fault, so ssh's own
    failure and the remote command's failure cannot be told apart differently
    in two callers.

    Args:
        host: The node, for the message.
        context: What was being attempted, e.g. ``"sending /home/x/a.sh"``.
        result: What the command did.

    Returns:
        The failure, or None when the command succeeded.
    """
    detail = stderr_tail(result["stderr"])
    if result["timed_out"]:
        # A peer that stopped answering mid-command is the tailnet's fault,
        # not the work's: the same code as an ssh that never connected.
        return RemoteFailure(
            code=FleetErrorCode.NODE_UNREACHABLE,
            message=f"ssh to {host} timed out while {context}: {detail}",
        )
    if result["returncode"] == SSH_FAILURE:
        return RemoteFailure(
            code=FleetErrorCode.NODE_UNREACHABLE,
            message=f"ssh to {host} failed while {context}: {detail}",
        )
    if result["returncode"] != 0:
        return RemoteFailure(
            code=FleetErrorCode.DISPATCH_FAILED,
            message=f"{context} on {host} exited {result['returncode']}: {detail}",
        )
    return None


def _raise_on(failure: RemoteFailure | None) -> None:
    """Turn a failure value into the typed exception, if there is one.

    Args:
        failure: What :func:`_failure_for` decided.

    Raises:
        AppError: Carrying the failure's own code and message.
    """
    if failure is not None:
        raise AppError(failure["code"], failure["message"])


#: The deadline every ssh this module runs carries, in seconds.
#:
#: The keepalive above ends a DEAD peer; this ends a LIVE one that will not
#: finish, which the keepalive cannot see. Every command this module sends
#: is short by construction -- a probe, a file write, a script that lists a
#: directory or launches a detached run and returns -- so two minutes is
#: generous for the work and still forty times shorter than the scheduled
#: tick's old 72-hour ceiling. The one exception is :func:`send_file`,
#: whose work grows with the file: its deadline is
#: :func:`send_deadline_seconds`.
SSH_TIMEOUT_SECONDS = 120

#: The slowest rate a copy to a node is allowed before it is called stuck,
#: in bytes a second.
#:
#: Measured 2026-10-14 (MCPs board task bfe2b3da): scp of 52,428,800 bytes
#: from the hub took loki 25,188 ms and sedona 21,592 ms, about 2.0 and 2.3
#: MB/s, and the archives the hub copies reach 239,780,392 bytes (the API
#: companion bundle). Under the flat :data:`SSH_TIMEOUT_SECONDS` those copies
#: were cut at 120 s and read as ``NODE_UNREACHABLE`` (fleet job 851300a6 on
#: loki). Half a MiB a second is a quarter of loki's rate, so a link has to
#: slow fourfold before a live copy is cut; a dead one still ends on the
#: keepalive in a minute.
SEND_FLOOR_BYTES_PER_SECOND = 524_288


def send_deadline_seconds(size_bytes: int) -> int:
    """The deadline for copying a file of this size to a node.

    Args:
        size_bytes: The file's length in bytes.

    Returns:
        :data:`SSH_TIMEOUT_SECONDS` for the connection and the remote side,
        plus the whole seconds the bytes take at
        :data:`SEND_FLOOR_BYTES_PER_SECOND`.
    """
    return SSH_TIMEOUT_SECONDS + math.ceil(size_bytes / SEND_FLOOR_BYTES_PER_SECOND)


#: ssh's own exit status when it cannot reach the host or the connection dies.
#:
#: A remote command that genuinely exits 255 is indistinguishable from this,
#: and that ambiguity is accepted deliberately: the scripts this package sends
#: exit 0 or a small status, so 255 from one of them would itself be a fault
#: worth surfacing as unreachable rather than as a result.
SSH_FAILURE = 255


def attempt_ssh(host: str, argv: tuple[str, ...]) -> RemoteOutcome:
    """Run one argv on a node, reporting failure as a value.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        argv: The remote command as a list of words. A list rather than a
            string so nothing local re-splits it.

    Returns:
        The command's standard output, or the reason there is none.
    """
    return _attempt_ssh_within(host, argv, timeout_seconds=SSH_TIMEOUT_SECONDS)


def _attempt_ssh_within(host: str, argv: tuple[str, ...], *, timeout_seconds: int) -> RemoteOutcome:
    """Run one argv on a node under a stated deadline, reporting failure as a value.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        argv: The remote command as a list of words.
        timeout_seconds: The deadline for the whole command.

    Returns:
        The command's standard output, or the reason there is none.
    """
    result = _test_hooks.run(["ssh", *SSH_OPTIONS, host, *argv], timeout_seconds=timeout_seconds)
    failure = _failure_for(host, f"running `{' '.join(argv)}`", result)
    return RemoteOutcome(output="" if failure is not None else result["stdout"], failure=failure)


def run_ssh(host: str, argv: tuple[str, ...]) -> str:
    """Run one argv on a node and return its standard output.

    The raising boundary over :func:`attempt_ssh`, for callers that have
    already committed to this node.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        argv: The remote command as a list of words.

    Returns:
        The command's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh itself fails to reach the
            node, and with ``DISPATCH_FAILED`` if the remote command runs and
            exits non-zero. The two are different faults with different
            fixes -- one is the tailnet, the other is the work -- and ssh
            reports its own failures with status 255.
    """
    outcome = attempt_ssh(host, argv)
    _raise_on(outcome["failure"])
    return outcome["output"]


def attempt_send(host: str, remote_path: str, body: str) -> RemoteFailure | None:
    """Place a script on a node, reporting failure as a value.

    The body is streamed over stdin into a file on the far side rather than
    passed as an argument, so its content cannot be interpreted by any shell
    between here and the disk -- see the module docstring for what that costs
    when it is not done. The command that receives the stream is
    :func:`fleet.core.dialect.write_command`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.

    Returns:
        The reason it did not land, or None when it did.
    """
    return _attempt_stream(
        host, dialect.write_command(remote_path), body, what=f"sending {remote_path}"
    )


def _attempt_stream(host: str, command: str, body: str, *, what: str) -> RemoteFailure | None:
    """Stream a body into one remote command's stdin, reporting failure as a value.

    Args:
        host: SSH destination.
        command: The remote command that reads standard input.
        body: The text streamed, UTF-8 encoded.
        what: What the stream does, for the failure's message.

    Returns:
        The reason it failed, or None when it did not.
    """
    result = _test_hooks.run(
        ["ssh", *SSH_OPTIONS, host, command],
        timeout_seconds=SSH_TIMEOUT_SECONDS,
        stdin_bytes=body.encode("utf-8"),
    )
    return _failure_for(host, what, result)


def send_file(host: str, source: pathlib.Path, remote_path: str, *, size_bytes: int) -> None:
    """Copy a local file's bytes to a path on a node, over scp.

    For an ARCHIVE, where :func:`send_script` is the wrong transport: that
    streams text through a remote shell. Measured 2026-09-26 (MCPs board task
    140e7042): an 83,210,142-byte archive, sent that way as one base64 line,
    did not land on serendipity (Windows then) within
    :data:`SSH_TIMEOUT_SECONDS`, and scp copied the same file there in 7.3 s
    with the digest intact. scp uses ssh's file transfer, so no shell reads
    the bytes and nothing re-encodes them.

    Its exit status keeps ssh's meaning: 255 when the node cannot be reached,
    1 when the node refused the write (a directory that does not exist),
    which :func:`_failure_for` already tells apart.

    Args:
        host: SSH destination.
        source: The local file.
        remote_path: Absolute path on the node to write, replacing any file
            there. Its directory must exist.
        size_bytes: The file's length, which sets the copy's deadline
            (:func:`send_deadline_seconds`).

    Raises:
        AppError: With ``NODE_UNREACHABLE`` when the node cannot be reached or
            the copy outlasts :func:`send_deadline_seconds`, or
            ``DISPATCH_FAILED`` when it refuses the write.
    """
    result = _test_hooks.run(
        ["scp", "-q", *SSH_OPTIONS, str(source), f"{host}:{remote_path}"],
        timeout_seconds=send_deadline_seconds(size_bytes),
    )
    _raise_on(_failure_for(host, f"copying {source.name} to {remote_path}", result))


def send_script(host: str, remote_path: str, body: str) -> None:
    """Place a script on a node.

    The raising boundary over :func:`attempt_send`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` as
            :func:`run_ssh` describes.
    """
    _raise_on(attempt_send(host, remote_path, body))


def attempt_script(host: str, remote_path: str, body: str) -> RemoteOutcome:
    """Send a script to a node and run it by path, reporting failure as a value.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute
            (:data:`fleet.core.dialect.SH_INVOCATION`).
        body: The script's complete text.

    Returns:
        The script's standard output, or the reason there is none. A send that
        fails short-circuits: running a path that was never written would
        answer with the far side's "file not found" rather than with the
        transport fault that actually happened.
    """
    failure = attempt_send(host, remote_path, body)
    if failure is not None:
        return RemoteOutcome(output="", failure=failure)
    return attempt_ssh(host, (*dialect.SH_INVOCATION, remote_path))


class Answered(TypedDict):
    """A script's output from a node that may not have answered.

    Attributes:
        output: The script's standard output, or None when the node did not
            answer.
        unreachable: Why it did not, when ssh could not reach it or timed
            out; None when it answered.
    """

    output: str | None
    unreachable: str | None


def read_script(host: str, remote_path: str, body: str) -> Answered:
    """Run a read on a node, reporting a node that did not answer as a value.

    For a serving runner's reads of the runs it holds (:mod:`fleet.cli.node_watch`):
    a node that misses one read, lavender-wsl on 2026-10-07 between 02:53Z
    and 02:55Z say, is read again at the next poll, while a node that
    answered and failed the read is a fault, raised as :func:`run_script`
    raises it.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.

    Returns:
        The output, or why the node did not answer.

    Raises:
        AppError: With ``DISPATCH_FAILED`` when the node answered and the
            script failed there.
    """
    outcome = attempt_script(host, remote_path, body)
    failure = outcome["failure"]
    if failure is None:
        return Answered(output=outcome["output"], unreachable=None)
    if failure["code"] is FleetErrorCode.NODE_UNREACHABLE:
        return Answered(output=None, unreachable=failure["message"])
    raise AppError(failure["code"], failure["message"])


def run_script(host: str, remote_path: str, body: str) -> str:
    """Send a script to a node and run it by path.

    The raising boundary over :func:`attempt_script`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.

    Returns:
        The script's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``.
    """
    outcome = attempt_script(host, remote_path, body)
    _raise_on(outcome["failure"])
    return outcome["output"]


#: How a node runs a command as root: its account's passwordless sudo, ``-n``
#: so a node without it refuses at once instead of waiting on a password
#: prompt nobody will answer.
ROOT_PREFIX: tuple[str, ...] = ("sudo", "-n")

#: How a node runs a root payload: bash by path, since the payloads rely on
#: ``pipefail`` and arrays, which the dialect's ``/bin/sh`` lacks.
ROOT_BASH: tuple[str, ...] = (*ROOT_PREFIX, "/bin/bash")


def attempt_root_script(
    host: str, remote_path: str, body: str, *, timeout_seconds: int
) -> RemoteOutcome:
    """Send a payload to a node and run it with full rights, reporting failure as a value.

    For a CI runner host's payloads (:mod:`fleet.core.runner_audit`,
    :mod:`fleet.core.runner_rebuild`): they read and write units, slices,
    services and the runner account's files, which only root may. The send
    keeps the short deadline; only the run takes the caller's, because a
    provision that installs packages was never sized for
    :data:`SSH_TIMEOUT_SECONDS`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then run under
            :data:`ROOT_BASH`.
        body: The payload's complete text, in bash.
        timeout_seconds: The deadline for running it.

    Returns:
        The payload's standard output, or the reason there is none. A send
        that fails short-circuits, as :func:`attempt_script` explains.
    """
    failure = attempt_send(host, remote_path, body)
    if failure is not None:
        return RemoteOutcome(output="", failure=failure)
    return _attempt_ssh_within(host, (*ROOT_BASH, remote_path), timeout_seconds=timeout_seconds)


def run_root_script(host: str, remote_path: str, body: str, *, timeout_seconds: int) -> str:
    """Send a payload to a node and run it with full rights.

    The raising boundary over :func:`attempt_root_script`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then run.
        body: The payload's complete text.
        timeout_seconds: The deadline for running it.

    Returns:
        The payload's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``; a payload
            still running at the deadline is ended and reported as the
            latter, and one exiting non-zero as the latter with its stderr.
    """
    outcome = attempt_root_script(host, remote_path, body, timeout_seconds=timeout_seconds)
    _raise_on(outcome["failure"])
    return outcome["output"]


__all__ = [
    "ROOT_BASH",
    "ROOT_PREFIX",
    "SSH_FAILURE",
    "SSH_TIMEOUT_SECONDS",
    "Answered",
    "RemoteFailure",
    "RemoteOutcome",
    "attempt_root_script",
    "attempt_script",
    "attempt_send",
    "attempt_ssh",
    "read_script",
    "run_root_script",
    "run_script",
    "run_ssh",
    "send_file",
    "send_script",
]
