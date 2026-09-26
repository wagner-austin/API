"""Running a script on a node, and the one rule about how.

THE RULE, AND IT WAS LEARNED THE EXPENSIVE WAY ON 2026-09-04. A remote command
is never built by interpolating text into a command line. Probing three nodes
by that route failed twice: quotes are stripped passing through the local
shell, ssh, and ``cmd`` into ``powershell``, and the second attempt arrived as
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

THE NODE'S PLATFORM DECIDES HOW A FILE IS WRITTEN AND HOW A SCRIPT IS RUN.
Until 2026-09-20 both were PowerShell for every node; with the first Linux
node they are one of two dialects (:mod:`fleet.core.dialect`), chosen by the
platform the caller passes. Passed rather than probed, because the very
first thing sent to a node is a probe, and it has to be written somehow.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, dialect


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
        context: What was being attempted, e.g. ``"sending C:/x.ps1"``.
        result: What the command did.

    Returns:
        The failure, or None when the command succeeded.
    """
    detail = result["stderr"].strip() or "<no stderr>"
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


#: Options every ssh invocation carries.
#:
#: ``BatchMode=yes`` makes a missing key fail immediately instead of prompting
#: for a password no automated caller can answer -- the failure is the point,
#: because a prompt would hang a dispatch forever.
#:
#: ``ServerAliveInterval`` and ``ServerAliveCountMax`` make ssh notice a peer
#: that went away AFTER the handshake: ``ConnectTimeout`` bounds only the
#: connection, and a laptop that sleeps mid-command leaves the TCP session
#: ESTABLISHED with nothing on either end to say so. Measured 2026-09-17
#: 11:15Z on pendragon (board tasks 35940277 and 41ac6ed2): one such ssh sat
#: three days. Four missed probes fifteen seconds apart end it in a minute.
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)

#: The deadline every ssh this module runs carries, in seconds.
#:
#: The keepalive above ends a DEAD peer; this ends a LIVE one that will not
#: finish, which the keepalive cannot see. Every command this module sends
#: is short by construction -- a probe, a file write, a script that lists a
#: directory or launches a detached run and returns -- so two minutes is
#: generous for the work and still forty times shorter than the scheduled
#: tick's old 72-hour ceiling.
SSH_TIMEOUT_SECONDS = 120

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


def attempt_send(
    host: str, remote_path: str, body: str, *, platform: NodePlatform
) -> RemoteFailure | None:
    """Place a script on a node, reporting failure as a value.

    The body is streamed over stdin into a file on the far side rather than
    passed as an argument, so its content cannot be interpreted by any shell
    between here and the disk -- see the module docstring for what that costs
    when it is not done. The command that receives the stream is the
    platform's (:meth:`fleet.core.dialect.Dialect.write_command`), and the
    incidents behind each are recorded on the dialect.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.
        platform: The node's declared platform.

    Returns:
        The reason it did not land, or None when it did.
    """
    return _attempt_stream(
        host,
        dialect.for_platform(platform).write_command(remote_path),
        body,
        what=f"sending {remote_path}",
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


def stream_to_command(host: str, command: str, body: str, *, what: str) -> None:
    """Stream a body into one remote command's stdin.

    For a write whose bytes the platform's own :meth:`write_command` would
    change: on Windows that command re-encodes with a BOM and CRLF line
    ends, which a PowerShell script needs and a bash payload for a WSL
    distro cannot survive (:mod:`fleet.core.runner_distro`).

    Args:
        host: SSH destination.
        command: The remote command that reads standard input.
        body: The text streamed, UTF-8 encoded.
        what: What the stream does, for the error.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` as
            :func:`run_ssh` describes.
    """
    _raise_on(_attempt_stream(host, command, body, what=what))


def send_script(host: str, remote_path: str, body: str, *, platform: NodePlatform) -> None:
    """Place a script on a node.

    The raising boundary over :func:`attempt_send`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.
        platform: The node's declared platform.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` as
            :func:`run_ssh` describes.
    """
    _raise_on(attempt_send(host, remote_path, body, platform=platform))


def attempt_script(
    host: str, remote_path: str, body: str, *, platform: NodePlatform
) -> RemoteOutcome:
    """Send a script to a node and run it by path, reporting failure as a value.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.
        platform: The node's declared platform, which decides how the path
            is executed (:meth:`fleet.core.dialect.Dialect.invocation`).

    Returns:
        The script's standard output, or the reason there is none. A send that
        fails short-circuits: running a path that was never written would
        answer with the far side's "file not found" rather than with the
        transport fault that actually happened.
    """
    failure = attempt_send(host, remote_path, body, platform=platform)
    if failure is not None:
        return RemoteOutcome(output="", failure=failure)
    return attempt_ssh(host, (*dialect.for_platform(platform).invocation(), remote_path))


def run_script(host: str, remote_path: str, body: str, *, platform: NodePlatform) -> str:
    """Send a script to a node and run it by path.

    The raising boundary over :func:`attempt_script`.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.
        platform: The node's declared platform.

    Returns:
        The script's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``.
    """
    return run_script_within(
        host, remote_path, body, platform=platform, timeout_seconds=SSH_TIMEOUT_SECONDS
    )


def run_script_within(
    host: str, remote_path: str, body: str, *, platform: NodePlatform, timeout_seconds: int
) -> str:
    """Send a script to a node and run it by path under a stated deadline.

    For the one kind of remote work that is long by construction: a runner
    host's rebuild stages download a distro image and install packages,
    which :data:`SSH_TIMEOUT_SECONDS` was never sized for (board task
    1aa6a021). Sending the file keeps the short deadline; only the run
    takes the caller's.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.
        platform: The node's declared platform.
        timeout_seconds: The deadline for running the script.

    Returns:
        The script's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``; a script
            still running at the deadline is ended and reported as the
            latter.
    """
    _raise_on(attempt_send(host, remote_path, body, platform=platform))
    outcome = _attempt_ssh_within(
        host,
        (*dialect.for_platform(platform).invocation(), remote_path),
        timeout_seconds=timeout_seconds,
    )
    _raise_on(outcome["failure"])
    return outcome["output"]


__all__ = [
    "SSH_FAILURE",
    "SSH_OPTIONS",
    "SSH_TIMEOUT_SECONDS",
    "RemoteFailure",
    "RemoteOutcome",
    "attempt_script",
    "attempt_send",
    "attempt_ssh",
    "run_script",
    "run_script_within",
    "run_ssh",
    "send_script",
    "stream_to_command",
]
