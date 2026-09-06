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
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks


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
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

#: How the node is asked to run a script file it has just been handed.
#:
#: ``-NoProfile`` because a profile is the node owner's, and a dispatch that
#: inherited it would run different code on different machines for reasons
#: nobody recorded. ``-ExecutionPolicy Bypass`` because the script arrived over
#: ssh and is unsigned by construction.
POWERSHELL_INVOCATION = ("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")

#: ssh's own exit status when it cannot reach the host or the connection dies.
#:
#: A remote command that genuinely exits 255 is indistinguishable from this,
#: and that ambiguity is accepted deliberately: the scripts this package sends
#: exit 0 or a small status, so 255 from one of them would itself be a fault
#: worth surfacing as unreachable rather than as a result.
SSH_FAILURE = 255

#: How a script's body is written on the far side.
#:
#: Streamed from stdin rather than passed as an argument, so no shell between
#: here and the disk can interpret it. ``-LiteralPath`` because a path is a
#: path: without it PowerShell treats ``[`` and ``]`` as wildcards.
#:
#: THE OUTER QUOTES ARE LOad-BEARING AND THEIR ABSENCE WAS A REAL BUG. Windows
#: OpenSSH hands a remote command to ``cmd.exe``, not to PowerShell. Unquoted,
#: cmd sees the ``|`` as ITS OWN pipe, runs ``powershell -Command $input``, and
#: pipes the result to ``Set-Content`` -- which cmd does not have. Measured
#: 2026-09-04 on the first real dispatch: ``'Set-Content' is not recognized as
#: an internal or external command``. Quoting makes the whole thing one
#: argument to powershell.
#:
#: The directory is created in the same command because the alternative is a
#: second round trip that would itself need a path to already exist. Paths
#: given to this must be ABSOLUTE and literal: a ``$env:TEMP`` inside the
#: single quotes below is not expanded by PowerShell, so it would create a
#: directory named ``$env:TEMP`` rather than resolving one.
_WRITE_COMMAND = (
    'powershell -NoProfile -Command "'
    "New-Item -ItemType Directory -Force -Path (Split-Path -Parent '{path}') | Out-Null; "
    "$input | Set-Content -LiteralPath '{path}' -Encoding utf8"
    '"'
)


def attempt_ssh(host: str, argv: tuple[str, ...]) -> RemoteOutcome:
    """Run one argv on a node, reporting failure as a value.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        argv: The remote command as a list of words. A list rather than a
            string so nothing local re-splits it.

    Returns:
        The command's standard output, or the reason there is none.
    """
    result = _test_hooks.run(["ssh", *SSH_OPTIONS, host, *argv])
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
    when it is not done.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.

    Returns:
        The reason it did not land, or None when it did.
    """
    result = _test_hooks.run(
        ["ssh", *SSH_OPTIONS, host, _WRITE_COMMAND.format(path=remote_path)],
        stdin_bytes=body.encode("utf-8"),
    )
    return _failure_for(host, f"sending {remote_path}", result)


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
        remote_path: Absolute path on the node to write and then execute.
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
    return attempt_ssh(host, (*POWERSHELL_INVOCATION, remote_path))


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


__all__ = [
    "POWERSHELL_INVOCATION",
    "SSH_FAILURE",
    "SSH_OPTIONS",
    "RemoteFailure",
    "RemoteOutcome",
    "attempt_script",
    "attempt_send",
    "attempt_ssh",
    "run_script",
    "run_ssh",
    "send_script",
]
