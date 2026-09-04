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
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks

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


def run_ssh(host: str, argv: tuple[str, ...]) -> str:
    """Run one argv on a node and return its standard output.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        argv: The remote command as a list of words. A list rather than a
            string so nothing local re-splits it.

    Returns:
        The command's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh itself fails to reach the
            node, and with ``DISPATCH_FAILED`` if the remote command runs and
            exits non-zero. The two are different faults with different
            fixes -- one is the tailnet, the other is the work -- and ssh
            reports its own failures with status 255.
    """
    result = _test_hooks.run(["ssh", *SSH_OPTIONS, host, *argv])
    if result["returncode"] == SSH_FAILURE:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"ssh to {host} failed: {result['stderr'].strip() or '<no stderr>'}",
        )
    if result["returncode"] != 0:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"`{' '.join(argv)}` on {host} exited {result['returncode']}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )
    return result["stdout"]


def send_script(host: str, remote_path: str, body: str) -> None:
    """Place a script on a node.

    The body is streamed over stdin into a file on the far side rather than
    passed as an argument, so its content cannot be interpreted by any shell
    between here and the disk -- see the module docstring for what that costs
    when it is not done.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write.
        body: The script's complete text.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` as
            :func:`run_ssh` describes.
    """
    result = _test_hooks.run(
        ["ssh", *SSH_OPTIONS, host, _WRITE_COMMAND.format(path=remote_path)],
        stdin_bytes=body.encode("utf-8"),
    )
    if result["returncode"] == SSH_FAILURE:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"ssh to {host} failed while sending {remote_path}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )
    if result["returncode"] != 0:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"writing {remote_path} on {host} exited {result['returncode']}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )


def run_script(host: str, remote_path: str, body: str) -> str:
    """Send a script to a node and run it by path.

    Args:
        host: SSH destination.
        remote_path: Absolute path on the node to write and then execute.
        body: The script's complete text.

    Returns:
        The script's standard output.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``.
    """
    send_script(host, remote_path, body)
    return run_ssh(host, (*POWERSHELL_INVOCATION, remote_path))


__all__ = [
    "POWERSHELL_INVOCATION",
    "SSH_FAILURE",
    "SSH_OPTIONS",
    "run_script",
    "run_ssh",
    "send_script",
]
