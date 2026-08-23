"""Running commands on the cluster and sending bytes to it.

Every remote call goes through ``ssh`` with an argv list rather than a shell
string. Corpus filenames and payload commands are attacker-adjacent only in
the sense that they are arbitrary text, but arbitrary text is enough: a
filename holding a quote or a semicolon would be interpreted rather than
transferred, and the resulting failure would look like a corrupt file.

Nothing here catches. Commands run with ``check=False`` and their exit status
is inspected, so a remote failure becomes an :class:`~platform_core.errors.AppError`
that names the command and carries the cluster's own stderr.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import _test_hooks


def _shell_single_quote(value: str) -> str:
    """Quote a value for safe interpolation into a remote shell command.

    Args:
        value: Text to quote.

    Returns:
        The value wrapped in single quotes, with embedded single quotes
        escaped by closing, inserting an escaped quote, and reopening. This
        is the only form that is safe for arbitrary text in POSIX ``sh``.
    """
    return "'" + value.replace("'", "'\\''") + "'"


def run_remote(host: str, command: str) -> str:
    """Run a command on the cluster and return its standard output.

    Args:
        host: SSH destination, an alias from the user's ssh config.
        command: Command line to execute remotely.

    Returns:
        The command's standard output.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if the
            command exits non-zero, carrying the exit status and the
            cluster's stderr. The message is the diagnostic; discarding it
            would send the reader back to the cluster to rediscover it.
    """
    result = _test_hooks.run(["ssh", "-o", "BatchMode=yes", host, command])
    if result["returncode"] != 0:
        raise AppError(
            Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
            f"`{command}` on {host} exited {result['returncode']}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )
    return result["stdout"]


def put_bytes(host: str, remote_path: str, payload: bytes) -> None:
    """Write bytes to a path on the cluster.

    The transfer streams through ``cat`` on the remote side rather than using
    ``scp``, so exactly the bytes given arrive: no line-ending translation, no
    mode inference, and no dependence on the local platform's idea of text.

    Args:
        host: SSH destination.
        remote_path: Absolute destination path on the cluster.
        payload: Exact bytes to write.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if the
            write exits non-zero.
    """
    quoted = _shell_single_quote(remote_path)
    result = _test_hooks.run(
        ["ssh", "-o", "BatchMode=yes", host, f"cat > {quoted}"],
        stdin_bytes=payload,
    )
    if result["returncode"] != 0:
        raise AppError(
            Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
            f"writing {remote_path} on {host} exited {result['returncode']}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )


def make_directory(host: str, remote_dir: str) -> None:
    """Create a directory on the cluster, including parents.

    Args:
        host: SSH destination.
        remote_dir: Absolute directory to create.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if the
            creation exits non-zero.
    """
    run_remote(host, f"mkdir -p {_shell_single_quote(remote_dir)}")


def remote_digest(host: str, remote_path: str) -> str:
    """Compute a file's sha256 on the cluster.

    Args:
        host: SSH destination.
        remote_path: Absolute path to digest.

    Returns:
        Raw ``sha256sum`` output for the caller to parse.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if the
            command exits non-zero, which includes the file being absent.
    """
    return run_remote(host, f"sha256sum {_shell_single_quote(remote_path)}")


__all__ = ["make_directory", "put_bytes", "remote_digest", "run_remote"]
