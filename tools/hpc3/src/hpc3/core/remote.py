"""Running commands on the cluster and sending bytes to it.

Every remote call goes through ``ssh`` with an argv list rather than a shell
string. Corpus filenames and payload commands are attacker-adjacent only in
the sense that they are arbitrary text, but arbitrary text is enough: a
filename holding a quote or a semicolon would be interpreted rather than
transferred, and the resulting failure would look like a corrupt file.

Nothing here catches. Commands run with ``check=False`` and their exit status
is inspected, so a remote failure becomes an :class:`~platform_core.errors.AppError`
that names the command and carries the cluster's own stderr.

A command line also has a LENGTH, and on this submitter that is the limit that
fires first -- before Slurm, before ssh, before the cluster is reached at all.
:func:`token_batches` and :func:`run_remote_batched` are how a query over a
list stays inside it; :data:`MAX_COMMAND_CHARS` carries the two failures that
put them here.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import _test_hooks

#: Longest remote command line this package will build out of a list. Not a
#: Slurm limit and not an ssh limit: the cap is LOCAL, on the submitter, and it
#: fires before anything reaches the cluster. Two have now been met here, at
#: different sizes and wearing different disguises:
#:
#: * A 136-member campaign packed every artifact into one shell line, ~10 KB,
#:   past cmd.exe's 8191-character argument limit. The command arrived at the
#:   remote bash TRUNCATED and died on "unexpected end of file"
#:   (vhsearch2-r0, 2026-09-02).
#: * ``hpc3-triage`` asked accounting about every open ledger entry -- 6645
#:   rows -- building a ~70 KB argv, past CreateProcess's 32767-character
#:   limit. Python raised ``FileNotFoundError: [WinError 206] The filename or
#:   extension is too long``, which names no command and so reads as a missing
#:   executable rather than as an argument that outgrew its call (2026-09-05).
#:
#: 4000 sits under the lower of the two with room for the ssh argv around it,
#: and is deliberately far from both: a query worth splitting at all is worth
#: splitting well inside every limit it can meet.
MAX_COMMAND_CHARS = 4000


def token_batches(tokens: Sequence[str], *, overhead: int, separator: str) -> list[list[str]]:
    """Group a command's variable part so each group fits one command line.

    Split by MEASURED WIDTH rather than by a token count. A count is a guess
    about how long a token is, and it is wrong the moment ids or paths get
    longer than the day it was chosen -- producing exactly the over-long
    command it was meant to prevent, at a size nobody re-derives.

    Args:
        tokens: The variable part of a command, in the order it must be sent.
            Already quoted if the caller quotes: what is measured has to be
            what is sent.
        overhead: Characters the command spends on everything else -- program
            name, flags, format string. Counted by the caller, the only party
            that knows its own command.
        separator: What will join the tokens in the built command.

    Returns:
        The tokens, in order, grouped so each group's joined width plus
        ``overhead`` is at most :data:`MAX_COMMAND_CHARS`. One group when they
        all fit, which is the ordinary case and rebuilds exactly the single
        command this replaced; no groups for no tokens, which every caller
        refuses by name before reaching here.

    Raises:
        ValueError: If one token cannot fit even alone. Emitting it anyway
            would rebuild the over-long command this exists to split, and the
            failure would again name neither the token nor the limit.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    width = overhead
    for token in tokens:
        alone = overhead + len(token)
        if alone > MAX_COMMAND_CHARS:
            raise ValueError(
                f"{token!r} needs {alone} characters in a command carrying {overhead} "
                f"of overhead, over the {MAX_COMMAND_CHARS}-character limit"
            )
        # Charged only when there is something to separate from, so a batch's
        # measured width is the width it will really have.
        addition = len(token) + (len(separator) if current != [] else 0)
        if width + addition > MAX_COMMAND_CHARS:
            batches.append(current)
            current = []
            width = overhead
            addition = len(token)
        current.append(token)
        width += addition
    if current != []:
        batches.append(current)
    return batches


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


def run_remote_batched(host: str, commands: Sequence[str]) -> str:
    """Run the batches of one split query and return their combined output.

    The commands are one question that did not fit in a single call, so their
    outputs are the rows of one answer and are returned as one.

    Recombined line by line rather than concatenated: a batch whose output did
    not end in a newline would otherwise fuse its last row onto the next
    batch's first, and every parser here reads rows -- so the fusion would
    present as one malformed row rather than as a transport defect, and the
    two rows it ate would be gone.

    Args:
        host: SSH destination.
        commands: Command lines to run, in order.

    Returns:
        Every output line, in order, each newline-terminated. Empty when the
        batches reported nothing, which is what an unqueued account looks
        like.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.REMOTE_COMMAND_FAILED` if
            any batch fails. One failed batch fails the whole query: a partial
            answer reads as the complete one, and the rows it is missing are
            the ones nobody will go looking for.
    """
    lines = [line for command in commands for line in run_remote(host, command).splitlines()]
    return "".join(f"{line}\n" for line in lines)


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


__all__ = [
    "MAX_COMMAND_CHARS",
    "make_directory",
    "put_bytes",
    "remote_digest",
    "run_remote",
    "run_remote_batched",
    "token_batches",
]
