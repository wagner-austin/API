"""Read the session host's enrolments without moving announcement ownership off the hub."""

from __future__ import annotations

import pathlib
import re
import shlex

from ci_wake import _test_hooks
from ci_wake.enrolment import PushAttempt, decode_attempts


def enrolment_argv(host: str, path: pathlib.Path) -> tuple[str, ...]:
    """Build a noninteractive read of the explicitly named source.

    Args:
        host: SSH host name or configured alias, never SSH options.
        path: Absolute source path on that host.

    Returns:
        An SSH argv with the remote path quoted for its shell.

    Raises:
        ValueError: The host or source path is unsafe or relative.
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host) is None or not path.is_absolute():
        raise ValueError("CI_WAKE_ENROLMENT_SOURCE_INVALID: expected host and absolute path")
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        f"cat -- {shlex.quote(str(path))}",
    )


def read_remote_attempts(host: str, path: pathlib.Path) -> tuple[PushAttempt, ...]:
    """Read remote enrolments, refusing missing files and transport errors.

    Args:
        host: The source's SSH host.
        path: Absolute enrolment path, independent of the hub position file.

    Returns:
        Rows decoded by the same parser used for local enrolment.

    Raises:
        ValueError: Invalid source or nonzero transport exit.
        subprocess.TimeoutExpired: SSH did not finish within thirty seconds.
        JSONTypeError: The source contains malformed rows.
    """
    completed = _test_hooks.run_process(enrolment_argv(host, path), timeout=30)
    if completed.returncode != 0:
        raise ValueError(
            f"CI_WAKE_ENROLMENT_UNREADABLE: {host}:{path}: "
            f"exit {completed.returncode}: {completed.stderr}"
        )
    return decode_attempts(completed.stdout, f"{host}:{path}")


__all__ = ["enrolment_argv", "read_remote_attempts"]
