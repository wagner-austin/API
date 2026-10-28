"""Compute-node effects, shared by the real worker and its executable tests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import pathlib
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from platform_core.config import _test_hooks as config_hooks
from platform_core.environment_record import HostRecord, capture_host_record, stdlib_host_probe

from hpc3.contracts.pins import normalise_name
from hpc3.contracts.wrds_run import WrdsRuntime


def _default_digest(path: pathlib.Path) -> str:
    """Hash a file without loading its dataset into memory.

    Args:
        path: File to hash.

    Returns:
        Content SHA256.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _default_runtime() -> WrdsRuntime:
    """Capture the executing interpreter and every resolved distribution.

    Returns:
        Interpreter bytes/version and normalized installed versions.
    """
    names: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name: str | None = distribution.metadata["Name"]
        if name is None:
            raise ValueError("WRDS_RUNTIME: distribution has no Name metadata")
        names.add(normalise_name(name))
    return WrdsRuntime(
        python=sys.executable,
        sha256=_default_digest(pathlib.Path(sys.executable)),
        version=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in sorted(names)},
    )


def _default_host() -> HostRecord:
    """Capture the machine fingerprint used by the shared run-record contract.

    Returns:
        Operating system, architecture and processor count.
    """
    return capture_host_record(stdlib_host_probe(os.cpu_count))


class Execute(Protocol):
    """Run a bounded payload with logs written directly to its run directory."""

    def __call__(
        self,
        argv: Sequence[str],
        directory: pathlib.Path,
        environment: Mapping[str, str],
        seconds: int,
    ) -> int:
        """Execute the payload and return its exit status.

        Args:
            argv: Executable and literal arguments.
            directory: Working directory containing output and log paths.
            environment: Exact child environment.
            seconds: Wall-clock deadline.

        Returns:
            Child exit status.
        """
        ...


def _default_execute(
    argv: Sequence[str], directory: pathlib.Path, environment: Mapping[str, str], seconds: int
) -> int:
    """Execute the real child; timeout and process errors propagate.

    Args:
        argv: Executable and arguments, never shell text.
        directory: Working directory.
        environment: Child environment.
        seconds: Deadline.

    Returns:
        Child exit status.
    """
    with (
        (directory / "payload.out").open("wb") as stdout,
        (directory / "payload.err").open("wb") as stderr,
    ):
        return subprocess.run(
            list(argv),
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            timeout=seconds,
            check=False,
        ).returncode


def _default_read(path: pathlib.Path) -> bytes:
    """Read an explicitly requested file.

    Args:
        path: File path.

    Returns:
        Its bytes.
    """
    return path.read_bytes()


def _default_write(path: pathlib.Path, payload: bytes) -> None:
    """Write a new result, refusing replacement of an earlier attempt.

    Args:
        path: New file path, with existing parent.
        payload: Bytes to persist.
    """
    with path.open("xb") as stream:
        stream.write(payload)


def _default_new_directory(path: pathlib.Path) -> None:
    """Claim a new local collection directory, refusing existing attempts.

    Args:
        path: New directory with an existing parent.
    """
    path.mkdir(mode=0o700)


digest_file: Callable[[pathlib.Path], str] = _default_digest
runtime: Callable[[], WrdsRuntime] = _default_runtime
host: Callable[[], HostRecord] = _default_host
execute: Execute = _default_execute
environment: Callable[[], dict[str, str]] = config_hooks.get_environment
read: Callable[[pathlib.Path], bytes] = _default_read
write: Callable[[pathlib.Path, bytes], None] = _default_write
new_directory: Callable[[pathlib.Path], None] = _default_new_directory


def reset_hooks() -> None:
    """Restore all production implementations."""
    global digest_file, runtime, host, execute, environment, read, write, new_directory
    digest_file, runtime, host = _default_digest, _default_runtime, _default_host
    execute, environment, read, write = (
        _default_execute,
        config_hooks.get_environment,
        _default_read,
        _default_write,
    )
    new_directory = _default_new_directory


__all__ = [
    "Execute",
    "digest_file",
    "environment",
    "execute",
    "host",
    "new_directory",
    "read",
    "reset_hooks",
    "runtime",
    "write",
]
