"""Run one metadata job on WRDS Grid Engine through the cluster transport.

Both WRDS inventories -- the filesystem scan and the SQL catalog -- upload a
program, wait for a bounded synchronous job and copy back one verified
artifact. The steps and their refusals live here once, so the two commands
cannot drift on what counts as a verified upload, a scheduler receipt or a
complete download.
"""

from __future__ import annotations

import hashlib
import pathlib
import shlex

from hpc3.core import _test_hooks
from hpc3.core.digest import check_remote_digest, parse_remote_digest
from hpc3.core.remote import put_bytes, remote_digest, run_remote


def receipt_path(output: pathlib.Path) -> pathlib.Path:
    """Name the receipt saved beside a local artifact.

    Args:
        output: Local artifact path.

    Returns:
        ``<output>.receipt.json``.
    """
    return output.with_suffix(output.suffix + ".receipt.json")


def refuse_existing(paths: list[pathlib.Path]) -> None:
    """Refuse to overwrite any earlier run's local artifact.

    Checked before anything is submitted, so a refused run costs no job.

    Args:
        paths: Every local file a run will write.

    Raises:
        FileExistsError: Naming the first that already exists.
    """
    for path in paths:
        if _test_hooks.file_exists(path):
            raise FileExistsError(str(path))


def remote_directory(remote_dir: str) -> pathlib.PurePosixPath:
    """Validate a new job directory beneath the user's WRDS home.

    Args:
        remote_dir: Requested absolute directory.

    Returns:
        The validated path.

    Raises:
        ValueError: ``WRDS_CATALOG_PATH`` when the path is relative, climbs
            with ``..`` or lies outside ``/home``.
    """
    remote = pathlib.PurePosixPath(remote_dir)
    if not remote.is_absolute() or ".." in remote.parts or not str(remote).startswith("/home/"):
        raise ValueError("WRDS_CATALOG_PATH: remote directory must be beneath /home")
    return remote


def create_directory(host: str, remote: pathlib.PurePosixPath) -> None:
    """Create the job directory, private and new.

    Args:
        host: WRDS SSH alias.
        remote: Directory from :func:`remote_directory`.

    Raises:
        AppError: When the directory exists or cannot be created; a reused
            directory could return an earlier run's artifact as this one's.
    """
    run_remote(host, shlex.join(["mkdir", "-m", "700", "--", str(remote)]))


def upload(host: str, path: str, payload: bytes) -> str:
    """Write a file on WRDS and confirm the bytes that arrived.

    Args:
        host: WRDS SSH alias.
        path: Absolute destination.
        payload: Exact bytes.

    Returns:
        The payload's sha256.

    Raises:
        AppError: When the transfer fails or the remote digest differs.
    """
    digest = hashlib.sha256(payload).hexdigest()
    put_bytes(host, path, payload)
    check_remote_digest(path, digest, parse_remote_digest(remote_digest(host, path), path))
    return digest


def submit(
    host: str, remote: pathlib.PurePosixPath, name: str, command: list[str], *, environment: bool
) -> str:
    """Run one job synchronously under a one-hour bound.

    Args:
        host: WRDS SSH alias.
        remote: Job directory receiving the scheduler's stdout and stderr.
        name: Grid Engine job name.
        command: Program and arguments executed on the compute node.
        environment: Export the login shell's environment to the job. The
            SQL catalog needs it: WRDS Cloud sets the ident-authenticated
            database host there, and the job has no password to offer.

    Returns:
        The scheduler's job id.

    Raises:
        AppError: When the job exits non-zero; ``qsub -sync y`` exits with
            the job's status.
        ValueError: ``WRDS_JOB_ID`` when the scheduler's first line is not an id.
    """
    exported = ["-V"] if environment else []
    argv = [
        "qsub",
        "-terse",
        "-sync",
        "y",
        *exported,
        "-b",
        "y",
        "-N",
        name,
        "-l",
        "h_rt=3600",
        "-o",
        str(remote / f"{name}.out"),
        "-e",
        str(remote / f"{name}.err"),
        *command,
    ]
    scheduler = run_remote(host, shlex.join(argv))
    job_id = scheduler.splitlines()[0] if scheduler.splitlines() else ""
    if not job_id.isdecimal():
        raise ValueError(f"WRDS_JOB_ID: unrecognized scheduler receipt {scheduler!r}")
    return job_id


__all__ = [
    "create_directory",
    "receipt_path",
    "refuse_existing",
    "remote_directory",
    "submit",
    "upload",
]
