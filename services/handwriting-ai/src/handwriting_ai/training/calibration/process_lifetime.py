"""Bound a calibration candidate and its loader workers to one lifetime.

A candidate is a spawned child that builds a ``DataLoader`` with worker
processes. Killing the candidate alone used to leave those workers behind
(board task 49899fd2): on Windows nothing tied them to it, and on POSIX a
forkserver sat between them and it. The candidate now claims ownership of
everything it will create before creating any of it, and the runner closes
that ownership in its ``finally`` whatever the outcome was: a kill-on-close
job on Windows (:mod:`platform_core.process_job`), its own process group on
POSIX (:mod:`platform_core.process_group`).
"""

from __future__ import annotations

from typing import Protocol

from platform_core.process_group import isolate_in_new_group, kill_group
from platform_core.process_job import join_kill_on_close_job

from handwriting_ai import _test_hooks

#: How long the runner waits for a killed candidate to be reaped.
EXIT_TIMEOUT_SECONDS = 10.0


class OwnedProcess(Protocol):
    """The lifecycle operations used on a started multiprocessing process."""

    @property
    def pid(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


def isolate_candidate() -> None:
    """Claim ownership of every process this candidate is about to create.

    Called in the candidate itself, before any loader exists: a POSIX
    candidate becomes the leader of its own group, a Windows candidate joins
    a kill-on-close job that every descendant inherits.

    Raises:
        RuntimeError: ``CALIBRATION_OWNERSHIP`` when the platform refused,
            naming the call and its error; a candidate that cannot own its
            workers must not start them.
    """
    if _test_hooks.os_name == "posix":
        problem = isolate_in_new_group(_test_hooks.group_api())
    else:
        problem = join_kill_on_close_job(_test_hooks.job_api())
    if problem:
        raise RuntimeError(f"CALIBRATION_OWNERSHIP: {problem}")


def close_candidate(process: OwnedProcess) -> None:
    """Stop the owned candidate, then remove every process it left behind.

    Args:
        process: The started candidate created by this runner.

    Raises:
        RuntimeError: ``CALIBRATION_NOT_STARTED`` for a process with no pid,
            ``CALIBRATION_EXIT_TIMEOUT`` for one still alive after a kill
            and a bounded join, ``CALIBRATION_GROUP_SWEEP`` when the POSIX
            group could not be signalled.
    """
    pid = process.pid
    if pid is None:
        raise RuntimeError("CALIBRATION_NOT_STARTED: candidate has no process id")
    if process.is_alive():
        process.kill()
    process.join(timeout=EXIT_TIMEOUT_SECONDS)
    if process.is_alive():
        raise RuntimeError("CALIBRATION_EXIT_TIMEOUT: candidate did not exit")
    if _test_hooks.os_name == "posix":
        # The leader has exited, so it cannot create another member after
        # this sweep; the group id is its pid, and the group survives while
        # any worker does. On Windows the job's last handle closed with it.
        problem = kill_group(_test_hooks.group_api(), pid)
        if problem:
            raise RuntimeError(f"CALIBRATION_GROUP_SWEEP: {problem}")


__all__ = ["EXIT_TIMEOUT_SECONDS", "OwnedProcess", "close_candidate", "isolate_candidate"]
