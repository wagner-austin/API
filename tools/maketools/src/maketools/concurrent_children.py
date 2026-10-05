"""Run several children side by side, each with its own captured output.

The real binding of :data:`maketools._test_hooks.run_concurrently`, held in
its own module the way :mod:`maketools.processes` holds the process-table
readers, so the hooks module stays a list of bindings.

EACH CHILD WRITES TO ITS OWN TEMPORARY FILE, never to a pipe. A pipe this
process did not drain while it waited on a sibling would fill and stall the
child writing to it, and a suite's verbose output fills one in seconds; a
file has no such limit, and is read back once the child has exited.

THE BOUND COVERS THE BATCH. Past it, every child still running is killed and
reaped before :class:`subprocess.TimeoutExpired` propagates, through an exit
callback registered as each child starts, so a child started before a later
one failed to start is never left behind either. A kill reaches the child
itself (a ``make``), as :func:`subprocess.run`'s own timeout does for
:data:`maketools._test_hooks.run_inheriting`; the descendants of a wedged
suite are the pre-run sweep's (:mod:`maketools.reap`).
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Final

from maketools.commands import ConcurrentOutcome

#: How often the batch is looked at for exits. A child's recorded seconds
#: are late by at most this, which is nothing against a lint or a suite.
POLL_SECONDS: Final[float] = 0.2


def _kill_if_running(child: subprocess.Popen[bytes]) -> None:
    """Kill and reap a child that has not exited; leave one that has.

    Args:
        child: The child.
    """
    if child.poll() is None:
        child.kill()
        child.wait()


def run_children_concurrently(
    argvs: Sequence[Sequence[str]],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> list[ConcurrentOutcome]:
    """Start every child, then wait for all of them.

    Args:
        argvs: One executable-and-arguments list per child.
        cwd: The working directory every child shares.
        env: Every child's complete environment.
        timeout_seconds: Wall-clock bound on the whole batch.

    Returns:
        One outcome per child, in the order of ``argvs``.

    Raises:
        subprocess.TimeoutExpired: When the batch outlives the bound, after
            every child still running has been killed. Its ``cmd`` lists the
            commands that had not exited.
    """
    started = time.monotonic()
    with ExitStack() as stack:
        captures: list[IO[bytes]] = []
        children: list[subprocess.Popen[bytes]] = []
        for argv in argvs:
            capture = stack.enter_context(tempfile.TemporaryFile())
            child = subprocess.Popen(
                list(argv), cwd=cwd, env=dict(env), stdout=capture, stderr=subprocess.STDOUT
            )
            stack.callback(_kill_if_running, child)
            captures.append(capture)
            children.append(child)
        exited: dict[int, float] = {}
        while True:
            for index, child in enumerate(children):
                if index not in exited and child.poll() is not None:
                    exited[index] = time.monotonic() - started
            if len(exited) == len(children):
                break
            if time.monotonic() - started > timeout_seconds:
                running = [" ".join(argvs[i]) for i in range(len(argvs)) if i not in exited]
                raise subprocess.TimeoutExpired(cmd=running, timeout=timeout_seconds)
            time.sleep(POLL_SECONDS)
        outcomes: list[ConcurrentOutcome] = []
        for index, (child, written) in enumerate(zip(children, captures, strict=True)):
            written.seek(0)
            outcomes.append(
                ConcurrentOutcome(
                    returncode=child.returncode,
                    output=written.read().decode("utf-8", errors="replace"),
                    seconds=exited[index],
                )
            )
        return outcomes


__all__ = ["POLL_SECONDS", "run_children_concurrently"]
