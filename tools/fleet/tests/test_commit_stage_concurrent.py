"""Exports run at once from one hub mirror all succeed (MCPs board task 695b0b02).

Fleet job 2243695b was refused ``EXPORT_FAILED`` at 2026-11-05T06:17:50Z
because its ``git update-ref -d`` of its bundle ref found the mirror's
``packed-refs.lock`` held past git's default 1 s wait, by one of the 13 other
exports of the same commit or by git's own maintenance on a loaded hub.
Here every part is real: one bare mirror, :data:`EXPORTS` real
:func:`fleet.core.commit_stage.bundle_commit` calls in threads, and a real
``git update-ref --stdin`` transaction holding ``packed-refs.lock`` as any
other writer of the mirror holds it. The holder keeps the lock
:data:`HELD_PAST_SECONDS` past the instant every export reached its ref
deletion, three times git's default wait, so without git waiting longer
every export is refused with the job's own words; and the test asserts by
order, never by elapsed time, that each export finished only after the
holder gave the lock back.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time
from collections.abc import Sequence

from platform_core.errors import AppError

from fleet.core import _test_hooks, commit_stage
from tests.test_commit_stage import GIT_SECONDS, _commit, _git

#: How many exports of one commit run at once, as a push gate queues them.
EXPORTS = 8

#: How long the holder keeps ``packed-refs.lock`` after every export has
#: started deleting its bundle ref, in seconds: three times git's default
#: ``core.packedRefsTimeout`` of 1 s, so a deletion that waits only that long
#: has given up before the lock is free.
HELD_PAST_SECONDS = 3.0

#: The ref the holder's transaction deletes, which takes ``packed-refs.lock``.
HELD_REF = "refs/hold/writer"


class DeletionsReached:
    """The real process seam, which counts every export into its ref deletion.

    Attributes:
        reached: Passed by every export once its deletion is about to run,
            and by the holder, so the holder's wait starts when all are there.
    """

    def __init__(self, exports: int) -> None:
        """Expect ``exports`` deletions and the holder at the barrier."""
        self.reached = threading.Barrier(exports + 1, timeout=GIT_SECONDS)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Run the command for real, first meeting the holder when it deletes a ref."""
        if "-d" in argv:
            self.reached.wait()
        return _test_hooks._default_run(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


class PackedRefsHolder:
    """A real git transaction holding the mirror's ``packed-refs.lock``.

    ``git update-ref --stdin`` takes every lock of a prepared transaction,
    ``packed-refs.lock`` among them for a deletion, and keeps them until it
    reads ``commit``.
    """

    def __init__(self, mirror: pathlib.Path, sha: str) -> None:
        """Prepare a deletion of :data:`HELD_REF` and require the lock taken.

        Args:
            mirror: The hub's mirror.
            sha: A commit :data:`HELD_REF` is written at first.
        """
        _git(mirror, "update-ref", HELD_REF, sha)
        git_reads, self._sends = os.pipe()
        self._answers, git_writes = os.pipe()
        self._process = subprocess.Popen(
            ("git", "-C", str(mirror), "update-ref", "--stdin"),
            stdin=git_reads,
            stdout=git_writes,
        )
        os.close(git_reads)
        os.close(git_writes)
        self._said(f"start\ndelete {HELD_REF}\nprepare\n", ("start: ok", "prepare: ok"))
        assert (mirror / "packed-refs.lock").is_file()

    def _said(self, lines: str, answers: tuple[str, ...]) -> None:
        """Send the transaction lines and require git's answers, in order."""
        os.write(self._sends, lines.encode("utf-8"))
        said = b""
        while said.count(b"\n") < len(answers):
            chunk = os.read(self._answers, 4096)
            assert chunk, f"git ended the transaction after {said!r}"
            said += chunk
        assert tuple(said.decode("utf-8").splitlines()) == answers

    def release(self) -> None:
        """Commit the transaction, which gives ``packed-refs.lock`` back."""
        self._said("commit\n", ("commit: ok",))
        os.close(self._sends)
        assert self._process.wait(timeout=GIT_SECONDS) == 0
        assert os.read(self._answers, 4096) == b""
        os.close(self._answers)


def test_exports_waiting_on_a_held_packed_refs_lock_all_succeed(tmp_path: pathlib.Path) -> None:
    sha = _commit(tmp_path / "work", {"a.txt": "one\n"})
    mirror = tmp_path / "hub.git"
    _git(tmp_path, "clone", "--quiet", "--bare", str(tmp_path / "work"), str(mirror))
    seam = DeletionsReached(EXPORTS)
    _test_hooks.run = seam
    holder = PackedRefsHolder(mirror, sha)
    order: list[str] = []
    noted = threading.Lock()

    def export(index: int) -> None:
        try:
            commit_stage.bundle_commit(
                mirror,
                sha=sha,
                run_id=f"run-{index}",
                base=None,
                destination=tmp_path / "bundles" / f"run-{index}.bundle",
            )
        except AppError as refused:
            with noted:
                order.append(f"refused run-{index}: {refused.message}")
            return
        with noted:
            order.append(f"exported run-{index}")

    threads = [threading.Thread(target=export, args=(index,)) for index in range(EXPORTS)]
    for thread in threads:
        thread.start()
    seam.reached.wait()
    time.sleep(HELD_PAST_SECONDS)
    with noted:
        order.append("released")
    holder.release()
    for thread in threads:
        thread.join(timeout=GIT_SECONDS)

    # Every export waited for the lock and finished once it was given back.
    assert order[0] == "released"
    assert sorted(order[1:]) == sorted(f"exported run-{index}" for index in range(EXPORTS))
    # Each run's ref went once its bundle was written, and the holder's
    # deletion landed too, so both sides wrote packed-refs.
    assert _git(mirror, "for-each-ref", "refs/fleet/bundle/", "refs/hold/") == ""
    for index in range(EXPORTS):
        bundle = tmp_path / "bundles" / f"run-{index}.bundle"
        heads = _git(mirror, "bundle", "list-heads", str(bundle))
        assert heads == f"{sha} refs/fleet/bundle/run-{index}"
