"""No clone out of a node's mirror runs beside a write into it (MCPs board task 5280a25d).

``git clone --local`` lists a mirror's object directory and links each file
it listed, refusing one that changed in between: packages/wiki-check's job
392cd878 was refused on lavender at 2026-11-05T20:48:59Z with 'fatal:
hardlink different from source at .../objects/info/commit-graphs/tmp_graph_IWyF5I',
a commit graph the detached maintenance of a fetch into that mirror was
writing. Here every part is real: a node mirror made by a real stage, a
real ``flock`` holder that keeps the mirror's lock, and the node's own
scripts run by ``/bin/sh``. Each case asserts that the script was still
waiting when the holder let go, which a script that takes no lock fails
whether it then succeeds or is refused, and that it then did its work.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time

from fleet.core import commit_stage, dialect, stage_repository
from tests.test_commit_stage import GIT_SECONDS, _commit, _git, _node_mirror_at

#: How long a holder keeps the lock while the script waits, in seconds: a
#: clone or a fetch of these few-file repositories takes milliseconds, so a
#: script still running after this long is waiting.
HELD_SECONDS = 1.0


class MirrorLockHolder:
    """A real ``flock`` on a node mirror's lock, held until :meth:`release`.

    While it holds the lock it appends to ``writing`` without pause, as
    ``git index-pack`` writes a fetch's pack or maintenance a commit graph,
    and removes the file when it lets go, as git renames its temporary files.
    """

    def __init__(self, mirror: pathlib.Path, writing: pathlib.Path) -> None:
        """Take the lock and require it taken.

        The lock's path is spelled out rather than asked of
        :func:`fleet.core.stage_repository.mirror_lock`, so the case pins the
        file every node script and this holder agree on.

        Args:
            mirror: The node's mirror.
            writing: The file appended to while the lock is held.
        """
        self._release = mirror.with_name(f"{mirror.name}.release")
        holding = 'echo held; while [ ! -e "$1" ]; do printf x >> "$2"; done; rm -f "$2"'
        self._answers, holder_writes = os.pipe()
        self._process = subprocess.Popen(
            (
                "flock",
                f"{mirror}.objects.lock",
                "/bin/sh",
                "-c",
                holding,
                "holder",
                str(self._release),
                str(writing),
            ),
            stdout=holder_writes,
        )
        os.close(holder_writes)
        assert os.read(self._answers, 4096) == b"held\n"

    def release(self) -> None:
        """Let go of the lock and require the holder ended cleanly."""
        self._release.touch()
        assert self._process.wait(timeout=GIT_SECONDS) == 0
        assert os.read(self._answers, 4096) == b""
        os.close(self._answers)


class Script:
    """A node script run by ``/bin/sh`` in a thread, its exit and words kept."""

    def __init__(self, text: str) -> None:
        """Start it.

        Args:
            text: The script.
        """
        self.done: subprocess.CompletedProcess[str] | None = None
        self._thread = threading.Thread(target=self._run, args=(text,))
        self._thread.start()

    def _run(self, text: str) -> None:
        self.done = subprocess.run(
            ("/bin/sh", "-c", text),
            capture_output=True,
            text=True,
            timeout=GIT_SECONDS,
            check=False,
        )

    def waiting(self) -> bool:
        """Whether it is still running."""
        return self._thread.is_alive()

    def finished(self) -> subprocess.CompletedProcess[str]:
        """Wait for it and answer how it ended."""
        self._thread.join(timeout=GIT_SECONDS)
        assert self.done is not None
        return self.done


def _staged_mirror(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, str]:
    """A hub repository and the node mirror its first stage made.

    Args:
        tmp_path: Where both are made.

    Returns:
        The hub, the node mirror and the commit it holds.
    """
    hub = tmp_path / "hub"
    sha = _commit(hub, {"src/a.py": "print(1)\n"})
    node = tmp_path / "node" / "repo.git"
    node.parent.mkdir()
    _node_mirror_at(hub, node, "run-1", None)
    return hub, node, sha


def test_a_clone_out_of_the_mirror_waits_while_a_write_into_it_holds_the_lock(
    tmp_path: pathlib.Path,
) -> None:
    hub, node, sha = _staged_mirror(tmp_path)
    target = node.parent / "run-2"
    target.mkdir()
    (node.parent / "run-2.stage").mkdir()
    holder = MirrorLockHolder(node, node / "objects" / "pack" / "tmp_pack_live")
    script = Script(
        commit_stage.unpack_script(
            mirror=str(node),
            bundle=None,
            run_id="run-2",
            sha=sha,
            scope=(),
            staged=f"{target}.stage",
            target=str(target),
            base_branch=None,
        )
    )

    time.sleep(HELD_SECONDS)
    waited = script.waiting()
    landed_while_held = sorted(path.name for path in target.iterdir())
    holder.release()
    done = script.finished()

    assert waited
    assert landed_while_held == []
    assert done.returncode == 0, done.stderr
    assert _git(target, "rev-parse", "HEAD^{tree}") == _git(hub, "rev-parse", "HEAD^{tree}")
    assert not (target / ".git" / "objects" / "pack" / "tmp_pack_live").exists()


def test_a_companion_clone_waits_while_a_write_into_its_mirror_holds_the_lock(
    tmp_path: pathlib.Path,
) -> None:
    _, node, sha = _staged_mirror(tmp_path)
    tree = tmp_path / "MCPs"
    tree.mkdir()
    holder = MirrorLockHolder(node, node / "objects" / "info" / "tmp_graph_live")
    script = Script(
        dialect.checked_script(
            stage_repository.companion_repository_commands(str(tree), str(node), sha, "main")
        )
    )

    time.sleep(HELD_SECONDS)
    waited = script.waiting()
    holder.release()
    done = script.finished()

    assert waited
    assert done.returncode == 0, done.stderr
    assert _git(tree, "rev-parse", "HEAD") == sha
    assert _git(tree, "rev-parse", "refs/remotes/origin/main") == sha


def test_a_fetch_into_the_mirror_waits_while_a_clone_holds_the_lock(
    tmp_path: pathlib.Path,
) -> None:
    hub, node, first = _staged_mirror(tmp_path)
    second = _commit(hub, {"src/a.py": "print(2)\n"})
    bundle = tmp_path / "run-2.bundle"
    commit_stage.bundle_commit(
        hub, sha=second, run_id="run-2", base=first, base_tip=None, destination=bundle
    )
    holder = MirrorLockHolder(node, tmp_path / "beside")
    script = Script(
        dialect.checked_script(
            commit_stage.fetch_commands(
                mirror=str(node), bundle=str(bundle), run_id="run-2", base_branch=False
            )
        )
    )

    time.sleep(HELD_SECONDS)
    waited = script.waiting()
    tip_while_held = _git(node, "rev-parse", commit_stage.NODE_TIP_REF)
    holder.release()
    done = script.finished()

    assert waited
    assert tip_while_held == first
    assert done.returncode == 0, done.stderr
    assert _git(node, "rev-parse", commit_stage.NODE_TIP_REF) == second


def test_the_maintenance_a_fetch_starts_has_ended_when_the_fetch_has(
    tmp_path: pathlib.Path,
) -> None:
    """A fetch's automatic maintenance runs inside it, under its lock, never
    detached behind it: with every fetch kept as a pack and one pack the
    limit, the fetch's own ``gc --auto`` packs the mirror's two into one,
    and that is done by the time the script that fetched has ended."""
    hub, node, first = _staged_mirror(tmp_path)
    # The first stage's few objects were written loose; packed, the next
    # fetch's pack is the mirror's second.
    _git(node, "repack", "-d", "--quiet")
    assert len(list((node / "objects" / "pack").glob("*.pack"))) == 1
    _git(node, "config", "fetch.unpackLimit", "1")
    _git(node, "config", "gc.autoPackLimit", "1")
    second = _commit(hub, {"src/a.py": "print(2)\n"})
    bundle = tmp_path / "run-2.bundle"
    commit_stage.bundle_commit(
        hub, sha=second, run_id="run-2", base=first, base_tip=None, destination=bundle
    )

    done = Script(
        dialect.checked_script(
            commit_stage.fetch_commands(
                mirror=str(node), bundle=str(bundle), run_id="run-2", base_branch=False
            )
        )
    ).finished()

    assert done.returncode == 0, done.stderr
    assert len(list((node / "objects" / "pack").glob("*.pack"))) == 1
    assert not (node / "gc.pid").exists()
    assert _git(node, "rev-parse", commit_stage.NODE_TIP_REF) == second
