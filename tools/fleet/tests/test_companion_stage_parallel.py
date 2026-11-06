"""Companion staging executes real bundles, transfers, digests, fetches and clones.

Only SSH routing is replaced: the same rendered scripts run locally, scp
becomes a real cp process, and the hub's git runs on a real hub mirror. A
barrier makes serial staging fail rather than relying on a timing assertion
(board task 007bfc7e). A second stage of a commit the node's mirror reaches
sends nothing, and a companion clean at that commit is kept (MCPs board task
5280a25d).
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, companion_stage, export
from tests.test_stage_repository import _git, source_repository

HOST = "stage-test"


@pytest.fixture(name="real_pool", autouse=True)
def _real_pool() -> None:
    """Stage on the real pool, whose overlap each case is about."""
    _test_hooks.executor = _test_hooks._default_executor


class LocalTransport:
    """Route the real remote scripts to local processes with bounded overlap.

    Attributes:
        peak: Greatest number of simultaneous transfers.
        active: Transfers still running when a stage returns or fails.
        copies: How many transfers ran.
        hub: Each git command run on the hub's side, in order.
    """

    def __init__(self, width: int, *, spoil: str = "", removing: bool = False) -> None:
        """Require one full admitted wave to reach its transfer boundary.

        Args:
            width: How many transfers must be under way at once.
            spoil: A companion directory whose bundle is damaged in transit.
            removing: Whether that bundle is removed rather than truncated.
        """
        self.peak = 0
        self.active = 0
        self.copies = 0
        self.hub: list[tuple[str, ...]] = []
        self._spoil = spoil
        self._removing = removing
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(width, timeout=10)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Execute each script, transfer and hub git command through the real process seam."""
        if argv[0] == "scp":
            source = pathlib.Path(argv[-2])
            if self._spoil and f"/{self._spoil}.stage/" in argv[-1]:
                if self._removing:
                    source.unlink()
                else:
                    source.write_bytes(b"truncated in transit")
            with self._lock:
                self.active += 1
                self.copies += 1
                self.peak = max(self.peak, self.active)
            self._barrier.wait()
            result = _test_hooks._default_run(
                ("cp", str(source), argv[-1].removeprefix(f"{HOST}:")),
                timeout_seconds=timeout_seconds,
            )
            with self._lock:
                self.active -= 1
            return result
        if argv[0] == "git":
            self.hub.append(tuple(argv))
            return _test_hooks._default_run(argv, timeout_seconds=timeout_seconds)
        remote = argv[argv.index(HOST) + 1 :]
        command = ("sh", "-c", remote[0]) if len(remote) == 1 else tuple(remote)
        return _test_hooks._default_run(
            command,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


def companions_at(tmp_path: pathlib.Path, count: int) -> tuple[export.CompanionExport, ...]:
    """Companions of their own remotes, each held by a real hub mirror at one tip."""
    source, _first, tip = source_repository(tmp_path)
    hub = tmp_path / "hub" / "companion-wagner-austin-MCPs.git"
    _git("init", "--bare", "--quiet", str(hub))
    _git("-C", str(hub), "fetch", "--quiet", str(source), f"+{tip}:refs/fleet/companion")
    return tuple(
        export.CompanionExport(
            directory=f"companion-{index}",
            ref="main",
            sha=tip,
            mirror=hub,
            key=f"companion-wagner-austin-remote-{index}",
            bundle=tmp_path / "attempt" / f"companion-{index}-{tip}.bundle",
        )
        for index in range(count)
    )


def _stage(
    stage_root: pathlib.Path, companions: tuple[export.CompanionExport, ...], workers: int
) -> tuple[companion_stage.StagedCompanion, ...]:
    """Stage the companions on the local node.

    Args:
        stage_root: The node's stage root.
        companions: What to stage.
        workers: The job's worker grant.

    Returns:
        Each companion's stage.
    """
    return companion_stage.stage_companions(
        HOST, stage_root=str(stage_root), run_id="run-1", companions=companions, workers=workers
    )


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_real_staging_overlaps_only_within_allocation_and_preserves_history(
    tmp_path: pathlib.Path, workers: int
) -> None:
    companions = companions_at(tmp_path, 8)
    stage_root = tmp_path / "node"
    stage_root.mkdir()
    dirty = stage_root / "companion-0"
    dirty.mkdir()
    (dirty / "removed.txt").write_text("must disappear", encoding="utf-8")
    transport = LocalTransport(min(workers, companion_stage.STAGE_WIDTH))
    _test_hooks.run = transport

    staged = _stage(stage_root, companions, workers)

    assert [row["tree"] for row in staged] == [
        str(stage_root / row["directory"]) for row in companions
    ]
    assert transport.peak == min(workers, companion_stage.STAGE_WIDTH)
    assert transport.active == 0
    assert not (dirty / "removed.txt").exists()
    for row in staged:
        path = row["tree"]
        assert _git("-C", path, "rev-parse", "HEAD") == companions[0]["sha"]
        assert _git("-C", path, "rev-parse", "origin/main") == companions[0]["sha"]
        assert _git("-C", path, "show", "HEAD~1:packages/maketools/run.py") == "print(1)"
        assert " over an empty mirror, to " in row["detail"]
    assert not any(row["bundle"].exists() for row in companions)


def test_a_second_stage_sends_nothing_and_keeps_a_clean_companion(tmp_path: pathlib.Path) -> None:
    """The node's mirror reaches the commit, so nothing is bundled or sent;
    the companion is clean at it, so it is kept. Dirtied, it is replaced from
    the mirror, still with nothing sent."""
    (companion,) = companions_at(tmp_path, 1)
    stage_root = tmp_path / "node"
    stage_root.mkdir()
    first = LocalTransport(1)
    _test_hooks.run = first
    (staged,) = _stage(stage_root, (companion,), 1)
    tree = pathlib.Path(staged["tree"])
    kept = tree / "packages" / "maketools" / "run.py"
    inode = kept.stat().st_ino

    again = LocalTransport(1)
    _test_hooks.run = again
    (reused,) = _stage(stage_root, (companion,), 1)

    assert (first.copies, again.copies) == (1, 0)
    assert reused["detail"] == (
        f"0 bytes, the companion-0 companion at {companion['sha']} kept clean at {tree}"
    )
    assert kept.stat().st_ino == inode

    (tree / "left-by-a-build.txt").write_text("stale", encoding="utf-8")
    replaced_by = LocalTransport(1)
    _test_hooks.run = replaced_by
    (replaced,) = _stage(stage_root, (companion,), 1)

    assert replaced_by.copies == 0
    assert replaced["detail"] == (
        f"0 bytes, the companion-0 companion at {companion['sha']} already in the node's "
        f"mirror, to {tree}"
    )
    assert not (tree / "left-by-a-build.txt").exists()
    assert _git("-C", str(tree), "rev-parse", "HEAD") == companion["sha"]


def test_a_later_commit_is_sent_over_the_node_mirror_s_tip(tmp_path: pathlib.Path) -> None:
    """The companion's ref moved since its last stage on this node: the
    bundle leaves out the history the node's mirror already reaches."""
    (at_tip,) = companions_at(tmp_path, 1)
    first = _git("-C", str(at_tip["mirror"]), "rev-parse", f"{at_tip['sha']}~1")
    at_first = export.CompanionExport(
        directory=at_tip["directory"],
        ref=at_tip["ref"],
        sha=first,
        mirror=at_tip["mirror"],
        key=at_tip["key"],
        bundle=at_tip["bundle"],
    )
    stage_root = tmp_path / "node"
    stage_root.mkdir()
    _test_hooks.run = LocalTransport(1)
    (earlier,) = _stage(stage_root, (at_first,), 1)
    transport = LocalTransport(1)
    _test_hooks.run = transport

    (later,) = _stage(stage_root, (at_tip,), 1)

    assert earlier["detail"].endswith(f"over an empty mirror, to {earlier['tree']}")
    assert later["detail"].endswith(f"at {at_tip['sha']} over {first}, to {later['tree']}")
    bundled = [call[-2:] for call in transport.hub if "bundle" in call and "create" in call]
    assert bundled == [("refs/fleet/bundle/run-1-companion-0", f"^{first}")]
    assert _git("-C", later["tree"], "rev-parse", "HEAD") == at_tip["sha"]


@pytest.mark.parametrize("missing", [False, True])
def test_failed_transfer_refuses_and_joins_other_real_staging(
    tmp_path: pathlib.Path, missing: bool
) -> None:
    companions = companions_at(tmp_path, 2)
    stage_root = tmp_path / "node"
    stage_root.mkdir()
    transport = LocalTransport(2, spoil="companion-0", removing=missing)
    _test_hooks.run = transport

    with pytest.raises(AppError) as excinfo:
        _stage(stage_root, companions, 2)

    assert excinfo.value.code is (
        FleetErrorCode.DISPATCH_FAILED if missing else FleetErrorCode.STAGE_DIGEST_MISMATCH
    )
    assert transport.peak == 2
    assert transport.active == 0
    assert not (stage_root / "companion-0" / ".git").exists()
    assert _git("-C", str(stage_root / "companion-1"), "rev-parse", "HEAD") == companions[1]["sha"]


def test_no_companions_stage_nothing(tmp_path: pathlib.Path) -> None:
    assert _stage(tmp_path / "absent", (), 1) == ()
    assert not (tmp_path / "absent").exists()
