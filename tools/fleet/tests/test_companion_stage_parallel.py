"""Bounded companion staging executes real transfers, digests and Git clones.

Only SSH routing is replaced: the same rendered scripts run locally, and
scp becomes a real cp process. A barrier makes serial staging fail rather
than relying on a timing assertion. Board task 007bfc7e.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, export, staging
from tests.test_stage_repository import _bundled_companion, _git


class LocalTransport:
    """Route the real remote scripts to local processes with bounded overlap.

    Attributes:
        peak: Greatest number of simultaneous transfers.
        active: Transfers still running when a stage returns or fails.
    """

    def __init__(self, width: int) -> None:
        """Require one full admitted wave to reach its transfer boundary."""
        self.peak = 0
        self.active = 0
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
        """Execute each script and file transfer through the real process seam."""
        if argv[0] == "scp":
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            self._barrier.wait()
            result = _test_hooks._default_run(
                ("cp", argv[-2], argv[-1].removeprefix("stage-test:")),
                timeout_seconds=timeout_seconds,
            )
            with self._lock:
                self.active -= 1
            return result
        remote = argv[argv.index("stage-test") + 1 :]
        command = ("sh", "-c", remote[0]) if len(remote) == 1 else tuple(remote)
        return _test_hooks._default_run(
            command,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


def companions_at(tmp_path: pathlib.Path, count: int) -> tuple[export.CompanionExport, ...]:
    """Make independent bundle files from one real two-commit repository."""
    bundle, _first, tip = _bundled_companion(tmp_path)
    payload = bundle.read_bytes()
    companions: list[export.CompanionExport] = []
    for index in range(count):
        source = tmp_path / f"bundle-{index}"
        source.write_bytes(payload)
        companions.append(
            export.CompanionExport(
                directory=f"companion-{index}",
                ref="main",
                sha=tip,
                path=source,
                data=payload,
            )
        )
    return tuple(companions)


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
    transport = LocalTransport(min(workers, 4))
    _test_hooks.run = transport

    paths = staging.stage_companions(
        "stage-test", stage_root=str(stage_root), companions=companions, workers=workers
    )

    assert paths == tuple(str(stage_root / row["directory"]) for row in companions)
    assert transport.peak == min(workers, 4)
    assert transport.active == 0
    assert not (dirty / "removed.txt").exists()
    for path, row in zip(paths, companions, strict=True):
        assert _git("-C", path, "rev-parse", "HEAD") == row["sha"]
        assert _git("-C", path, "rev-parse", "origin/main") == row["sha"]
        assert _git("-C", path, "show", "HEAD~1:packages/maketools/run.py") == "print(1)"


@pytest.mark.parametrize("missing", [False, True])
def test_failed_transfer_refuses_and_joins_other_real_staging(
    tmp_path: pathlib.Path, missing: bool
) -> None:
    companions = companions_at(tmp_path, 2)
    if missing:
        companions[0]["path"].unlink()
    else:
        companions[0]["path"].write_bytes(b"truncated in transit")
    stage_root = tmp_path / "node"
    stage_root.mkdir()
    transport = LocalTransport(2)
    _test_hooks.run = transport

    with pytest.raises(AppError) as excinfo:
        staging.stage_companions(
            "stage-test", stage_root=str(stage_root), companions=companions, workers=2
        )

    assert excinfo.value.code is (
        FleetErrorCode.DISPATCH_FAILED if missing else FleetErrorCode.STAGE_DIGEST_MISMATCH
    )
    assert transport.peak == 2
    assert transport.active == 0
    assert not (stage_root / "companion-0" / ".git").exists()
    assert _git("-C", str(stage_root / "companion-1"), "rev-parse", "HEAD") == companions[1]["sha"]


def test_no_companions_stage_nothing(tmp_path: pathlib.Path) -> None:
    assert (
        staging.stage_companions(
            "stage-test", stage_root=str(tmp_path / "absent"), companions=(), workers=1
        )
        == ()
    )
    assert not (tmp_path / "absent").exists()
