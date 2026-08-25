"""Proving a run can FINISH before letting it start.

This module exists because of a specific, expensive failure. A 20-epoch run
trained for 49 minutes, completed every epoch, wrote its final checkpoint --
and then died on the artifact-upload step, because a configuration guard
belonging to a different artifact-store implementation ran before the hook
that would have handled it. Forty-nine minutes of A100 time to learn that a
string was empty.

The epoch-boundary checkpoint had failed the same way one run earlier, on a
directory that was not writable. Both were checkable in under a second, and
both were checked only after the expensive part was already spent.

So the rule this module encodes:

    Everything a run needs at the END is exercised at the START, for real,
    against the same objects the run will use.

"For real" is the load-bearing part. A configuration check would not have
caught either failure -- the config was fine, the *path* was not. So this
writes an actual file to each output root and reads it back, and pushes an
actual directory through the artifact store and pulls it back out. If the
finish line is unreachable, the run refuses to start rather than discovering
it an hour later.

The cost is a few hundred milliseconds against runs measured in hours.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.logging import get_logger

from model_trainer.core._hook_protocols import ArtifactStoreProto

_log = get_logger(__name__)

PROBE_NAME = ".preflight-probe"
"""Filename written and removed when testing a directory for writability."""

PROBE_ARTIFACT = "preflight-probe-artifact"
"""Artifact name used for the round-trip through the store."""

_PROBE_BYTES = b"preflight\n"


def check_writable(roots: dict[str, Path]) -> None:
    """Prove every output directory can actually be written to.

    Creates each directory, writes a probe file, reads it back and removes
    it. Creation alone is not enough: a directory can exist and still refuse
    a write, which is exactly how the checkpoint save failed.

    Byte fidelity is deliberately NOT asserted here -- that is
    :func:`check_artifact_round_trip`'s job, where it is checked against the
    store a run actually saves through and can be tested. Asserting it here
    too would add a branch no test can reach on a real filesystem.

    Args:
        roots: Human-readable name to directory, so a failure names the
            setting the operator has to change rather than only a path.

    Raises:
        AppError: With ``ARTIFACT_UPLOAD_FAILED`` naming the first root that
            could not be created, written or read, and the underlying reason.
    """
    for name, root in roots.items():
        probe = root / PROBE_NAME
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(_PROBE_BYTES)
            probe.read_bytes()
            probe.unlink()
        except OSError as unwritable:
            raise AppError(
                ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
                f"{name} at {root} is not writable: {unwritable}. "
                "A run would train to completion and then fail saving.",
            ) from unwritable
    _log.info("output roots writable", extra={"roots": ",".join(sorted(roots))})


def _discard(tree: Path) -> None:
    """Remove a probe tree and everything under it.

    No existence guard: the only caller creates this tree immediately before,
    so a check for its absence would be a branch nothing can take. If that
    ever stops being true the ``FileNotFoundError`` is the correct outcome --
    it means the probe went somewhere other than where it was cleaned up.

    Args:
        tree: Directory to remove, deepest entries first.
    """
    for child in sorted(tree.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    tree.rmdir()


def check_artifact_round_trip(store: ArtifactStoreProto, scratch: Path, written_to: Path) -> None:
    """Prove the artifact store can store and retrieve, before it is needed.

    Pushes a real directory through ``upload_artifact`` and pulls it back
    with ``download_artifact``. A configuration check would not have caught
    the failure this exists for: the store was fine and the caller refused to
    reach it.

    The probe is removed afterwards, including whatever the store wrote for
    it. A check that leaves 300-byte tarballs beside a run's real output
    makes the output directory harder to read every time it passes, which is
    every time.

    Args:
        store: The artifact store the run will actually use.
        scratch: Directory to build the probe in and extract it back into.
            Removed before returning.
        written_to: Directory the store writes into, swept for the probe's
            own artifact once the round trip has proven it works.

    Raises:
        AppError: With ``ARTIFACT_UPLOAD_FAILED`` if the round trip does not
            return the bytes that went in.
    """
    source = scratch / PROBE_ARTIFACT
    source.mkdir(parents=True, exist_ok=True)
    (source / "probe.txt").write_bytes(_PROBE_BYTES)

    stored = store.upload_artifact(source, artifact_name=PROBE_ARTIFACT, request_id="preflight")
    restored = store.download_artifact(
        stored["file_id"],
        dest_dir=scratch / "preflight-restore",
        request_id="preflight",
        expected_root=PROBE_ARTIFACT,
    )
    round_tripped = (restored / "probe.txt").read_bytes()
    _discard(scratch)
    for leftover in written_to.glob(f"{PROBE_ARTIFACT}*"):
        leftover.unlink()

    if round_tripped != _PROBE_BYTES:
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
            "The artifact store returned different bytes than it was given; "
            "a finished run would be saved wrong rather than not at all.",
        )
    _log.info("artifact store round-trips", extra={"file_id": stored["file_id"]})


__all__ = ["PROBE_ARTIFACT", "PROBE_NAME", "check_artifact_round_trip", "check_writable"]
