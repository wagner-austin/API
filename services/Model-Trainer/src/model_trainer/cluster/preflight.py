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

import hashlib
import re
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.logging import get_logger

from model_trainer.core._hook_protocols import ArtifactStoreProto

_log = get_logger(__name__)

CERTIFICATION_SUFFIX = "-digests.txt"
"""Suffix of the record naming every corpus this directory is allowed to train.

Deliberately a suffix rather than a fixed filename, so one directory can hold
records from several certification runs and a new one does not overwrite the
last.
"""

_DIGEST_TOKEN = re.compile(r"\b[0-9a-f]{64}\b")

_DIGEST_CHUNK = 1 << 20

PROBE_NAME = ".preflight-probe"
"""Filename stem written and removed when testing a directory for writability.

Always suffixed with the run's own token. Two arms of one experiment share an
output root on a shared filesystem and start seconds apart, so a fixed name
means the first run's cleanup deletes the probe the second just wrote. That is
exactly how arm B of the Kazakh A/B died 19 seconds in -- killed by the check
that exists to stop runs dying.
"""

PROBE_ARTIFACT = "preflight-probe-artifact"
"""Artifact name stem used for the round-trip through the store.

Suffixed with the run's token for the same reason, and swept by that exact
name rather than by prefix: a glob over the bare stem would remove a
concurrent run's probe artifact out from under its round trip.
"""

_PROBE_BYTES = b"preflight\n"


def check_writable(roots: dict[str, Path], *, token: str) -> None:
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
        token: Unique to this run, appended to the probe filename. Required
            rather than defaulted: sibling arms share an output root, and a
            shared probe name makes one run's cleanup the other's failure.

    Raises:
        AppError: With ``ARTIFACT_UPLOAD_FAILED`` naming the first root that
            could not be created, written or read, and the underlying reason.
    """
    for name, root in roots.items():
        probe = root / f"{PROBE_NAME}-{token}"
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


def check_artifact_round_trip(
    store: ArtifactStoreProto, scratch: Path, written_to: Path, *, token: str
) -> None:
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
        token: Unique to this run, appended to the probe artifact's name. The
            sweep is then scoped to that exact name -- globbing the bare stem
            would delete a sibling arm's probe artifact mid-round-trip.

    Raises:
        AppError: With ``ARTIFACT_UPLOAD_FAILED`` if the round trip does not
            return the bytes that went in.
    """
    artifact_name = f"{PROBE_ARTIFACT}-{token}"
    source = scratch / artifact_name
    source.mkdir(parents=True, exist_ok=True)
    (source / "probe.txt").write_bytes(_PROBE_BYTES)

    stored = store.upload_artifact(source, artifact_name=artifact_name, request_id="preflight")
    restored = store.download_artifact(
        stored["file_id"],
        dest_dir=scratch / "preflight-restore",
        request_id="preflight",
        expected_root=artifact_name,
    )
    round_tripped = (restored / "probe.txt").read_bytes()
    _discard(scratch)
    for leftover in written_to.glob(f"{artifact_name}*"):
        leftover.unlink()

    if round_tripped != _PROBE_BYTES:
        raise AppError(
            ModelTrainerErrorCode.ARTIFACT_UPLOAD_FAILED,
            "The artifact store returned different bytes than it was given; "
            "a finished run would be saved wrong rather than not at all.",
        )
    _log.info("artifact store round-trips", extra={"file_id": stored["file_id"]})


def _digest_of(path: Path) -> str:
    """Hash a file's exact bytes.

    Args:
        path: File to read.

    Returns:
        Lowercase hex SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def check_corpus_certified(corpus_dir: Path, file_id: str) -> None:
    """Prove the corpus is the one it claims to be, and that it was certified.

    Two questions, and they fail differently.

    **Is this file what it says it is?** The corpus is addressed by digest --
    ``corpus_dir/<file_id>`` -- but nothing was checking that the bytes there
    actually hash to that name. A file named after a digest it does not have
    is indistinguishable from the real one until the results are wrong.

    **Was it certified?** A digest proves identity, never provenance. The
    corpus this exists for hashed correctly to its own name and was still the
    wrong thing: raw OSCAR English concatenated with a wiki export, never
    language-ID filtered, never transliterated, assembled by hand and copied
    up. It trained for hours, twice, and produced numbers that described
    cookie banners. So the run also requires the digest to appear in a
    ``*-digests.txt`` record placed beside the corpus by whatever certified
    it -- ``hpc3-stage`` writes one, and a file put there by hand has none.

    Neither check is expensive: a 15 MB corpus hashes in well under a second,
    against runs measured in hours.

    Args:
        corpus_dir: Directory holding staged corpora, keyed by digest.
        file_id: Digest of the corpus this run asked for.

    Raises:
        AppError: With ``CORPUS_EMPTY`` when the corpus is absent, its bytes
            do not hash to its name, or no certification record admits it.
    """
    corpus = corpus_dir / file_id
    if not corpus.is_file():
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"No corpus at {corpus}. The run asked for {file_id}; stage it first.",
        )

    actual = _digest_of(corpus)
    if actual != file_id:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"{corpus} is named {file_id} but its bytes hash to {actual}. "
            "The file is not what the run asked for.",
        )

    records = sorted(corpus_dir.glob(f"*{CERTIFICATION_SUFFIX}"))
    certified: set[str] = set()
    for record in records:
        # finditer, not findall: findall is typed list[Any], which strict mode
        # rejects and which would silently admit non-string groups anyway.
        for match in _DIGEST_TOKEN.finditer(record.read_text(encoding="utf-8")):
            certified.add(match.group(0))

    if file_id not in certified:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"{file_id} is not named by any *{CERTIFICATION_SUFFIX} record in "
            f"{corpus_dir} ({len(records)} record(s) found, {len(certified)} "
            "digest(s) between them). A corpus with no certification is one "
            "nobody checked -- the last such corpus was raw web scrape that "
            "trained for hours before anyone read it.",
        )

    _log.info(
        "corpus certified",
        extra={"file_id": file_id, "records": len(records), "certified": len(certified)},
    )


__all__ = [
    "CERTIFICATION_SUFFIX",
    "PROBE_ARTIFACT",
    "PROBE_NAME",
    "check_artifact_round_trip",
    "check_corpus_certified",
    "check_writable",
]
