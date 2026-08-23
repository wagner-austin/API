"""Holding a manifest against a record written before it.

A manifest is self-consistent by construction: whoever emitted the files
computed the digests from those same files, so they always agree. That
agreement proves the emitter was deterministic and nothing else. The question
staging cannot otherwise answer is whether those are the digests the published
work actually used.

So staging requires a second document -- one that was written by a different
act, at a different time, and is not derived from the files being staged. Every
digest in the manifest must appear in it. That is a real check because the two
cannot be made to agree by repeating the mistake: re-emitting a corpus from the
wrong source state produces new digests, and new digests are not in the record.

The record's format is deliberately not specified. Any text is read and every
64-character lowercase-hex token in it is taken as a published digest, so a
``sha256sum`` listing, a JSON manifest, a run log or a hand-kept file all work
without conversion. Reading loosely is safe here because the direction of the
check is one-way: extra digests in the record admit nothing, while a digest
missing from it refuses.
"""

from __future__ import annotations

import pathlib
import re

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import SHA256_HEX_LENGTH, StageManifest
from hpc3.core import _test_hooks

_DIGEST_TOKEN = re.compile(rf"\b[0-9a-f]{{{SHA256_HEX_LENGTH}}}\b")


def read_expected_digests(path: pathlib.Path) -> set[str]:
    """Collect every published digest a record file names.

    Args:
        path: The record to read.

    Returns:
        Every lowercase-hex sha256 token found anywhere in the text.

    Raises:
        AppError: With ``STAGED_DIGEST_UNEXPECTED`` if the file holds no
            digest at all. An empty record would refuse every file with a
            message about the manifest, when the fault is that the record
            named nothing -- most often a path that does not point where the
            caller thought.
    """
    text = _test_hooks.read_bytes(path).decode("utf-8")
    found: set[str] = {match.group(0) for match in _DIGEST_TOKEN.finditer(text)}
    if found == set():
        raise AppError(
            Hpc3ErrorCode.STAGED_DIGEST_UNEXPECTED,
            f"{path} names no sha256 digest, so it cannot vouch for anything. "
            "Point --expect-from at the record of digests this work published.",
        )
    return found


def check_expected(manifest: StageManifest, expected: set[str], *, source: pathlib.Path) -> None:
    """Refuse a manifest naming bytes the published record does not.

    Args:
        manifest: What is about to be staged.
        expected: Digests the record vouches for.
        source: The record's path, named in the message so the operator knows
            which document disagreed.

    Raises:
        AppError: With ``STAGED_DIGEST_UNEXPECTED`` on the first file whose
            digest is absent from the record. This is the check that catches
            a corpus re-emitted from the wrong source state -- the one case
            where every digest matches its own file and the experiment is
            still void.
    """
    for staged in manifest["files"]:
        if staged["sha256"] not in expected:
            raise AppError(
                Hpc3ErrorCode.STAGED_DIGEST_UNEXPECTED,
                f"{staged['name']} has digest {staged['sha256']}, which {source} does "
                "not name. These are not the bytes the published work used: either "
                "they were regenerated from a different source state, or the record "
                "is the wrong one. Nothing was staged.",
            )


__all__ = ["check_expected", "read_expected_digests"]
