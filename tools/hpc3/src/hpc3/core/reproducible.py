"""Refuse to stage bytes the repository cannot produce.

WHY THIS EXISTS, AND WHY --expect-from DOES NOT COVER IT.
:func:`~hpc3.core.stage.stage_manifest` records each file's sha256 and writes
it beside the data, so a later reader can prove which bytes a run used. That
record is computed from THE BYTES ON DISK. When the file is tracked and the
working tree disagrees with the repository -- most often a text file checked
out with CRLF against an LF blob -- the record is perfectly accurate about
what was staged and REPRODUCIBLE BY NOBODY. Another checkout produces
different bytes, hashes them differently, and the manifest refuses its own
file.

:mod:`hpc3.core.expected` already demands a second document that is not
derived from the files being staged. It cannot catch this: the manifest and
the expected-digest record are both written on one machine from one checkout,
so they agree with each other by construction. Two artifacts agreeing because
they were made together is the defect this whole family of checks exists to
separate from two artifacts agreeing because the bytes are right.

MEASURED BEFORE IT WAS WRITTEN. Across every ``*stage.json`` in
``tools/hpc3/runs`` on 2026-09-09 -- 363 files -- three violate this, all in
one project, all with Windows-checkout digests against LF blobs. They are the
v2 training payload and both generation specs of ``code-style``. This check
therefore lands REFUSING three files that are already on the cluster, which
is the correct state: the digests are honest about a staging that happened,
and the bytes are still not reproducible from the repository.

WHAT IT DELIBERATELY DOES NOT DO. An UNTRACKED file is not this check's
business. There is no repository copy for the bytes to disagree with, so
there is nothing to compare and nothing to refuse -- corpora and images are
staged from outside the tree and always will be. The predicate fires only on
tracked-AND-differing, which is one-directional and needs no exemption list.

BLOB HASHES, NOT CONTENTS. Both sides are compared as git blob hashes rather
than by reading bytes back, because equal blob hashes means equal bytes and
the hashes are hex text. :class:`~hpc3.core._test_hooks.CommandResult`
decodes stdout with ``errors="replace"``; a comparison done on contents would
silently mangle any file that is not valid UTF-8 and report two different
binaries as identical.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.stage import StageManifest
from hpc3.core import _test_hooks


def committed_blob(source_dir: pathlib.Path, name: str) -> str:
    """Ask the repository for its committed blob hash of one file.

    Args:
        source_dir: Directory holding the file, used as git's working
            directory so the answer comes from whichever repository actually
            contains it.
        name: File name within that directory.

    Returns:
        The 40-character blob hash HEAD records for the path, or the empty
        string when the path is not in HEAD. Empty is a real answer here and
        not a failure: an untracked file has no repository copy, which is the
        case this check passes rather than the case it refuses.
    """
    result = _test_hooks.run(
        ["git", "-C", str(source_dir), "rev-parse", f"HEAD:./{name}"],
    )
    if result["returncode"] != 0:
        return ""
    return result["stdout"].strip()


def working_blob(source_dir: pathlib.Path, name: str) -> str:
    """Compute the blob hash of the file as it sits on disk.

    ``--no-filters`` IS LOAD-BEARING, AND ITS ABSENCE IS UNDETECTABLE BY ANY
    TEST THAT SCRIPTS GIT'S OUTPUT. Without it, ``hash-object`` applies the
    same clean filter as ``git add``: a CRLF working file is normalised to LF
    before hashing, so it returns exactly HEAD's blob hash and this check
    passes on the one case it was written for. Measured on
    ``code-style-gen-v2-base.json``, whose bytes ARE the motivating defect --
    filtered 28ba697b (equal to HEAD, hides it), unfiltered 1007163890
    (differs, catches it). The first version of this module shipped without
    the flag, passed its entire unit suite, and was caught only by running it
    against the real repository. ``test_reproducible_against_git.py`` drives
    real git for that reason.

    Args:
        source_dir: Directory holding the file.
        name: File name within that directory.

    Returns:
        The 40-character blob hash of the bytes AS THEY SIT ON DISK.

    Raises:
        AppError: With ``STAGE_SOURCE_NOT_REPRODUCIBLE`` when git cannot hash
            the file at all. That means the path is unreadable or git is
            absent, and staging on a guess about which of those it was would
            be worse than stopping.
    """
    result = _test_hooks.run(
        ["git", "-C", str(source_dir), "hash-object", "--no-filters", name],
    )
    if result["returncode"] != 0:
        raise AppError(
            Hpc3ErrorCode.STAGE_SOURCE_NOT_REPRODUCIBLE,
            f"could not hash {name} in {source_dir}: {result['stderr'].strip()}. "
            "This check compares the file against the repository's copy and "
            "cannot do so without git.",
        )
    return result["stdout"].strip()


def unreproducible(source_dir: pathlib.Path, manifest: StageManifest) -> tuple[str, ...]:
    """Name the files whose staged bytes the repository does not hold.

    Args:
        source_dir: Directory holding the files.
        manifest: What is about to be staged.

    Returns:
        The names, in manifest order, of files that ARE tracked and whose
        working-tree bytes differ from the committed ones. A file absent from
        HEAD is not reported: see the module docstring.

    Raises:
        AppError: Propagated from :func:`working_blob`.
    """
    return tuple(
        staged["name"]
        for staged in manifest["files"]
        if (committed := committed_blob(source_dir, staged["name"]))
        and committed != working_blob(source_dir, staged["name"])
    )


def require_sources_reproducible(source_dir: pathlib.Path, manifest: StageManifest) -> None:
    """Refuse a staging whose digests only this machine could produce.

    Args:
        source_dir: Directory holding the files.
        manifest: What is about to be staged.

    Raises:
        AppError: With ``STAGE_SOURCE_NOT_REPRODUCIBLE``, naming every file
            whose bytes differ from the repository's. Raised BEFORE any
            transfer, so a refused staging leaves the cluster untouched and
            no certification record is written for bytes nobody can rebuild.
    """
    drifted = unreproducible(source_dir, manifest)
    if not drifted:
        return
    listed = "\n  ".join(drifted)
    raise AppError(
        Hpc3ErrorCode.STAGE_SOURCE_NOT_REPRODUCIBLE,
        f"{len(drifted)} file(s) differ from the repository's copy, so the digests "
        f"this staging would record can only be reproduced on this machine:\n  {listed}\n"
        "Commit the files, or check out a tree that matches what you intend to stage. "
        "A text file is most often here because it was checked out with CRLF against an "
        "LF blob -- `git check-attr text -- <file>` says whether a rule covers it, and a "
        "`-text` entry in .gitattributes makes the bytes identical on every checkout. "
        "An UNTRACKED file is never reported: it has no repository copy to disagree with.",
    )


__all__ = [
    "committed_blob",
    "require_sources_reproducible",
    "unreproducible",
    "working_blob",
]
