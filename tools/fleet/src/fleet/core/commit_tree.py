"""Extract named trees of one commit into a scratch directory, or say why not.

Lifted from :mod:`fleet.core.published_tree` (board task 465689f5) when a
second caller came to need exactly what the session verbs had: code read
from a named commit and never from a working tree. The session verbs read
MCPs' ``origin/main`` (MCPs board task f4cd489f); the fleet's own agent
reads the commit ``make fleet-roll`` recorded after its execution suites
passed (:mod:`fleet.core.rolled`). Both resolve a ref, archive a few paths
of that commit, extract them into one directory per commit under this
machine's scratch root, and check the extraction holds what will be
imported. Each caller names its own refusal codes, so a refusal says which
extraction failed, and adds whatever else its extraction needs in between.

NO FALLBACK TO THE WORKING TREE, for either caller. Every step that fails
returns a ``CODE: message`` refusal and the caller runs nothing; code run
from a working tree on the one day the extraction fails is the defect both
callers exist to remove.

EACH CALLER NAMES ITS DESTINATION. The session verbs extract one directory
per commit, which a repeated commit rewrites with the same bytes, because
only one verb extracts at a time. The scheduled ticks start five at once,
so each extracts into a directory of its own (:mod:`fleet.core.rolled`).
Either way a new commit gets a fresh directory, so a module deleted
upstream can never be imported from an older extraction.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Sequence
from typing import Final, TypedDict

from fleet.core import _test_hooks
from fleet.core._test_hooks import CommandResult

#: A full commit id as ``git rev-parse`` prints it.
COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")

#: The deadline for each local git and tar step, in seconds. Each reads or
#: writes a few megabytes on this machine's own disk.
TREE_STEP_TIMEOUT_SECONDS: Final[int] = 120

#: The tarball's name inside an extraction directory.
TARBALL_NAME: Final = "published.tar"


class ResolvedCommit(TypedDict):
    """A ref that resolved.

    Attributes:
        commit: The full commit id it names.
    """

    commit: str


def step_refusal(code: str, step: str, result: CommandResult) -> str:
    """Compose the refusal for one failed step.

    Args:
        code: The detail prefix.
        step: What was run, for the reader.
        result: Its outcome.

    Returns:
        ``CODE: <step> exited <n>: <stderr>``, the stderr trimmed.
    """
    return f"{code}: {step} exited {result['returncode']}: {result['stderr'].strip()[:400]}"


def resolve_commit(repo_root: pathlib.Path, ref: str, code: str) -> ResolvedCommit | str:
    """Resolve a ref in a repository to the commit it names.

    Args:
        repo_root: The repository whose object store holds the ref.
        ref: The full ref name.
        code: The refusal's prefix when the ref names no commit.

    Returns:
        The commit, or a ``CODE: message`` refusal naming the exit status
        and both streams.
    """
    resolved = _test_hooks.run(
        ("git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    commit = resolved["stdout"].strip()
    if resolved["returncode"] != 0 or COMMIT_PATTERN.fullmatch(commit) is None:
        return (
            f"{code}: {ref} in {repo_root} did not resolve to a commit (exit "
            f"{resolved['returncode']}, stdout {commit[:80]!r}, stderr "
            f"{resolved['stderr'].strip()[:200]!r}); nothing ran"
        )
    return ResolvedCommit(commit=commit)


def extract_paths(
    repo_root: pathlib.Path,
    commit: str,
    paths: Sequence[str],
    *,
    destination: pathlib.Path,
    archive_code: str,
    extract_code: str,
) -> pathlib.Path | str:
    """Archive ``paths`` of ``commit`` and extract them into ``destination``.

    Args:
        repo_root: The repository holding the commit.
        commit: The full commit id.
        paths: The ``git archive`` pathspecs.
        destination: The directory to extract into, created with its
            parents. The caller decides whose it is: a caller that can run
            concurrently with itself must name one no other run writes,
            because two ``tar`` runs extracting over one tree collide on
            each other's files (:mod:`fleet.core.rolled` carries the
            incident).
        archive_code: The refusal's prefix when ``git archive`` fails.
        extract_code: The refusal's prefix when ``tar`` fails.

    Returns:
        ``destination``, now holding the paths, or a refusal.
    """
    _test_hooks.make_directory(destination)
    tarball = destination / TARBALL_NAME
    archived = _test_hooks.run(
        (
            "git",
            "-C",
            str(repo_root),
            "archive",
            "--format=tar",
            "-o",
            str(tarball),
            commit,
            "--",
            *paths,
        ),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    if archived["returncode"] != 0:
        return step_refusal(archive_code, f"git archive {commit}", archived)
    extracted = _test_hooks.run(
        ("tar", "-x", "-f", str(tarball), "-C", str(destination)),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    if extracted["returncode"] != 0:
        return step_refusal(extract_code, f"tar -x {tarball}", extracted)
    return destination


def incompleteness(
    destination: pathlib.Path, commit: str, required: Sequence[str], *, code: str
) -> str | None:
    """Say which required files an extraction lacks, if any.

    Args:
        destination: The extraction directory.
        commit: The commit it was taken from.
        required: Repository-relative files that must exist in it.
        code: The refusal's prefix.

    Returns:
        A refusal naming every missing file in order, or None when all are
        present.
    """
    missing = [
        path
        for path in required
        if not _test_hooks.file_exists(destination / pathlib.PurePosixPath(path))
    ]
    if not missing:
        return None
    return (
        f"{code}: the extraction of {commit} at {destination} lacks "
        f"{', '.join(missing)}; nothing ran"
    )


__all__ = [
    "COMMIT_PATTERN",
    "TARBALL_NAME",
    "TREE_STEP_TIMEOUT_SECONDS",
    "ResolvedCommit",
    "extract_paths",
    "incompleteness",
    "resolve_commit",
    "step_refusal",
]
