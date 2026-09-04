"""Getting a project's working tree onto a node, verified before it is used.

WHY TAR AND NOT rsync, OR git. Measured 2026-09-04: none of the three
reachable nodes has ``rsync``, and all three have ``tar`` (Windows ships
bsdtar). So the transport is a tar archive, which removes a dependency rather
than adding one.

Git is the tempting alternative and is wrong here. Uncommitted work is the
normal state in this repo -- the standing rule is to stay on main with no
branches -- so "push and pull" would either refuse to dispatch the thing
somebody is actually working on, or demand a branch nobody wants. The dispatch
carries the working tree as it is.

WHY THE ARCHIVE TRAVELS AS BASE64. The transport is ssh into PowerShell, and
raw bytes do not survive it: the stream is decoded as text at more than one
layer, and a single mangled byte in a gzip member is a corrupt archive that
extracts partially. Base64 is text by construction, costs a third more bytes,
and removes the failure mode entirely rather than making it rarer.

THE DIGEST IS COMPARED BEFORE ANYTHING IS EXTRACTED. The node reassembles the
archive, digests it, and reports; only then is it told to unpack. Verifying
after extraction would mean an unverified tree had already landed where the
build will look for it, and a truncated tree builds and fails in a way that
reads as the code's fault.

WHAT IS NOT SENT. ``.venv`` leads the exclusion list, and not for tidiness: it
is the thing this whole package exists to stop two dispatches sharing, and one
machine's has absolute paths baked into it. The node builds its own from the
lockfile that IS sent.
"""

from __future__ import annotations

import base64
import hashlib
import pathlib

from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, remote

#: Directory names never carried to a node.
#:
#: ``.git`` is excluded because a dispatch runs a build, not a history, and the
#: repository is by far the largest thing in most projects. The caches are
#: reproducible by definition.
EXCLUDED_DIRECTORIES = (
    ".venv",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)

#: What the reassembled archive is called on the node.
ARCHIVE_NAME = "tree.tgz"

#: What the base64 text is called on the node before it is decoded.
ENCODED_NAME = "tree.b64"


def archive(
    project_root: pathlib.Path, members: tuple[str, ...], destination: pathlib.Path
) -> bytes:
    """Build a gzipped tar of everything a dispatch has to carry.

    Written to a file and read back rather than captured from tar's standard
    output, because the command runner decodes streams as UTF-8 with
    replacement -- which is right for a diagnostic and destroys an archive.

    Args:
        project_root: Absolute path to the monorepo root.
        members: Repo-relative directories to include, from
            :func:`~fleet.core.manifest.build_tree`. A SET rather than one
            project, because a project here is not self-contained: its
            lockfile resolves against sibling path dependencies and its
            Makefile calls a launcher at the repository root. Every member is
            named relative to the root and extracted relative to the stage
            directory, so ``../../libs/platform_core`` resolves on the node
            exactly as it does here.
        destination: Local file to write the archive to.

    Returns:
        The archive bytes.

    Raises:
        ValueError: If no members were given. An empty archive would stage
            successfully, extract to nothing, and fail at ``make`` with a
            missing-target error that says nothing about staging.
        AppError: With ``STAGE_ARCHIVE_UNREADABLE`` if tar exits non-zero,
            carrying its own stderr. The usual cause is a member that does
            not exist locally, and tar names it better than a pre-check
            would.
    """
    if not members:
        raise ValueError(
            "an archive needs at least one member; staging nothing extracts to nothing and "
            "fails later at make, with a message about a missing target rather than staging"
        )
    excludes: list[str] = []
    for name in EXCLUDED_DIRECTORIES:
        excludes.extend(("--exclude", name))
    result = _test_hooks.run(
        ["tar", "-czf", str(destination), "-C", str(project_root), *excludes, *members]
    )
    if result["returncode"] != 0:
        raise AppError(
            FleetErrorCode.STAGE_ARCHIVE_UNREADABLE,
            f"could not archive {', '.join(members)} under {project_root}: "
            f"{result['stderr'].strip() or '<no stderr>'}",
        )
    return _test_hooks.read_bytes(destination)


def digest(payload: bytes) -> str:
    """Digest an archive.

    Args:
        payload: The archive bytes.

    Returns:
        The lowercase hex SHA-256, full length. This is compared rather than
        displayed, and a truncated digest is a weaker comparison for no
        benefit to the only reader that matters.
    """
    return hashlib.sha256(payload).hexdigest()


def encode(payload: bytes) -> str:
    """Render an archive as the text that will cross the transport.

    Args:
        payload: The archive bytes.

    Returns:
        Standard base64, one line. One line rather than wrapped, because the
        node reads it with a single ``Get-Content -Raw`` and wrapping would
        make the decode depend on how the writer chose to fold it.
    """
    return base64.b64encode(payload).decode("ascii")


def make_directory_script(target: str) -> str:
    """Render the script that creates a dispatch's directory.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        The script's text.
    """
    return f"New-Item -ItemType Directory -Force -LiteralPath '{target}' | Out-Null\n"


def reassemble_script(target: str) -> str:
    """Render the script that rebuilds the archive and reports its digest.

    It deliberately does NOT extract. The digest it prints is what the sender
    compares against, and unpacking here would mean an unverified tree had
    already landed where the build will look for it.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        The script's text, whose only output is the digest.
    """
    return (
        f"$encoded = Get-Content -Raw -LiteralPath '{target}/{ENCODED_NAME}'\n"
        f"$bytes = [Convert]::FromBase64String($encoded.Trim())\n"
        f"[IO.File]::WriteAllBytes('{target}/{ARCHIVE_NAME}', $bytes)\n"
        f"(Get-FileHash -Algorithm SHA256 -LiteralPath "
        f"'{target}/{ARCHIVE_NAME}').Hash.ToLower()\n"
    )


def extract_script(target: str) -> str:
    """Render the script that unpacks a verified archive.

    ``-m`` makes extracted files take the node's clock rather than the
    sender's. Without it a tree staged from a machine whose clock is ahead
    produces make targets that look newer than their sources, and the build
    does nothing at all -- which reads as a suite that passed instantly.

    Args:
        target: Absolute remote directory for this dispatch.

    Returns:
        The script's text.
    """
    return f"tar -xzmf '{target}/{ARCHIVE_NAME}' -C '{target}'\n"


def stage(
    host: str,
    *,
    run_id: str,
    stage_root: str,
    payload: bytes,
) -> str:
    """Send a project's tree to a node and verify it before unpacking.

    Args:
        host: SSH destination.
        run_id: The dispatch, which names its own directory so two dispatches
            of one project cannot extract over each other.
        stage_root: Absolute directory on the node holding staged trees.
        payload: The archive bytes.

    Returns:
        The absolute remote directory the tree was extracted into.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, or ``STAGE_DIGEST_MISMATCH`` when the node's digest
            differs from the sender's. The mismatch is fatal rather than
            retried: a transfer that truncated once will do it again, and a
            retry loop turns a diagnosable fault into an intermittent one.
    """
    target = f"{stage_root}/{run_id}"
    remote.run_script(host, f"{stage_root}/mkdir-{run_id}.ps1", make_directory_script(target))
    remote.send_script(host, f"{target}/{ENCODED_NAME}", encode(payload))

    received = remote.run_script(
        host, f"{target}/reassemble.ps1", reassemble_script(target)
    ).strip()
    expected = digest(payload)
    if received != expected:
        raise AppError(
            FleetErrorCode.STAGE_DIGEST_MISMATCH,
            f"{host} reassembled an archive digesting {received or '<nothing>'} where "
            f"{expected} was sent; nothing has been unpacked",
        )

    remote.run_script(host, f"{target}/extract.ps1", extract_script(target))
    return target


__all__ = [
    "ARCHIVE_NAME",
    "ENCODED_NAME",
    "EXCLUDED_DIRECTORIES",
    "archive",
    "digest",
    "encode",
    "extract_script",
    "make_directory_script",
    "reassemble_script",
    "stage",
]
