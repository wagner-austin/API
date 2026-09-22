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

from fleet.contracts.node import NodePlatform
from fleet.core import _test_hooks, dialect, names, remote

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

#: The archive's deadline, in seconds. The largest tree ever staged here
#: (tools/hpc3, 5.7 MB compressed) took under ten seconds; ten minutes
#: holds a tree a hundred times that and still ends a tar wedged on a
#: locked file before the tick's own bound does.
ARCHIVE_TIMEOUT_SECONDS = 600

#: WHY `runs` IS NOT ON THAT LIST, though it was for an hour.
#:
#: Five consecutive dispatches of tools/fleet grew 185 KB, 1.4 MB, 4.5 MB,
#: 13.5 MB, 20.7 MB, because each staged the archives the previous ones had
#: left in the tree. Excluding `runs` fixed the size and broke something
#: worse: `tools/hpc3` commits 294 run documents under `tools/hpc3/runs`,
#: force-added past the monorepo's `**/runs/` ignore, and its suite reads
#: them -- so the exclusion made four of its tests fail on lavender for a
#: reason that read as hpc3's fault.
#:
#: The archives were the error, not the directory. They are scratch and now
#: live outside the repository entirely
#: (:attr:`fleet.cli._config.LoadedWorkspace.archives`), so nothing about a
#: project's own tree has to be hidden to keep them out.


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
    # tar will not create the directory it is asked to write into, and a
    # workspace whose records directory does not exist yet is the ordinary
    # first-run case: the archive is built BEFORE any record is appended, so
    # nothing has made it. Without this the first dispatch into a fresh
    # workspace fails at tar with a message about a path.
    _test_hooks.make_directory(destination.parent)
    excludes: list[str] = []
    for name in EXCLUDED_DIRECTORIES:
        excludes.extend(("--exclude", name))
    result = _test_hooks.run(
        ["tar", "-czf", str(destination), "-C", str(project_root), *excludes, *members],
        timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
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
        node reads it in one go (``Get-Content -Raw`` on one platform,
        ``base64 -d`` on the other) and wrapping would make the decode depend
        on how the writer chose to fold it.
    """
    return base64.b64encode(payload).decode("ascii")


def send_verified(
    host: str,
    *,
    platform: NodePlatform,
    into: str,
    payload: bytes,
    described_as: str,
) -> None:
    """Send an archive to a directory on a node and verify it lands whole.

    The half of staging that is the same for a project's export and for a
    companion beside it: the encoded archive lands, the node reassembles and
    digests it WITHOUT extracting, and the sender compares. Nothing is
    unpacked here -- the caller decides where the verified bytes go, which
    is the one thing the two cases do differently.

    Args:
        host: SSH destination.
        platform: The node's declared platform.
        into: Absolute remote directory, which must already exist, holding
            the encoded archive and then the archive.
        payload: The archive bytes.
        described_as: What the archive is, for the mismatch's message.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, or ``STAGE_DIGEST_MISMATCH`` when the node's digest
            differs from the sender's. The mismatch is fatal rather than
            retried: a transfer that truncated once will do it again, and a
            retry loop turns a diagnosable fault into an intermittent one.
    """
    spoken = dialect.for_platform(platform)
    remote.send_script(host, f"{into}/{names.ENCODED_NAME}", encode(payload), platform=platform)
    received = remote.run_script(
        host,
        spoken.script_path(into, names.REASSEMBLE_STEM),
        spoken.reassemble_script(into),
        platform=platform,
    ).strip()
    expected = digest(payload)
    if received != expected:
        raise AppError(
            FleetErrorCode.STAGE_DIGEST_MISMATCH,
            f"{host} reassembled {described_as} digesting {received or '<nothing>'} where "
            f"{expected} was sent; nothing has been unpacked",
        )


def stage_companion(
    host: str,
    *,
    platform: NodePlatform,
    stage_root: str,
    directory: str,
    sha: str,
    payload: bytes,
) -> str:
    """Put one companion repository on a node, beside the exports.

    The directory is REPLACED rather than unpacked over, and the archive
    never enters it: :func:`fleet.core.names.companion_directory` and
    :data:`fleet.core.names.COMPANION_STAGE_SUFFIX` carry the reasons. What
    lands is the export of one commit and nothing else, committed on the
    node so the check that reads it reads a HEAD.

    Args:
        host: SSH destination.
        platform: The node's declared platform.
        stage_root: Absolute directory on the node holding staged trees.
        directory: The companion's declared directory name.
        sha: The commit the archive was written from.
        payload: The archive bytes.

    Returns:
        The absolute remote directory the companion was extracted into.

    Raises:
        AppError: As :func:`send_verified` describes, and from the transport
            for any of the scripts around it.
    """
    spoken = dialect.for_platform(platform)
    tree = names.companion_directory(stage_root, directory)
    staged = names.companion_stage_directory(stage_root, directory)
    staged_stem = names.make_directory_stem(names.companion_stage_name(directory))
    remote.run_script(
        host,
        spoken.script_path(stage_root, names.reset_directory_stem(directory)),
        spoken.reset_directory_script(tree),
        platform=platform,
    )
    remote.run_script(
        host,
        spoken.script_path(stage_root, staged_stem),
        spoken.make_directory_script(staged),
        platform=platform,
    )
    send_verified(
        host,
        platform=platform,
        into=staged,
        payload=payload,
        described_as=f"the {directory} companion at {sha}",
    )
    remote.run_script(
        host,
        spoken.script_path(staged, names.EXTRACT_STEM),
        dialect.extract_script(f"{staged}/{names.ARCHIVE_NAME}", tree),
        platform=platform,
    )
    remote.run_script(
        host,
        spoken.script_path(staged, names.COMPANION_REPOSITORY_STEM),
        dialect.companion_repository_script(tree, sha),
        platform=platform,
    )
    return tree


def stage(
    host: str,
    *,
    platform: NodePlatform,
    run_id: str,
    stage_root: str,
    payload: bytes,
) -> str:
    """Send a project's tree to a node and verify it before unpacking.

    The scripts are the node's dialect (:mod:`fleet.core.dialect`): the
    directory is made, the encoded archive lands, the node reassembles and
    digests it WITHOUT extracting, and only a digest that matches the
    sender's is followed by the extract and the ``git init`` that makes ruff
    honour ``.gitignore`` there.

    Args:
        host: SSH destination.
        platform: The node's declared platform.
        run_id: The dispatch, which names its own directory so two dispatches
            of one project cannot extract over each other.
        stage_root: Absolute directory on the node holding staged trees.
        payload: The archive bytes.

    Returns:
        The absolute remote directory the tree was extracted into.

    Raises:
        AppError: As :func:`send_verified` describes, and from the transport
            for any of the scripts around it.
    """
    spoken = dialect.for_platform(platform)
    target = f"{stage_root}/{run_id}"
    remote.run_script(
        host,
        spoken.script_path(stage_root, names.make_directory_stem(run_id)),
        spoken.make_directory_script(target),
        platform=platform,
    )
    send_verified(host, platform=platform, into=target, payload=payload, described_as="an archive")
    remote.run_script(
        host,
        spoken.script_path(target, names.EXTRACT_STEM),
        dialect.extract_script(f"{target}/{names.ARCHIVE_NAME}", target),
        platform=platform,
    )
    remote.run_script(
        host,
        spoken.script_path(target, names.INIT_REPOSITORY_STEM),
        dialect.init_repository_script(target),
        platform=platform,
    )
    return target


__all__ = [
    "EXCLUDED_DIRECTORIES",
    "archive",
    "digest",
    "encode",
    "send_verified",
    "stage",
    "stage_companion",
]
