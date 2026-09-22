"""Exporting one commit of a project as the archive a node is staged with.

THE TREE ON THE NODE IS THE COMMIT, BY CONSTRUCTION. Until MCPs board task
fd5cabfa a dispatch carried the hub's working tree, tarred as it stood, so no
run of it could be cited as the check of a commit. Here the payload is the
bytes ``git archive --format=tar.gz <sha>`` writes for the commit the queue
row names, read out of a bare mirror on the hub that was fetched from the
project's declared remote. Nothing on the hub's disk but the mirror is
consulted, and the mirror holds only what the remote served, so an uncommitted
edit, a stale checkout or a different branch cannot reach the node.

WHY A MIRROR ON THE HUB AND NOT A FETCH ON THE NODE. The hub holds the git
credentials for every private remote and is the one machine the tailnet
policy lets reach the others; a node holds no credential and cannot open a
socket to the hub (docker-compose.diphtheria.yml's header measures the
packet filter). So the commit travels with the work, through the same
verified base64 staging every dispatch uses (:mod:`fleet.core.staging`).

WHY A SHA THE REMOTE LACKS IS ITS OWN REFUSAL. The ordinary way this fails is
a session submitting HEAD before pushing it. That is not a fleet fault and not
a network fault; it is one ``git push`` away from working, and the code says
so rather than folding it into the transport's words.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.source import ProjectCompanion, ProjectSource
from fleet.core import _test_hooks

#: The directory under the workspace's ``runs`` that holds one bare mirror
#: per project, named after the project key with its slashes folded.
MIRRORS_DIRECTORY = "mirrors"

#: The local ref a companion's declared ref is fetched into, inside that
#: companion's own mirror.
#:
#: Fetched into a ref rather than read off ``FETCH_HEAD``, so the commit is
#: REFERENCED between the fetch and the archive: an unreferenced tip is
#: exactly what ``git gc`` is entitled to remove, and the window would be
#: rare enough to be undebuggable.
COMPANION_REF = "refs/fleet/companion"

#: A fetch's deadline, in seconds. A first fetch of a large repository over
#: the operator's uplink is minutes; ten holds it and ends a hung remote
#: before the tick's own bound does.
FETCH_TIMEOUT_SECONDS = 600

#: The archive's deadline, in seconds; ``git archive`` of the largest
#: repository here (MCPs, 30 MB compressed) takes well under a minute.
ARCHIVE_TIMEOUT_SECONDS = 300

#: A ``git init --bare``'s deadline, in seconds.
INIT_TIMEOUT_SECONDS = 60

#: What git says when the remote will not serve the object asked for: the
#: three spellings across protocol versions and server settings. Any of
#: them means the sha is not fetchable from the remote, which for a project
#: on GitHub means it was never pushed.
NOT_ON_REMOTE_MARKERS = (
    "couldn't find remote ref",
    "unadvertised object",
    "not our ref",
)


def mirror_path(mirrors_root: pathlib.Path, project: str) -> pathlib.Path:
    """Where a project's bare mirror lives on the hub.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        project: The project key, as the registry spells it.

    Returns:
        ``<mirrors_root>/<key with / as ->.git``; the key's grammar admits
        nothing else a path could misread.
    """
    return mirrors_root / f"{project.replace('/', '-')}.git"


def companion_mirror_key(remote: str) -> str:
    """Name the bare mirror one companion remote is fetched into.

    Derived from the REMOTE and not from the directory a companion lands in,
    because the directory is a per-project choice: two projects could both
    stage a companion as ``MCPs`` from different remotes, and one mirror
    serving both would flip between two repositories under one ref. The
    remote's owner and repository name identify it the way a clone does.

    Args:
        remote: The companion's declared remote, in the source grammar.

    Returns:
        ``companion-<owner>-<repository>``; the grammar admits only
        characters :func:`mirror_path` can spell.
    """
    trimmed = remote.removesuffix(".git").replace(":", "/")
    owner, repository = trimmed.rsplit("/", 2)[-2:]
    return f"companion-{owner}-{repository}"


def _failure(code: FleetErrorCode, verb: str, stderr: str) -> AppError[FleetErrorCode]:
    """Compose the refusal for a git command that exited non-zero.

    Args:
        code: The refusal's code.
        verb: What was being done, for the message.
        stderr: git's own words, carried verbatim and bounded.

    Returns:
        The error to raise.
    """
    words = stderr.strip()[:600] if stderr.strip() else "(git wrote nothing to stderr)"
    return AppError(code=code, message=f"{verb}: {words}")


def ensure_mirror(mirror: pathlib.Path) -> None:
    """Create a project's bare mirror when it does not exist yet.

    Args:
        mirror: The mirror's path.

    Raises:
        AppError: ``EXPORT_FAILED`` when ``git init --bare`` exits non-zero.
    """
    if _test_hooks.directory_exists(mirror):
        return
    _test_hooks.make_directory(mirror.parent)
    made = _test_hooks.run(
        ("git", "init", "--bare", "--quiet", str(mirror)),
        timeout_seconds=INIT_TIMEOUT_SECONDS,
    )
    if made["returncode"] != 0:
        raise _failure(FleetErrorCode.EXPORT_FAILED, f"git init --bare {mirror}", made["stderr"])


def has_commit(mirror: pathlib.Path, sha: str) -> bool:
    """Whether the mirror already holds the commit.

    Args:
        mirror: The mirror's path.
        sha: The commit.

    Returns:
        True iff ``git cat-file -e <sha>^{{commit}}`` exits 0.
    """
    probe = _test_hooks.run(
        ("git", "-C", str(mirror), "cat-file", "-e", f"{sha}^{{commit}}"),
        timeout_seconds=INIT_TIMEOUT_SECONDS,
    )
    return probe["returncode"] == 0


def fetch_commit(mirror: pathlib.Path, remote: str, sha: str) -> None:
    """Fetch one commit from the remote into the mirror.

    Skipped when the mirror already holds it: a second job for the same sha
    (a re-run, or two packages of one commit) costs no round trip.

    Args:
        mirror: The mirror's path, already initialised.
        remote: The project's declared remote.
        sha: The commit, forty hex.

    Raises:
        AppError: ``SHA_NOT_ON_REMOTE`` when the remote will not serve the
            object (one of :data:`NOT_ON_REMOTE_MARKERS` in git's words),
            ``EXPORT_FAILED`` for any other non-zero fetch, git's words
            verbatim.
    """
    if has_commit(mirror, sha):
        return
    fetched = _test_hooks.run(
        ("git", "-C", str(mirror), "fetch", "--quiet", "--no-tags", remote, sha),
        timeout_seconds=FETCH_TIMEOUT_SECONDS,
    )
    if fetched["returncode"] == 0:
        return
    stderr = fetched["stderr"]
    if any(marker in stderr for marker in NOT_ON_REMOTE_MARKERS):
        raise AppError(
            code=FleetErrorCode.SHA_NOT_ON_REMOTE,
            message=(
                f"{remote} does not serve {sha}; a commit the remote has never seen is one "
                f"git push away from checkable, and this runner exports only what the remote "
                f"serves. git said: {stderr.strip()[:300]}"
            ),
        )
    raise _failure(FleetErrorCode.EXPORT_FAILED, f"git fetch {remote} {sha} into {mirror}", stderr)


def fetch_ref(mirror: pathlib.Path, remote: str, ref: str) -> str:
    """Fetch a ref's tip into the mirror and answer the commit it names.

    Never skipped the way :func:`fetch_commit` is: a sha the mirror holds is
    that sha forever, while a ref's tip is a moving answer and the question
    a companion asks is what it points at NOW.

    Args:
        mirror: The companion's mirror, already initialised.
        remote: The companion's declared remote.
        ref: The declared ref, ``main`` or ``refs/heads/main``.

    Returns:
        The forty-hex commit the ref names.

    Raises:
        AppError: ``COMPANION_REF_NOT_ON_REMOTE`` when the remote does not
            serve the ref (one of :data:`NOT_ON_REMOTE_MARKERS` in git's
            words), which is a fleet.json line to correct rather than a
            commit to push; ``EXPORT_FAILED`` for any other non-zero fetch
            or for a fetched ref that does not resolve to a commit, git's
            words verbatim.
    """
    fetched = _test_hooks.run(
        (
            "git",
            "-C",
            str(mirror),
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            remote,
            f"+{ref}:{COMPANION_REF}",
        ),
        timeout_seconds=FETCH_TIMEOUT_SECONDS,
    )
    if fetched["returncode"] != 0:
        stderr = fetched["stderr"]
        if any(marker in stderr for marker in NOT_ON_REMOTE_MARKERS):
            raise AppError(
                code=FleetErrorCode.COMPANION_REF_NOT_ON_REMOTE,
                message=(
                    f"{remote} does not serve {ref!r}, which fleet.json declares as a "
                    f"companion ref; every job of the project carrying it fails here until "
                    f"that line names a ref the remote has. git said: {stderr.strip()[:300]}"
                ),
            )
        raise _failure(
            FleetErrorCode.EXPORT_FAILED, f"git fetch {remote} {ref} into {mirror}", stderr
        )
    resolved = _test_hooks.run(
        ("git", "-C", str(mirror), "rev-parse", f"{COMPANION_REF}^{{commit}}"),
        timeout_seconds=INIT_TIMEOUT_SECONDS,
    )
    if resolved["returncode"] != 0:
        raise _failure(
            FleetErrorCode.EXPORT_FAILED,
            f"git rev-parse {COMPANION_REF} in {mirror}",
            resolved["stderr"],
        )
    return resolved["stdout"].strip()


def archive_commit(mirror: pathlib.Path, sha: str, destination: pathlib.Path) -> bytes:
    """Write ``git archive --format=tar.gz <sha>`` and read the bytes back.

    Written to a file and read back rather than captured from git's
    standard output, because the command runner decodes streams as UTF-8
    and a gzip stream is not text (the same reason :mod:`fleet.core.staging`
    gives for tar).

    Args:
        mirror: The mirror's path, holding the commit.
        sha: The commit.
        destination: Where the archive is written; its directory is made.

    Returns:
        The archive's bytes, which are the payload
        :func:`fleet.core.staging.stage` sends and digests.

    Raises:
        AppError: ``EXPORT_FAILED`` when ``git archive`` exits non-zero.
    """
    _test_hooks.make_directory(destination.parent)
    archived = _test_hooks.run(
        (
            "git",
            "-C",
            str(mirror),
            "archive",
            "--format=tar.gz",
            "-o",
            str(destination),
            sha,
        ),
        timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    )
    if archived["returncode"] != 0:
        raise _failure(
            FleetErrorCode.EXPORT_FAILED, f"git archive {sha} from {mirror}", archived["stderr"]
        )
    return _test_hooks.read_bytes(destination)


def require_source(project: str, source: ProjectSource | None) -> ProjectSource:
    """The project's source, or the refusal for a project declared without one.

    Args:
        project: The project key.
        source: The project's declared source, or None.

    Returns:
        The source.

    Raises:
        AppError: ``PROJECT_REMOTE_MISSING``, naming the registry line to
            write and the working-tree path that still works without it.
    """
    if source is None:
        raise AppError(
            code=FleetErrorCode.PROJECT_REMOTE_MISSING,
            message=(
                f"project {project!r} declares no source in fleet.json, so there is no remote "
                "to fetch a commit from; give it a source (remote, path, install) or dispatch "
                "it from a working tree with fleet-run"
            ),
        )
    return source


def prepare_mirror(
    mirrors_root: pathlib.Path, *, project: str, remote: str, sha: str
) -> pathlib.Path:
    """Make sure the project's mirror exists and holds the commit.

    The fetch happens HERE, before any lease is taken by the caller, so a sha
    the remote has never seen is refused with nothing held.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        project: The project key.
        remote: The project's declared remote.
        sha: The commit the queue row names.

    Returns:
        The mirror's path, holding the commit.

    Raises:
        AppError: As :func:`ensure_mirror` and :func:`fetch_commit` describe.
    """
    mirror = mirror_path(mirrors_root, project)
    ensure_mirror(mirror)
    fetch_commit(mirror, remote, sha)
    return mirror


class PreparedCompanion(TypedDict):
    """A companion's mirror on the hub and the commit its ref names now.

    Attributes:
        mirror: The bare mirror, holding that commit under
            :data:`COMPANION_REF`.
        sha: The commit the declared ref resolved to on this fetch, which
            the feed records so a verdict can be read against the workspace
            it was measured with.
    """

    mirror: pathlib.Path
    sha: str


class CompanionExport(TypedDict):
    """One companion's archive, ready to be staged beside an export.

    Bytes and not a builder, unlike a project's payload: a companion's
    archive is named by its own commit rather than by the run, and it is
    built where the ref is resolved -- before any lease is taken, so a ref
    the remote does not serve refuses with nothing held.

    Attributes:
        directory: The declared directory it lands in on the node.
        sha: The commit its ref resolved to on this fetch.
        data: The gzipped tar's bytes.
    """

    directory: str
    sha: str
    data: bytes


def export_companions(
    mirrors_root: pathlib.Path,
    archive_dir: pathlib.Path,
    companions: tuple[ProjectCompanion, ...],
) -> tuple[CompanionExport, ...]:
    """Fetch and archive every companion a project's export carries.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        archive_dir: Local directory the archives are written in, which must
            be run output rather than anywhere a build reads.
        companions: The project's declarations, in order.

    Returns:
        One export per companion, in the same order.

    Raises:
        AppError: As :func:`prepare_companion` and :func:`archive_commit`
            describe.
    """
    exported: list[CompanionExport] = []
    for companion in companions:
        prepared = prepare_companion(mirrors_root, companion)
        key = companion_mirror_key(companion["remote"])
        exported.append(
            CompanionExport(
                directory=companion["directory"],
                sha=prepared["sha"],
                data=archive_commit(
                    prepared["mirror"],
                    prepared["sha"],
                    archive_dir / f"{key}-{prepared['sha']}.tgz",
                ),
            )
        )
    return tuple(exported)


def prepare_companion(mirrors_root: pathlib.Path, companion: ProjectCompanion) -> PreparedCompanion:
    """Make sure a companion's mirror exists and holds its ref's tip.

    The twin of :func:`prepare_mirror`, and called in the same place for the
    same reason: before any lease is taken, so a ref the remote does not
    serve is refused with nothing held.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        companion: The declaration, from the project's source.

    Returns:
        The mirror and the commit its ref names.

    Raises:
        AppError: As :func:`ensure_mirror` and :func:`fetch_ref` describe.
    """
    mirror = mirror_path(mirrors_root, companion_mirror_key(companion["remote"]))
    ensure_mirror(mirror)
    sha = fetch_ref(mirror, companion["remote"], companion["ref"])
    return PreparedCompanion(mirror=mirror, sha=sha)


__all__ = [
    "ARCHIVE_TIMEOUT_SECONDS",
    "COMPANION_REF",
    "FETCH_TIMEOUT_SECONDS",
    "INIT_TIMEOUT_SECONDS",
    "MIRRORS_DIRECTORY",
    "NOT_ON_REMOTE_MARKERS",
    "CompanionExport",
    "PreparedCompanion",
    "archive_commit",
    "companion_mirror_key",
    "ensure_mirror",
    "export_companions",
    "fetch_commit",
    "fetch_ref",
    "has_commit",
    "mirror_path",
    "prepare_companion",
    "prepare_mirror",
    "require_source",
]
