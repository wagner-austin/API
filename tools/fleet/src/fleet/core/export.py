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
verified staging every dispatch uses (:mod:`fleet.core.staging`).

WHY A SHA THE REMOTE LACKS IS ITS OWN REFUSAL. The ordinary way this fails is
a session submitting HEAD before pushing it. That is not a fleet fault and not
a network fault; it is one ``git push`` away from working, and the code says
so rather than folding it into the transport's words.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.source import PATH_PATTERN, ProjectCompanion, ProjectSource
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

#: The deadline of an archive or a companion's bundle, in seconds; either of
#: the largest repository here (MCPs: a 38 MB archive, a 62 MB bundle) takes
#: well under a minute.
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


def archive_commit(
    mirror: pathlib.Path, sha: str, destination: pathlib.Path, paths: tuple[str, ...]
) -> bytes:
    """Write ``git archive --format=tar.gz <sha>`` and read the bytes back.

    Written to a file and read back rather than captured from git's
    standard output, because the command runner decodes streams as UTF-8
    and a gzip stream is not text (the same reason :mod:`fleet.core.staging`
    gives for tar).

    Args:
        mirror: The mirror's path, holding the commit.
        sha: The commit.
        destination: Where the archive is written; its directory is made.
        paths: The pathspec, from
            :func:`fleet.core.archive_scope.archive_pathspec`, scoping the
            archive to what the project's check reads. Empty for the whole
            commit, which is what a project whose repository declares no
            data directories gets: an empty tuple appends nothing, so the
            command run is the one this function has always run.

            REQUIRED AND NOT DEFAULTED, though every caller but one passes
            the same thing. The unscoped archive was the defect (board task
            140e7042, 216,826,747 bytes to check a 0.25 MB package), and a
            default would be the value a new call site reaches for without
            deciding -- which is exactly how the original had no pathspec.

    Returns:
        The archive's bytes, which are the payload
        :func:`fleet.core.staging.stage` sends and digests.

    Raises:
        AppError: ``EXPORT_FAILED`` when ``git archive`` exits non-zero.
    """
    _test_hooks.make_directory(destination.parent)
    scope = ("--", *paths) if paths else ()
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
            *scope,
        ),
        timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    )
    if archived["returncode"] != 0:
        raise _failure(
            FleetErrorCode.EXPORT_FAILED, f"git archive {sha} from {mirror}", archived["stderr"]
        )
    return _test_hooks.read_bytes(destination)


def bundle_ref(mirror: pathlib.Path, destination: pathlib.Path) -> bytes:
    """Write ``git bundle create`` of the companion ref and read the bytes back.

    The companion's payload since MCPs board task 2026dfbc: the ref with
    its whole history, which the node clones from
    (:func:`fleet.core.stage_repository.companion_repository_commands`), where an
    archive gave it one synthetic commit. Measured on the hub for MCPs main:
    61.8 MB in 2.2 s, against 38.1 MB and 3.0 s for the archive it replaces.

    Args:
        mirror: The companion's mirror, holding :data:`COMPANION_REF`.
        destination: Where the bundle is written; its directory is made.

    Returns:
        The bundle's bytes, which the node's digest is compared against.

    Raises:
        AppError: ``EXPORT_FAILED`` when ``git bundle create`` exits non-zero.
    """
    _test_hooks.make_directory(destination.parent)
    bundled = _test_hooks.run(
        ("git", "-C", str(mirror), "bundle", "create", "--quiet", str(destination), COMPANION_REF),
        timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    )
    if bundled["returncode"] != 0:
        raise _failure(
            FleetErrorCode.EXPORT_FAILED,
            f"git bundle create {COMPANION_REF} from {mirror}",
            bundled["stderr"],
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


def install_paths(step: tuple[str, ...]) -> tuple[str, ...]:
    """The tokens of one install step that name a path in the repository.

    A token names a path when it holds a slash, is not a flag, and is spelt
    in the registry's path alphabet (:data:`PATH_PATTERN`): ``scripts/x.sh``
    does, ``ci`` and ``--container`` do not, and neither does
    ``--workspace=packages/db``, whose path the tool resolves itself.

    Args:
        step: One declared install step, argv form.

    Returns:
        The path tokens, in the step's order.
    """
    return tuple(
        token
        for token in step
        if "/" in token and not token.startswith("-") and PATH_PATTERN.match(token) is not None
    )


def require_install_paths(
    mirror: pathlib.Path, sha: str, install: tuple[tuple[str, ...], ...]
) -> None:
    """Refuse a commit that lacks a path its declared install steps name.

    The steps come from today's registry and the tree from the commit, so a
    step declared after the commit was made names a file that commit does
    not carry. Run as it stood, the step exits 127 and the verdict reads as
    a broken commit (MCPs board task 8454b6a9: packages/db at a 2026-09-21
    commit, ``bash scripts/testdb-setup.sh``). Asked of the hub's mirror
    before any lease, like :func:`fetch_commit`'s refusal.

    Args:
        mirror: The project's mirror, holding the commit.
        sha: The commit.
        install: The project's declared install steps.

    Raises:
        AppError: ``INSTALL_PATH_NOT_IN_COMMIT`` naming the first step and
            path the commit lacks.
    """
    for step in install:
        for path in install_paths(step):
            probe = _test_hooks.run(
                ("git", "-C", str(mirror), "cat-file", "-e", f"{sha}:{path}"),
                timeout_seconds=INIT_TIMEOUT_SECONDS,
            )
            if probe["returncode"] != 0:
                raise AppError(
                    code=FleetErrorCode.INSTALL_PATH_NOT_IN_COMMIT,
                    message=(
                        f"the install step {' '.join(step)!r} names {path}, which commit {sha} "
                        "does not contain: the step was declared in fleet.json after this "
                        "commit, so today's registry cannot build it. Check a commit that "
                        f"carries {path}, or run this one from a working tree with fleet-run"
                    ),
                )


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
    """One companion's bundle, ready to be staged beside an export.

    Bytes and not a builder, unlike a project's payload: a companion's
    bundle is named by its own commit rather than by the run, and it is
    built where the ref is resolved -- before any lease is taken, so a ref
    the remote does not serve refuses with nothing held.

    Attributes:
        directory: The declared directory it lands in on the node.
        ref: The declared ref, whose branch the node checks out.
        sha: The commit its ref resolved to on this fetch.
        path: The local file the bundle was written to, which scp copies.
        data: The same bundle's bytes, which the node's digest is compared
            against.
    """

    directory: str
    ref: str
    sha: str
    path: pathlib.Path
    data: bytes


def export_companions(
    mirrors_root: pathlib.Path,
    archive_dir: pathlib.Path,
    companions: tuple[ProjectCompanion, ...],
) -> tuple[CompanionExport, ...]:
    """Fetch and bundle every companion a project's export carries.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        archive_dir: Local directory the bundles are written in, which must
            be run output rather than anywhere a build reads.
        companions: The project's declarations, in order.

    Returns:
        One export per companion, in the same order.

    Raises:
        AppError: As :func:`prepare_companion` and :func:`bundle_ref`
            describe.
    """
    exported: list[CompanionExport] = []
    for companion in companions:
        prepared = prepare_companion(mirrors_root, companion)
        key = companion_mirror_key(companion["remote"])
        path = archive_dir / f"{key}-{prepared['sha']}.bundle"
        exported.append(
            CompanionExport(
                directory=companion["directory"],
                ref=companion["ref"],
                sha=prepared["sha"],
                path=path,
                # The WHOLE repository with its history, unscoped by any
                # data path: the project's check reads the companion as a
                # workspace (slime lints its lifted code against the
                # committed MCPs workspace, corvis-stick reads MCPs at the
                # commit its hooks pin names), and nothing here knows which
                # parts or which past commits that check opens.
                data=bundle_ref(prepared["mirror"], path),
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
    "bundle_ref",
    "companion_mirror_key",
    "ensure_mirror",
    "export_companions",
    "fetch_commit",
    "fetch_ref",
    "has_commit",
    "install_paths",
    "mirror_path",
    "prepare_companion",
    "prepare_mirror",
    "require_install_paths",
    "require_source",
]
