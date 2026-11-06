"""Exporting one commit of a project as the archive a node is staged with.

THE TREE ON THE NODE IS THE COMMIT, BY CONSTRUCTION. Until MCPs board task
fd5cabfa a dispatch carried the hub's working tree, tarred as it stood, so no
run of it could be cited as the check of a commit. Here the commit the queue
row names is read out of a bare mirror on the hub that was fetched from the
project's declared remote, and since MCPs board task 1e0ee821 it travels as
the ``git bundle`` of what the node's own mirror lacks, from which the node
writes the commit's ``git archive`` (:mod:`fleet.core.commit_stage`).
Nothing on the hub's disk but the mirror is consulted, and the mirror holds
only what the remote served, so an uncommitted edit, a stale checkout or a
different branch cannot reach the node.

WHY A MIRROR ON THE HUB AND NOT A FETCH ON THE NODE. The hub holds the git
credentials for every private remote and is the one machine the tailnet
policy lets reach the others; a node holds no credential and cannot open a
socket to the hub (docker-compose.diphtheria.yml's header measures the
packet filter). So the commit travels with the work, through the same
verified staging every dispatch uses (:mod:`fleet.core.staging`).

WHY A SHA THE REMOTE LACKS IS ITS OWN REFUSAL. The ordinary way this fails is
a session submitting HEAD before pushing it. That is not a fleet fault and not
a network fault; it is one ``git push`` away from working, and the code says
so rather than folding it into the transport's words. The fetches, and the
attempts a lost one is given before the job is refused, are
:mod:`fleet.core.mirror_fetch`'s.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.source import PATH_PATTERN, InstallStep, ProjectCompanion, ProjectSource
from fleet.core import _test_hooks, mirror_fetch

#: The deadline of a commit's or a companion's bundle, in seconds; a bundle
#: of the largest repository's whole history, a node's first, is MCPs' 318
#: MB pack (MCPs board task 1e0ee821), which git writes from its packs well
#: inside it.
ARCHIVE_TIMEOUT_SECONDS = 300


def mirror_path(mirrors_root: pathlib.Path, key: str) -> pathlib.Path:
    """Where one bare mirror lives on the hub.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        key: :func:`remote_mirror_key` or :func:`companion_mirror_key`.

    Returns:
        ``<mirrors_root>/<key>.git``.
    """
    return mirrors_root / f"{key}.git"


def remote_mirror_key(remote: str) -> str:
    """Name the bare mirror a project's commits are fetched into.

    Keyed by REMOTE, not project key, so all packages of one repository share
    one history; per project, 107 hub mirrors held 122.8 GB on 2026-10-08.

    Args:
        remote: The project's declared remote, in the source grammar.

    Returns:
        ``<owner>-<repository>``, which :func:`mirror_path` can spell.
    """
    trimmed = remote.removesuffix(".git").replace(":", "/")
    owner, repository = trimmed.rsplit("/", 2)[-2:]
    return f"{owner}-{repository}"


def companion_mirror_key(remote: str) -> str:
    """Name the bare mirror one companion remote is fetched into.

    Derived from the REMOTE and not from the directory a companion lands in,
    because the directory is a per-project choice: two projects could both
    stage a companion as ``MCPs`` from different remotes, and one mirror
    serving both would flip between two repositories under one ref.

    Args:
        remote: The companion's declared remote, in the source grammar.

    Returns:
        ``companion-<owner>-<repository>``.
    """
    return f"companion-{remote_mirror_key(remote)}"


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
        timeout_seconds=mirror_fetch.INIT_TIMEOUT_SECONDS,
    )
    if made["returncode"] != 0:
        raise mirror_fetch.git_failure(
            FleetErrorCode.EXPORT_FAILED, f"git init --bare {mirror}", made["stderr"]
        )


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
                f"project {project!r} declares no source in the registry, so there is no remote "
                "to fetch a commit from; give it a source (remote, path, install) or dispatch "
                "it from a working tree with fleet-run"
            ),
        )
    return source


def prepare_mirror(mirrors_root: pathlib.Path, *, remote: str, sha: str) -> pathlib.Path:
    """Make sure the remote's mirror exists and holds the commit.

    The fetch happens HERE, before any lease is taken by the caller, so a sha
    the remote has never seen is refused with nothing held.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        remote: The project's declared remote.
        sha: The commit the queue row names.

    Returns:
        The mirror's path, holding the commit.

    Raises:
        AppError: As :func:`ensure_mirror` and
            :func:`fleet.core.mirror_fetch.fetch_commit` describe.
    """
    mirror = mirror_path(mirrors_root, remote_mirror_key(remote))
    ensure_mirror(mirror)
    mirror_fetch.fetch_commit(mirror, remote, sha)
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


def require_install_paths(mirror: pathlib.Path, sha: str, install: tuple[InstallStep, ...]) -> None:
    """Refuse a commit that lacks a path its declared install steps name.

    The steps come from today's registry and the tree from the commit, so a
    step declared after the commit was made names a file that commit does
    not carry. Run as it stood, the step exits 127 and the verdict reads as
    a broken commit (MCPs board task 8454b6a9: packages/db at a 2026-09-21
    commit, ``bash scripts/testdb-setup.sh``). Asked of the hub's mirror
    before any lease, like :func:`fleet.core.mirror_fetch.fetch_commit`'s refusal.

    Args:
        mirror: The project's mirror, holding the commit.
        sha: The commit.
        install: The project's declared install steps.

    Raises:
        AppError: ``INSTALL_PATH_NOT_IN_COMMIT`` naming the first step and
            path the commit lacks.
    """
    for step in install:
        for path in install_paths(step["argv"]):
            probe = _test_hooks.run(
                ("git", "-C", str(mirror), "cat-file", "-e", f"{sha}:{path}"),
                timeout_seconds=mirror_fetch.INIT_TIMEOUT_SECONDS,
            )
            if probe["returncode"] != 0:
                command = " ".join(step["argv"])
                raise AppError(
                    code=FleetErrorCode.INSTALL_PATH_NOT_IN_COMMIT,
                    message=(
                        f"the install step {command!r} names {path}, which commit {sha} "
                        "does not contain: the step was declared in the registry after this "
                        "commit, so today's registry cannot build it. Check a commit that "
                        f"carries {path}, or run this one from a working tree with fleet-run"
                    ),
                )


class PreparedCompanion(TypedDict):
    """A companion's mirror on the hub and the commit its ref names now.

    Attributes:
        mirror: The bare mirror, holding that commit under
            :data:`fleet.core.mirror_fetch.COMPANION_REF`.
        sha: The commit the declared ref resolved to on this fetch, which
            the feed records so a verdict can be read against the workspace
            it was measured with.
    """

    mirror: pathlib.Path
    sha: str


class CompanionExport(TypedDict):
    """One companion, its ref resolved, ready to be staged beside an export.

    Resolved where the job is prepared, before any lease is taken, so a ref
    the remote does not serve refuses with nothing held. Its bundle is
    written only once the node has said what its mirror of the remote
    already holds (:mod:`fleet.core.companion_stage`, MCPs board task
    5280a25d), into the directory its caller gives the claimed job alone
    (MCPs board task f9ee56aa).

    Attributes:
        directory: The declared directory it lands in on the node.
        ref: The declared ref, whose branch the node checks out.
        sha: The commit its ref resolved to on this fetch.
        mirror: The hub's mirror of the companion's remote, holding ``sha``.
        key: That mirror's key (:func:`companion_mirror_key`), which names
            the node's mirror of the same remote.
        bundle: The local file its bundle is written to, if one is sent.
    """

    directory: str
    ref: str
    sha: str
    mirror: pathlib.Path
    key: str
    bundle: pathlib.Path


#: How many companions' refs are fetched at once: packages/wiki-check
#: declares 17, each a round trip to GitHub inside its own mirror's turn.
COMPANION_FETCHES: Final = 8


def export_companions(
    mirrors_root: pathlib.Path,
    archive_dir: pathlib.Path,
    companions: tuple[ProjectCompanion, ...],
) -> tuple[CompanionExport, ...]:
    """Fetch every companion a project's export carries, :data:`COMPANION_FETCHES` at once.

    AT ONCE, NOT ONE AFTER ANOTHER (MCPs board task 5280a25d). Each
    companion is its own remote in its own mirror under its own turn, and
    one after another the 17 of packages/wiki-check, each fetched and then
    bundled whole, kept lavender idle from 18:35:02 to 18:35:53Z on
    2026-11-05 between job 30708103's claim and its first phase.

    Args:
        mirrors_root: The mirrors directory under the workspace's records.
        archive_dir: Local directory the bundles are written in, which must
            be run output rather than anywhere a build reads.
        companions: The project's declarations, in order.

    Returns:
        One export per companion, in the same order.

    Raises:
        AppError: As :func:`prepare_companion` describes, the first
            companion's in declared order of those that failed, once every
            fetch under way has ended.
    """
    if not companions:
        return ()

    def export_one(companion: ProjectCompanion) -> CompanionExport:
        prepared = prepare_companion(mirrors_root, companion)
        key = companion_mirror_key(companion["remote"])
        return CompanionExport(
            directory=companion["directory"],
            ref=companion["ref"],
            sha=prepared["sha"],
            mirror=prepared["mirror"],
            key=key,
            bundle=archive_dir / f"{key}-{prepared['sha']}.bundle",
        )

    with _test_hooks.executor(
        workers=min(COMPANION_FETCHES, len(companions)), name="fleet-companion-fetch"
    ) as pool:
        return tuple(pool.map(export_one, companions))


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
        AppError: As :func:`ensure_mirror` and
            :func:`fleet.core.mirror_fetch.fetch_ref` describe.
    """
    mirror = mirror_path(mirrors_root, companion_mirror_key(companion["remote"]))
    ensure_mirror(mirror)
    sha = mirror_fetch.fetch_ref(mirror, companion["remote"], companion["ref"])
    return PreparedCompanion(mirror=mirror, sha=sha)


__all__ = [
    "ARCHIVE_TIMEOUT_SECONDS",
    "COMPANION_FETCHES",
    "CompanionExport",
    "PreparedCompanion",
    "companion_mirror_key",
    "ensure_mirror",
    "export_companions",
    "install_paths",
    "mirror_path",
    "prepare_companion",
    "prepare_mirror",
    "remote_mirror_key",
    "require_install_paths",
    "require_source",
]
