"""Staging a companion out of the node's own mirror of its remote, sending only what it lacks.

WHY (MCPs board task 5280a25d). Every launch sent each companion it carries
as one ``git bundle`` of the companion ref's whole history, written on the
hub and copied to the node, whatever the node had been sent a minute before.
On lacrymal on 2026-11-05 the MCPs companion's bundle was 266 MB and took
22.8 s to write, API's 263 MB and 6.9 s and the personal wiki's 108 MB and
2.5 s, and packages/wiki-check carries 17 companions: its job 30708103 was
claimed at 18:35:00Z on lavender and reached its first phase at 18:36:14Z,
against the 10 s that task holds a gate job to.

SO A COMPANION TRAVELS AS A COMMIT DOES (:mod:`fleet.core.commit_stage`).
The node keeps a bare mirror of each companion remote beside the mirrors of
its projects' (:func:`fleet.core.names.node_mirror`, keyed by
:func:`fleet.core.export.companion_mirror_key`), the hub bundles the
companion's commit less what that mirror's :data:`fleet.core.commit_stage.NODE_TIP_REF`
reaches, a commit it already reaches is sent nothing at all, and the
companion is cloned out of the mirror by hard links
(:func:`fleet.core.stage_repository.companion_repository_commands`).

AND A COMPANION ALREADY CLEAN AT ITS COMMIT IS LEFT WHERE IT IS. The tree is
replaced before each run that carries it, so a file the workspace has since
deleted cannot survive into the tree a check reads as the workspace
(:func:`fleet.core.names.companion_directory`). A tree whose only refs are
``<branch>`` and ``origin/<branch>`` at the commit, with ``<branch>``
checked out and ``git status --ignored`` reporting nothing, is the tree a
fresh clone would make, so it is kept and nothing is sent or cloned.
"""

from __future__ import annotations

import shlex
from typing import Final

from typing_extensions import TypedDict

from fleet.core import _test_hooks, commit_stage, dialect, names, remote, stage_repository
from fleet.core.export import CompanionExport

#: What the companion's prepare script prints when its tree is kept.
REUSED_ANSWER: Final = "reused"

#: How many companions one launch stages at once, within its worker grant.
STAGE_WIDTH: Final = 4


class StagedCompanion(TypedDict):
    """Where a companion is on the node, and how the feed describes its stage.

    Attributes:
        tree: The absolute remote directory of the companion's repository.
        detail: The ``staged`` feed line's words: the bytes sent, the
            commit, and whether it was kept or what it was sent over.
    """

    tree: str
    detail: str


def prepare_script(*, tree: str, staged: str, mirror: str, sha: str, ref: str) -> str:
    """Keep a companion clean at its commit, or replace its directory and probe the mirror.

    Args:
        tree: The companion's directory.
        staged: Its staging directory.
        mirror: The node's mirror of the companion's remote.
        sha: The commit its ref resolved to.
        ref: The declared ref, whose branch is checked out.

    Returns:
        The script's text. It prints :data:`REUSED_ANSWER` and ends when
        the tree is kept; else it empties the tree, ``rm -rf`` needing no
        guard for an absent path and ``-f`` removing the read-only objects
        of the repository a previous run made there, makes both directories
        and prints what :func:`fleet.core.commit_stage.mirror_probe_lines`
        prints.
    """
    branch = ref.removeprefix("refs/heads/")
    quoted = shlex.quote(tree)
    refs = shlex.quote(f"refs/heads/{branch} {sha}\nrefs/remotes/origin/{branch} {sha}")
    listed = f"git -C {quoted} for-each-ref --format='%(refname) %(objectname)'"
    return (
        f"{dialect.PROLOGUE}"
        f'if [ "$({listed} 2>/dev/null || true)" = {refs} ]'
        f' && [ "$(git -C {quoted} symbolic-ref --quiet HEAD 2>/dev/null || true)"'
        f" = {shlex.quote(f'refs/heads/{branch}')} ]"
        f' && [ -z "$(git -C {quoted} status --porcelain --ignored --untracked-files=all 2>&1)" ];'
        f" then\n"
        f"  echo {REUSED_ANSWER}\n"
        "  exit 0\n"
        "fi\n"
        f"rm -rf {quoted}\n"
        f"mkdir -p {quoted} {shlex.quote(staged)}\n"
        f"{commit_stage.mirror_probe_lines(mirror=mirror, sha=sha, base_tip=None)}"
    )


def stage_companion(
    host: str, *, stage_root: str, run_id: str, companion: CompanionExport
) -> StagedCompanion:
    """Put one companion repository in a node's stage root.

    Args:
        host: SSH destination.
        stage_root: Absolute directory on the node holding staged trees.
        run_id: The dispatch, which with the companion's directory names its
            bundle's ref on the hub.
        companion: The companion, its ref resolved on the hub.

    Returns:
        Where the companion is and the feed's words for its stage.

    Raises:
        AppError: ``EXPORT_FAILED`` from the bundle, and as
            :func:`fleet.core.staging.send_verified` and the transport
            describe for the node's side.
    """
    directory = companion["directory"]
    sha = companion["sha"]
    tree = names.companion_directory(stage_root, directory)
    staged = names.staging_directory(tree)
    mirror = names.node_mirror(stage_root, companion["key"])
    said = remote.run_script(
        host,
        dialect.script_path(stage_root, names.reset_directory_stem(directory)),
        prepare_script(tree=tree, staged=staged, mirror=mirror, sha=sha, ref=companion["ref"]),
    )
    if REUSED_ANSWER in said.split():
        return StagedCompanion(
            tree=tree,
            detail=f"0 bytes, the {directory} companion at {sha} kept clean at {tree}",
        )
    node = commit_stage.read_node_mirror(said)
    bundle = None if node["held"] else f"{staged}/{names.ARCHIVE_NAME}"
    bundle_id = f"{run_id}-{directory}"
    sent = 0
    if bundle is not None:
        sent = commit_stage.send_bundle(
            host,
            staged=staged,
            run_id=bundle_id,
            commit=commit_stage.CommitExport(
                mirror=companion["mirror"],
                key=companion["key"],
                sha=sha,
                scope=(),
                bundle=companion["bundle"],
                base_branch=None,
            ),
            base=node["tip"],
            described_as=f"the {directory} companion at {sha}",
        )
    remote.run_script(
        host,
        dialect.script_path(staged, names.COMPANION_REPOSITORY_STEM),
        dialect.checked_script(
            (
                *commit_stage.fetch_commands(
                    mirror=mirror, bundle=bundle, run_id=bundle_id, base_branch=False
                ),
                *stage_repository.companion_repository_commands(
                    tree, mirror, sha, companion["ref"]
                ),
            )
        ),
        timeout_seconds=remote.send_deadline_seconds(sent),
    )
    over = (
        "already in the node's mirror"
        if bundle is None
        else f"over {node['tip'] or 'an empty mirror'}"
    )
    return StagedCompanion(
        tree=tree,
        detail=f"{sent} bytes, the {directory} companion at {sha} {over}, to {tree}",
    )


def stage_companions(
    host: str,
    *,
    stage_root: str,
    run_id: str,
    companions: tuple[CompanionExport, ...],
    workers: int,
) -> tuple[StagedCompanion, ...]:
    """Stage independent companions concurrently within the admitted allocation.

    The dispatch already holds every companion directory's node-local lease.
    Each stage uses its own staging directory and bundle, and each tree is
    kept clean or replaced before use. At most :data:`STAGE_WIDTH` run at
    once, never more than the job's worker allocation. All finish before a
    build can launch; a failed stage propagates after the pool has joined
    its in-flight work, so the caller cannot release leases over active
    writers.

    Args:
        host: SSH destination.
        stage_root: The node's declared stage directory.
        run_id: The dispatch.
        companions: Distinct companion directories, in declaration order.
        workers: Positive worker allocation granted to this dispatch.

    Returns:
        Each companion's stage, in declaration order.

    Raises:
        AppError: A companion's bundle, transfer, digest or checkout failure.
    """
    if not companions:
        return ()

    def stage_one(companion: CompanionExport) -> StagedCompanion:
        return stage_companion(host, stage_root=stage_root, run_id=run_id, companion=companion)

    with _test_hooks.executor(
        workers=min(STAGE_WIDTH, workers, len(companions)), name="fleet-companion-stage"
    ) as pool:
        return tuple(pool.map(stage_one, companions))


__all__ = [
    "REUSED_ANSWER",
    "STAGE_WIDTH",
    "StagedCompanion",
    "prepare_script",
    "stage_companion",
    "stage_companions",
]
