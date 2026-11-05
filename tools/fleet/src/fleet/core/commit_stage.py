"""Staging a commit out of the node's own mirror of its remote, sending only what it lacks.

WHY (MCPs board task 1e0ee821). Every queued check carried its commit as one
``git archive``: 169,980,029 bytes for MCPs at 92f35c248, whatever the node
had been sent a minute before. The hub copies that at the speed of the
node's link, measured from lacrymal on 2026-11-04 at 94 MB/s to colossus,
11 MB/s to pendragon and 7.6 MB/s to sedona, so a gate job's stage took 24
to 29 s on the slower two (fleet jobs 5a398139, 7dd7579e, 07e373b5) against
the 150 s the push gate holds MCPs/execution-linux and -deploy to from claim
to verdict. MCPs' whole history packs to 318 MB and API's to 264 MB, and
the commit a node is asked for is nearly always a few commits past the last
one it was sent.

SO THE NODE KEEPS A BARE MIRROR OF EACH REMOTE (:func:`fleet.core.names.node_mirror`),
and the hub sends a ``git bundle`` of the commit less everything the node's
mirror already holds: the first stage of a remote on a node carries its
history once, and every later one its newest commits. The node fetches the
bundle into its mirror, which git checks for completeness against what it
holds, and writes the commit's ``git archive`` from there with the pathspec
the hub wrote its own archive with until then
(:func:`fleet.core.archive_scope.archive_pathspec`), so the tree that lands
is the commit's by construction, as before. A commit
the node's mirror already holds, a second project of one commit say, is
sent nothing at all.

THE TRANSPORT IS UNCHANGED. The bundle crosses over scp and its digest is
compared before the node reads it (:func:`fleet.core.staging.send_verified`),
as a companion's bundle is; the tree is unpacked beside its staging
directory and committed as one commit
(:func:`fleet.core.stage_repository.init_repository_commands`).

EVERY REF WRITE WAITS ON GIT'S OWN LOCKS (MCPs board task 695b0b02). Each
export writes its run's bundle ref into the hub's one mirror and deletes it
once the bundle is written, and every runner on the hub exports at once: a
push gate queued 14 jobs of MCPs 340c7e9af within 0.13 s on 2026-11-05. A
ref deletion takes the mirror's ``packed-refs.lock``, which git waits for
only 1 s by default (``core.packedRefsTimeout``), so on lacrymal at load,
where another export or git's own maintenance held it longer, fleet job
2243695b was refused ``EXPORT_FAILED`` with git's ``Unable to create
'.../packed-refs.lock': File exists`` and its bundle ref was left in the
mirror. A turn of the fleet's own around the write would serialize the
exports and still not the maintenance git starts after a fetch, so the
waiting is git's: every command that writes a mirror's refs, the hub's and
the node's alike, passes :data:`REF_LOCK_WAITS`, under which git waits
:data:`REF_LOCK_TIMEOUT_MS` for whichever process holds the lock, whoever it
is. ``tests/test_commit_stage_concurrent.py`` runs exports at once against a
real mirror whose ``packed-refs.lock`` a real git transaction holds past
git's default wait.
"""

from __future__ import annotations

import pathlib
import re
import shlex
from typing import Final

from platform_core.errors import FleetErrorCode
from typing_extensions import TypedDict

from fleet.core import (
    _test_hooks,
    dialect,
    export,
    mirror_fetch,
    names,
    remote,
    stage_repository,
    staging,
)

#: The node mirror's ref that names the last commit fetched into it, the
#: base the hub's next bundle for that node leaves out.
NODE_TIP_REF: Final = "refs/fleet/staged"

#: The prefix of the ref the hub writes in its own mirror for one run's
#: bundle, since ``git bundle create`` bundles refs and refuses a bare sha;
#: the node fetches it into :data:`NODE_TIP_REF`.
BUNDLE_REF_PREFIX: Final = "refs/fleet/bundle/"

#: How long, in milliseconds, a command writing a mirror's refs waits for
#: another git process to give a ref's lock or ``packed-refs.lock`` back:
#: several runs stage on one node at once and export from one hub mirror at
#: once, and git's own 100 ms for a ref and 1 s for ``packed-refs`` refused
#: the second.
REF_LOCK_TIMEOUT_MS: Final = 60_000

#: The options every command writing a mirror's refs passes after ``git -C
#: <mirror>``: git waits :data:`REF_LOCK_TIMEOUT_MS` for a ref's own lock and
#: for ``packed-refs.lock``, which a ref deletion takes.
REF_LOCK_WAITS: Final = (
    "-c",
    f"core.filesRefLockTimeout={REF_LOCK_TIMEOUT_MS}",
    "-c",
    f"core.packedRefsTimeout={REF_LOCK_TIMEOUT_MS}",
)

#: What the node's prepare script prints when its mirror already holds the commit.
HELD_ANSWER: Final = "held"

#: A commit id as the prepare script prints it.
_SHA: Final = re.compile(r"[0-9a-f]{40}")


class CommitExport(TypedDict):
    """One queued commit, as the hub stages it.

    Attributes:
        mirror: The hub's mirror, holding the commit
            (:func:`fleet.core.export.prepare_mirror`).
        key: That mirror's key, which names the node's mirror of the same
            remote (:func:`fleet.core.export.remote_mirror_key`).
        sha: The commit.
        scope: The archive's pathspec, empty for the whole commit.
        bundle: The local file the bundle is written to, removed once sent.
    """

    mirror: pathlib.Path
    key: str
    sha: str
    scope: tuple[str, ...]
    bundle: pathlib.Path


class NodeMirror(TypedDict):
    """What the node's mirror holds, as its prepare script said.

    Attributes:
        tip: :data:`NODE_TIP_REF`'s commit, or None for a mirror that has
            never been fetched into or does not exist.
        held: Whether the mirror's :data:`NODE_TIP_REF` already reaches the
            commit being staged (:func:`mirror_probe_lines`).
    """

    tip: str | None
    held: bool


class StagedCommit(TypedDict):
    """Where a commit landed, and how the feed describes its stage.

    Attributes:
        target: The absolute remote directory the tree was unpacked into.
        detail: The ``staged`` feed line's words: the bytes sent, the
            commit, and the base it was sent over.
    """

    target: str
    detail: str


def prepare_script(*, target: str, staged: str, mirror: str, sha: str) -> str:
    """Make the export's directories and say what the node's mirror holds.

    Args:
        target: The export's directory.
        staged: Its staging directory.
        mirror: The node's mirror of the commit's remote.
        sha: The commit.

    Returns:
        The script's text. It prints what :func:`mirror_probe_lines` prints
        once both directories exist.
    """
    return (
        f"{dialect.PROLOGUE}mkdir -p {shlex.quote(target)} {shlex.quote(staged)}\n"
        f"{mirror_probe_lines(mirror=mirror, sha=sha)}"
    )


def mirror_probe_lines(*, mirror: str, sha: str) -> str:
    """The lines of a node script that say what the node's mirror holds.

    HELD MEANS :data:`NODE_TIP_REF` REACHES THE COMMIT (MCPs board task
    5280a25d), not that the object is there: two runs staging one remote on
    a node at once fetch into one mirror, and a small fetch writes its
    objects loose, the commit before its tree, while git moves the ref only
    once every object is written. A commit off the ref's history is sent
    again, which costs a bundle and never a tree missing its blobs.

    Args:
        mirror: The node's mirror of the remote.
        sha: The commit.

    Returns:
        The lines. They print :data:`NODE_TIP_REF`'s commit when the mirror
        has one and :data:`HELD_ANSWER` when that ref reaches ``sha``, and
        nothing about a mirror that does not exist yet.
    """
    quoted = shlex.quote(mirror)
    tip = shlex.quote(f"{NODE_TIP_REF}^{{commit}}")
    return (
        f"if [ -d {quoted} ]; then\n"
        f"  git -C {quoted} rev-parse --verify --quiet {tip} || true\n"
        f"  if git -C {quoted} merge-base --is-ancestor {shlex.quote(sha)} {NODE_TIP_REF}"
        f" 2>/dev/null; then echo {HELD_ANSWER}; fi\n"
        "fi\n"
    )


def read_node_mirror(said: str) -> NodeMirror:
    """Read the prepare script's answer.

    Args:
        said: What :func:`prepare_script` printed.

    Returns:
        The mirror's tip and whether it holds the commit.
    """
    words = said.split()
    tips = [word for word in words if _SHA.fullmatch(word) is not None]
    return NodeMirror(tip=tips[0] if tips else None, held=HELD_ANSWER in words)


def _git(mirror: pathlib.Path, args: tuple[str, ...], *, verb: str, timeout_seconds: int) -> None:
    """Run one git command in the hub's mirror, refusing the export when it fails.

    Args:
        mirror: The hub's mirror.
        args: The command after ``git -C <mirror>``.
        verb: What the command does, for the refusal.
        timeout_seconds: Its deadline.

    Raises:
        AppError: ``EXPORT_FAILED`` with git's words.
    """
    ran = _test_hooks.run(("git", "-C", str(mirror), *args), timeout_seconds=timeout_seconds)
    if ran["returncode"] != 0:
        raise mirror_fetch.git_failure(FleetErrorCode.EXPORT_FAILED, verb, ran["stderr"])


def bundle_commit(
    mirror: pathlib.Path, *, sha: str, run_id: str, base: str | None, destination: pathlib.Path
) -> bytes:
    """Write the bundle of a commit less what the node's mirror holds, and read it back.

    Args:
        mirror: The hub's mirror, holding ``sha``.
        sha: The commit.
        run_id: The dispatch, which names the ref the bundle is made of.
        base: The node mirror's tip, left out of the bundle with its history
            when the hub's mirror holds it; a base the hub's mirror lacks, or
            none, sends the commit's whole history.
        destination: Where the bundle is written; its directory is made.

    Returns:
        The bundle's bytes, which the node's digest is compared against.

    Raises:
        AppError: ``EXPORT_FAILED`` when a git command exits non-zero.
    """
    ref = f"{BUNDLE_REF_PREFIX}{run_id}"
    # A ref write may spend git's whole lock wait before it starts its own work.
    seconds = REF_LOCK_TIMEOUT_MS // 1000 + mirror_fetch.INIT_TIMEOUT_SECONDS
    _git(
        mirror,
        (*REF_LOCK_WAITS, "update-ref", ref, sha),
        verb=f"git update-ref {ref} in {mirror}",
        timeout_seconds=seconds,
    )
    try:
        known = base is not None and mirror_fetch.has_commit(mirror, base)
        excluded = (f"^{base}",) if known else ()
        _test_hooks.make_directory(destination.parent)
        _git(
            mirror,
            ("bundle", "create", "--quiet", str(destination), ref, *excluded),
            verb=f"git bundle create {sha} from {mirror}",
            timeout_seconds=export.ARCHIVE_TIMEOUT_SECONDS,
        )
    finally:
        _git(
            mirror,
            (*REF_LOCK_WAITS, "update-ref", "-d", ref),
            verb=f"git update-ref -d {ref} in {mirror}",
            timeout_seconds=seconds,
        )
    return _test_hooks.read_bytes(destination)


def fetch_commands(*, mirror: str, bundle: str | None, run_id: str) -> tuple[tuple[str, ...], ...]:
    """The commands that fetch a verified bundle into the node's mirror.

    Shared by a project's commit and a companion's
    (:mod:`fleet.core.companion_stage`, MCPs board task 5280a25d), which
    are staged out of the node's mirror alike.

    Args:
        mirror: The node's mirror.
        bundle: The verified bundle on the node, or None when the mirror
            already holds the commit and nothing was sent.
        run_id: What names the bundle's ref (:data:`BUNDLE_REF_PREFIX`).

    Returns:
        No command when nothing was sent; else the mirror made if absent and
        the bundle's ref fetched into :data:`NODE_TIP_REF`.
    """
    if bundle is None:
        return ()
    return (
        ("git", "init", "--bare", "--quiet", mirror),
        (
            "git",
            "-C",
            mirror,
            *REF_LOCK_WAITS,
            "fetch",
            "--quiet",
            "--no-tags",
            bundle,
            f"+{BUNDLE_REF_PREFIX}{run_id}:{NODE_TIP_REF}",
        ),
    )


def unpack_commands(
    *,
    mirror: str,
    bundle: str | None,
    run_id: str,
    sha: str,
    scope: tuple[str, ...],
    staged: str,
    target: str,
) -> tuple[tuple[str, ...], ...]:
    """The commands that put a commit's tree in the export from the node's mirror.

    Args:
        mirror: The node's mirror.
        bundle: The verified bundle on the node, or None when the mirror
            already holds the commit and nothing was sent.
        run_id: The dispatch, whose bundle ref is fetched.
        sha: The commit.
        scope: The archive's pathspec, empty for the whole commit.
        staged: The export's staging directory, where the tar is written.
        target: The export's directory.

    Returns:
        The commands, in order, for :func:`fleet.core.dialect.checked_script`:
        the mirror made if absent and the bundle fetched into it, the
        commit's tar written and unpacked, and the tree committed.
    """
    tar = f"{staged}/{names.TREE_TAR_NAME}"
    pathspec = ("--", *scope) if scope else ()
    return (
        *fetch_commands(mirror=mirror, bundle=bundle, run_id=run_id),
        ("git", "-C", mirror, "archive", "--format=tar", "-o", tar, sha, *pathspec),
        *dialect.extract_commands(tar, target),
        *stage_repository.init_repository_commands(target, run_id),
    )


def send_bundle(
    host: str,
    *,
    staged: str,
    run_id: str,
    commit: CommitExport,
    base: str | None,
    described_as: str,
) -> int:
    """Bundle the commit less the node mirror's tip and land it, verified.

    Args:
        host: SSH destination.
        staged: The staging directory the bundle lands in.
        run_id: What names the bundle's ref: the dispatch for its commit,
            the dispatch and directory for a companion.
        commit: The commit and the hub's mirror holding it.
        base: The node mirror's tip, or None.
        described_as: What the bundle is, for a digest mismatch's message.

    Returns:
        The bundle's length in bytes.

    Raises:
        AppError: As :func:`bundle_commit` and
            :func:`fleet.core.staging.send_verified` describe.
    """
    sha = commit["sha"]
    try:
        payload = bundle_commit(
            commit["mirror"],
            sha=sha,
            run_id=run_id,
            base=base,
            destination=commit["bundle"],
        )
        staging.send_verified(
            host,
            into=staged,
            source=commit["bundle"],
            payload=payload,
            described_as=described_as,
        )
    finally:
        commit["bundle"].unlink(missing_ok=True)
    return len(payload)


def stage_commit(host: str, *, stage_root: str, run_id: str, commit: CommitExport) -> StagedCommit:
    """Stage one commit on a node from its mirror, sending only what the mirror lacks.

    Args:
        host: SSH destination.
        stage_root: The node's declared stage root.
        run_id: The dispatch.
        commit: The commit and the hub's mirror holding it.

    Returns:
        Where the tree landed and the feed's words for it.

    Raises:
        AppError: ``EXPORT_FAILED`` from the bundle, and as
            :func:`fleet.core.staging.send_verified` and the transport
            describe for the node's side.
    """
    target = names.dispatch_directory(stage_root, run_id)
    staged = names.staging_directory(target)
    mirror = names.node_mirror(stage_root, commit["key"])
    sha = commit["sha"]
    said = remote.run_script(
        host,
        dialect.script_path(stage_root, names.make_directory_stem(run_id)),
        prepare_script(target=target, staged=staged, mirror=mirror, sha=sha),
    )
    node = read_node_mirror(said)
    bundle = None if node["held"] else f"{staged}/{names.ARCHIVE_NAME}"
    sent = 0
    if bundle is not None:
        sent = send_bundle(
            host,
            staged=staged,
            run_id=run_id,
            commit=commit,
            base=node["tip"],
            described_as=f"the bundle of {sha}",
        )
    remote.run_script(
        host,
        dialect.script_path(staged, names.UNPACK_STEM),
        dialect.checked_script(
            unpack_commands(
                mirror=mirror,
                bundle=bundle,
                run_id=run_id,
                sha=sha,
                scope=commit["scope"],
                staged=staged,
                target=target,
            )
        ),
        timeout_seconds=remote.send_deadline_seconds(sent),
    )
    over = (
        "already in the node's mirror"
        if bundle is None
        else f"over {node['tip'] or 'an empty mirror'}"
    )
    return StagedCommit(
        target=target, detail=f"{sent} bytes, git bundle of {sha} {over}, to {target}"
    )


__all__ = [
    "BUNDLE_REF_PREFIX",
    "HELD_ANSWER",
    "NODE_TIP_REF",
    "REF_LOCK_TIMEOUT_MS",
    "REF_LOCK_WAITS",
    "CommitExport",
    "NodeMirror",
    "StagedCommit",
    "bundle_commit",
    "fetch_commands",
    "mirror_probe_lines",
    "prepare_script",
    "read_node_mirror",
    "send_bundle",
    "stage_commit",
    "unpack_commands",
]
