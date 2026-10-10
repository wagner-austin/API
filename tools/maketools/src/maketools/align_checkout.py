"""Move this main checkout to a pushed commit: ``make align-commit COMMIT=<sha>``.

THE ONE SANCTIONED MOVER OF A MAIN CHECKOUT WITH LINKED TREES. The hub's
Claude Code hook ``block-shared-checkout-commit`` refuses every hand-run git
verb that moves such a checkout (commit, rebase, reset, pull, ...), because
its index and branch are shared and it is what scheduled work runs from: a
rebase run by hand in MCPs' checkout moved it under a deploy on 2026-10-10
(MCPs board task 43d66fb3). The hook's own design names ``make
align-commit`` as the route, since git inside a make recipe is what the
recipe does rather than what a session typed. MCPs has that target;
this repository did not, so its pump checkout (``C:\\Users\\Test\\PROJECTS\\API``,
which the hpc-wake scheduled task runs ``tools/hpc-wake`` from) sat 536
commits behind origin with ten local commits and sixteen dirty paths that
nobody could move (MCPs board task f2f89c69).

WHAT DIFFERS FROM MCPs' ALIGN. MCPs' is forwards-only: the hub checkout is
the deploy source and its local commits are what it deploys, so a target
that does not contain HEAD is refused. The pump checkout's local commits
are RE-COMMITS of content origin already carried, made because the hook
refused a rebase, so moving it to origin's head must be allowed to DROP a
local commit, and must refuse to drop one whose change the target does not
carry. Each local-only commit is read against the target, path by path:

* ``carried``: the path's blob at the target equals the commit's.
* ``absent``: the target's blob equals the commit's PARENT's, so the change
  is not there. Refused unless the commit is named in ``DROP``.
* ``evolved``: the target's blob differs from both, so the file moved on and
  whether the change survived cannot be read from blobs. Refused unless
  named in ``DROP``.

A commit ``git cherry`` reads as patch-equivalent to one on the target is
carried whole. The report names every commit, what became of it and why, so
the board post that cites the move carries the evidence.

WHAT IT REFUSES. A dirty tracked path (someone's work: commit, restore or
preserve it first); an untracked path the target tracks (the move would
overwrite it); a commit origin/main does not contain (push first); a target
that reads back as anything but the commit. Untracked paths the target does
not track are carried unchanged and named.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks

#: The remote and branch a main checkout aligns to.
REMOTE: Final[str] = "origin"
TRACKING_BRANCH: Final[str] = "origin/main"

#: A git status row's untracked marker.
UNTRACKED: Final[str] = "??"

#: The three readings of one path of a local-only commit against the target.
Fate = Literal["carried", "absent", "evolved"]


class PathReading(TypedDict):
    """One path of a local-only commit, read against the target.

    Attributes:
        path: The repository-relative path.
        fate: What became of the commit's change to it.
    """

    path: str
    fate: Fate


class LocalCommit(TypedDict):
    """One commit on HEAD that the target does not contain.

    Attributes:
        sha: The full sha.
        subject: Its first message line.
        patch_equivalent: Whether ``git cherry`` found its patch on the target.
        paths: Each path it touches, read against the target; empty when
            the commit is patch-equivalent, since the patch itself is there.
    """

    sha: str
    subject: str
    patch_equivalent: bool
    paths: list[PathReading]


class AlignReport(TypedDict):
    """What an align did.

    Attributes:
        before: HEAD before the move.
        after: HEAD after it, the target.
        local: Every commit the move dropped, with its reading.
        carried_untracked: Untracked paths left as they were.
    """

    before: str
    after: str
    local: list[LocalCommit]
    carried_untracked: list[str]


def _capture(argv: Sequence[str], repo_root: Path, doing: str) -> str:
    """Run a short git question and return its stdout as printed.

    Not stripped: a ``git status --porcelain`` row begins with its status
    letters, and the first row's leading space is part of them.

    Args:
        argv: The command.
        repo_root: The checkout.
        doing: What the command was for, for the refusal.

    Returns:
        Standard output, as printed.

    Raises:
        AppError: ``MAKETOOLS_ALIGN`` when the command failed.
    """
    result = _test_hooks.run_capturing(argv, cwd=repo_root)
    if result["returncode"] != 0:
        raise AppError(
            MaketoolsErrorCode.ALIGN,
            f"{doing} failed (exit {result['returncode']}): {result['stderr'].strip()}",
        )
    return result["stdout"]


def _capture_word(argv: Sequence[str], repo_root: Path, doing: str) -> str:
    """Run a git question whose whole answer is one sha or one line.

    Args:
        argv: The command.
        repo_root: The checkout.
        doing: What the command was for, for the refusal.

    Returns:
        Standard output, stripped.
    """
    return _capture(argv, repo_root, doing).strip()


def _succeeds(argv: Sequence[str], repo_root: Path) -> bool:
    """Whether a git question answers zero.

    Args:
        argv: The command.
        repo_root: The checkout.

    Returns:
        True on exit 0.
    """
    return _test_hooks.run_capturing(argv, cwd=repo_root)["returncode"] == 0


def _blob(repo_root: Path, revision: str, path: str) -> str | None:
    """The blob a path has at a revision.

    Args:
        repo_root: The checkout.
        revision: The commit.
        path: The path.

    Returns:
        The blob sha, or None when the revision has no such path.
    """
    result = _test_hooks.run_capturing(
        ["git", "rev-parse", "--verify", "--quiet", f"{revision}:{path}"], cwd=repo_root
    )
    return result["stdout"].strip() if result["returncode"] == 0 else None


def read_path(repo_root: Path, sha: str, target: str, path: str) -> PathReading:
    """Read what became of one path of a local-only commit at the target.

    Args:
        repo_root: The checkout.
        sha: The local-only commit.
        target: The commit the checkout is moving to.
        path: A path the commit touches.

    Returns:
        The reading.
    """
    at_target = _blob(repo_root, target, path)
    if at_target == _blob(repo_root, sha, path):
        return PathReading(path=path, fate="carried")
    if at_target == _blob(repo_root, f"{sha}^", path):
        return PathReading(path=path, fate="absent")
    return PathReading(path=path, fate="evolved")


def read_local_commits(repo_root: Path, head: str, target: str) -> list[LocalCommit]:
    """Read every commit on HEAD the target does not contain.

    Args:
        repo_root: The checkout.
        head: The checkout's HEAD.
        target: The commit it is moving to.

    Returns:
        The commits, oldest first, each with its reading.
    """
    cherry = _capture(["git", "cherry", target, head], repo_root, "git cherry")
    commits: list[LocalCommit] = []
    for line in cherry.splitlines():
        mark, sha = line.split(" ", 1)
        subject = _capture_word(["git", "log", "-1", "--format=%s", sha], repo_root, "git log")
        equivalent = mark == "-"
        paths: list[PathReading] = []
        if not equivalent:
            touched = _capture(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                repo_root,
                "git diff-tree",
            )
            paths = [read_path(repo_root, sha, target, path) for path in touched.splitlines()]
        commits.append(
            LocalCommit(sha=sha, subject=subject, patch_equivalent=equivalent, paths=paths)
        )
    return commits


def _refuse_uncarried(local: Sequence[LocalCommit], drop: Sequence[str]) -> None:
    """Refuse a local commit the target does not carry and DROP does not name.

    Args:
        local: The local-only commits, read.
        drop: The shas DROP named, full or abbreviated.

    Raises:
        AppError: ``MAKETOOLS_ALIGN`` naming each such commit and path.
    """
    problems: list[str] = []
    for commit in local:
        uncarried = [reading for reading in commit["paths"] if reading["fate"] != "carried"]
        if not uncarried or any(commit["sha"].startswith(name) for name in drop if name):
            continue
        detail = ", ".join(f"{reading['path']} {reading['fate']}" for reading in uncarried)
        problems.append(f"{commit['sha'][:12]} {commit['subject'][:60]!r}: {detail}")
    if problems:
        raise AppError(
            MaketoolsErrorCode.ALIGN,
            "the target does not carry every local commit; each below has a path whose change "
            "is absent at the target or whose file has moved on there. Read each one, and when "
            "its content is on origin by another route, name it in DROP=<sha>,<sha> to drop it; "
            "otherwise push it first:\n" + "\n".join(problems),
        )


def _local_blob(repo_root: Path, path: str) -> str | None:
    """The blob a working-tree file would be stored as.

    Args:
        repo_root: The checkout.
        path: The file.

    Returns:
        The blob sha, or None when there is no such file.
    """
    result = _test_hooks.run_capturing(["git", "hash-object", "--", path], cwd=repo_root)
    return result["stdout"].strip() if result["returncode"] == 0 else None


def overwritten_paths(
    repo_root: Path, untracked: Sequence[str], target: str, target_paths: Sequence[str]
) -> list[str]:
    """The target's files that sit under an untracked path with DIFFERENT bytes.

    A ``git status`` row names an untracked file, or an untracked directory
    with a trailing slash. A file of the target under either is overwritten
    by the move; that loses something only when the working tree's bytes
    differ from the target's, so a file the target tracks with the very
    bytes already there (a cache both sides hold) stops nothing, and one
    missing locally is created rather than overwritten.

    Args:
        repo_root: The checkout.
        untracked: The untracked rows, as ``git status`` names them.
        target: The commit the checkout is moving to.
        target_paths: Every path the target tracks.

    Returns:
        The paths whose local bytes the move would replace.
    """
    covered = [
        name
        for name in target_paths
        if any(name == path or (path.endswith("/") and name.startswith(path)) for path in untracked)
    ]
    differing: list[str] = []
    for name in covered:
        local = _local_blob(repo_root, name)
        if local is not None and local != _blob(repo_root, target, name):
            differing.append(name)
    return differing


def _refuse_dirty(repo_root: Path, target: str) -> list[str]:
    """Refuse tracked dirt and untracked files the target would overwrite.

    Args:
        repo_root: The checkout.
        target: The commit the checkout is moving to.

    Returns:
        The untracked paths the move carries unchanged.

    Raises:
        AppError: ``MAKETOOLS_ALIGN`` naming the paths.
    """
    status = _capture(["git", "status", "--porcelain"], repo_root, "git status")
    tracked: list[str] = []
    untracked: list[str] = []
    for row in status.splitlines():
        (untracked if row.startswith(UNTRACKED) else tracked).append(row[3:])
    if tracked:
        raise AppError(
            MaketoolsErrorCode.ALIGN,
            f"{repo_root} has {len(tracked)} uncommitted tracked path(s); they are somebody's "
            "work, so commit, restore or preserve them first:\n" + "\n".join(tracked),
        )
    target_paths = _capture(
        ["git", "ls-tree", "-r", "--name-only", target], repo_root, "git ls-tree"
    ).splitlines()
    overwritten = overwritten_paths(repo_root, untracked, target, target_paths)
    if overwritten:
        raise AppError(
            MaketoolsErrorCode.ALIGN,
            f"{len(overwritten)} untracked file(s) differ from what {target[:12]} tracks at the "
            "same path and would be overwritten by the move; preserve and remove them first:\n"
            + "\n".join(overwritten),
        )
    return untracked


def align_checkout(repo_root: Path, commit: str, drop: Sequence[str]) -> AlignReport:
    """Move a main checkout to a commit origin/main contains.

    Args:
        repo_root: The checkout.
        commit: The commit, as any revision git resolves.
        drop: Local-only commits the caller has read and allows dropping.

    Returns:
        What moved.

    Raises:
        AppError: ``MAKETOOLS_ALIGN`` for every refusal the module docstring
            names.
    """
    _capture(["git", "fetch", REMOTE], repo_root, f"git fetch {REMOTE}")
    target = _capture_word(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], repo_root, f"resolving {commit}"
    )
    if not _succeeds(["git", "merge-base", "--is-ancestor", target, TRACKING_BRANCH], repo_root):
        raise AppError(
            MaketoolsErrorCode.ALIGN,
            f"{target[:12]} is not on {TRACKING_BRANCH}; a main checkout moves only to a pushed "
            "commit, so push it first",
        )
    carried_untracked = _refuse_dirty(repo_root, target)
    head = _capture_word(["git", "rev-parse", "HEAD"], repo_root, "git rev-parse HEAD")
    local = [] if head == target else read_local_commits(repo_root, head, target)
    _refuse_uncarried(local, drop)
    if head != target:
        _capture(["git", "reset", "--hard", target], repo_root, f"git reset --hard {target[:12]}")
        after = _capture_word(["git", "rev-parse", "HEAD"], repo_root, "git rev-parse HEAD")
        if after != target:
            raise AppError(
                MaketoolsErrorCode.ALIGN, f"the checkout reads back {after[:12]}, not {target[:12]}"
            )
    return AlignReport(before=head, after=target, local=local, carried_untracked=carried_untracked)


def render_report(report: AlignReport) -> list[str]:
    """The lines a recipe prints about an align.

    Args:
        report: What moved.

    Returns:
        One line for the move, one per dropped commit, one per carried path.
    """
    if report["before"] == report["after"]:
        lines = [f"align: already at {report['after'][:12]}; nothing moved"]
    else:
        lines = [
            f"align: {report['before'][:12]} -> {report['after'][:12]}, "
            f"{len(report['local'])} local commit(s) dropped"
        ]
    for commit in report["local"]:
        how = (
            "patch-equivalent to a commit on the target"
            if commit["patch_equivalent"]
            else ", ".join(f"{reading['path']} {reading['fate']}" for reading in commit["paths"])
        )
        lines.append(f"  dropped {commit['sha'][:12]} {commit['subject'][:60]!r}: {how}")
    for path in report["carried_untracked"]:
        lines.append(f"  carried untracked {path}")
    return lines


__all__ = [
    "REMOTE",
    "TRACKING_BRANCH",
    "UNTRACKED",
    "AlignReport",
    "Fate",
    "LocalCommit",
    "PathReading",
    "align_checkout",
    "overwritten_paths",
    "read_local_commits",
    "read_path",
    "render_report",
]
