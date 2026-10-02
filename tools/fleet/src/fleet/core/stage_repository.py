"""The git commands that make a staged tree, or a companion, a repository.

Split out of :mod:`fleet.core.dialect` when turning automatic gc off in a
staged repository took that module past the 600-line ceiling (MCPs board
task 939ec5c7). They are the same COMMANDS on both platforms (every node has
git), so they are functions rather than dialect methods, and each dialect
wraps them in its own status handling (:meth:`fleet.core.dialect.Dialect.checked_script`),
as it does :func:`fleet.core.dialect.extract_commands`.
"""

from __future__ import annotations

from fleet.core.export import COMPANION_REF

#: Who a project's staged tree's one commit is authored by, passed to ``git``
#: with ``-c`` on the commit itself. A companion is cloned with its real
#: commits (:func:`companion_repository_commands`) and gets none.
#:
#: A node has no git identity and is not given one: a dispatch that
#: configured ``user.email`` would leave that setting behind on somebody's
#: workstation, and the next thing they committed there would carry it. The
#: address is in the reserved ``.invalid`` domain (RFC 2606), so it cannot
#: reach anybody even if a tree staged here were ever pushed.
EXPORT_AUTHOR_NAME = "fleet"
EXPORT_AUTHOR_EMAIL = "fleet@corvis.invalid"


def _no_auto_gc_command(target: str) -> tuple[str, ...]:
    """The setting that keeps git from packing a staged repository behind the build.

    A STAGED REPOSITORY NEVER RUNS AN AUTOMATIC GC (MCPs board task
    939ec5c7). ``git commit`` and ``git fetch`` end with ``git gc --auto``,
    which, past ``gc.auto``'s 6700 loose objects, detaches and packs and
    prunes them in the background. The API export is 9292 files, so its
    one commit always crossed that line, and the Linux execution build's
    ``sudo cp -a`` of the stage into execdocker's home raced the detached
    gc: on diphtheria on 2026-10-02 the copy died at ``cp: cannot stat
    .../.git/objects/60/9252c7...: No such file or directory`` and
    tools/fleet-execution-linux job 56e1aa54 ran no test. Measured there
    the same day: one commit of 7500 files left 258 object directories,
    and three seconds later 2 and a pack. A stage is thrown away after one
    build, so it has nothing to gain from a pack. The setting is the
    repository's own, never a node's: the next thing someone runs on
    that machine is not changed by it.

    Args:
        target: Absolute remote directory holding the repository, just
            initialised.

    Returns:
        The command, as an argument vector.
    """
    return ("git", "-C", target, "config", "gc.auto", "0")


def _commit_command(target: str, message: str) -> tuple[str, ...]:
    """The one commit of a staged tree, under the export identity.

    Args:
        target: Absolute remote directory holding the repository.
        message: The commit message (a sha or a run id in words).

    Returns:
        The command, as an argument vector.
    """
    return (
        "git",
        "-C",
        target,
        "-c",
        f"user.name={EXPORT_AUTHOR_NAME}",
        "-c",
        f"user.email={EXPORT_AUTHOR_EMAIL}",
        "commit",
        "--quiet",
        "--message",
        message,
    )


def companion_repository_commands(
    target: str, bundle: str, sha: str, ref: str
) -> tuple[tuple[str, ...], ...]:
    """The commands that clone a staged companion from its bundle.

    A companion exists to be read as a workspace, and the things that read
    it read git: slime's lift check compares its lifted files against
    ``git show HEAD:<path>`` so that an uncommitted edit is not mistaken
    for the code, and a check that runs MCPs' published maketools
    (hardware-wiki's and metabolomics-dashboard's host-code, chat's
    ps-harness) archives the command from ``../MCPs`` at ``origin/main``
    (MCPs board task a8ee9b21).

    A REAL CLONE, WITH ITS HISTORY (MCPs board task 2026dfbc). Until then
    the companion landed as ``git archive`` of the ref's tip, committed on
    the node as one synthetic commit: no history, and a HEAD whose sha was
    not the real one. A check that reads a companion's past therefore failed
    on every node and passed on every workstation. corvis-stick's
    HookCommands suite reads ``packages/claude-hooks/registration.json`` at
    the MCPs commit its hooks pin names, and on serendipity that read
    'fatal: path ... exists on disk, but not in cd4d6aa2', a commit that is
    an ancestor of MCPs main (job 972977be, 2026-09-30). The hub now sends
    ``git bundle create`` of the companion ref
    (:func:`fleet.core.export.bundle_ref`), and the node fetches it into
    ``refs/remotes/origin/<branch>`` and checks ``<branch>`` out there, so
    HEAD and ``origin/<branch>`` are the ref's real commit and every
    ancestor the hub's mirror holds is reachable from them.

    THE LAST TWO COMMANDS PROVE HEAD IS ``sha``: each is an ancestor of the
    other only when they are the same commit. The bundle is the one the hub
    wrote after resolving the ref to ``sha``, and its digest is checked
    before this runs, so a mismatch is not expected; it would mean the
    bundle named another commit, and the stage stops rather than checking a
    tree the feed would then misreport.

    The fetch ends in ``git gc --auto`` like a commit does, so the clone
    turns automatic gc off first (:func:`_no_auto_gc_command`).

    Args:
        target: Absolute remote directory the companion is cloned into,
            emptied and created just before.
        bundle: Absolute path to the verified bundle on the node.
        sha: The commit the hub resolved the ref to and bundled.
        ref: The companion's declared ref, ``main`` or ``refs/heads/main``.

    Returns:
        The six commands, in order, for a dialect's
        :meth:`fleet.core.dialect.Dialect.checked_script`.
    """
    branch = ref.removeprefix("refs/heads/")
    tracking = f"refs/remotes/origin/{branch}"
    return (
        ("git", "-C", target, "init", "--quiet"),
        _no_auto_gc_command(target),
        (
            "git",
            "-C",
            target,
            "fetch",
            "--quiet",
            "--no-tags",
            bundle,
            f"+{COMPANION_REF}:{tracking}",
        ),
        ("git", "-C", target, "checkout", "--quiet", "-B", branch, tracking),
        ("git", "-C", target, "merge-base", "--is-ancestor", "HEAD", sha),
        ("git", "-C", target, "merge-base", "--is-ancestor", sha, "HEAD"),
    )


def init_repository_commands(target: str, run_id: str) -> tuple[tuple[str, ...], ...]:
    """The commands that make a staged tree a one-commit git repository.

    THE SAME ON BOTH PLATFORMS, and without them a staged build lints
    different files from a local one. Ruff honours ``.gitignore`` and applies
    it ONLY inside a git repository -- so a tree that carries the file but no
    ``.git`` silently widens what gets linted to include everything the
    repository deliberately excludes.

    Measured on lavender 2026-09-04, dispatching ``tools/hpc3``: 902 ruff
    errors, all in ``tools/hpc3/runs``, which ``.gitignore`` line 170 excludes
    as build artifacts while explicitly tracking the run documents beside
    them. The same tree with ``git init`` run in it reports ``All checks
    passed``. Locally ``ruff check .`` passes and
    ``ruff check . --no-respect-gitignore`` reports exactly 902 -- the same
    number, which identifies the mechanism rather than suggesting it.

    THE ALTERNATIVE WAS TO ADD AN EXCLUDE TO THE PROJECT, AND IT WOULD HAVE
    BEEN WRONG. The repository already states which paths are build output;
    a ruff ``exclude`` restating it is a second copy of one policy, and the
    copy that drifts is the one nobody looks at. Reproducing the environment
    a build is defined against is this package's job, not the project's.

    THE INDEX IS FILLED TOO, since the first fleet verdict (MCPs board task
    fd5cabfa, 2026-09-21T10:03Z): ``MCPs/packages/maketools`` at 9ee70275
    on diphtheria read ``1 failed, 416 passed``, the one failure
    ``test_git_lists_the_tracked_files_matching_a_pattern`` asserting that
    ``git ls-files CLAUDE.md`` names the file, and in a repository with an
    empty index it names nothing. A checkout's suite reads its own tracked
    set (that package's ``lint-makefiles`` lints exactly the tracked
    Makefiles), so a staged tree whose index is empty is not the tree the
    suite was written against.

    AND IT IS FORCED, since API board task 0b3591d7 (2026-10-03). A plain
    ``git add --all`` drops every tracked file ``.gitignore`` ignores, which
    is every force-added one: ``tools/hpc3`` at f75c4a94 on lavender-wsl
    read ``1 failed``, its run-document audit reading ``git archive HEAD``
    and finding only the one sweep a ``.gitignore`` negation re-includes,
    the nine force-added beside it missing. The tree here is exactly the
    commit's archive, the transport files staged beside it, so ``--force``
    indexes the commit's files and nothing else; ruff reads ``.gitignore``
    patterns rather than the index, so the lint above is unchanged.

    AND THE INDEX IS COMMITTED, since MCPs board task 6bbfd171
    (2026-09-26). This docstring said until then that nothing reads a commit
    in the tree the recipe runs in, and it was false for every package whose
    suite migrates a test database: ``packages/db``'s migrator admits a test
    migration only against ``git rev-parse HEAD`` and ``git ls-tree HEAD``
    (``src/migrate/test-admission.ts``), and on diphtheria that day a staged
    MCPs tree with no commit failed it with ``TEST_MIGRATION_PROVENANCE_UNAVAILABLE:
    git rev-parse: fatal: ambiguous argument 'HEAD'``. The same tree with
    one commit applied 557 migrations and ``packages/tenant`` read ``ALL
    CHECKS PASSED``. The commit is made here, before any install step runs,
    so no hook the project's install configures (husky's
    ``core.hooksPath``) exists yet to run on it.

    AUTOMATIC GC IS TURNED OFF before the commit (:func:`_no_auto_gc_command`),
    so the commit leaves its loose objects where the build copies them.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, recorded in the message so the tree on the
            node names the run it was staged for.

    Returns:
        The four commands, in order, for a dialect's
        :meth:`fleet.core.dialect.Dialect.checked_script`.
    """
    return (
        ("git", "-C", target, "init", "--quiet"),
        _no_auto_gc_command(target),
        ("git", "-C", target, "add", "--all", "--force"),
        _commit_command(target, f"fleet export {run_id}"),
    )


__all__ = [
    "EXPORT_AUTHOR_EMAIL",
    "EXPORT_AUTHOR_NAME",
    "companion_repository_commands",
    "init_repository_commands",
]
