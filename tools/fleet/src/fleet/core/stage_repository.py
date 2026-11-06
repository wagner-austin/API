"""The git commands that make a staged tree, or a companion, a repository.

Split out of :mod:`fleet.core.dialect` when turning automatic gc off in a
staged repository took that module past the 600-line ceiling (MCPs board
task 939ec5c7). They are argument vectors, which
:func:`fleet.core.dialect.checked_script` renders into a script that ends at
the first failure, as it does :func:`fleet.core.dialect.extract_commands`.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

#: The settings every write into a node's mirror runs under: the automatic
#: maintenance git starts at the end of a fetch runs to its end inside that
#: fetch, under its lock, rather than detached behind it
#: (:func:`under_mirror_lock`). ``maintenance.autoDetach`` is read by git
#: 2.47 and later, a NixOS node's, and ``gc.autoDetach`` by the
#: ``git gc --auto`` an Ubuntu node's 2.43 runs instead.
INLINE_MAINTENANCE: Final = ("-c", "gc.autoDetach=false", "-c", "maintenance.autoDetach=false")

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

#: The setting a staged tree's ``git add`` runs under: its loose objects
#: stored without zlib (:func:`init_repository_commands`).
UNCOMPRESSED_OBJECTS = "core.looseCompression=0"


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


def _no_auto_maintenance_command(target: str) -> tuple[str, ...]:
    """The setting that keeps a newer git from packing a staged repository behind the build.

    ``gc.auto`` 0 is not enough on every node (MCPs board task c2e3f31f):
    git 2.54, a NixOS node's, ends ``git commit`` with ``git maintenance run
    --auto`` instead of ``git gc --auto``, and its tasks repack the loose
    objects whatever ``gc.auto`` says. Measured on loki on 2026-10-26: three
    loose blobs in one object directory, ``gc.auto`` 0, and the commit left
    a pack and no loose object; with ``maintenance.auto`` false beside it
    every loose object stayed, on loki's 2.54 and lavender's 2.43 alike.

    Args:
        target: Absolute remote directory holding the repository, just
            initialised.

    Returns:
        The command, as an argument vector.
    """
    return ("git", "-C", target, "config", "maintenance.auto", "false")


def _commit_command(target: str, message: str) -> tuple[str, ...]:
    """The one commit of a staged tree, under the export identity.

    ``--allow-empty`` because a tree staged on its commit's history
    (:func:`base_branch_commands`) is committed on top of that commit, and a
    commit exported whole leaves the index equal to its parent's tree.

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
        "--allow-empty",
        "--message",
        message,
    )


class BaseBranch(TypedDict):
    """The branch a project's commits are measured against, resolved on the hub.

    Attributes:
        ref: The declared branch, ``main`` or ``refs/heads/main``.
        sha: Its tip when the hub prepared the job
            (:func:`fleet.core.mirror_fetch.fetch_ref`).
    """

    ref: str
    sha: str


def base_branch_commands(
    target: str, sha: str, base_branch: BaseBranch
) -> tuple[tuple[str, ...], ...]:
    """The commands that put a staged tree on its commit's history, beside its base branch.

    A STAGED TREE COULD NOT SAY WHAT ITS COMMIT CHANGED (MCPs board task
    a48b5236). Its one commit was a root commit of the extracted tree, in a
    repository with no ref to the queued commit and no branch, so a check
    asking which files the commit changes against ``origin/main`` (the
    mutation step every MCPs TypeScript package runs on the files it
    changes) had nothing to ask. The repository is cloned from the node's
    mirror, which holds the commit's whole history and, for a project
    declaring a base branch, that branch's tip (both fetched from one
    bundle); so ``HEAD`` is set to the commit before the tree is indexed,
    making the staged commit its child, and ``refs/remotes/origin/<branch>``
    names the tip, as a checkout's does. ``git merge-base HEAD origin/main``
    then reads in a stage what it reads on a workstation, and ``git
    update-ref`` refuses either commit the mirror lacks, ending the script.

    Args:
        target: Absolute remote directory holding the cloned repository,
            before its tree is indexed.
        sha: The queued commit.
        base_branch: The branch and the tip the hub resolved.

    Returns:
        The two ref writes, in order.
    """
    branch = base_branch["ref"].removeprefix("refs/heads/")
    return (
        ("git", "-C", target, "update-ref", f"refs/remotes/origin/{branch}", base_branch["sha"]),
        ("git", "-C", target, "update-ref", "HEAD", sha),
    )


def companion_repository_commands(
    target: str, mirror: str, sha: str, ref: str
) -> tuple[tuple[str, ...], ...]:
    """The commands that clone a staged companion out of the node's mirror of its remote.

    A companion exists to be read as a workspace, and the things that read
    it read git: slime's lift check compares its lifted files against
    ``git show HEAD:<path>`` so that an uncommitted edit is not mistaken
    for the code, and a check that runs MCPs' published maketools
    (hardware-wiki's and metabolomics-dashboard's host-code) archives the
    command from ``../MCPs`` at ``origin/main`` (MCPs board task a8ee9b21).

    A REAL CLONE, WITH ITS HISTORY (MCPs board task 2026dfbc). Until then
    the companion landed as ``git archive`` of the ref's tip, committed on
    the node as one synthetic commit: no history, and a HEAD whose sha was
    not the real one. A check that reads a companion's past therefore failed
    on every node and passed on every workstation. corvis-stick's
    HookCommands suite reads ``packages/claude-hooks/registration.json`` at
    the MCPs commit its hooks pin names, and on serendipity that read
    'fatal: path ... exists on disk, but not in cd4d6aa2', a commit that is
    an ancestor of MCPs main (job 972977be, 2026-09-30). So the companion's
    ``refs/remotes/origin/<branch>`` is the ref's real commit, ``<branch>``
    is checked out there, and every ancestor the node's mirror holds is
    reachable from them.

    OUT OF THE NODE'S MIRROR (MCPs board task 5280a25d). Until then every
    launch sent each companion's whole history as one bundle the node
    fetched into a fresh repository: on lacrymal on 2026-11-05 the MCPs
    companion's bundle was 266 MB and took 22.8 s to write and API's 263 MB
    and 6.9 s, and packages/wiki-check carries 17 companions. The node now
    keeps a mirror of each companion's remote as it does of each project's
    (:mod:`fleet.core.companion_stage`), which receives only what it lacks,
    and the companion is cloned from it by hard links
    (:func:`clone_commands`). Its tracking ref is written to the commit
    itself, which ``git update-ref`` refuses when the mirror lacks it,
    ending the checked script there.

    Args:
        target: Absolute remote directory the companion is cloned into,
            emptied and created just before.
        mirror: The node's mirror of the companion's remote, holding ``sha``.
        sha: The commit the hub resolved the ref to.
        ref: The companion's declared ref, ``main`` or ``refs/heads/main``.

    Returns:
        The six commands, in order, for
        :func:`fleet.core.dialect.checked_script`.
    """
    branch = ref.removeprefix("refs/heads/")
    tracking = f"refs/remotes/origin/{branch}"
    return (
        *clone_commands(target, mirror),
        ("git", "-C", target, "update-ref", tracking, sha),
        ("git", "-C", target, "checkout", "--quiet", "-B", branch, tracking),
    )


def init_repository_commands(target: str, run_id: str) -> tuple[tuple[str, ...], ...]:
    """The commands that make a staged tree a one-commit git repository.

    Without them a staged build lints
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

    AUTOMATIC GC AND MAINTENANCE ARE TURNED OFF before the commit
    (:func:`_no_auto_gc_command`, :func:`_no_auto_maintenance_command`), so
    the commit leaves its loose objects where the build copies them. Until
    board task e705f08c it also set ``core.autocrlf`` false for a Windows
    node's git, whose add warned once per file (MCPs board task ee3d6cdc); a
    Linux node's git converts no line endings unless it is asked to.

    THE ADD WRITES ITS OBJECTS UNCOMPRESSED (MCPs board task 1e0ee821). It
    hashes and stores every file of the tree, and zlib was most of its
    time: on lacrymal on 2026-11-04 the add of an
    extracted MCPs tree took 6.3 s at git's default loose compression and
    1.9 s at ``core.looseCompression`` 0, every gate job paying it between
    its claim and its first phase. The setting is the add's alone
    (:data:`UNCOMPRESSED_OBJECTS`), so the repository stores nothing, and the
    objects it leaves, 417 MB for that tree against 216 MB, go with the run.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, recorded in the message so the tree on the
            node names the run it was staged for.

    Returns:
        The five commands, in order, for
        :func:`fleet.core.dialect.checked_script`.
    """
    return (
        ("git", "-C", target, "init", "--quiet"),
        _no_auto_gc_command(target),
        _no_auto_maintenance_command(target),
        *index_commands(target, run_id),
    )


def index_commands(target: str, run_id: str) -> tuple[tuple[str, ...], ...]:
    """The commands that index a staged tree, forced, and make it one commit.

    The half of :func:`init_repository_commands` that a staged commit's
    unpack runs as well, after :func:`clone_commands` and the extraction:
    there every object the tree holds is already the node mirror's, linked
    into the repository, so the add hashes each file and writes almost
    nothing (MCPs board task 5280a25d).

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, recorded in the message.

    Returns:
        The add and the commit.
    """
    return (
        ("git", "-C", target, "-c", UNCOMPRESSED_OBJECTS, "add", "--all", "--force"),
        _commit_command(target, f"fleet export {run_id}"),
    )


def clone_commands(target: str, mirror: str) -> tuple[tuple[str, ...], ...]:
    """The commands that make an empty directory a repository holding the node mirror's objects.

    BY HARD LINKS, NOT BY COPIES (MCPs board task 5280a25d). ``git clone
    --local`` links the mirror's object files into the new repository, so a
    companion is cloned without copying its history and a staged commit's
    add finds every object it would have written already there: timed by
    hand against the MCPs mirror at b96091cf9 on 2026-11-05, the add of the
    extracted tree took 0.83 s on pendragon with them and 2.11 s without,
    and 2.3 s on lenovold against 5.4 s, the committed tree the same. The
    clone names its source as ``origin``; that section is removed, so the
    repository names no remote, as before. Automatic gc and maintenance are
    turned off before anything else writes (:func:`_no_auto_gc_command`,
    :func:`_no_auto_maintenance_command`).

    The clone runs under the mirror's lock (:func:`under_mirror_lock`),
    since it links the mirror's files one at a time.

    Args:
        target: Absolute remote directory, empty, the repository is made in.
        mirror: The node's mirror of the remote.

    Returns:
        The clone and the three settings after it.
    """
    return (
        under_mirror_lock(
            mirror, ("git", "clone", "--quiet", "--local", "--no-checkout", mirror, target)
        ),
        _no_auto_gc_command(target),
        _no_auto_maintenance_command(target),
        ("git", "-C", target, "config", "--remove-section", "remote.origin"),
    )


def mirror_lock(mirror: str) -> str:
    """The file a node's mirror is locked on, beside it in the mirrors directory.

    Args:
        mirror: The node's mirror.

    Returns:
        ``<mirror>.objects.lock``, which nothing git runs writes or reads.
    """
    return f"{mirror}.objects.lock"


def under_mirror_lock(mirror: str, command: tuple[str, ...]) -> tuple[str, ...]:
    """A command that writes a node's mirror or links its files, run holding the mirror's lock.

    A CLONE OF A MIRROR NEVER RUNS BESIDE A WRITE INTO IT (MCPs board task
    5280a25d). ``git clone --local`` lists the mirror's object directory and
    links each file it listed, and refuses a file that changed in between:
    on lavender at 20:48:59Z on 2026-11-05 packages/wiki-check's job
    392cd878 was refused ``DISPATCH_FAILED`` on 'fatal: hardlink different
    from source at .../chat/.git/objects/info/commit-graphs/tmp_graph_IWyF5I',
    the commit graph that the automatic maintenance of a fetch into the chat
    companion's mirror was writing, detached, while the clone ran. A fetch
    of another run into the same mirror writes its pack and its loose
    objects through temporary files the same way. Counted by hand on ziggurat
    the same day, 16 of 30 clones of a mirror were refused so while another
    process appended to a file in its pack directory.

    So every write into a node's mirror and every clone out of it holds one
    exclusive ``flock`` on :func:`mirror_lock`, which the kernel gives back
    when its holder exits, however it exits; a write's automatic maintenance
    runs inside it (:data:`INLINE_MAINTENANCE`). A clone holds it for
    milliseconds, and a fetch for the transfer of a bundle that carries only
    what the mirror lacks. What reads the mirror, ``git archive`` and the
    probes, takes no lock: git's readers find a pack that maintenance
    replaced.

    Args:
        mirror: The node's mirror.
        command: The write or the clone, as an argument vector.

    Returns:
        The command under ``flock``, as an argument vector.
    """
    return ("flock", mirror_lock(mirror), *command)


__all__ = [
    "EXPORT_AUTHOR_EMAIL",
    "EXPORT_AUTHOR_NAME",
    "INLINE_MAINTENANCE",
    "UNCOMPRESSED_OBJECTS",
    "BaseBranch",
    "base_branch_commands",
    "clone_commands",
    "companion_repository_commands",
    "index_commands",
    "init_repository_commands",
    "mirror_lock",
    "under_mirror_lock",
]
