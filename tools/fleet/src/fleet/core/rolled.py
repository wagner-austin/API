"""The fleet code a scheduled tick runs: the rolled commit, never the checkout.

THE GAP (board task 465689f5, b6eb30c8's R3). The hub's scheduled ticks,
then ``run-agent-tick.ps1`` and ``run-node-agent-tick.ps1`` and since
2026-09-29 :mod:`fleet.cli.tick`, ran ``poetry run fleet-agent`` in
``C:/Users/Test/PROJECTS/API/tools/fleet``. So the fleet ran
whatever that checkout held: every commit anyone pulled into it went live at
the next three-minute tick, and so did every uncommitted edit a session left
there. No step between a change and the fleet running it asked whether its
execution suites had ever run.

WHAT A ROLL IS. ``make fleet-roll`` in API's root first runs ``executed``,
MCPs' ``publish-executed`` for tools/fleet-execution and
tools/fleet-execution-linux, which refuses unless the fleet queue records a
passed run of each at exactly HEAD and the tree is exactly HEAD. Only then
does it point :data:`ROLLED_REF` at HEAD. A tick resolves that ref, extracts
the trees the agent imports and the registry it reads from that commit
(:mod:`fleet.core.commit_tree`), and runs the agent from the extraction.
The dependencies stay the checkout's tools/fleet environment; the code and
the configuration are pinned.

WHY A REF. It lives in the object store every worktree of the repository
shares, so a roll made from any worktree is what the next tick reads, it
names a commit git already holds, and nothing that edits files can move it.

NOTHING RUNS WITHOUT A ROLL. A ref that does not resolve, a failed archive
or extraction, or an extraction lacking an entry point refuses the tick by
code, and the agent does not start: running the checkout on the day the
extraction fails is the defect this removes.

ONE EXTRACTION PER COMMIT, MADE IN A TURN (MCPs board task fb505198). The
hub's ticks start together, ten in one second by 2026-10-24. Extracting
into one directory per commit with no turn, they ran several ``tar``
processes over one tree, and at 10:24Z on 2026-09-27 two of them failed on
files another was writing (FLEET_ROLL_EXTRACT_FAILED on diphtheria's and
sedona's ticks, each naming a different file). From then until 2026-10-24
each launch extracted its own copy under its process id and removed it when
its agent exited, which wrote the same tree to diphtheria's SSD every few
seconds, about 150 GiB a day with its tarball and bytecode, and left 873
copies behind from ticks killed before the removal. Now a launch takes
:data:`EXTRACT_TURN`, and inside it either finds the commit's tree, or
extracts into a directory named for its own process and renames it into
place once the required files are there, so a tree is only ever whole and
no ``tar`` writes where another launch reads. The same turn reaps what
:mod:`fleet.core.scratch` counts as left behind.

NO DEADLOCK. The execution suites a roll needs are dispatched and collected
by the agent already rolled, and each node runs the suite of the commit it
was handed, so an older roll is what proves the next one.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Final, TypedDict

from fleet.core import _test_hooks, commit_tree, file_turn, scratch
from fleet.core.workspace_files import SETTINGS_FILE

#: The ref ``make fleet-roll`` sets and every tick reads.
ROLLED_REF: Final = "refs/fleet/rolled"

#: The source trees the agents import: tools/fleet and its three runtime
#: path dependencies (tools/fleet/pyproject.toml), and the registry they read.
FLEET_SRC: Final = "tools/fleet/src"
PLATFORM_CORE_SRC: Final = "libs/platform_core/src"
MONOREPO_GUARDS_SRC: Final = "libs/monorepo_guards/src"
BOARD_WATCH_SRC: Final = "tools/board-watch/src"
#: The registry directory (:mod:`fleet.core.workspace_files`; one file,
#: ``tools/fleet/fleet.json``, until MCPs board task 4a78c916).
REGISTRY_DIRECTORY: Final = "tools/fleet/registry"

#: The source trees, in the order they go on ``PYTHONPATH``.
SOURCE_TREES: Final[tuple[str, ...]] = (
    FLEET_SRC,
    PLATFORM_CORE_SRC,
    MONOREPO_GUARDS_SRC,
    BOARD_WATCH_SRC,
)

ARCHIVED_PATHS: Final[tuple[str, ...]] = (*SOURCE_TREES, REGISTRY_DIRECTORY)

#: Files whose absence means the extraction cannot run a tick: both agents'
#: entry points, one module of each imported package, and the registry's
#: fixed file (its machines and projects directories are checked by the
#: agent's own reader).
REQUIRED_FILES: Final[tuple[str, ...]] = (
    f"{FLEET_SRC}/fleet/cli/agent.py",
    f"{FLEET_SRC}/fleet/cli/node_agent.py",
    f"{PLATFORM_CORE_SRC}/platform_core/__init__.py",
    f"{MONOREPO_GUARDS_SRC}/monorepo_guards/__init__.py",
    f"{BOARD_WATCH_SRC}/board_watch/__init__.py",
    f"{REGISTRY_DIRECTORY}/{SETTINGS_FILE}",
)

#: Where the registry's relative record paths resolve in the checkout: the
#: ledger, feed and leases are this machine's running state, never rolled.
RECORDS_DIR: Final = "tools/fleet"

#: The lock file the extraction turn is held by, in the extractions
#: directory beside the trees.
TURN_LOCK: Final = "turn.lock"

#: The turn a launch finds or makes its commit's tree in, and reaps in. The
#: longest turn is an extraction, whose two steps each have
#: :data:`fleet.core.commit_tree.TREE_STEP_TIMEOUT_SECONDS`, so a lock a
#: minute older than both was left by a launch that died holding it.
EXTRACT_TURN: Final = file_turn.TurnBounds(
    what="extraction turn",
    waited_for="another launch finding, extracting or reaping the rolled tree",
    stale_seconds=2 * commit_tree.TREE_STEP_TIMEOUT_SECONDS + 60,
    poll_seconds=1,
)

#: Detail prefixes, one per way a tick can be refused.
REF_UNRESOLVED_CODE: Final = "FLEET_ROLL_REF_UNRESOLVED"
ARCHIVE_FAILED_CODE: Final = "FLEET_ROLL_ARCHIVE_FAILED"
EXTRACT_FAILED_CODE: Final = "FLEET_ROLL_EXTRACT_FAILED"
INCOMPLETE_CODE: Final = "FLEET_ROLL_INCOMPLETE"


class RolledTree(TypedDict):
    """The rolled fleet code one tick runs.

    Attributes:
        commit: The full commit id the extraction was taken from.
        directory: The commit's one extraction, shared by every launch of
            it and kept until no launch has started from it for
            :data:`fleet.core.scratch.STALE_SECONDS`.
        python_path: The ``PYTHONPATH`` value putting the four extracted
            source trees ahead of the checkout's editable installs.
        config: The extracted registry, passed as ``--config``.
        reaped: What this launch's turn removed as left behind, each as
            ``<scratch subdirectory>/<name>``, for the tick's log.
    """

    commit: str
    directory: str
    python_path: str
    config: str
    reaped: tuple[str, ...]


def staging_directory(extractions: pathlib.Path, commit: str) -> pathlib.Path:
    """Where this launch extracts a commit before renaming it into place.

    Args:
        extractions: The extractions directory.
        commit: The rolled commit.

    Returns:
        ``<extractions>/<commit>.pid-<this process id>``, a name no tree has.
    """
    return extractions / f"{commit}.pid-{os.getpid()}"


def extract_rolled_tree(api_root: pathlib.Path) -> RolledTree | str:
    """Find or extract the rolled fleet code, or say why a tick cannot run.

    Args:
        api_root: The API checkout whose object store holds the ref.

    Returns:
        The commit's extraction, or a ``CODE: message`` refusal when any
        step fails; nothing is run from the working tree instead.
    """
    resolved = commit_tree.resolve_commit(api_root, ROLLED_REF, REF_UNRESOLVED_CODE)
    if isinstance(resolved, str):
        return resolved
    commit = resolved["commit"]
    extractions = scratch.directory(scratch.EXTRACTIONS_DIR)
    with file_turn.held(
        extractions / TURN_LOCK, holder=f"the launch of {commit}", bounds=EXTRACT_TURN
    ):
        destination = _ready_tree(api_root, commit, extractions)
        reaped = tuple(
            f"{name}/{entry}"
            for name, keep in (
                (scratch.EXTRACTIONS_DIR, frozenset({commit, TURN_LOCK})),
                (scratch.ARCHIVES_DIR, frozenset[str]()),
            )
            for entry in scratch.reap(scratch.directory(name), keep=keep)
        )
    if isinstance(destination, str):
        return destination
    return RolledTree(
        commit=commit,
        directory=str(destination),
        python_path=os.pathsep.join(
            str(destination / pathlib.PurePosixPath(tree)) for tree in SOURCE_TREES
        ),
        config=str(destination / pathlib.PurePosixPath(REGISTRY_DIRECTORY)),
        reaped=reaped,
    )


def _ready_tree(
    api_root: pathlib.Path, commit: str, extractions: pathlib.Path
) -> pathlib.Path | str:
    """Find the commit's whole tree or make it, inside the extraction turn.

    Args:
        api_root: The API checkout.
        commit: The rolled commit.
        extractions: The extractions directory.

    Returns:
        ``<extractions>/<commit>``, whole and touched now so the reaper
        counts its age from this launch, or a refusal.
    """
    destination = extractions / commit
    if _test_hooks.directory_exists(destination):
        # A TREE MISSING A REQUIRED FILE IS REMOVED AND REFUSED, so the next
        # launch extracts it again rather than every launch refusing on it.
        # Only a whole tree is ever renamed into place, so this is a tree
        # something outside the fleet edited.
        damaged = commit_tree.incompleteness(
            destination, commit, REQUIRED_FILES, code=INCOMPLETE_CODE
        )
        if damaged is not None:
            shutil.rmtree(destination)
            return damaged
    else:
        made = _extract(api_root, commit, extractions)
        if isinstance(made, str):
            return made
        os.rename(made, destination)
    now = _test_hooks.now()
    os.utime(destination, (now, now))
    return destination


def _extract(api_root: pathlib.Path, commit: str, extractions: pathlib.Path) -> pathlib.Path | str:
    """Extract the commit into this launch's staging directory.

    Args:
        api_root: The API checkout.
        commit: The rolled commit.
        extractions: The extractions directory.

    Returns:
        The staging directory, holding every required file and not the
        tarball it was extracted from, or a refusal with the directory
        removed.
    """
    staging = staging_directory(extractions, commit)
    extracted = commit_tree.extract_paths(
        api_root,
        commit,
        ARCHIVED_PATHS,
        destination=staging,
        archive_code=ARCHIVE_FAILED_CODE,
        extract_code=EXTRACT_FAILED_CODE,
    )
    # A REFUSED LAUNCH REMOVES ITS OWN DIRECTORY, which extract_paths made
    # before either step could fail: 23 such directories stayed under one
    # commit on the hub after the 2026-09-28 extraction timeouts (MCPs board
    # task 94ac1c4f).
    if isinstance(extracted, str):
        shutil.rmtree(staging)
        return extracted
    incomplete = commit_tree.incompleteness(staging, commit, REQUIRED_FILES, code=INCOMPLETE_CODE)
    if incomplete is not None:
        shutil.rmtree(staging)
        return incomplete
    (staging / commit_tree.TARBALL_NAME).unlink()
    return staging


def rolled_state(checkout: pathlib.Path) -> str:
    """What :data:`ROLLED_REF` names now, as a serving agent compares it.

    A node runner serves across its scheduled starts (:mod:`fleet.cli.node_serve`,
    MCPs board task 8993c306), so it reads this when it starts and again at
    each fire boundary, and hands over once the two differ
    (:func:`fleet.cli.node_serve.handover_policy`). It is read by the agent
    rather than handed down by the launcher because the launcher runs from
    the checkout, which no roll updates, and a flag it does not yet pass
    would refuse every start.

    Args:
        checkout: A directory inside the checkout whose ref the launcher
            read: the agent's records directory, ``tools/fleet`` there.

    Returns:
        The commit the ref names, or the :data:`REF_UNRESOLVED_CODE`
        refusal saying why it names none; either compares unequal to a
        different answer, so a ref that stops resolving hands over too, and
        the next start reports the refusal as its own.
    """
    resolved = commit_tree.resolve_commit(checkout, ROLLED_REF, REF_UNRESOLVED_CODE)
    if isinstance(resolved, str):
        return resolved
    return resolved["commit"]


__all__ = [
    "ARCHIVED_PATHS",
    "ARCHIVE_FAILED_CODE",
    "BOARD_WATCH_SRC",
    "EXTRACT_FAILED_CODE",
    "EXTRACT_TURN",
    "FLEET_SRC",
    "INCOMPLETE_CODE",
    "MONOREPO_GUARDS_SRC",
    "PLATFORM_CORE_SRC",
    "RECORDS_DIR",
    "REF_UNRESOLVED_CODE",
    "REGISTRY_DIRECTORY",
    "REQUIRED_FILES",
    "ROLLED_REF",
    "SOURCE_TREES",
    "TURN_LOCK",
    "RolledTree",
    "extract_rolled_tree",
    "rolled_state",
    "staging_directory",
]
