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

ONE EXTRACTION PER LAUNCH, REMOVED WHEN THE AGENT EXITS. The hub's five
scheduled ticks start in the same second. Extracting into one directory per
commit, as the session verbs do, they ran five ``tar`` processes over one
tree, and at 10:24Z on 2026-09-27 two of them failed on files another was
writing (FLEET_ROLL_EXTRACT_FAILED on diphtheria's and sedona's ticks, each
naming a different file). So each launch extracts under its own process id
and :func:`discard_rolled_tree` removes the directory after the agent
returns: no run writes a tree another reads, and no tick trusts a copy an
earlier one left behind.

NO DEADLOCK. The execution suites a roll needs are dispatched and collected
by the agent already rolled, and each node runs the suite of the commit it
was handed, so an older roll is what proves the next one.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Final, TypedDict

from fleet.core import _test_hooks, commit_tree

#: The ref ``make fleet-roll`` sets and every tick reads.
ROLLED_REF: Final = "refs/fleet/rolled"

#: The source trees the agents import: tools/fleet and its three runtime
#: path dependencies (tools/fleet/pyproject.toml), and the registry they read.
FLEET_SRC: Final = "tools/fleet/src"
PLATFORM_CORE_SRC: Final = "libs/platform_core/src"
MONOREPO_GUARDS_SRC: Final = "libs/monorepo_guards/src"
BOARD_WATCH_SRC: Final = "tools/board-watch/src"
REGISTRY_FILE: Final = "tools/fleet/fleet.json"

#: The source trees, in the order they go on ``PYTHONPATH``.
SOURCE_TREES: Final[tuple[str, ...]] = (
    FLEET_SRC,
    PLATFORM_CORE_SRC,
    MONOREPO_GUARDS_SRC,
    BOARD_WATCH_SRC,
)

ARCHIVED_PATHS: Final[tuple[str, ...]] = (*SOURCE_TREES, REGISTRY_FILE)

#: Files whose absence means the extraction cannot run a tick: both agents'
#: entry points, one module of each imported package, and the registry.
REQUIRED_FILES: Final[tuple[str, ...]] = (
    f"{FLEET_SRC}/fleet/cli/agent.py",
    f"{FLEET_SRC}/fleet/cli/node_agent.py",
    f"{PLATFORM_CORE_SRC}/platform_core/__init__.py",
    f"{MONOREPO_GUARDS_SRC}/monorepo_guards/__init__.py",
    f"{BOARD_WATCH_SRC}/board_watch/__init__.py",
    REGISTRY_FILE,
)

#: Where the registry's relative record paths resolve in the checkout: the
#: ledger, feed and leases are this machine's running state, never rolled.
RECORDS_DIR: Final = "tools/fleet"

#: The scratch-root subdirectory the extractions live in.
EXTRACTIONS_DIR: Final = "fleet-rolled"

#: Detail prefixes, one per way a tick can be refused.
REF_UNRESOLVED_CODE: Final = "FLEET_ROLL_REF_UNRESOLVED"
ARCHIVE_FAILED_CODE: Final = "FLEET_ROLL_ARCHIVE_FAILED"
EXTRACT_FAILED_CODE: Final = "FLEET_ROLL_EXTRACT_FAILED"
INCOMPLETE_CODE: Final = "FLEET_ROLL_INCOMPLETE"


class RolledTree(TypedDict):
    """The rolled fleet code one tick runs.

    Attributes:
        commit: The full commit id the extraction was taken from.
        directory: This launch's own extraction, removed after the agent
            exits.
        python_path: The ``PYTHONPATH`` value putting the four extracted
            source trees ahead of the checkout's editable installs.
        config: The extracted registry, passed as ``--config``.
    """

    commit: str
    directory: str
    python_path: str
    config: str


def launch_directory(commit: str) -> pathlib.Path:
    """This launch's own extraction directory for a commit.

    Args:
        commit: The rolled commit.

    Returns:
        ``<scratch root>/fleet-rolled/<commit>/pid-<this process id>``.
    """
    return _test_hooks.temp_root() / EXTRACTIONS_DIR / commit / f"pid-{os.getpid()}"


def discard_rolled_tree(tree: RolledTree) -> None:
    """Remove a launch's extraction once its agent has exited.

    Args:
        tree: The extraction :func:`extract_rolled_tree` returned.

    Raises:
        OSError: When the directory cannot be removed; it propagates, so a
            tick that could not clean up says so in its log.
    """
    shutil.rmtree(tree["directory"])


def extract_rolled_tree(api_root: pathlib.Path) -> RolledTree | str:
    """Extract the rolled fleet code, or say why a tick cannot run.

    Args:
        api_root: The API checkout whose object store holds the ref.

    Returns:
        The extraction, or a ``CODE: message`` refusal when any step fails;
        nothing is run from the working tree instead.
    """
    resolved = commit_tree.resolve_commit(api_root, ROLLED_REF, REF_UNRESOLVED_CODE)
    if isinstance(resolved, str):
        return resolved
    commit = resolved["commit"]
    directory = launch_directory(commit)
    destination = commit_tree.extract_paths(
        api_root,
        commit,
        ARCHIVED_PATHS,
        destination=directory,
        archive_code=ARCHIVE_FAILED_CODE,
        extract_code=EXTRACT_FAILED_CODE,
    )
    # A REFUSED LAUNCH REMOVES ITS OWN DIRECTORY, which extract_paths made
    # before either step could fail. Until 2026-09-29 only a launch whose
    # agent ran removed it, and the refused ones stayed: 23 pid directories
    # under one commit on the hub after the 2026-09-28 extraction timeouts
    # (MCPs board task 94ac1c4f). Nothing reads one again, since each launch
    # extracts under its own process id.
    if isinstance(destination, str):
        shutil.rmtree(directory)
        return destination
    incomplete = commit_tree.incompleteness(
        destination, commit, REQUIRED_FILES, code=INCOMPLETE_CODE
    )
    if incomplete is not None:
        shutil.rmtree(directory)
        return incomplete
    return RolledTree(
        commit=commit,
        directory=str(destination),
        python_path=os.pathsep.join(
            str(destination / pathlib.PurePosixPath(tree)) for tree in SOURCE_TREES
        ),
        config=str(destination / pathlib.PurePosixPath(REGISTRY_FILE)),
    )


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
    "EXTRACTIONS_DIR",
    "EXTRACT_FAILED_CODE",
    "FLEET_SRC",
    "INCOMPLETE_CODE",
    "MONOREPO_GUARDS_SRC",
    "PLATFORM_CORE_SRC",
    "RECORDS_DIR",
    "REF_UNRESOLVED_CODE",
    "REGISTRY_FILE",
    "REQUIRED_FILES",
    "ROLLED_REF",
    "SOURCE_TREES",
    "RolledTree",
    "discard_rolled_tree",
    "extract_rolled_tree",
    "launch_directory",
    "rolled_state",
]
