"""Freezing the code a batch's matches import, and checking a frozen tree.

Split from :mod:`rw_bot.harness.runner` when that file passed the six-hundred
line ceiling, and the boundary is a real one: this module is about what a
batch CARRIES -- which sources are captured, whether the capture is complete,
and whether the prebuilt agent jar being captured is current -- while the
runner is about PLAYING the batch. A new tree source or freshness rule
touches this file; a new way to play a job touches the runner.

Every filesystem operation is reached through
:mod:`rw_bot.harness._test_hooks`, exactly as in the runner, so a test drives
the real control flow against fakes rather than a rehearsal of it.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import (
    AGENT_BUILD_DIR,
    AGENT_MANIFEST,
    AGENT_SOURCE_DIR,
    AGENT_SOURCE_SUFFIX,
    FROZEN_AGENT_JAR,
)
from rw_bot.harness.launch import (
    CATALOGUE,
    FROZEN_CATALOGUE,
    FROZEN_TYPE_DUMP,
    TYPE_DUMP,
)
from rw_bot.harness.runner import SweepConfig
from rw_bot.harness.sweep import SweepError

#: What a batch freezes: everything a match imports or reads at launch.
#:
#: ``sweeps`` joined when the tree became a STAGED artifact. A compute node
#: reads its job file from the payload like everything else, and the file
#: naming which arms and seeds a batch played is as much the experiment as the
#: doctrines are -- the same argument that put those here.
#:
#: The two registry dumps joined for the same reason and were the last to.
#: They had been left out on the reasoning that a dump is an artifact of the
#: game build rather than code that changes between batches -- true, and not
#: the question the tree answers. The question is whether a match can READ it
#: where the match runs, and the first cluster member to reach the planner
#: died on ``FileNotFoundError: 'wiki/sources/m0-probe/printunits.log'``
#: having already patched, seeded and held the world at frame one. A compute
#: node has no repository for a repository-relative path to mean anything
#: against (job 55663569, 2026-08-30).
TREE_SOURCES = (
    "scripts",
    "doctrines",
    "sweeps",
    # The fitted heads ride with the code that scores them: a braced arm
    # on a cluster node reads models/razebrace.ndjson out of the payload,
    # and a tree without it would fail at model load, member by member
    # ([[impossible-step-three-design]]).
    "models",
    f"{AGENT_BUILD_DIR}/{FROZEN_AGENT_JAR}",
    CATALOGUE,
    TYPE_DUMP,
)

#: The frozen tree's directory name, under the batch's results directory.
TREE_DIR = ".tree"

#: Written into the tree last, so its presence certifies a complete freeze.
TREE_MARKER = ".complete"

#: A frozen tree handed to a run was incomplete.
_TREE_INCOMPLETE = "RW-SWEEP-006"

#: The prebuilt agent jar was absent, or older than an agent source, when a
#: tree was about to freeze it.
_JAR_STALE = "RW-SWEEP-007"

#: What a match actually READS out of a frozen tree, as it is laid out inside
#: one. Not the same strings as :data:`TREE_SOURCES`: a copy lands under its
#: own basename, so ``agent/build/rw-agent.jar`` arrives flat as
#: ``rw-agent.jar`` -- which is where
#: :data:`~rw_bot.harness.agent_build.FROZEN_AGENT_JAR` looks for it. Checking
#: the source spelling instead would refuse every tree ever frozen, and that
#: is exactly what the first version of this did.
FROZEN_ENTRIES = (
    "src/rw_bot/__init__.py",
    "doctrines",
    "scripts",
    "sweeps",
    "models",
    FROZEN_AGENT_JAR,
    FROZEN_CATALOGUE,
    FROZEN_TYPE_DUMP,
)


def check_agent_jar_fresh() -> None:
    """Raise unless the prebuilt agent jar is at least as new as its sources.

    Only ``make agent`` rebuilds ``agent/build/rw-agent.jar``. A single match
    compiles its own per-stamp jar, so play always runs current agent code
    while the frozen jar silently ages -- and a frozen tree carries the aged
    one onto machines that cannot rebuild it. That is exactly what happened
    on 2026-09-11: both pin-recertification batches (dettwinpin96, 96b)
    measured a jar built before either pin existed, because nothing between
    the pin commits and submission ever ran ``make agent``, and their floor
    numbers were void as pin measurements. The freeze is the last moment the
    workstation -- the only depot with a compiler -- can notice.

    Equal times pass: a checkout or copy can land a source and the jar in the
    same second, and only a source strictly newer than the jar is evidence
    the jar predates it.

    Raises:
        SweepError: ``RW-SWEEP-007`` when the jar has never been built, or
            naming every agent source newer than it. Both refusals end the
            same way: run ``make agent``, then freeze.
        OSError: When the agent source directory or manifest cannot be read.
    """
    jar = Path(AGENT_BUILD_DIR) / FROZEN_AGENT_JAR
    if not _test_hooks.path_exists(jar):
        raise SweepError(
            _JAR_STALE,
            f"{jar.as_posix()} does not exist: the tree freezes the PREBUILT agent jar, "
            "and a compute node cannot build one -- run `make agent`, then freeze",
        )
    jar_mtime = _test_hooks.file_mtime(jar)
    sources = [Path(AGENT_MANIFEST)] + [
        Path(AGENT_SOURCE_DIR) / name
        for name in _test_hooks.list_names(Path(AGENT_SOURCE_DIR))
        if name.endswith(AGENT_SOURCE_SUFFIX)
    ]
    stale = [source.as_posix() for source in sources if _test_hooks.file_mtime(source) > jar_mtime]
    if stale:
        raise SweepError(
            _JAR_STALE,
            f"{jar.as_posix()} is older than {', '.join(stale)}: the tree would freeze "
            "an agent that predates its own sources, and every match in the batch would "
            "attach it -- run `make agent`, then freeze",
        )


def prepare_tree(config: SweepConfig) -> None:
    """Freeze the code the batch's matches will import, once, at launch.

    A match imports the source tree at launch, so before this existed an edit
    landed mid-batch meant later matches ran different code from earlier ones
    -- an arm's twelve seeds were only one experiment if nobody touched the
    tree for the batch's whole runtime, and the working tree was frozen for
    hours at a stretch. The batch copies what its matches need into its own
    results directory instead: the tree is editable the moment the sweep
    starts, and the batch carries a record of exactly what it ran.

    **An existing snapshot is reused, never refreshed.** That is what makes a
    resumed batch a continuation rather than a new experiment: the matches
    played after the interruption import the same frozen code as the ones
    played before it, whatever has happened to the working tree in between.

    Args:
        config: How the batch is being played.

    Raises:
        SweepError: ``RW-SWEEP-007`` when the prebuilt agent jar is absent or
            older than an agent source (:func:`check_agent_jar_fresh`) --
            refused before the first copy, so no tree exists to mistake for a
            good one.
        OSError: When a copy fails.
    """
    tree = Path(config["tree"])
    # Judged by the marker, not by the directory: a directory can survive a
    # partial delete -- Windows file locks kept one alive with its doctrines
    # gone -- and reusing a gutted tree fails ten matches at once with the
    # freeze reporting success (log: 2026-07-31). The marker is written last,
    # so its presence certifies every copy before it finished.
    if _test_hooks.path_exists(tree / TREE_MARKER):
        _test_hooks.write_line(f"[sweep] reusing the frozen tree at {config['tree']}")
        return
    check_agent_jar_fresh()
    _test_hooks.make_dirs(tree / "src")
    _test_hooks.copy_entry(Path("src/rw_bot"), tree / "src")
    for entry in TREE_SOURCES:
        _test_hooks.copy_entry(Path(entry), tree)
    _test_hooks.write_text_lines(tree / TREE_MARKER, ("frozen",))
    _test_hooks.write_line(f"[sweep] tree frozen at {config['tree']}")


def check_frozen_tree(tree: Path) -> None:
    """Raise unless a tree carries everything a match reads out of it.

    The counterpart to :func:`prepare_tree` for a run that does NOT freeze its
    own. A cluster member is handed a tree that was frozen before submission
    and staged, so it must check what it was given rather than build it: the
    sources ``prepare_tree`` copies from are repository-relative, and a
    compute node has no repository, so a freeze there would report success
    having copied nothing.

    The marker alone is not enough. It certifies that every copy BEFORE it
    finished, and says nothing about a source that was absent when the copy
    ran -- the agent jar is the case, because ``make agent`` builds it and the
    repository does not carry it.

    Args:
        tree: The frozen tree.

    Raises:
        SweepError: ``RW-SWEEP-006`` naming every missing entry rather than
            the first. The failure it replaces is a member dying on a node
            with an import error, and one look should account for all of it.
    """
    absent = [
        name for name in (TREE_MARKER, *FROZEN_ENTRIES) if not _test_hooks.path_exists(tree / name)
    ]
    if absent:
        raise SweepError(
            _TREE_INCOMPLETE,
            f"the frozen tree at {tree} is missing {', '.join(absent)}: a match reads its "
            "planner, its doctrines and its agent jar from here, and none of them can be "
            "rebuilt on a compute node -- the Linux depot ships a JRE with no compiler",
        )


__all__ = [
    "FROZEN_ENTRIES",
    "TREE_DIR",
    "TREE_MARKER",
    "TREE_SOURCES",
    "check_agent_jar_fresh",
    "check_frozen_tree",
    "prepare_tree",
]
