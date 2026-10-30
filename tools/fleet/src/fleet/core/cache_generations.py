"""Removing the cache generations no run on a node can still read.

The generations are :mod:`fleet.core.tool_caches`'s (MCPs board task
346abc45). After each retire the runner sends the node
:func:`removal_script`, which removes every generation under the node's cache
root, and under execdocker's on a node that runs docker projects' builds as
that user (:func:`fleet.core.retire.runs_isolated`), that is not current,
that no run directory still on the node names in its marker, and that has
not been current for :data:`~fleet.core.tool_caches.REMOVAL_GRACE_SECONDS`,
by :data:`~fleet.core.tool_caches.SH_REMOVE_UNUSED` over
:data:`~fleet.core.tool_caches.SH_STAGE_GENERATIONS_IN_USE`. ci-clean runs
the same rule over the CI runner hosts' caches, against the runner installs
running a job.
"""

from __future__ import annotations

import shlex
import threading
from typing import Final

from platform_core.logging import get_logger

from fleet.contracts.node import NodeConfig
from fleet.core import dialect, names, remote, retire
from fleet.core.linux_isolated_build import EXEC_CACHE, EXEC_HOME_LINE, EXEC_STAGE
from fleet.core.tool_caches import (
    SH_REMOVE_UNUSED,
    SH_STAGE_GENERATIONS_IN_USE,
    tool_caches_directory,
)

_log = get_logger(__name__)

#: The removal script's name under the stage root. One name for every run,
#: so no two removals of one process run at once (:data:`REMOVING`).
REMOVAL_STEM: Final = "fleet-cache-generations"

#: Held across each removal. A serving node runner settles several runs at
#: once (:mod:`fleet.cli.run_locks`, MCPs board task 8993c306), and two
#: removals of one node would write the one script path above together.
REMOVING: Final = threading.Lock()


def removal_script(stage_root: str, *, isolated: bool) -> str:
    """The removal as a POSIX sh script, over the node's caches and execdocker's.

    Args:
        stage_root: The node's declared stage root.
        isolated: Whether the node runs docker projects' builds as
            execdocker (:func:`fleet.core.retire.runs_isolated`), whose
            caches are then judged too, as root, against that user's copies.

    Returns:
        The script's text, printing one line per cache root it judged.
    """
    tools = tool_caches_directory(names.cache_root(stage_root))
    lines = (
        *SH_REMOVE_UNUSED,
        *SH_STAGE_GENERATIONS_IN_USE,
        "as=",
        f"remove_unused {shlex.quote(tools)} {shlex.quote(stage_root)}",
    )
    isolated_lines = (
        EXEC_HOME_LINE,
        "as='sudo -n'",
        f'remove_unused "{tool_caches_directory(EXEC_CACHE)}" "{EXEC_STAGE}"',
    )
    return dialect.PROLOGUE + "".join(
        f"{line}\n" for line in lines + (isolated_lines if isolated else ())
    )


def remove_unused_on_node(node: NodeConfig) -> str:
    """Remove the cache generations no run on one node can still read.

    A NODE THAT DOES NOT ANSWER IS LEFT TO THE NEXT REMOVAL (MCPs board task
    c1d48330). Every caller removes after its row is closed, and a removal
    judges every generation, not one run's, so one the node missed leaves
    nothing the next one does not remove. Raising there ended lavender-wsl's
    serve at 04:05Z on 2026-10-07, after the row it settled had closed.

    Args:
        node: The node.

    Returns:
        The node's report, or the line saying it did not answer; either is
        also logged.

    Raises:
        AppError: With ``DISPATCH_FAILED`` carrying the node's own error
            when it answered and a generation could not be read or removed.
    """
    stage_root = node["stage_root"]
    with REMOVING:
        read = remote.read_script(
            node["host"],
            dialect.script_path(stage_root, REMOVAL_STEM),
            removal_script(stage_root, isolated=retire.runs_isolated(node)),
        )
    output = read["output"]
    if output is None:
        missed = (
            f"cache-generations: {node['host']} did not answer; the next removal judges its "
            f"generations: {read['unreachable']}"
        )
        _log.info("%s", missed)
        return missed
    report = output.strip()
    _log.info("%s on %s", report, node["host"])
    return report


__all__ = [
    "REMOVAL_STEM",
    "REMOVING",
    "removal_script",
    "remove_unused_on_node",
]
