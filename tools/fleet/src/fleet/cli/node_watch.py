"""The watch at the end of a node runner's tick: close a run when it ends.

WHY A TICK WATCHES (MCPs board task c1d48330). Each node's runner is a
scheduled task repeating every 3 minutes under IgnoreNew, and a tick ran its
collect pass and its fill pass once, so a run that ended was closed, and its
verdict posted, only on the next tick: claimed-to-closed was a whole number
of ticks and every session waiting on a fleet verdict waited up to 3 minutes
after its check had ended. Measured on 2026-10-04 (board task 74b13c20):
serendipity's osm-mcp row ended 19:27:05Z and closed 19:30:21Z, its search
row ended 19:52:18Z and closed 19:54:17Z, lavender-wsl's store-hit rows read
exactly 181 s claimed-to-closed whatever their work, and a tick spent about
170 of its 180 s doing nothing.

WHAT IT DOES. After the tick's passes, while the runner holds a run this
machine's ledger still calls running and the workspace's
``node_watch_seconds`` window (counted from the tick's start) leaves room for
another poll, it waits :data:`POLL_SECONDS` and reads each held run's result
off the node (:func:`fleet.core.collect.poll_result`, one ssh read). The
moment one has ended it runs the passes again, which closes that run, posts
its verdict and fills the room it freed, and watches whatever they leave.

WHAT IT COSTS, AND WHAT IT DOES NOT. A runner holding no running job makes
no call at all beyond its passes, and a poll that finds nothing still
running writes nothing anywhere: absence of a result is the ordinary answer
(:mod:`fleet.core.collect`). The window is bounded by the workspace's
decoder (:data:`fleet.contracts.workspace.NODE_WATCH_CEILING_SECONDS`) so
the last pass a watch starts ends before the next tick is due.
"""

from __future__ import annotations

import datetime
from typing import Final, Protocol

from platform_core.logging import get_logger

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks, collect

_log = get_logger(__name__)

#: How long the watch waits between reads of a held run's result.
POLL_SECONDS: Final[int] = 10


class TickPasses(Protocol):
    """A runner's collect pass followed by its fill pass."""

    def __call__(self) -> frozenset[str]:
        """Collect what finished, then fill the node's room.

        Returns:
            The run ids of the running jobs the collect pass found held and
            of the jobs the fill pass launched.
        """


def live_among(loaded: _config.LoadedWorkspace, run_ids: frozenset[str]) -> frozenset[str]:
    """The run ids this machine's ledger still calls running.

    Args:
        loaded: The workspace and its resolved record paths.
        run_ids: The runs to look for.

    Returns:
        Those of them that are still live, read from the local ledger, so
        asking costs no ssh and no queue call.
    """
    return frozenset(
        row["run_id"]
        for row in collect_cli.live_rows(loaded, run_id=None)
        if row["run_id"] in run_ids
    )


def first_ended(node: NodeConfig, run_ids: frozenset[str]) -> str | None:
    """Read each run's result off the node until one has ended.

    Args:
        node: The node the runs are on.
        run_ids: The runs to read, in sorted order.

    Returns:
        The first run whose result the node has written, or None when every
        one is still going.

    Raises:
        AppError: As :func:`fleet.core.collect.poll_result` describes.
    """
    for run_id in sorted(run_ids):
        if collect.poll_result(node, run_id=run_id) is not None:
            return run_id
    return None


def run_tick(
    loaded: _config.LoadedWorkspace,
    *,
    alias: str,
    node: NodeConfig,
    passes: TickPasses,
) -> int:
    """Run a tick's passes, then watch what they left running.

    Args:
        loaded: The workspace and its resolved record paths.
        alias: This node's workspace name.
        node: Its declaration.
        passes: The runner's collect pass followed by its fill pass.

    Returns:
        How many times the watch ran the passes again.

    Raises:
        AppError: From the passes or from a poll, neither caught: the next
            tick meets the same runs.
    """
    started = _test_hooks.now()
    watched = live_among(loaded, passes())
    if not watched:
        _log.info("%s holds no running job; no watch this tick", alias)
        return 0
    deadline = started + loaded.workspace["node_watch_seconds"]
    polls = 0
    reruns = 0
    while watched and _test_hooks.now() + POLL_SECONDS <= deadline:
        _test_hooks.sleep(POLL_SECONDS)
        polls += 1
        watched = live_among(loaded, watched)
        ended = first_ended(node, watched)
        if ended is None:
            continue
        _log.info("%s: %s has ended; collecting it now", alias, ended)
        watched = live_among(loaded, passes())
        reruns += 1
    until = datetime.datetime.fromtimestamp(deadline, tz=datetime.UTC).isoformat()
    _log.info(
        "%s watch until %s: %d poll(s), %d pass(es) rerun, %d run(s) still watched",
        alias,
        until,
        polls,
        reruns,
        len(watched),
    )
    return reruns


__all__ = ["POLL_SECONDS", "TickPasses", "first_ended", "live_among", "run_tick"]
