"""A serve claims only while ``refs/fleet/rolled`` names the commit it runs.

THE GAP (MCPs board task 1fd01f09). A node runner serves across its
scheduled starts (:mod:`fleet.cli.node_serve`) and read the ref only at a
fire boundary's handover, so for up to a boundary after a roll it went on
claiming and launching with the registry of the commit it was extracted
from. Measured 2026-11-04: ``make fleet-roll`` moved the ref to
84bc5ff0b6a9, which gave MCPs/execution-twin's workspace build
``ROOTS=execution:mcp-proxy``, and returned before 21:20:42Z; job 0fa14fab,
queued then, was claimed by pendragon's serve at 21:20:44Z, 16 s before the
21:21:00Z boundary, ran ``CI_BUILD_PLAN: roots=execution``, the install
steps of the registry before the roll, and failed on mcp-proxy's relay.js.
The same check queued at 21:22:03Z ran on the rolled registry and passed
(job 4a18e72b). Runs of 2026-10-08 (90c135ac) and 2026-10-12 (e309c14a)
met the same window.

THE RULE. A serve compares the ref with the commit its own registry was
extracted from (:func:`fleet.core.rolled.extraction_commit`), never with a
reading of the ref, and reads it twice around every claim
(:func:`fleet.cli.node_agent.claim_pass`): before asking the queue, when a
move means the serve asks for nothing, and after a job is handed to it, when a move
means the job is given back unlaunched (:func:`give_back`). The read after
the claim comes after the job was queued, and a job queued after a roll
returned was queued after the roll moved the ref, so no such job is
launched on the older registry. Either read that finds the ref moved
latches the guard: the serve claims nothing more and hands over at the next
boundary (:func:`fleet.cli.node_serve.handover_policy`), where the next
serve, extracted from the rolled commit, claims.

The read before the claim keeps the give-back for the few milliseconds a
claim call spans: without it every stale serve would claim a job after each
roll and hand it back, and a job given back twice is failed by the queue
(MCPs ``reportDispatchRunLost``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli.node_claim import decided
from fleet.contracts.dispatch import DispatchJob
from fleet.contracts.runner_tick import RunnerTick
from fleet.core import tick_report
from fleet.core.queue_lost import report_lost
from fleet.core.rolled import ROLLED_REF

_log = get_logger(__name__)

#: The code a job given back for a move opens its trail row with.
MOVED_CODE: Final = "FLEET_ROLL_MOVED"


class RollGuard:
    """Whether ``refs/fleet/rolled`` still names the commit this serve runs.

    Attributes:
        runs: The commit the serve's registry and code were extracted from.
    """

    runs: str

    def __init__(self, *, runs: str, rolled: Callable[[], str]) -> None:
        """Bind the serve's commit and the ref's reader; nothing is read.

        Args:
            runs: The commit the serve runs.
            rolled: Reads what the ref names now, or the refusal saying why
                it names none (:func:`fleet.core.rolled.rolled_state`).
        """
        self.runs = runs
        self._rolled = rolled
        self._moved: str | None = None

    def latched(self) -> str | None:
        """The move an earlier read found, read nothing now.

        Returns:
            Why the serve claims nothing more, or None while no read has
            found the ref moved.
        """
        return self._moved

    def moved(self) -> str | None:
        """Read the ref, unless an earlier read already found it moved.

        A refusal compares unequal to the commit, so a ref that stops
        resolving stops the claiming too, and the next start reports the
        refusal as its own.

        Returns:
            Why the serve claims nothing more, or None while the ref names
            the commit it runs.
        """
        if self._moved is not None:
            return self._moved
        current = self._rolled()
        if current == self.runs:
            return None
        self._moved = f"{ROLLED_REF} moved from {self.runs} to {current}"
        return self._moved


def skip_claim(
    credentials: McpCredentials,
    identity: JSONObject,
    tick: RunnerTick,
    *,
    alias: str,
    moved: str,
) -> None:
    """Record a ready node's tick that asked the queue for nothing after a roll.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments.
        tick: The tick the gate composed for the ready node.
        alias: This node's workspace name.
        moved: What :meth:`RollGuard.moved` said.

    Raises:
        AppError: From the queue call.
    """
    verdict = f"asked the queue for nothing: {moved}; the next serve claims on the rolled commit"
    _log.info("%s %s", alias, verdict)
    tick_report.record_tick(
        credentials, decided(tick, claiming=False, verdict=verdict), identity=identity
    )


def give_back(
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    alias: str,
    moved: str,
) -> DispatchJob:
    """Give back a job claimed after a roll, before anything of it is staged.

    Args:
        credentials: The queue's endpoint and headers.
        job: The job the queue handed this serve.
        identity: This runner's identity arguments.
        alias: This node's workspace name.
        moved: What :meth:`RollGuard.moved` said.

    Returns:
        The job as the queue left it: ``queued`` the first time it is given
        back, ``failed`` when it had been given back before.

    Raises:
        AppError: From the queue call.
    """
    answered = report_lost(
        credentials,
        job_id=job["job_id"],
        detail=(
            f"{MOVED_CODE}: {alias}'s serve was handed {job['job_id']} after {moved}; "
            "nothing was staged, so the job is given back for a serve on the rolled commit"
        ),
        identity=identity,
    )
    _log.info(
        "%s gave back %s, which the queue left %s: %s",
        alias,
        job["job_id"],
        answered["status"].value,
        moved,
    )
    return answered


__all__ = ["MOVED_CODE", "RollGuard", "give_back", "skip_claim"]
