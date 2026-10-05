"""The claiming half of a serve, which a queue that did not answer costs a poll.

Split out of :mod:`fleet.cli.node_serve`, which runs it between fire
boundaries, when the queue's silence became something it must ride
(MCPs board task 8993c306).

WHY. A serve lists the queued jobs every poll and runs the collect and fill
passes, and every one of those asks the queue. Deploys recreate mcp-fleet,
which then refuses every call for a minute or two: on 2026-10-05 a refused
listing at 07:24:56Z ended all 7 serves, and a refusal at 11:54:58Z, met
again by a settle, ended diphtheria's, so row bde22e57 closed 105 s after
its check. A call nothing answered now raises ``QUEUE_UNANSWERED``
(:mod:`fleet.core.queue_transport`), and here it marks the serve as
RECOVERING instead: each poll after that runs the collect pass and then
the fill pass, the two a fresh start runs, until the queue answers both,
so what a refusal used to cost, the next start, now comes within a poll of
the queue answering. That includes a claim whose start report went
unanswered (:meth:`fleet.cli.node_launch.Launcher.take_unreported`), which
the collect pass adopts (:func:`fleet.cli.node_collect.reconcile_claim`).
Every other failure still raises, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.errors import AppError
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.core import queue
from fleet.core.queue_transport import unanswered

_log = get_logger(__name__)


def queued_here(credentials: McpCredentials, *, alias: str) -> frozenset[str] | None:
    """The queued jobs a serve watches for: those naming this node or no node.

    Tags are not read here: a job whose tags this runner lacks starts one
    fill pass, whose claim the queue answers with nothing, and its id is
    then seen, so it starts no other.

    Args:
        credentials: The queue's endpoint and headers.
        alias: This node's workspace name.

    Returns:
        Their job ids, or None when the queue did not answer the listing
        (``QUEUE_UNANSWERED``, logged).

    Raises:
        AppError: Any other failure of the queue listing.
    """
    try:
        listed = queue.queued_for(credentials, project=None)
    except AppError as refusal:
        if not unanswered(refusal):
            raise
        _log.info("%s: the queue did not answer its listing: %s", alias, refusal.message)
        return None
    return frozenset(job["job_id"] for job in listed if job["requested_node"] in (None, alias))


def answered(step: Callable[[], None], *, alias: str, what: str) -> bool:
    """Run one of a serve's passes, saying whether the queue answered it.

    Args:
        step: The pass.
        alias: This node's workspace name, for the log.
        what: The pass's name, for the log.

    Returns:
        False when the queue did not answer it (``QUEUE_UNANSWERED``),
        which is logged.

    Raises:
        AppError: Any other failure of the pass. Not caught.
    """
    try:
        step()
    except AppError as refusal:
        if not unanswered(refusal):
            raise
        _log.info(
            "%s: the queue did not answer its %s pass; it runs again at the next poll: %s",
            alias,
            what,
            refusal.message,
        )
        return False
    return True


class Claiming:
    """What a serve claims between its fire boundaries, and how it recovers.

    Attributes:
        fills: How many fill passes it has run, one the queue left
            unanswered included.
        recovering: Whether a listing or a pass went unanswered and the
            queue has not answered both passes since.
    """

    fills: int
    recovering: bool

    def __init__(
        self,
        *,
        alias: str,
        collect: Callable[[], None],
        fill: Callable[[], None],
        queued: Callable[[], frozenset[str] | None],
        start_unreported: Callable[[], bool],
        closed: Callable[[], int],
    ) -> None:
        """Bind the serve's passes and questions; nothing is run.

        Args:
            alias: This node's workspace name, for the log.
            collect: The collect pass.
            fill: The fill pass.
            queued: The queued job ids, or None when the queue did not answer.
            start_unreported: Whether a launch's start report went
                unanswered since the last ask.
            closed: How many runs the watch has settled.
        """
        self._alias = alias
        self._collect = collect
        self._fill = fill
        self._queued = queued
        self._start_unreported = start_unreported
        self._closed = closed
        self._seen: frozenset[str] = frozenset()
        self._settled = closed()
        self.fills = 0
        self.recovering = False

    def passes(self, *, fill: bool) -> None:
        """Run the collect pass, then the fill pass if asked and the first was answered.

        Args:
            fill: Whether to run the fill pass after the collect pass.

        Raises:
            AppError: Any failure of a pass but ``QUEUE_UNANSWERED``.
        """
        self.recovering = not answered(self._collect, alias=self._alias, what="collect")
        if self.recovering or not fill:
            return
        self.recovering = not answered(self._fill, alias=self._alias, what="fill")
        self.fills += 1
        self._settled = self._closed()

    def poll(self) -> None:
        """One poll between boundaries: recover, or fill on an arrival or a settle.

        A job counts as arrived when a listing shows it and the last one did
        not, the first listing's every job included, so a job that waited
        through the opening fill pass, untaken for its tags or the node's
        room, costs one more fill pass at the first poll and none after.

        Raises:
            AppError: Any failure of the listing or a pass but ``QUEUE_UNANSWERED``.
        """
        if self._start_unreported():
            self.recovering = True
        if self.recovering:
            self.passes(fill=True)
            return
        queued = self._queued()
        if queued is None:
            self.recovering = True
            return
        arrived = queued - self._seen
        self._seen = queued
        if arrived or self._closed() != self._settled:
            self._settled = self._closed()
            self.recovering = not answered(self._fill, alias=self._alias, what="fill")
            self.fills += 1


__all__ = ["Claiming", "answered", "queued_here"]
