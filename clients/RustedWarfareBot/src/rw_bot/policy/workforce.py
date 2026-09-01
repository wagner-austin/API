"""Who is free to be given a job, and who is already carrying one out.

The builder is the only channel through which credits become structures, so
"which workers are available" is the question the whole economy is gated on. It
is answered from movement and from what is rising on the map rather than from a
record of what was ordered, because an order the engine dropped and an order
being carried out look identical to a sender.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.policy.build_order import BUILDER_TYPE
from rw_bot.policy.observation import has_moved
from rw_bot.policy.siting import RING_SLOT_RADIUS, is_refused
from rw_bot.wire.state import Entity, Sample

#: World-unit radius within which a rising structure counts as a worker's job.
#:
#: A structure goes up on the point it was ordered at, so this only has to
#: survive float noise and the engine's own placement snapping. It is the ring
#: slot radius for the same reason that constant exists: wide enough to match
#: the building actually placed, narrow enough that a neighbouring one is not
#: mistaken for it.
JOB_RADIUS = RING_SLOT_RADIUS


#: The most builders worth holding.
#:
#: Set by measurement, not by argument. Uncapped, the bot bought 33 in a
#: 1500-sample match -- 16,500 credits of labour placing 13 extractors, while the
#: army it was supposed to be funding stayed at a dozen units
#: ([[policy-production]]).
DEFAULT_MAX_WORKERS = 4

#: Samples a stationary builder may sit on an unstarted expansion before the
#: order is presumed lost and sent again.
#:
#: The same reasoning as the plan's stall window, used for the opposite purpose:
#: there it ends the plan, here it retries. A builder that has neither moved nor
#: started building for this many samples is not on its way anywhere, and the
#: cost of being wrong is one duplicate order the engine collapses onto the same
#: waypoint ([[policy-loop]]).
EXPAND_RETRY_SAMPLES = 45


class Workforce:
    """Which builders are free, and what each was last sent to do.

    **One builder was an assumption baked into every layer.** The plan found
    "the" builder, the economy found "the" builder, and both meant
    ``the first one in the roster`` -- so a second would have stood idle for the
    whole match, and the guards that stopped the two rules fighting over the
    first were duplicated in both of them and still let one re-task the other's
    worker ([[policy-loop]]).

    Availability has one owner now, and it is this. A worker is busy when the
    world says so and free otherwise; the rules below are handed the free ones
    and never ask who exists.

    Busy means one of two observable things, neither a deadline this class
    invents:

    * it moved since the previous observation, so it is walking to a site;
    * something of the type it was sent to build is going up where it was sent.

    A worker that is neither, and has an outstanding job, is given
    :data:`EXPAND_RETRY_SAMPLES` observations before the order is presumed lost
    and it is freed to be given another -- the same window, and the same
    reasoning, as the plan's stall clock.
    """

    def __init__(self, retry_samples: int) -> None:
        """Open a workforce.

        Args:
            retry_samples: Observations a stationary worker may sit on an
                unstarted job before the order is presumed lost.
        """
        self._retry = retry_samples
        self._was: dict[int, tuple[float, float]] = {}
        self._moved: dict[int, bool] = {}
        self._job: dict[int, tuple[str, tuple[float, float]]] = {}
        self._quiet: dict[int, int] = {}
        self._refused: list[tuple[float, float]] = []

    def free(self, sample: Sample) -> tuple[Entity, ...]:
        """Return the workers not visibly carrying out an order.

        Args:
            sample: One observation of the world.

        Returns:
            The free workers, in roster order.
        """
        workers = [
            entity
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == BUILDER_TYPE
        ]
        self._forget(workers)
        return tuple(worker for worker in workers if self._is_free(sample, worker))

    def size(self, sample: Sample) -> int:
        """Return how many workers are owned at all, free or not.

        Args:
            sample: One observation of the world.

        Returns:
            The count.
        """
        return sum(
            1
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == BUILDER_TYPE
        )

    def working(self, unit_id: int) -> bool:
        """Report whether this worker was moving at the last observation.

        The plan's stall clock asks about the unit it actually ordered rather
        than about "the" builder, which is what lets two of them build at once
        without either one's travel resetting the other's clock.

        Args:
            unit_id: The worker to ask about.

        Returns:
            True when it moved between the last two observations.
        """
        return self._moved.get(unit_id, False)

    def assign(self, unit_id: int, type_name: str, site: tuple[float, float]) -> None:
        """Record what a worker has just been ordered to build.

        Args:
            unit_id: The worker ordered.
            type_name: What it was told to build.
            site: Where it was told to build it.
        """
        self._job[unit_id] = (type_name, site)
        self._quiet[unit_id] = 0

    def claims(self) -> tuple[tuple[float, float], ...]:
        """Return the sites workers are currently under orders to build on.

        **What this answers is "who already has that pool".** Occupancy is judged
        by what is standing on a pool, so one a builder is merely walking toward
        reads as free -- and with several workers free at once, each was offered
        the same nearest pool in turn. One instrumented run granted 23 extractor
        orders, lost nothing at all, and finished with four extractors
        ([[policy-holding-ground]]).

        A job outlives the structure it built by up to the retry window, because
        nothing clears it until the worker is judged free again. That is
        harmless here: by then something is standing on the pool and it is
        occupied on the ordinary test anyway.

        Returns:
            One site per worker with an order outstanding, in no order.
        """
        return tuple(site for _, site in self._job.values())

    def record_refusal(self, site: tuple[float, float]) -> None:
        """Record a site the engine refused, reported by the agent's watch.

        The second writer to the ledger, and the fast one: the agent watches
        every dispatched build order and reports the engine dropping its
        waypoint the sample after it happens, where the presumed-lost clock
        in :meth:`free` needs the whole retry window of silence. Both feed
        the same ledger because both observe the same fact by different
        means; the slow clock stays as the fallback for whatever the watch
        cannot see.

        Args:
            site: The refused placement, as ordered.
        """
        if not is_refused(site, self._refused):
            self._refused.append(site)

    def refused(self) -> tuple[tuple[float, float], ...]:
        """Return the sites the engine has silently refused this match.

        Written by the presumed-lost judgement in :meth:`free` -- a worker
        stationary on an unstarted job for the whole retry window -- which is
        the only observable trace a silent refusal leaves. Read by every ring
        chooser, so a refused spot is the NEXT slot rather than the same
        doomed order re-sent for the rest of the match
        ([[policy-loop]]; wiki log 2026-08-31, verdict-withheld).

        Returns:
            The refused sites, in refusal order.
        """
        return tuple(self._refused)

    def underway(self, type_name: str) -> int:
        """Count workers currently under orders to build one type.

        **What this answers is "how many of these are already coming".** A
        headcount read off standing structures alone is blind to the walk:
        the nuke channel counted zero launchers while its first builder was
        in transit and assigned a fresh builder to the same job every tick
        -- eight granted claims, 360,000 credits, one structure
        (`runs/sweeps/vh-nuke`, log 2026-08-05). The same disease
        :meth:`claims` exists for, asked by type instead of by site.

        Args:
            type_name: The structure type to count.

        Returns:
            Workers with an outstanding order for that type.
        """
        return sum(1 for job_type, _ in self._job.values() if job_type == type_name)

    def _is_free(self, sample: Sample, worker: Entity) -> bool:
        """Judge one worker against what it was last sent to do.

        Args:
            sample: One observation of the world.
            worker: The worker to judge.

        Returns:
            True when nothing observable says it is working.
        """
        unit_id = worker["unit_id"]
        before = self._was.get(unit_id)
        now = (worker["x"], worker["y"])
        self._was[unit_id] = now
        self._moved[unit_id] = has_moved(before, now)
        if self._moved[unit_id]:
            self._quiet[unit_id] = 0
            return False
        job = self._job.get(unit_id)
        if job is None:
            return True
        # A job whose site the ledger already carries is over: the agent's
        # watch reported the engine dropping the order, so there is nothing
        # at the site to rise and no reason to spend the retry window below
        # confirming silence. The entry is already written -- this only frees
        # the worker, the same tick the report lands.
        if is_refused(job[1], self._refused):
            del self._job[unit_id]
            self._quiet[unit_id] = 0
            return True
        # **A finished job frees its worker on the tick it finishes.** The
        # freeing used to happen only through the quiet window below, which
        # requires the site to show nothing rising for the whole retry
        # window -- and the defence cover ring packs structures densely
        # enough that a NEIGHBOUR'S rising turret inside the job radius kept
        # re-marking finished workers busy, chaining into a full-workforce
        # freeze: four of ten rich Hard matches ended `army 0 -> 0` with the
        # plan waiting all match on "every unit that can make landFactory is
        # busy" (log: 2026-07-31). The structure the worker was sent to
        # build, standing complete at the site, is the one unambiguous
        # completion signal and it outranks every inference.
        if _complete_at(sample, job[0], job[1]):
            del self._job[unit_id]
            self._quiet[unit_id] = 0
            return True
        if _rising_at(sample, job[0], job[1]):
            self._quiet[unit_id] = 0
            return False
        quiet = self._quiet.get(unit_id, 0) + 1
        self._quiet[unit_id] = quiet
        if quiet < self._retry:
            return False
        # Stationary, nothing going up, and long enough that the order is not
        # merely slow. The engine refuses some placements silently, so the
        # worker is freed to be given a different one -- and the SITE is
        # recorded as refused, because freeing the worker alone re-ran the
        # same doomed order forever: the refused spot stays structure-free,
        # every chooser offered it again, and the Hard panel's scorecards
        # read `expansions 64 (0 factories)` (wiki log 2026-08-31,
        # verdict-withheld). This is the one place a silent refusal is
        # observable, so it is the one place the ledger is written.
        self._refused.append(job[1])
        del self._job[unit_id]
        self._quiet[unit_id] = 0
        return True

    def _forget(self, workers: Sequence[Entity]) -> None:
        """Drop bookkeeping for workers that are no longer in the roster.

        Args:
            workers: The workers currently owned.
        """
        alive = {worker["unit_id"] for worker in workers}
        for unit_id in set(self._was) - alive:
            self._was.pop(unit_id, None)
            self._moved.pop(unit_id, None)
            self._job.pop(unit_id, None)
            self._quiet.pop(unit_id, None)


def _complete_at(sample: Sample, type_name: str, site: tuple[float, float]) -> bool:
    """Report whether a finished structure of this type stands at this point.

    The completion signal :meth:`Workforce._is_free` trusts ahead of the
    quiet-window inference. Ownership checked for the same reason
    :func:`_rising_at` checks it.

    Args:
        sample: One observation of the world.
        type_name: The type that was ordered.
        site: Where it was ordered.

    Returns:
        True when an owned finished entity of that type stands there.
    """
    limit = JOB_RADIUS**2
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"] or entity["type_name"] != type_name:
            continue
        if (entity["x"] - site[0]) ** 2 + (entity["y"] - site[1]) ** 2 <= limit:
            return True
    return False


def _rising_at(sample: Sample, type_name: str, site: tuple[float, float]) -> bool:
    """Report whether something of this type is going up at this point.

    Ownership is checked, or an opponent's half-built structure nearby would
    read as our worker's job and hold it busy forever.

    Args:
        sample: One observation of the world.
        type_name: The type that was ordered.
        site: Where it was ordered.

    Returns:
        True when an owned unfinished entity of that type stands there.
    """
    limit = JOB_RADIUS**2
    for entity in sample["entities"]:
        if not entity["mine"] or entity["complete"] or entity["type_name"] != type_name:
            continue
        if (entity["x"] - site[0]) ** 2 + (entity["y"] - site[1]) ** 2 <= limit:
            return True
    return False


__all__ = ["EXPAND_RETRY_SAMPLES", "JOB_RADIUS", "Workforce"]
