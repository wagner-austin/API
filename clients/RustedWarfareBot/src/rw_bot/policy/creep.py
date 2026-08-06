"""Turrets walked to the enemy's door, one covered step at a time.

The documented human answer to the cheating difficulties is not to outgrow
them -- the compounding arithmetic forbids it -- but to put turrets where the
enemy lives while its attack groups still sit on their thousand-tick opening
delay ([[ai-opponent-strategy]]). A turret outranges and outlasts anything the
AI fields early, and each one built covers the builder walking to place the
next. The engine's own AI cannot answer ground it has already lost.

The creep is geometry over what every sample carries, like the rush it aims
the same way ([[policy-rush]]): the goal is the mirror of our anchor through
the pool centroid, the front is our forward-most turret, and the next site is
one turret-reach step from the front toward the goal -- mutual support by the
engine's own figure, not an invented spacing.

**The walk holds at a chosen fraction of the way, and heals what it holds.**
Walked all the way to the enemy start, the line died at their door faster
than it stood up -- 164 turrets, 82,000 credits, refuted (log 2026-07-31).
Held at a choke, the same wall is the community's whole answer to the
cheating difficulties: the enemy's army funnels into standing fire, and the
bridge probes measured the second-deepest economy dent of the Impossible
campaign from a wall that was not even AT the bridge. The doctrine names
the hold as a percent of the anchor-to-goal line, and every third structure
the walk lays is a repair bay -- an unhealed turret is a turret bought
twice ([[community-play-strategies]]).

Pure in the usual sense: samples in, orders out, the campaign sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile, profile_of
from rw_bot.policy.budget import Budget
from rw_bot.policy.defence import TURRET_TYPE
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.siting import clear_point_near, find_anchor
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import BuildOrder, build_order
from rw_bot.wire.state import Entity, Sample

#: What heals the held line, every third creep structure: a wall that
#: cannot heal loses to a wave it survived ([[community-play-strategies]]).
REPAIR_TYPE = "repairbay"

#: How many wall structures per healing one.
WALL_CYCLE = 3


class Creeper:
    """Advances the turret line, one order outstanding at a time.

    The same shape as the other stateful controllers: the siting decision is
    a pure read of the sample, and what lives here is the count for the
    ledger. One turret at a time, because the second site is chosen from
    where the first one STANDS -- ordering two at once would project both
    from a front that neither has advanced yet.

    Attributes:
        ordered: Creep build orders sent so far, for the report.
    """

    def __init__(self) -> None:
        """Open a creeper."""
        self.ordered = 0

    def advance(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        budget: Budget,
        free: Sequence[Entity],
        workforce: Workforce,
        hold: int = 100,
    ) -> tuple[BuildOrder, ...]:
        """Order the next creep turret when the line is ready to advance.

        Ready means: the goal is known, no creep turret is still going up, a
        worker is free, the site is clear, and the budget grants the price.
        Any of those failing is an ordinary tick, not an error -- the creep
        resumes the moment the world allows.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for prices and immobility.
            profiles: Combat profiles by type name, for the turret's reach.
            budget: The tick's credits.
            free: Workers not already carrying out an order.
            workforce: Told what the worker was sent to build, so the next
                observation sees it working and does not re-draft it.
            hold: Percent of the anchor-to-goal line the walk stops at.
                One hundred is the old walk to the enemy's door; a bridge
                map's choke sits near fifty.

        Returns:
            The build order to send, or nothing this tick.
        """
        if not free or _rising_wall(sample):
            return ()
        anchor = find_anchor(sample, catalogue)
        goal = mirror_point(sample, catalogue)
        if anchor is None or goal is None:
            return ()
        stop = _hold_point((anchor["x"], anchor["y"]), goal, hold)
        front = _front(sample, stop)
        if front is None:
            front = (anchor["x"], anchor["y"])
        step = profile_of(profiles, TURRET_TYPE)["attack_range"]
        if _held(front, stop, step):
            return ()
        # Every third structure heals the wall rather than lengthening it.
        wall_type = REPAIR_TYPE if self.ordered % WALL_CYCLE == WALL_CYCLE - 1 else TURRET_TYPE
        point = _next_step(front, stop, step)
        site = clear_point_near(sample, point, catalogue)
        if site is None:
            return ()
        claim = budget.claim(f"creep:{wall_type}", catalogue[wall_type]["price"])
        if not claim["granted"]:
            # A refused wall piece saves toward itself, the same gated use
            # the tech unlock bought (:meth:`~rw_bot.policy.budget.Budget
            # .withhold`). The walk is sequential by design, so its head
            # entry blocks everything behind it -- and the bastion probe
            # measured the block: `creep:repairbay asked 2,378 got 0`, the
            # 1,500-credit healer never fitting a tick and the whole wall
            # standing at two turrets while the bridge went unmanned (log
            # 2026-07-31). For a creep arm the wall IS the army, and the
            # saving binds the spenders behind it accordingly.
            budget.withhold(catalogue[wall_type]["price"])
            return ()

        def range_to_site(unit: Entity) -> float:
            return _range_to(unit, site)

        builder = min(free, key=range_to_site)
        workforce.assign(builder["unit_id"], wall_type, site)
        self.ordered += 1
        return (build_order(unit_id=builder["unit_id"], type_name=wall_type, x=site[0], y=site[1]),)


def _next_step(
    front: tuple[float, float], goal: tuple[float, float], step: float
) -> tuple[float, float]:
    """Return the next site: one step from the front, toward the goal.

    The front is where the line actually stands -- the forward-most turret,
    or the anchor before one exists -- so the walk advances from reality
    rather than from intent. A front within one step of the goal builds AT
    the goal: that is the kill zone the whole walk exists to reach.

    Args:
        front: Where the line stands now.
        goal: The estimated enemy start.
        step: The turret's own reach, so each site covers the next walk.

    Returns:
        The projected site.
    """
    span_x = goal[0] - front[0]
    span_y = goal[1] - front[1]
    span = sqrt(span_x**2 + span_y**2)
    if span <= step:
        return goal
    return (front[0] + span_x / span * step, front[1] + span_y / span * step)


def _hold_point(
    anchor: tuple[float, float], goal: tuple[float, float], hold: int
) -> tuple[float, float]:
    """Return the point the walk stops at: ``hold`` percent of the way out.

    Args:
        anchor: Our own base.
        goal: The estimated enemy start.
        hold: Percent of the line to walk before holding.

    Returns:
        The hold point.
    """
    share = hold / 100.0
    return (
        anchor[0] + (goal[0] - anchor[0]) * share,
        anchor[1] + (goal[1] - anchor[1]) * share,
    )


def _held(front: tuple[float, float], stop: tuple[float, float], step: float) -> bool:
    """Report whether the line already stands at its hold point.

    Held means a wall structure stands within half a step of the stop --
    close enough that the next projected site would pile onto it rather
    than advance.

    Args:
        front: Where the line stands now.
        stop: The hold point.
        step: The turret's reach.

    Returns:
        True when the walk is done.
    """
    span = sqrt((stop[0] - front[0]) ** 2 + (stop[1] - front[1]) ** 2)
    return span <= step / 2.0


def _front(sample: Sample, goal: tuple[float, float]) -> tuple[float, float] | None:
    """Return the owned complete turret nearest the goal, if any stands."""
    best: tuple[float, float] | None = None
    best_span = 0.0
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"]:
            continue
        if entity["type_name"] not in (TURRET_TYPE, REPAIR_TYPE):
            continue
        span = (entity["x"] - goal[0]) ** 2 + (entity["y"] - goal[1]) ** 2
        if best is None or span < best_span:
            best = (entity["x"], entity["y"])
            best_span = span
    return best


def _rising_wall(sample: Sample) -> bool:
    """Report whether an owned wall structure is still under construction."""
    return any(
        entity["mine"]
        and not entity["complete"]
        and entity["type_name"] in (TURRET_TYPE, REPAIR_TYPE)
        for entity in sample["entities"]
    )


def _range_to(unit: Entity, site: tuple[float, float]) -> float:
    """Squared distance from a unit to a site, for choosing the nearest worker."""
    return (unit["x"] - site[0]) ** 2 + (unit["y"] - site[1]) ** 2


__all__ = ["REPAIR_TYPE", "WALL_CYCLE", "Creeper"]
