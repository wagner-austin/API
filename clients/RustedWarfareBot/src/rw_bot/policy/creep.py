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

        Returns:
            The build order to send, or nothing this tick.
        """
        if not free or _rising_turret(sample):
            return ()
        anchor = find_anchor(sample, catalogue)
        goal = mirror_point(sample, catalogue)
        if anchor is None or goal is None:
            return ()
        front = _front(sample, goal)
        if front is None:
            front = (anchor["x"], anchor["y"])
        point = _next_step(front, goal, profile_of(profiles, TURRET_TYPE)["attack_range"])
        site = clear_point_near(sample, point, catalogue)
        if site is None:
            return ()
        claim = budget.claim(f"creep:{TURRET_TYPE}", catalogue[TURRET_TYPE]["price"])
        if not claim["granted"]:
            return ()

        def range_to_site(unit: Entity) -> float:
            return _range_to(unit, site)

        builder = min(free, key=range_to_site)
        workforce.assign(builder["unit_id"], TURRET_TYPE, site)
        self.ordered += 1
        return (
            build_order(unit_id=builder["unit_id"], type_name=TURRET_TYPE, x=site[0], y=site[1]),
        )


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


def _front(sample: Sample, goal: tuple[float, float]) -> tuple[float, float] | None:
    """Return the owned complete turret nearest the goal, if any stands."""
    best: tuple[float, float] | None = None
    best_span = 0.0
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"]:
            continue
        if entity["type_name"] != TURRET_TYPE:
            continue
        span = (entity["x"] - goal[0]) ** 2 + (entity["y"] - goal[1]) ** 2
        if best is None or span < best_span:
            best = (entity["x"], entity["y"])
            best_span = span
    return best


def _rising_turret(sample: Sample) -> bool:
    """Report whether an owned creep-type turret is still under construction."""
    return any(
        entity["mine"] and not entity["complete"] and entity["type_name"] == TURRET_TYPE
        for entity in sample["entities"]
    )


def _range_to(unit: Entity, site: tuple[float, float]) -> float:
    """Squared distance from a unit to a site, for choosing the nearest worker."""
    return (unit["x"] - site[0]) ** 2 + (unit["y"] - site[1]) ** 2


__all__ = ["Creeper"]
