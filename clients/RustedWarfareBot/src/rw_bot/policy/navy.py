"""Establishing the sea factory: terrain discovery by attempt, as a channel.

The naval theater's gate ([[policy-exact-timing]], the naval wall): the
enemy's fleet core declares ``No anti-air or anti-sub`` while the attack
submarine outranges it submerged and untouchable -- a hard counter by the
engine's own stat sheet, and the first response priced positive-when-armed
BEFORE its trigger was built (law ten; log 2026-08-10). Everything routes
through a ``seaFactory``, which "can only be built on water", and the
planner has no terrain map.

It never needed one. The sea probe proved the sensor is the engine itself:
offer a build order along the anchor-to-mirror line -- the water lies
between the starts on every symmetric water map -- and dry land swallows
the order in silence while a wet point grows a factory. This channel is
that walk with a budget: one candidate at a time, a patience window per
candidate, the claim-or-withhold discipline every saving channel uses.

The submarines themselves need nothing new: once the factory stands,
:class:`~rw_bot.policy.medic.Medic` keeps the headcount alive through the
same hire machinery that staffs combat engineers, because a sub is just a
hire whose producer floats.
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.budget import Budget
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import BuildOrder, build_order
from rw_bot.wire.state import Sample

#: The structure the walk stands, and the reason it exists.
FACTORY_TYPE = "seaFactory"

#: The fleet's own anti-air, hired BEFORE the submarines: navy96c's
#: factories died to lightGunships, and the stat sheet had already
#: assigned this job -- the missile ship is the only hull that shoots
#: air. A fleet without one is a free kill queue (log 2026-08-10).
GUARD_TYPE = "missileShip"

#: Fractions of the anchor-to-mirror line offered to the engine, nearest
#: first and FINE near the shore: navy96c's factories stood at 0.25 --
#: 460 world units out, alone -- and died to gunships before hiring a
#: single submarine, so the walk hugs our edge of the water at one-percent
#: steps before conceding distance (log 2026-08-10).
FRACTIONS: tuple[float, ...] = (
    0.2,
    0.21,
    0.22,
    0.23,
    0.24,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
)

#: Samples watched per candidate before the walk advances. Builders walk
#: before they build, so patience is part of the sensor -- and each
#: candidate costs ONE claim and ONE order, never a stream of either.
PATIENCE = 40


class Shipyard:
    """Walks the line until a sea factory stands, one candidate at a time.

    Stateful the way every saving channel is: which candidate is being
    offered and for how long. Decisions stay pure -- a sample and a budget
    in, at most one build order out -- and the engine's acceptance is read
    from the roster, never assumed from the order.
    """

    def __init__(self) -> None:
        """Open the walk at the nearest candidate."""
        self._candidate = 0
        self._waited = 0
        self._paid = False

    def establish(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        wanted: bool,
    ) -> tuple[BuildOrder, ...]:
        """Offer the current candidate, advancing when patience runs out.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor and the
                factory's price.
            budget: The tick's credits; the factory price is claimed while
                the walk is live and withheld toward when refused, the
                saving pattern every strategic purchase uses
                ([[policy-budget]]).
            wanted: Whether the doctrine plays the water at all.

        Returns:
            At most one build order -- the current candidate, re-sent every
            tick once the price is claimed ONCE. Both halves were measured
            separately: claiming per tick consumed 369,000 credits and the
            economy never existed (navy96), and ordering once let the
            expander re-task the builder a tick later and the factory never
            stood (navy96b interim). The navy sends after the expander in
            the tick, so the re-sent order lands last and wins the builder
            (log 2026-08-10).
        """
        if not wanted or self._candidate >= len(FRACTIONS):
            return ()
        for entity in sample["entities"]:
            if entity["mine"] and entity["type_name"] == FACTORY_TYPE:
                # Standing or under construction either way: the walk's job
                # is done and the headcount channel takes over.
                return ()
        stats = catalogue.get(FACTORY_TYPE)
        if stats is None:
            # A catalogue without the type cannot price the claim; the
            # doctrine asked for water the build simply cannot describe.
            return ()
        anchor = find_anchor(sample, catalogue)
        goal = mirror_point(sample, catalogue)
        builders = [
            entity
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"] and entity["type_name"] == "builder"
        ]
        if anchor is None or goal is None or not builders:
            return ()
        if not self._paid:
            claim = budget.claim(f"navy:{FACTORY_TYPE}", stats["price"])
            if not claim["granted"]:
                budget.withhold(stats["price"])
                return ()
            self._paid = True
            self._waited = 0
        share = FRACTIONS[self._candidate]
        self._waited += 1
        if self._waited > PATIENCE:
            self._candidate += 1
            self._waited = 0
            if self._candidate >= len(FRACTIONS):
                return ()
            share = FRACTIONS[self._candidate]
        return (
            build_order(
                # The NEWEST builder, not the opening's: builders[-1] by
                # roster order is the latest hire, and dragging the first
                # builder across the map is dragging the opening with it.
                unit_id=builders[-1]["unit_id"],
                type_name=FACTORY_TYPE,
                x=anchor["x"] + (goal[0] - anchor["x"]) * share,
                y=anchor["y"] + (goal[1] - anchor["y"]) * share,
            ),
        )


__all__ = ["FACTORY_TYPE", "FRACTIONS", "PATIENCE", "Shipyard"]
