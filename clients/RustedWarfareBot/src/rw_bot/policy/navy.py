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

#: Fractions of the anchor-to-mirror line offered to the engine, nearest
#: first: the shore closest to our base is the edge a builder can reach,
#: and the live probe accepted the second candidate on duel_lake
#: (log 2026-08-10, the sea probe).
FRACTIONS: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)

#: Samples offered per candidate before the walk advances. Builders walk
#: before they build, so patience is part of the sensor.
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
            At most one build order -- the current candidate, re-offered
            each tick so the assigned builder keeps walking toward it.
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
        claim = budget.claim(f"navy:{FACTORY_TYPE}", stats["price"])
        if not claim["granted"]:
            budget.withhold(stats["price"])
            return ()
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
                unit_id=builders[0]["unit_id"],
                type_name=FACTORY_TYPE,
                x=anchor["x"] + (goal[0] - anchor["x"]) * share,
                y=anchor["y"] + (goal[1] - anchor["y"]) * share,
            ),
        )


__all__ = ["FACTORY_TYPE", "FRACTIONS", "PATIENCE", "Shipyard"]
