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

The battery's five pilots re-audited this walk (log 2026-08-14): its
claim moved EARLY in the tick (a tail-of-tick withhold binds nobody --
the starvation pilot five measured), and the walk now holds its builder
on the incomplete factory (the expander re-tasks a released builder and
an abandoned construction dies unfinished -- pilot six's defect, masked
here only because water is unreachable by the land army).

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
from rw_bot.wire.state import Entity, Sample

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
        # Whether the factory has ever been observed standing -- what
        # tells "not built yet" (keep saving) from "stood and died"
        # (re-fund the rebuild): the engine charges per attempt, so the
        # books must too (the battery's second pilot; log 2026-08-14).
        self._stood = False
        # The walk's builder, pinned by id: the probe's factory stood
        # because ONE builder accumulated walking progress across
        # candidate windows, and navy96e's never did because
        # ``builders[-1]`` re-resolved to whichever builder had just
        # been hired -- unit 24, then 43, then 55, each starting the
        # trek from the base with forty ticks to live (log 2026-08-10).
        self._builder_id: int | None = None

    def _pin_builder(self, builders: list[Entity], avoid: int | None) -> int | None:
        """Return the walk's builder, re-picking only when the pinned one died.

        Pick the NEWEST builder not claimed by another walk -- dragging
        the opening's builder across the map is dragging the opening with
        it -- and then KEEP it until it dies. The patience window restarts
        with a replacement, because a fresh builder starts the trek from
        the base and inheriting a spent window refuses the fraction
        without ever having reached it.

        Args:
            builders: Our complete builders, roster order; never empty.
            avoid: A builder id another walk has pinned, or None.

        Returns:
            The pinned builder's unit id, or None when every builder is
            claimed elsewhere.
        """
        alive = {worker["unit_id"] for worker in builders}
        builder_id = self._builder_id
        if builder_id is None or builder_id not in alive or builder_id == avoid:
            candidates = [w["unit_id"] for w in builders if w["unit_id"] != avoid]
            if not candidates:
                return None
            builder_id = candidates[-1]
            self._builder_id = builder_id
            self._waited = 0
        return builder_id

    def pinned_builder(self) -> int | None:
        """Return the id of the builder this walk holds, or None.

        Exposed so a second walk can avoid pinning the same builder --
        two live walks re-sending against one builder would override
        each other every tick ([[policy-holding-ground]]).
        """
        return self._builder_id

    def _factory(self, sample: Sample) -> Entity | None:
        """Return our sea factory, standing or under construction."""
        for entity in sample["entities"]:
            if entity["mine"] and entity["type_name"] == FACTORY_TYPE:
                return entity
        return None

    def fund(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        wanted: bool,
    ) -> None:
        """Claim the factory price once, EARLY in the tick.

        The battery's fifth pilot measured what a tail-of-tick claim is
        worth: 4,866 refusals while the army spent every credit first,
        because a withhold at the end of the chain binds nobody (log
        2026-08-14). The shipyard carried the same defect, masked only by
        early-game balances; funding is now its own step, called before
        the hires and conversions.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the factory's price.
            budget: The tick's credits.
            wanted: Whether the doctrine plays the water at all.
        """
        if not wanted or self._candidate >= len(FRACTIONS):
            return
        if self._factory(sample) is not None:
            self._stood = True
            return
        if self._stood:
            # It stood and is gone: the walk resumes and the rebuild
            # re-funds, because the engine will charge again.
            self._stood = False
            self._paid = False
        if self._paid:
            return
        stats = catalogue.get(FACTORY_TYPE)
        if stats is None:
            return
        claim = budget.claim(f"navy:{FACTORY_TYPE}", stats["price"])
        if not claim["granted"]:
            budget.withhold(stats["price"])
            return
        self._paid = True
        self._waited = 0

    def establish(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        wanted: bool,
        avoid_builder: int | None = None,
    ) -> tuple[BuildOrder, ...]:
        """Offer the current candidate, advancing when patience runs out.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.
            budget: The tick's credits, unread: :meth:`fund` claims the
                factory early in the tick, where a refusal binds the
                spenders below it; kept for call-compatibility.
            wanted: Whether the doctrine plays the water at all.
            avoid_builder: A builder id another walk has pinned, never
                taken here.

        Returns:
            At most one build order. While walking: the current
            candidate, re-sent every tick once :meth:`fund` has paid.
            While the factory is INCOMPLETE: the same order at the
            standing factory, so the expander cannot re-task the builder
            away from an unfinished construction (the battery's sixth
            pilot; log 2026-08-14). Both re-sends land after the
            expander's, which is what wins the builder (log 2026-08-10).
        """
        del budget
        if not wanted or self._candidate >= len(FRACTIONS):
            return ()
        factory = self._factory(sample)
        if factory is not None and factory["complete"]:
            # Standing finished: the walk's job is done and the headcount
            # channel takes over.
            return ()
        if factory is None and not self._paid:
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
        builder_id = self._pin_builder(builders, avoid_builder)
        if builder_id is None:
            return ()
        if factory is not None:
            # The construction hold: keep the builder on the incomplete
            # factory until the engine reports it finished.
            return (
                build_order(
                    unit_id=builder_id,
                    type_name=FACTORY_TYPE,
                    x=factory["x"],
                    y=factory["y"],
                ),
            )
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
                unit_id=builder_id,
                type_name=FACTORY_TYPE,
                x=anchor["x"] + (goal[0] - anchor["x"]) * share,
                y=anchor["y"] + (goal[1] - anchor["y"]) * share,
            ),
        )


__all__ = ["FACTORY_TYPE", "FRACTIONS", "GUARD_TYPE", "PATIENCE", "Shipyard"]
