"""The artillery battery: a shore turret that outranges the enemy fleet.

The naval hole's cheapest candidate response (log 2026-08-13, the turret
probe): the ground turret's artillery fork reaches 350 against the
battleship's 240, so a battery on the shore-most land shells the parked
fleet from ground no ship can answer -- the standoff logic of the attack
submarine (log 2026-08-10) without the factory, the escort, or the
recurring hull bill. Total cost is the catalogue's own $2,100: a $500
ground turret and the engine's $1,600 conversion, priced by the option row
rather than assumed.

A builder cannot place the fork directly; the chain is the probe's --
stand a ``c_turret_t1`` near the water, then convert it in place through
the same produce-on-self the flame channel uses. Both halves transplant
proven mechanics whole (law eleven): the walk is the shipyard's -- one
candidate at a time, patience per candidate, claim once, re-send every
tick, the newest builder pinned by id until it dies -- and the fractions
are the probe's, descending toward our base because the turret wants the
LAST land before the water where the shipyard wanted the first water
after the land.

The fork order re-sends every tick until the fork stands, which is also
what wins the holder: the flame converter may claim the same turret in
the same tick, and the engine honors whoever sent last
([[policy-holding-ground]]; log 2026-08-10, the send-order law).
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.budget import Budget
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import BuildOrder, ProduceOrder, build_order, produce_order
from rw_bot.wire.state import Entity, Sample

#: The structure a builder can place, and the fork it becomes.
TURRET_TYPE = "c_turret_t1"
BATTERY_TYPE = "c_turret_t1_artillery"

#: Fractions of the anchor-to-mirror line offered to the engine, nearest
#: the water first: the probe measured the lake starting at 0.25 on this
#: line and the shore-most buildable land at 0.14 -- about 196 world units
#: from the first water, inside the fork's 350 reach -- with 0.22 through
#: 0.16 refused (log 2026-08-13). The refused fractions stay: terrain
#: varies by seed, and each refusal costs one patience window, not a
#: match.
FRACTIONS: tuple[float, ...] = (0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10)

#: Samples watched per candidate before the walk advances. Builders walk
#: before they build, so patience is part of the sensor.
PATIENCE = 40

#: Manhattan distance within which a standing turret is the walk's own:
#: placement snaps to the build grid, so the accepted structure never
#: lands exactly on the offered point, and a base-cover turret is
#: hundreds of units away.
SITE_SNAP = 64.0


class Battery:
    """Stands one artillery battery: walk the shore, then convert in place.

    Stateful the way every saving channel is: which candidate is offered,
    how long, and whether each half's price is claimed. Decisions stay
    pure -- a sample and a budget in, at most one order out per method --
    and the engine's acceptance is read from the roster, never assumed
    from the order.
    """

    def __init__(self) -> None:
        """Open the walk at the water-most candidate."""
        self._candidate = 0
        self._waited = 0
        self._paid_turret = False
        self._paid_fork = False
        # One battery per match: once the fork has stood, the channel is
        # closed for good. A battery that later dies is a loss the panel
        # measures, not a rebuild -- rebuilding would spend credits the
        # one-time claims never covered.
        self._done = False
        # Whether the walk's turret has ever been observed standing --
        # what tells "not built yet" (keep saving) from "stood and died"
        # (re-fund the rebuild).
        self._stood = False
        # EVERY point the walk has offered, so the conversion half
        # converts the WALK'S turret and never a cover turret standing at
        # the base ([[policy-holding-ground]]) -- all of them, not just
        # the current candidate's: the third pilot's turret was accepted
        # AFTER patience moved the walk on, stood at the abandoned point,
        # and a last-site-only check never recognized it (log
        # 2026-08-14).
        self._offered: dict[int, tuple[float, float]] = {}
        # The walk's builder, pinned by id until it dies: the shipyard's
        # own rule, learned when ``builders[-1]`` re-resolved to every
        # fresh hire and nobody ever reached the shore (log 2026-08-10).
        self._builder_id: int | None = None

    def _pin_builder(self, builders: list[Entity], avoid: int | None) -> int | None:
        """Return the walk's builder, re-picking only when the pinned one died.

        The newest builder not claimed by another walk: dragging the
        opening's builder across the map is dragging the opening with it,
        and two walks sharing one builder would override each other's
        orders every tick.

        Args:
            builders: Our complete builders, roster order; never empty.
            avoid: A builder id another channel has pinned, or None.

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

    def fund(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        wanted: bool,
    ) -> None:
        """Claim whichever half the channel needs next, EARLY in the tick.

        The fifth pilot's card: the walk's claim lived at the end of the
        spending chain, where its withhold binds nobody -- the exact trap
        the finisher's early claim documents -- and it starved through
        4,866 refusals while the army spent every credit first (log
        2026-08-14). Funding is now its own step, called before the
        hires and conversions so a refused half binds the whole tick.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the turret's price.
            budget: The tick's credits.
            wanted: Whether the doctrine stands a battery at all.
        """
        if not wanted or self._done:
            return
        if not self._site_holds(sample):
            self._fund_turret(catalogue, budget)
            return
        self._fund_fork(sample, budget)

    def _fund_turret(self, catalogue: Mapping[str, UnitStats], budget: Budget) -> None:
        """Claim the walk's turret price once, withholding while refused."""
        if self._candidate >= len(FRACTIONS) or self._paid_turret:
            return
        stats = catalogue.get(TURRET_TYPE)
        if stats is None:
            return
        claim = budget.claim(f"battery:{TURRET_TYPE}", stats["price"])
        if not claim["granted"]:
            budget.withhold(stats["price"])
            return
        self._paid_turret = True
        self._waited = 0

    def _fund_fork(self, sample: Sample, budget: Budget) -> None:
        """Claim the fork at the engine's own price once the turret offers."""
        turret = self._own_turret(sample)
        if turret is None or turret["type_name"] != TURRET_TYPE or not turret["complete"]:
            return
        if self._paid_fork:
            return
        price = None
        for option in sample["options"]:
            if option["unit_id"] == turret["unit_id"] and option["produces"] == BATTERY_TYPE:
                price = option["price"]
                break
        if price is None:
            # The engine is not offering the fork yet -- a turret still
            # settling, or an options row not yet published. Wait; the
            # claim needs the engine's price, not a guess.
            return
        claim = budget.claim(f"battery:{BATTERY_TYPE}", price)
        if not claim["granted"]:
            budget.withhold(price)
            return
        self._paid_fork = True

    def establish(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        wanted: bool,
        avoid_builder: int | None = None,
    ) -> tuple[BuildOrder, ...]:
        """Offer the current candidate a turret, advancing on patience.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.
            budget: The tick's credits, unread: :meth:`fund` claims both
                halves early in the tick, where a refusal binds the
                spenders below it; kept in the signature so the two walks
                stay call-compatible for the quartermaster.
            wanted: Whether the doctrine stands a battery at all.
            avoid_builder: A builder id another walk has pinned, never
                taken here.

        Returns:
            At most one build order. While walking: the current
            candidate, re-sent every tick once :meth:`fund` has paid the
            turret. While the site turret is INCOMPLETE: the same order
            at the standing turret, because the expander re-tasks the
            builder to distant pools the moment the walk goes silent and
            an abandoned construction dies unfinished -- pilot six stood
            three turrets and completed none (log 2026-08-14). Sent
            after the expander so the re-sent order lands last and wins
            the builder back every tick (log 2026-08-10, the send-order
            law).
        """
        del budget
        if not wanted or self._done or self._candidate >= len(FRACTIONS):
            return ()
        standing = self._own_turret(sample) if self._site_holds(sample) else None
        if standing is not None and (standing["type_name"] == BATTERY_TYPE or standing["complete"]):
            # Finished either way: the builder is released and the
            # conversion half takes over.
            return ()
        if standing is None and not self._paid_turret:
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
        if standing is not None:
            # The construction hold: keep the builder on the incomplete
            # turret until the engine reports it finished.
            return (
                build_order(
                    unit_id=builder_id,
                    type_name=TURRET_TYPE,
                    x=standing["x"],
                    y=standing["y"],
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
        x = anchor["x"] + (goal[0] - anchor["x"]) * share
        y = anchor["y"] + (goal[1] - anchor["y"]) * share
        self._offered[self._candidate] = (x, y)
        return (build_order(unit_id=builder_id, type_name=TURRET_TYPE, x=x, y=y),)

    def _site_holds(self, sample: Sample) -> bool:
        """Whether the site holds our structure; re-funds a razed one.

        A turret that stood and is gone -- died building, converting, or
        standing -- makes the walk resume at the proven fraction, and the
        rebuild RE-FUNDS both halves: the engine charges per attempt, so
        books that remembered the first payment would spend the second
        unaccounted (the pilot's second turret, log 2026-08-14).

        Args:
            sample: One observation of the world.

        Returns:
            True while our turret or battery stands at the site.
        """
        if self._own_turret(sample) is not None:
            self._stood = True
            return True
        if self._stood:
            self._stood = False
            self._paid_turret = False
            self._paid_fork = False
        return False

    def _own_turret(self, sample: Sample) -> Entity | None:
        """Return the walk's own turret or battery, identified by its site.

        A cover doctrine stands ground turrets at the base; the channel's
        is the one within a placement snap of ANY point the walk has
        offered. The engine may accept a candidate after patience has
        moved the walk on -- the builder was still walking -- so an
        abandoned point's turret is still the walk's own (log
        2026-08-14, the third pilot).

        Args:
            sample: One observation of the world.

        Returns:
            The entity at one of the walk's sites, or None.
        """
        if not self._offered:
            return None
        for entity in sample["entities"]:
            if entity["mine"] and entity["type_name"] in (TURRET_TYPE, BATTERY_TYPE):
                near = any(
                    abs(entity["x"] - x) + abs(entity["y"] - y) <= SITE_SNAP
                    for x, y in self._offered.values()
                )
                if near:
                    return entity
        return None

    def convert(self, sample: Sample, budget: Budget, wanted: bool) -> tuple[ProduceOrder, ...]:
        """Order the standing turret up the artillery fork, funded once.

        Args:
            sample: One observation of the world.
            budget: The tick's credits, unread: :meth:`fund` claims the
                fork early in the tick; kept for call-compatibility.
            wanted: Whether the doctrine stands a battery at all.

        Returns:
            At most one produce order, re-sent every tick until the fork
            stands: conversion never fills the queue, and the re-send is
            also what wins the holder against the flame converter's claim
            on the same turret ([[policy-holding-ground]]).
        """
        del budget
        if not wanted or self._done:
            return ()
        turret = self._own_turret(sample)
        if turret is None:
            return ()
        if turret["type_name"] == BATTERY_TYPE:
            # The fork stands; the channel closes for the match.
            self._done = True
            return ()
        if not turret["complete"] or not self._paid_fork:
            return ()
        return (produce_order(unit_id=turret["unit_id"], type_name=BATTERY_TYPE),)

    def pinned_builder(self) -> int | None:
        """Return the id of the builder this walk holds, or None."""
        return self._builder_id

    def holder_id(self, sample: Sample) -> int | None:
        """Return the site turret's id while it awaits its fork, or None.

        The pilot's void (log 2026-08-14): the flame converter's $700
        claim funds before the fork's $1,600, so it took the shore turret
        every time it stood and the battery never existed. The converters
        exclude this holder.

        Args:
            sample: One observation of the world.

        Returns:
            The unit id of the walk's standing base turret, or None when
            nothing at the site awaits conversion.
        """
        if self._done:
            return None
        turret = self._own_turret(sample)
        if turret is None or turret["type_name"] != TURRET_TYPE:
            return None
        return turret["unit_id"]


__all__ = ["BATTERY_TYPE", "FRACTIONS", "PATIENCE", "TURRET_TYPE", "Battery"]
