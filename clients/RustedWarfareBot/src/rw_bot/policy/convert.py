"""Structures converted up a chosen fork, funded the way the medics are.

The extractor walk (:func:`~rw_bot.policy.economy.upgradeable`) follows each
structure's single next tier and deliberately leaves forks alone -- and the
ground turret is a four-way fork: gun, artillery, lightning, flame. The branch
worth choosing is the doctrine's call, and the community's is unambiguous for
the horde matches this campaign keeps dying to: the flamethrower's wide-area
fire, self-repair and 1,600 hit points are their named answer to "hover
tanks and other early units that tend to send rushes in groups"
([[community-play-strategies]]).

A conversion has the same funding arithmetic as every other big purchase --
under pressure the balance never holds the price plus the reserve -- so a
refused conversion withholds toward itself ([[policy-budget]]). And it has
the same duplicate trap the income walk met first: converting never fills
the queue, the structure keeps offering the upgrade it is performing, and a
duplicate landing after completion names an action that no longer exists and
crashes the match. The ordered set remembers the pair, exactly as
:func:`~rw_bot.policy.spending.upgrade_income` does
([[policy-holding-ground]]).

Pure decisions, stateful memory: which conversions were ordered, the same
shape as every other controller.
"""

from __future__ import annotations

from typing import Final

from rw_bot.policy.budget import Budget
from rw_bot.wire.command import ProduceOrder, produce_order
from rw_bot.wire.state import Sample

#: The anti-horde branch of the ground turret's tier-two fork.
#:
#: 1,600 hit points, self-repair, shoot delay 5 and area damage at reach 155
#: -- short-range wide-area fire that scales with how crowded the attack is,
#: which is the exact shape of the waves that end every Impossible match. A
#: 1,000-credit conversion off the 500-credit turret cover already builds.
FLAME_TYPE: Final = "c_turret_t2_flame"


class Converter:
    """Holds the doctrine's count of one converted type, one order at a time.

    Attributes:
        converted: Conversion orders sent so far, for the report.
    """

    def __init__(self, type_name: str = FLAME_TYPE) -> None:
        """Open the conversion channel.

        Args:
            type_name: What holders are converted into, the flame turret
                unless told otherwise.
        """
        self.converted = 0
        self._type = type_name
        self._ordered: set[int] = set()

    def convert(self, sample: Sample, budget: Budget, wanted: int) -> tuple[ProduceOrder, ...]:
        """Order the next conversion when the headcount is short and funded.

        Args:
            sample: One observation of the world.
            budget: The tick's credits.
            wanted: Converted structures the doctrine keeps, zero for none.

        Returns:
            The produce order to send, or nothing this tick.
        """
        if wanted <= 0:
            return ()
        alive = 0
        roster: dict[int, str] = {}
        for entity in sample["entities"]:
            if not entity["mine"] or not entity["complete"]:
                continue
            if entity["type_name"] == self._type:
                alive += 1
            roster[entity["unit_id"]] = entity["type_name"]
        # A holder mid-conversion is a conversion already paid for: it was
        # ordered, still stands, and has not yet become the target. Counting
        # it prevents the duplicate the queue cannot signal -- converting
        # never fills ``queued`` ([[policy-holding-ground]]).
        underway = sum(
            1 for unit_id in self._ordered if roster.get(unit_id, self._type) != self._type
        )
        if alive + underway >= wanted:
            return ()
        offer = _conversion_offer(sample, self._type, self._ordered)
        if offer is None:
            return ()
        holder, price = offer
        claim = budget.claim(f"convert:{self._type}", price)
        if not claim["granted"]:
            # The conversion saves toward itself, the tech unlock's gated
            # pattern: bounded by the headcount, not an income ladder
            # ([[policy-budget]]).
            budget.withhold(price)
            return ()
        self._ordered.add(holder)
        self.converted += 1
        return (produce_order(unit_id=holder, type_name=self._type),)


#: The ground turret's gun chain, base to top.
#:
#: The community's fortified-zone recipe names the top of this chain as the
#: zone's teeth: "1 T3 turret can solo most ground units -- now imagine a
#: wall of them", and the AI's attack-move stands its army at that wall
#: until one of them dies (steam-impossible-playbook.txt,
#: [[community-play-strategies]]). The tier-two gun is the only step that
#: offers the tier three, so holding the top means walking the chain.
GUN_CHAIN: Final = ("c_turret_t1", "c_turret_t2_gun", "c_turret_t3_gun")


class TurretLadder:
    """Holds the doctrine's count of top-tier turrets by walking a chain.

    The extractor walk follows each structure's single next tier; the
    ground turret is a fork, so this walks one CHOSEN branch: convert an
    idle mid-tier holder to the top when one offers, otherwise feed the
    pipeline by converting a base holder to the mid tier -- one order a
    tick, each step funded through the withhold like every other purchase
    the balance never holds ([[policy-budget]]).

    Ordered conversions are remembered as (holder, target) pairs, the
    income walk's own key: a conversion keeps the engine identity, so a
    holder remembered by id alone could never take its second step and
    the chain would stop at the mid tier ([[policy-holding-ground]]).

    Attributes:
        converted: Conversion orders sent so far, for the report.
    """

    def __init__(self, chain: tuple[str, str, str] = GUN_CHAIN) -> None:
        """Open the ladder.

        Args:
            chain: Base, mid and top type of the walked branch.
        """
        self.converted = 0
        self._chain = chain
        self._ordered: set[tuple[int, str]] = set()

    def _underway(self, roster: dict[int, str], target: str) -> int:
        """Count holders told to become ``target`` that have not yet.

        Args:
            roster: Owned complete entities' types by unit id.
            target: The conversion target counted.

        Returns:
            Conversions in flight -- ordered, holder still standing, not
            yet the target. A dead holder is not in flight; its order died
            with it.
        """
        return sum(
            1
            for unit_id, tgt in self._ordered
            if tgt == target and unit_id in roster and roster[unit_id] != target
        )

    def _ordered_for(self, target: str) -> set[int]:
        """Return the holders already told to become ``target``.

        Args:
            target: The conversion target.

        Returns:
            Their unit ids -- the per-target exclusion that keeps a holder
            re-orderable for its NEXT tier while never duplicating the
            conversion it is already performing.
        """
        return {unit_id for unit_id, tgt in self._ordered if tgt == target}

    def convert(self, sample: Sample, budget: Budget, wanted: int) -> tuple[ProduceOrder, ...]:
        """Advance the chain one funded step when the top count is short.

        Args:
            sample: One observation of the world.
            budget: The tick's credits.
            wanted: Top-tier turrets the doctrine keeps, zero for none.

        Returns:
            The produce order to send, or nothing this tick.
        """
        if wanted <= 0:
            return ()
        _, mid, top = self._chain
        roster = {
            entity["unit_id"]: entity["type_name"]
            for entity in sample["entities"]
            if entity["mine"] and entity["complete"]
        }
        tops = sum(1 for name in roster.values() if name == top)
        need_top = wanted - tops - self._underway(roster, top)
        if need_top <= 0:
            return ()
        offer = _conversion_offer(sample, top, self._ordered_for(top))
        if offer is None:
            # No idle mid offers the top yet: feed the pipeline, bounded by
            # what the top still needs -- an unbounded feed would convert
            # every base turret while the top step saved, and the base tier
            # IS the cover the zone stands on.
            mids = sum(1 for name in roster.values() if name == mid)
            if mids + self._underway(roster, mid) >= need_top:
                return ()
            offer = _conversion_offer(sample, mid, self._ordered_for(mid))
            if offer is None:
                return ()
            target = mid
        else:
            target = top
        holder, price = offer
        claim = budget.claim(f"convert:{target}", price)
        if not claim["granted"]:
            # Each step saves toward itself, the gated pattern: the tier-two
            # hop is ~1,000 and the tier-three 11,000+, and neither ever
            # fits a contested tick unaided ([[policy-budget]]).
            budget.withhold(price)
            return ()
        self._ordered.add((holder, target))
        self.converted += 1
        return (produce_order(unit_id=holder, type_name=target),)


def _conversion_offer(sample: Sample, type_name: str, ordered: set[int]) -> tuple[int, int] | None:
    """Return an idle holder offering the conversion, and the engine's price.

    Args:
        sample: One observation of the world.
        type_name: What the channel converts holders into.
        ordered: Holders already told to convert, never asked twice -- the
            structure keeps offering the upgrade it is already performing,
            and a duplicate landing after completion crashes the match
            ([[policy-holding-ground]]).

    Returns:
        The holder's unit id and the option's price, or None when nothing
        idle offers the conversion.
    """
    idle = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["queued"] == 0
    }
    for option in sample["options"]:
        if (
            option["produces"] == type_name
            and option["available"]
            and not option["placed"]
            and option["unit_id"] in idle
            and option["unit_id"] not in ordered
        ):
            return (option["unit_id"], option["price"])
    return None


__all__ = ["FLAME_TYPE", "GUN_CHAIN", "Converter", "TurretLadder"]
