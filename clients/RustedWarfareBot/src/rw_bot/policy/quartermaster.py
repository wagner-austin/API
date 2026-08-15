"""The standing purchases, gathered: hires, conversions, and shore walks.

Split from :mod:`rw_bot.policy.campaign` when the battery pushed the loop
past the size cap -- a real seam, not a trim: these channels share one
shape (a doctrine count in, a funded order out, state that remembers what
was already bought) and the loop only ever dispatched them in a block.
Nothing here sends: orders return to the loop, which owns dispatch, so
the architecture's two-sender rule holds.

Two orderings in this module ARE policy, stated here once:

* the fleet guard hires before the submarines -- an unescorted fleet is a
  free kill queue for the first gunship (navy96c, log 2026-08-10);
* the battery converts AFTER the flame converter -- both may claim the
  same base turret in one tick, the engine honors whoever sent last, and
  the fork the doctrine paid $1,600 toward is the one that must win
  (log 2026-08-14).
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.battery import Battery
from rw_bot.policy.budget import Budget
from rw_bot.policy.convert import Converter, TurretLadder
from rw_bot.policy.medic import BUNKER_TYPE, Medic
from rw_bot.policy.navy import GUARD_TYPE, Shipyard
from rw_bot.wire.command import BuildOrder, ProduceOrder
from rw_bot.wire.state import Sample


class Quartermaster:
    """Holds every standing-purchase channel and its doctrine count.

    Attributes:
        shipyard: The sea factory walk, exposed for the builder-pin
            coordination the establish pass performs.
    """

    def __init__(
        self,
        *,
        medics: int,
        navy: int,
        bunkers: int,
        flame: int,
        guns: int,
        battery: int,
    ) -> None:
        """Open every channel with its doctrine count.

        Args:
            medics: Combat engineers kept alive via saving hires.
            navy: Attack submarines kept alive on the water.
            bunkers: Mobile turrets kept alive the same way.
            flame: Flame turrets held by converting ground turrets.
            guns: Top-tier gun turrets held by walking the turret chain.
            battery: Artillery batteries stood on the shore, 0 or 1 --
                the channel stands at most one per match.
        """
        self._medics = medics
        self._navy = navy
        self._bunkers = bunkers
        self._flame = flame
        self._guns = guns
        self._battery = battery
        self.shipyard = Shipyard()
        self._medic = Medic()
        self._bunker = Medic(BUNKER_TYPE)
        self._submarines = Medic("attackSubmarine")
        self._fleet_guard = Medic(GUARD_TYPE)
        self._flamer = Converter()
        self._gunner = TurretLadder()
        self._battery_walk = Battery()

    def produces(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
    ) -> tuple[ProduceOrder, ...]:
        """Return this tick's hires and conversions, in claim order.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the battery's prices.
            budget: The tick's credits.

        Returns:
            Produce orders for the loop to dispatch, guard before subs,
            battery's fork last so its re-send wins a contested holder.
        """
        # Both walks FUND first of all -- a claim at the end of the chain
        # starved through 4,866 refusals while the army spent every
        # credit (the fifth pilot, log 2026-08-14) -- while the battery's
        # fork order still SENDS last, below. And the site turret is
        # spoken for: neither converter may take it.
        self._battery_walk.fund(sample, catalogue, budget, self._battery > 0)
        self.shipyard.fund(sample, catalogue, budget, self._navy > 0)
        fork = self._battery_walk.convert(sample, budget, self._battery > 0)
        spoken_for = self._battery_walk.holder_id(sample)
        return (
            *self._fleet_guard.hire(sample, budget, min(self._navy, 1)),
            *self._submarines.hire(sample, budget, self._navy),
            *self._medic.hire(sample, budget, self._medics),
            *self._bunker.hire(sample, budget, self._bunkers),
            *self._flamer.convert(sample, budget, self._flame, exclude=spoken_for),
            *self._gunner.convert(sample, budget, self._guns, exclude=spoken_for),
            *fork,
        )

    def builds(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
    ) -> tuple[BuildOrder, ...]:
        """Return this tick's shore walks, called after the expander.

        Both walks re-send every tick and whoever sends last holds the
        builder (log 2026-08-10), so the loop calls this after the
        expander's block -- and the battery avoids the shipyard's pinned
        builder so two live walks never override each other.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name.
            budget: The tick's credits.

        Returns:
            Build orders for the loop to dispatch.
        """
        return (
            *self.shipyard.establish(
                sample,
                catalogue,
                budget,
                self._navy > 0,
                avoid_builder=self._battery_walk.pinned_builder(),
            ),
            *self._battery_walk.establish(
                sample,
                catalogue,
                budget,
                self._battery > 0,
                avoid_builder=self.shipyard.pinned_builder(),
            ),
        )


__all__ = ["Quartermaster"]
