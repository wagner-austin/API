"""Combat engineers hired the way the tech unlock was bought: by saving.

The healer the community meta is built on is priced out of every tick: the
legion probe's ledger read ``produce:combatEngineer asked 181 got 0`` -- the
tier-two factory OFFERS the 3,500-credit combat engineer, and under
Impossible pressure the balance never once held 3,500 plus the reserve
([[policy-budget]], log 2026-07-31). The same arithmetic starved the tech
unlock and the creep wall, and both were fixed the same way: **a refused
purchase that saves toward itself**, bounded so the saving cannot become
the refuted income-ladder pause.

The bound here is a headcount. The doctrine names how many medics to keep
alive; the channel hires one at a time, funds it through the withhold, and
never has more than one on order -- a healer mid-production is a healer
already paid for, and ordering a second against the same shortfall was the
duplicate-produce bug in every other channel that met it
([[policy-production]]).

Medics heal automatically in range, so joining the army is the whole job:
they ride the composition like any unit, and the flee reflex's wounded run
home past them ([[community-play-strategies]]).

Pure decisions, stateful memory: which producer holds the outstanding
order, the same shape as every other controller.
"""

from __future__ import annotations

from typing import Final

from rw_bot.policy.budget import Budget
from rw_bot.wire.command import ProduceOrder, produce_order
from rw_bot.wire.state import Sample

#: The healer the channel hires.
#:
#: Armed, a thousand hit points, heals units at twice the repair bay's rate,
#: and offered by the tier-two land factory -- the mid-game staple of the
#: community meta ([[community-play-strategies]]).
MEDIC_TYPE: Final = "combatEngineer"

#: The anti-horde unit the channel hires for the ``bunkers`` doctrine field.
#:
#: The Mobile Turret: 4,500 credits, 800 hit points, area damage 110 behind
#: a deploy shield -- the community's named counter to massed tier-one land,
#: which is what kills every arm at Impossible. Ordinary production never
#: once accumulated its price: ``produce:mechBunker asked 1178 got 0`` with
#: the economy healthy (log 2026-08-01), the same arithmetic that starved
#: the medics and the tech unlock ([[community-play-strategies]]).
BUNKER_TYPE: Final = "mechBunker"


class Medic:
    """Keeps the doctrine's count of one hired type alive, one hire at a time.

    Built for the combat engineer and generalised the day the mobile turret
    met the same funding wall: any composition entry priced out of every
    tick needs the withhold, and the machinery is identical -- only the type
    changes.

    Attributes:
        hired: Produce orders sent so far, for the report.
    """

    def __init__(self, type_name: str = MEDIC_TYPE) -> None:
        """Open the hiring channel.

        Args:
            type_name: What the channel hires, the medic unless told
                otherwise.
        """
        self.hired = 0
        self._type = type_name
        self._pending: int | None = None

    def hire(self, sample: Sample, budget: Budget, wanted: int) -> tuple[ProduceOrder, ...]:
        """Order the next medic when the headcount is short and funded.

        Args:
            sample: One observation of the world.
            budget: The tick's credits.
            wanted: Medics the doctrine keeps alive, zero for none.

        Returns:
            The produce order to send, or nothing this tick.
        """
        if wanted <= 0:
            return ()
        alive = 0
        producers: dict[int, int] = {}
        for entity in sample["entities"]:
            if not entity["mine"] or not entity["complete"]:
                continue
            if entity["type_name"] == self._type:
                alive += 1
            producers[entity["unit_id"]] = entity["queued"]
        if self._pending is not None:
            # One outstanding order at a time: while the producer's queue is
            # busy the hire is underway, and a roll-out or a lost factory
            # clears the slot either way.
            if producers.get(self._pending, 0) > 0:
                return ()
            self._pending = None
        if alive >= wanted:
            return ()
        offer = _hire_offer(sample, self._type)
        if offer is None:
            return ()
        producer, price = offer
        claim = budget.claim(f"medic:{self._type}", price)
        if not claim["granted"]:
            # The hire saves toward itself, the tech unlock's gated
            # pattern: without it the price never fits a tick, with it the
            # bound is the headcount ([[policy-budget]]).
            budget.withhold(price)
            return ()
        self._pending = producer
        self.hired += 1
        return (produce_order(unit_id=producer, type_name=self._type),)


def _hire_offer(sample: Sample, type_name: str) -> tuple[int, int] | None:
    """Return an idle producer offering the hired type, and the engine's price.

    Args:
        sample: One observation of the world.
        type_name: What the channel hires.

    Returns:
        The producer's unit id and the option's price, or None when nothing
        idle offers the type -- the tier is not unlocked yet, or every
        factory is mid-queue.
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
        ):
            return (option["unit_id"], option["price"])
    return None


__all__ = ["BUNKER_TYPE", "MEDIC_TYPE", "Medic"]
