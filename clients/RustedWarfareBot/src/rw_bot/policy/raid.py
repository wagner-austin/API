"""Attacking the opponent's income where intel remembers it standing.

Every Very Hard non-win now ends the same way: our economy holds, theirs
compounds, and five to eight enemy builders rebuild whatever the waves kill
([[policy-holding-ground]]). The waves cannot fix that -- they attack what is
visible near the army, and the rebuild engine is extractors standing in the
fog. The community corpus treats harassing income as ordinary play; this bot
has never once made the opponent's economy the target
([[community-play-strategies]]).

The raid is the composition of two proven parts: the intel memory knows where
enemy extractors stood ([[policy-loop]]), and attack-move fights its way to a
point ([[community-play-strategies]]). A small party -- the engine's own
first-group size -- is drafted from the army and sent at the nearest
remembered extractor; a raider standing on the memory of one that is no
longer there reports the death to the memory and moves to the next.

**V1 of this idea was refuted 0/12 and its rules are v2's spine.** The party
used to top itself back up one recruit at a time, and each recruit
attack-moved across the map alone -- a one-unit trickle into a fortified
base, issued forever. Raid arms reinforced as much as their control and
ended with half the army value, kills no higher, extractors bleeding
mid-game (log: 2026-07-29, "raid v1 refuted at 0/12"). So v2 holds the
waves' own discipline: a party reduced below the size that makes one is not
one any more -- it disbands and fights its way home -- and a fresh party is
drafted whole, from units already gathered at the anchor, and only when the
army holds more than the wave gate needs. Whether the army can *spare* a
party is the campaign's call, made against the wave controller's own figure
([[policy-raid]]).

Pure in the usual sense: samples and memory in, orders out, and the campaign
sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import RALLY_RADIUS
from rw_bot.policy.intel import Intel, Sighting
from rw_bot.policy.party import Detachment, draft_gathered
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackMoveOrder
from rw_bot.wire.state import Entity, Sample

#: Types whose remembered sightings are raid objectives.
#:
#: Income and nothing else. Raiding the army is what the waves are for, and
#: raiding defences is what the waves die to.
INCOME_TYPES = ("extractorT1", "extractorT2", "extractorT3")

#: Types whose remembered sightings are RAZE objectives -- the displacement
#: regime's targets (log 2026-09-11). The attrition anatomy's number-one
#: killer is massed ``c_artillery`` over turret lines, and every composition
#: answer to it measured flat with its mechanism verified; the displacement
#: hypothesis strikes the line's PRODUCTION instead. Factories and nothing
#: else, for the raid's own reasons: the army is the waves' business, and
#: defences are what parties die to.
PRODUCTION_TYPES = (
    "landFactory",
    "airFactory",
    "seaFactory",
    "mechFactory",
    "experimentalLandFactory",
)


def income_objectives(intel: Intel) -> tuple[Sighting, ...]:
    """Return every remembered enemy extractor.

    Args:
        intel: The fog memory.

    Returns:
        Income sightings in identity order.
    """
    return tuple(s for s in intel.remembered() if s["type_name"] in INCOME_TYPES)


def production_objectives(intel: Intel) -> tuple[Sighting, ...]:
    """Return every remembered enemy factory.

    The raze regime raids only what intel remembers, exactly as the income
    raid does: a fogged factory is not a target, and an empty memory stands
    the party down rather than substituting a lesser objective.

    Args:
        intel: The fog memory.

    Returns:
        Production sightings in identity order.
    """
    return tuple(s for s in intel.remembered() if s["type_name"] in PRODUCTION_TYPES)


class Raider(Detachment):
    """Keeps one small party assaulting remembered enemy income.

    The bookkeeping is :class:`~rw_bot.policy.party.Detachment`'s; what is
    the raid's own is the objective set, and the confirmation that reports
    a dead objective back to the memory.
    """

    def _confirmed_dead(self, sample: Sample, army: Sequence[Entity], target: Sighting) -> bool:
        """Report whether a party member stands on the memory and sees nothing.

        The arrival test is the engine's own rally radius, for the usual
        reason: when has a unit finished walking is one question with one
        answer ([[engine-ai-zones]]).

        Args:
            sample: One observation of the world.
            army: Units available to fight.
            target: The objective under assault.

        Returns:
            True when the sighting is confirmed gone.
        """
        visible = {e["unit_id"] for e in sample["entities"] if e["hostile"]}
        if target["unit_id"] in visible:
            return False
        limit = RALLY_RADIUS**2
        party = self.party()
        for unit in army:
            if unit["unit_id"] not in party:
                continue
            d2 = (unit["x"] - target["x"]) ** 2 + (unit["y"] - target["y"]) ** 2
            if d2 <= limit:
                return True
        return False

    def strike(
        self,
        sample: Sample,
        intel: Intel,
        army: Sequence[Entity],
        catalogue: Mapping[str, UnitStats],
        may_draft: bool,
        raze_now: bool,
    ) -> tuple[AttackMoveOrder, ...]:
        """Advance the raid by at most one objective's worth of orders.

        The objective is the remembered extractor nearest our anchor -- the
        frontier one, reachable before the deep ones -- or, when ``raze_now``,
        the remembered FACTORY nearest it: the displacement regime retasks
        the same party at the standoff line's production (Doctrine.raze).
        A party member standing where the memory says the objective is,
        seeing none, reports the death and the raid moves on.

        **A party or nothing.** Survivors below strength disband and
        attack-move home -- fighting their way back to the reserve rather
        than standing where the party broke. Replacing members one at a time
        is what v1 died of: each recruit crossed the map alone, forever
        (log: 2026-07-29).

        **Drafted whole, from the gathered.** A fresh party is only taken
        from units standing within the rally radius of the anchor -- the
        reserve's own gathering ground -- so it starts together the way a
        wave does, instead of forming up en route by lowest id. Lowest id
        still orders the draft, so two runs of one seed draft identically.

        Args:
            sample: One observation of the world.
            intel: The fog memory, corrected in place on confirmations.
            army: Units available to fight, scouts already excluded.
            catalogue: Unit stats by type name, for the anchor.
            may_draft: Whether the campaign judges the army able to spare a
                fresh party -- the wave gate's need plus the party size. A
                party already out is managed regardless: the gate arbitrates
                drafting, not the raid in progress.
            raze_now: Whether the displacement regime is live -- objectives
                become remembered factories instead of extractors. The
                campaign computes this from the ``raze`` sample gate.

        Returns:
            The attack-move orders to send, empty while the party is already
            en route or there is nothing remembered to raid.
        """
        objectives = production_objectives(intel) if raze_now else income_objectives(intel)
        if not objectives:
            self.muster(())
            self.forget_objective()
            return ()
        anchor = find_anchor(sample, catalogue)
        if anchor is None:
            return ()

        def nearness(s: Sighting) -> tuple[float, int]:
            dx = s["x"] - anchor["x"]
            dy = s["y"] - anchor["y"]
            return (dx * dx + dy * dy, s["unit_id"])

        target = min(objectives, key=nearness)

        survivors = self.survivors(army)
        if survivors and len(survivors) < self.size:
            return self.disband(survivors, anchor)
        party = survivors
        if not party and may_draft:
            party = draft_gathered(army, anchor, self.size)
        if not self.muster(party):
            return ()

        if self._confirmed_dead(sample, army, target):
            intel.forget(target["unit_id"])
            self.forget_objective()
            return ()

        return self.advance(target["unit_id"], target["x"], target["y"])


__all__ = [
    "INCOME_TYPES",
    "PRODUCTION_TYPES",
    "Raider",
    "income_objectives",
    "production_objectives",
]
