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
from rw_bot.policy.combat import FIRST_WAVE, RALLY_RADIUS
from rw_bot.policy.intel import Intel, Sighting
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackMoveOrder, attack_move_order
from rw_bot.wire.state import Entity, Sample

#: Types whose remembered sightings are raid objectives.
#:
#: Income and nothing else. Raiding the army is what the waves are for, and
#: raiding defences is what the waves die to.
INCOME_TYPES = ("extractorT1", "extractorT2", "extractorT3")


def income_objectives(intel: Intel) -> tuple[Sighting, ...]:
    """Return every remembered enemy extractor.

    Args:
        intel: The fog memory.

    Returns:
        Income sightings in identity order.
    """
    return tuple(s for s in intel.remembered() if s["type_name"] in INCOME_TYPES)


class Raider:
    """Keeps one small party assaulting remembered enemy income.

    The same shape as the other stateful controllers: decisions stay in pure
    reads, and what lives here is the memory between observations -- who is in
    the party, which objective it is on, and what has already been ordered
    ([[issuing-orders]]).

    Attributes:
        size: Party size, public because the campaign arbitrates the draft
            against it -- surplus is the wave gate's need plus this.
        raids: Objectives assaulted so far, for the report.
        marches: Outbound member-orders sent so far, for the report. The
            figure that would have convicted v1 on its first scorecard: its
            `raids` read 2-6 while dozens of lone replacements marched,
            because re-drafts against the same objective counted nothing.
    """

    def __init__(self, size: int = FIRST_WAVE) -> None:
        """Open a raider.

        Args:
            size: Party size. Defaults to the engine's own first-group size:
                below it the engine's AI calls a force a trickle, and so does
                ours ([[engine-ai-triggers]]).
        """
        self.size = size
        self.raids = 0
        self.marches = 0
        self._party: frozenset[int] = frozenset()
        self._objective = 0
        self._ordered: dict[int, int] = {}

    def party(self) -> frozenset[int]:
        """Return the engine ids currently drafted.

        The campaign withholds these from the wave controller: a unit cannot
        serve two commanders, and assignment is the arbitration
        ([[engine-ai-zones]]).

        Returns:
            The party, empty when there is no objective to raid.
        """
        return self._party

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
        for unit in army:
            if unit["unit_id"] not in self._party:
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
    ) -> tuple[AttackMoveOrder, ...]:
        """Advance the raid by at most one objective's worth of orders.

        The objective is the remembered extractor nearest our anchor -- the
        frontier one, reachable before the deep ones. A party member standing
        where the memory says an extractor is, seeing none, reports the death
        and the raid moves on.

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

        Returns:
            The attack-move orders to send, empty while the party is already
            en route or there is nothing remembered to raid.
        """
        objectives = income_objectives(intel)
        if not objectives:
            self._party = frozenset()
            self._objective = 0
            return ()
        anchor = find_anchor(sample, catalogue)
        if anchor is None:
            return ()

        def nearness(s: Sighting) -> tuple[float, int]:
            dx = s["x"] - anchor["x"]
            dy = s["y"] - anchor["y"]
            return (dx * dx + dy * dy, s["unit_id"])

        target = min(objectives, key=nearness)

        alive = {unit["unit_id"] for unit in army}
        survivors = sorted(self._party & alive)
        if survivors and len(survivors) < self.size:
            return self._disband(survivors, anchor)
        party = survivors
        if not party and may_draft:
            party = self._draft(army, anchor)
        self._party = frozenset(party)
        if not self._party:
            return ()

        if self._confirmed_dead(sample, army, target):
            intel.forget(target["unit_id"])
            self._objective = 0
            return ()

        if target["unit_id"] != self._objective:
            self._objective = target["unit_id"]
            self._ordered = {}
            self.raids += 1
        orders = tuple(
            attack_move_order(unit_id=member, x=target["x"], y=target["y"])
            for member in sorted(self._party)
            if self._ordered.get(member) != target["unit_id"]
        )
        for order in orders:
            self._ordered[order["unit_id"]] = target["unit_id"]
        self.marches += len(orders)
        return orders

    def _disband(self, survivors: Sequence[int], anchor: Entity) -> tuple[AttackMoveOrder, ...]:
        """Send the under-strength party home fighting, and dissolve it.

        Attack-move rather than move, because the road home crosses the same
        ground the road out did. Once home the survivors are the wave
        controller's again -- the campaign stops withholding whatever is no
        longer in the party.

        Args:
            survivors: The remaining members, in id order.
            anchor: The structure the reserve gathers at.

        Returns:
            The homeward orders.
        """
        self._party = frozenset()
        self._objective = 0
        self._ordered = {}
        return tuple(
            attack_move_order(unit_id=member, x=anchor["x"], y=anchor["y"]) for member in survivors
        )

    def _draft(self, army: Sequence[Entity], anchor: Entity) -> list[int]:
        """Pick a whole party from the units gathered at the anchor, or none.

        Args:
            army: Units available to fight.
            anchor: The structure the reserve gathers at.

        Returns:
            The new party in id order, empty when the gathering ground holds
            fewer than a party.
        """
        limit = RALLY_RADIUS**2
        gathered = sorted(
            unit["unit_id"]
            for unit in army
            if (unit["x"] - anchor["x"]) ** 2 + (unit["y"] - anchor["y"]) ** 2 <= limit
        )
        if len(gathered) < self.size:
            return []
        return gathered[: self.size]


__all__ = ["INCOME_TYPES", "Raider", "income_objectives"]
