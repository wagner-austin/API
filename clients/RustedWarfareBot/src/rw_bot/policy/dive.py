"""Closing on the gun that outranges the line.

Every Very Hard opener seed is the same opening: the enemy fields three to
twelve ``c_artillery`` behind a screen of scouts and hover tanks, and no
winning seed faces a single piece (pinbase48's binary split, log
2026-09-12). The army answers it the only way the waves know -- the
kill-groups take the nearest target, which is the screen, and trade into
the 290-reach guns behind it until the line stalls at eleven to thirteen
pieces while the enemy's income compounds (the loss-table anatomy, log
2026-09-13). Every composition answer measured flat: a share of our own
artillery joins too late or trades even. The dive is the tactical answer
the community corpus plays against a battery: **a fast party goes and
touches the gun**, because a piece that outranges the line by 130 world
units is fragile and slow, and the hover slot exists for that leg.

The party holds the raid's discipline (:mod:`rw_bot.policy.party`):
drafted whole from the gathered, disbanded home under strength,
arbitrated by the campaign against the wave gate's opening rung the way
the hunt is. What is the dive's own is the quarry and the draft: the
quarry is the VISIBLE hostile ground mover whose land gun starts beyond
every land gun the army fields -- the outranged clause's own membership
test, lifted rather than re-stated (:func:`~rw_bot.policy.counter.
outranges`) -- nearest the party's own centre; and the draft takes the
FASTEST gathered units, because the leg is a race across the gun's reach.
No memory fallback and no lesser objective: with no outranging gun in
sight a standing party fights its way home and no party is raised, which
is what makes a zero-artillery seed the champion's match bit for bit --
the property the static composition merge could not have and the
conditional join could only approximate.

Pure in the usual sense: samples and memory in, orders out, and the
campaign sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.counter import land_reach, outranges
from rw_bot.policy.hunt import centroid
from rw_bot.policy.party import Detachment, draft_fastest
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackMoveOrder
from rw_bot.wire.state import Entity, Sample


def outranging_guns(
    army: Sequence[Entity],
    targets: Sequence[Entity],
    profiles: Mapping[str, CombatProfile],
    catalogue: Mapping[str, UnitStats],
) -> tuple[Entity, ...]:
    """Return the visible hostile ground movers that outrange the army's guns.

    The dive's objective class. The reach is the longest land gun among
    the types the ARMY fields now -- not the doctrine's mix -- because the
    question is whether the pieces standing today can answer the gun, and
    an army with no land gun yet has no standoff to close.

    Args:
        army: Units available to fight, scouts already excluded.
        targets: The hostile entities visible this observation.
        profiles: Combat profiles by type name, for reach and layers.
        catalogue: Unit stats by type name, for the speed that tells a
            building from a unit.

    Returns:
        The outranging movers, in the order seen.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            a fielded type or a mobile ground hostile.
    """
    reach = land_reach(tuple(unit["type_name"] for unit in army), profiles)
    if reach is None:
        return ()
    return tuple(t for t in targets if outranges(t, reach, profiles, catalogue))


class Diver(Detachment):
    """Keeps one fast party closing on the nearest outranging gun.

    The bookkeeping is :class:`~rw_bot.policy.party.Detachment`'s; what is
    the dive's own is the quarry (only a gun that outranges the line) and
    the draft (the fastest gathered).
    """

    def dive(
        self,
        sample: Sample,
        army: Sequence[Entity],
        targets: Sequence[Entity],
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        may_draft: bool,
    ) -> tuple[AttackMoveOrder, ...]:
        """Advance the dive by at most one objective's worth of orders.

        The objective is the outranging gun nearest the party's own centre
        -- pursuit measures from where the party stands, the anchor when
        one would be raised. With no such gun in sight a standing party
        is sent home fighting and dissolved, and none is raised: the dive
        answers a gun that is there, never one that might be. Ties break
        on unit id, so two runs of one seed dive identically.

        Args:
            sample: One observation of the world.
            army: Units available to fight, scouts already excluded.
            targets: The hostile entities visible this observation.
            catalogue: Unit stats by type name, for the anchor, the draft's
                speed and the mover test.
            profiles: Combat profiles by type name, for the reach test.
            may_draft: Whether the campaign judges the army able to spare
                a fresh party. A party already out is managed regardless.

        Returns:
            The attack-move orders to send, empty while the party is
            already closing on its objective or there is nothing to dive.
        """
        anchor = find_anchor(sample, catalogue)
        if anchor is None:
            return ()
        survivors = self.survivors(army)
        if survivors and len(survivors) < self.size:
            return self.disband(survivors, anchor)
        guns = outranging_guns(army, targets, profiles, catalogue)
        if not guns:
            # Nothing outranges the line: a standing party is a diversion
            # with no objective, so it goes home and rejoins the reserve.
            return self.disband(survivors, anchor)
        if survivors:
            members = [unit for unit in army if unit["unit_id"] in self.party()]
            centre_x, centre_y = centroid(members)
        else:
            centre_x, centre_y = anchor["x"], anchor["y"]

        def nearness(entity: Entity) -> tuple[float, int]:
            dx = entity["x"] - centre_x
            dy = entity["y"] - centre_y
            return (dx * dx + dy * dy, entity["unit_id"])

        quarry = min(guns, key=nearness)
        party = survivors
        if not party and may_draft:
            party = draft_fastest(army, anchor, self.size, catalogue)
        if not self.muster(party):
            return ()
        return self.advance(quarry["unit_id"], quarry["x"], quarry["y"])


__all__ = ["Diver", "outranging_guns"]
