"""Marching at the enemy before the enemy is visible.

The waves attack what is visible, and at match start nothing is -- so an
all-in doctrine had no way to *start* the fight it exists for: the first wave
released and stood at the rally point waiting for an opponent who was busy
compounding a 3.7x income multiplier. Every measured Impossible match ended
defeated or wiped without the bot ever reaching the enemy base
([[policy-holding-ground]]).

The march target is the mirror of our anchor through the resource-pool
centroid. Skirmish duel maps are symmetric by construction -- both players
get the same pools reflected through the map centre -- so the reflection of
our Command Center is the opponent's, computed from nothing but the sample.
Attack-move, not move: the point of a rush is to fight whatever stands on
the way in ([[issuing-orders]]).

Pure in the usual sense: samples in, orders out, the campaign sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackMoveOrder, attack_move_order
from rw_bot.wire.state import Entity, Sample


def mirror_point(sample: Sample, catalogue: Mapping[str, UnitStats]) -> tuple[float, float] | None:
    """Return the reflection of our anchor through the pool centroid.

    The centroid of *all* resource pools approximates the map centre on the
    symmetric skirmish maps this bot plays, and reflecting the anchor through
    it lands on the opponent's start. No fog is consulted and nothing is
    remembered: the answer is pure geometry over what every sample carries.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the anchor.

    Returns:
        The estimated enemy start, or None when there is no anchor or no
        pool to reflect through.
    """
    anchor = find_anchor(sample, catalogue)
    pools = sample["pools"]
    if anchor is None or not pools:
        return None
    centre_x = sum(pool["x"] for pool in pools) / len(pools)
    centre_y = sum(pool["y"] for pool in pools) / len(pools)
    return (2.0 * centre_x - anchor["x"], 2.0 * centre_y - anchor["y"])


class Rusher:
    """Sends released units at the estimated enemy start until contact.

    The same shape as the other stateful controllers: the decision stays in
    :func:`mirror_point`, and what lives here is the memory of who has
    already been sent ([[issuing-orders]]).

    Attributes:
        marches: Outbound march orders sent so far, for the report -- pooled
            with the raid's own count, because both answer "how many units
            were sent across the map on purpose".
    """

    def __init__(self) -> None:
        """Open a rusher."""
        self.marches = 0
        self._ordered: set[int] = set()

    def ordered(self) -> frozenset[int]:
        """Return every unit this rusher has ever marched.

        The strike force's roster. A committed march that stays in the
        engagement pool is re-tasked onto the first thing it meets -- the
        income-aimed dump was wiped with its dip unmoved because every
        marcher stopped to fight the army it was built to walk past. The
        campaign withholds these ids from the engagement exactly as it
        withholds the raid party ([[policy-combat]], log 2026-07-31).

        Returns:
            The marched unit ids.
        """
        return frozenset(self._ordered)

    def march(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        released: Sequence[Entity],
        targets_visible: bool,
        *,
        force: bool = False,
    ) -> tuple[AttackMoveOrder, ...]:
        """Order every newly released unit at the enemy start, once each.

        Only while nothing is visible to fight: the moment contact happens,
        the engagement policy owns the released units and re-tasks them onto
        real targets -- the engine runs the newest waypoint, so no unwinding
        is needed here ([[policy-combat]]). Once per unit, because re-issuing
        a waypoint at the sampling rate resets the walk.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for the anchor.
            released: Units the wave controller has cleared to attack.
            targets_visible: Whether the combat policy has anything real to
                shoot at this observation.
            force: March regardless of contact. The all-in's override: its
                dump exists to cross the map, and at Impossible something
                is ALWAYS visible at home -- gated on contact, the first
                all-in probe never marched at all ([[policy-combat]]).

        Returns:
            The attack-move orders to send, in id order.
        """
        if targets_visible and not force:
            return ()
        point = mirror_point(sample, catalogue)
        if point is None:
            return ()
        # A forced march is aimed at the income field, not the start point.
        # Three time-triggered dumps crossed the map, traded up to even
        # against the standing army, and dented the economy by nothing --
        # dips 1,600-2,550 against six-figure rivals -- because everything
        # walked at one point and fought whatever held it. The army is the
        # 3.7x-replaceable thing; the extractors are what pays for it, they
        # stand on pools the sample already carries, and the marchers deal
        # themselves across the ones nearest the enemy start (log
        # 2026-07-31).
        posts = _income_posts(sample, point) if force else (point,)
        fresh = [u for u in sorted(released, key=_by_id) if u["unit_id"] not in self._ordered]
        orders = tuple(
            attack_move_order(
                unit_id=unit["unit_id"],
                x=posts[index % len(posts)][0],
                y=posts[index % len(posts)][1],
            )
            for index, unit in enumerate(fresh)
        )
        for order in orders:
            self._ordered.add(order["unit_id"])
        self.marches += len(orders)
        return orders


#: How many of the enemy's nearest pools a forced march spreads across.
#:
#: Few enough that each strike group keeps a first wave's weight, many
#: enough that one turret cluster cannot hold the whole dump.
INCOME_POSTS: Final = 4

#: How far from the enemy start a pool still counts as theirs.
#:
#: Generous by half a map: the near half of the pool field is what feeds
#: them, and a marcher sent to a contested middle pool is still burning
#: income they would otherwise take.
INCOME_REACH: Final = 1400.0


def _income_posts(
    sample: Sample,
    start: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Return the enemy-side pool points a forced march deals itself across.

    Args:
        sample: One observation of the world.
        start: The estimated enemy start.

    Returns:
        Up to :data:`INCOME_POSTS` pool points nearest the start, or the
        start alone when no pool stands within reach.
    """

    def nearness(pool_point: tuple[float, float]) -> float:
        return (pool_point[0] - start[0]) ** 2 + (pool_point[1] - start[1]) ** 2

    points = sorted(
        ((pool["x"], pool["y"]) for pool in sample["pools"]),
        key=nearness,
    )
    near = [p for p in points if nearness(p) <= INCOME_REACH * INCOME_REACH]
    return tuple(near[:INCOME_POSTS]) or (start,)


def _by_id(unit: Entity) -> int:
    """Order units by engine id, the ordering every draft in this codebase uses."""
    return unit["unit_id"]


__all__ = ["Rusher", "mirror_point"]
