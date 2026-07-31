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

    def march(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        released: Sequence[Entity],
        targets_visible: bool,
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

        Returns:
            The attack-move orders to send, in id order.
        """
        if targets_visible:
            return ()
        point = mirror_point(sample, catalogue)
        if point is None:
            return ()
        orders = tuple(
            attack_move_order(unit_id=unit["unit_id"], x=point[0], y=point[1])
            for unit in sorted(released, key=_by_id)
            if unit["unit_id"] not in self._ordered
        )
        for order in orders:
            self._ordered.add(order["unit_id"])
        self.marches += len(orders)
        return orders


def _by_id(unit: Entity) -> int:
    """Order units by engine id, the ordering every draft in this codebase uses."""
    return unit["unit_id"]


__all__ = ["Rusher", "mirror_point"]
