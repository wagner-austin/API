"""The party discipline both detached forces hold.

The raid learned these rules the expensive way -- v1 topped its party up
one recruit at a time and each replacement attack-moved across the map
alone, forever, refuted 0/12 (log 2026-07-29) -- and the hunt inherits
them rather than re-learning them: **drafted whole, from the gathered**
(a fresh party is only taken from units standing within the rally radius
of the anchor, lowest id first, so two runs of one seed draft
identically), and **a party or nothing** (survivors below strength
disband and attack-move home, fighting their way back rather than
standing where the party broke).

Extracted from :class:`~rw_bot.policy.raid.Raider` when the hunt became
the second holder of the same rules ([[policy-raid]]); a second copy
would have drifted on exactly the questions v1 settled.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.policy.combat import RALLY_RADIUS
from rw_bot.wire.command import AttackMoveOrder, attack_move_order
from rw_bot.wire.state import Entity


def draft_gathered(army: Sequence[Entity], anchor: Entity, size: int) -> list[int]:
    """Pick a whole party from the units gathered at the anchor, or none.

    Args:
        army: Units available to fight.
        anchor: The structure the reserve gathers at.
        size: The party size a draft must fill.

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
    if len(gathered) < size:
        return []
    return gathered[:size]


def homeward(survivors: Sequence[int], anchor: Entity) -> tuple[AttackMoveOrder, ...]:
    """Send an under-strength party home fighting.

    Attack-move rather than move, because the road home crosses the same
    ground the road out did. Once home the survivors are the wave
    controller's again -- the campaign stops withholding whatever is no
    longer in a party.

    Args:
        survivors: The remaining members, in id order.
        anchor: The structure the reserve gathers at.

    Returns:
        The homeward orders.
    """
    return tuple(
        attack_move_order(unit_id=member, x=anchor["x"], y=anchor["y"]) for member in survivors
    )


__all__ = ["draft_gathered", "homeward"]
